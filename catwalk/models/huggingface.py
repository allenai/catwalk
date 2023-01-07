import warnings
from typing import Any, Iterator, Dict, Sequence, Tuple, List, cast, Optional

import more_itertools
import torch
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from tango.integrations.torch.util import resolve_device
from transformers import (AutoModelForMultipleChoice,
                          AutoTokenizer,
                          AutoModelForQuestionAnswering,
                          QuestionAnsweringPipeline, PreTrainedModel, PreTrainedTokenizer,
                          AutoModelForSequenceClassification)
from torchmetrics.functional.classification import multiclass_accuracy
from transformers.tokenization_utils_base import LARGE_INTEGER

from catwalk import cached_transformers
from catwalk.model import Model, UnsupportedTaskError, TrainableModel, Instance
from catwalk.task import Task, InstanceFormat, WithAnswerOptionsMixin
from catwalk.tasks.huggingface import HFQAInstance, HFMCInstance, HFClassificationInstance


@Model.register("catwalk::hf")
class HFAutoModel(Model):
    VERSION = "004acc"

    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        
    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        device = resolve_device()

        if task.has_instance_conversion(InstanceFormat.HF_MC):
            mc_instances = cast(Sequence[HFMCInstance], self._convert_instances(instances, InstanceFormat.HF_MC, task))
            model = cached_transformers.get(AutoModelForMultipleChoice, self.pretrained_model_name_or_path, False).to(device)
            tokenizer = cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_model_name_or_path)
            return self._predict_mc(mc_instances, model, tokenizer, batch_size=batch_size)
        elif task.has_instance_conversion(InstanceFormat.HF_QA):
            qa_instances = cast(Sequence[HFQAInstance], self._convert_instances(instances, InstanceFormat.HF_QA, task))
            model = cached_transformers.get(AutoModelForQuestionAnswering, self.pretrained_model_name_or_path, False).to(device)
            tokenizer = cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_model_name_or_path)
            return self._predict_qa(qa_instances, model, tokenizer, batch_size=batch_size)
        elif task.has_instance_conversion(InstanceFormat.HF_CLASSIFICATION):
            classification_instances = cast(
                Sequence[HFClassificationInstance],
                self._convert_instances(instances, InstanceFormat.HF_CLASSIFICATION, task))
            model = cached_transformers.get(
                AutoModelForSequenceClassification,
                self.pretrained_model_name_or_path, False
            ).to(device)

            assert isinstance(task, WithAnswerOptionsMixin)
            model_num_labels = model.config.num_labels
            if model_num_labels == 1:
                model_num_labels = 2
            if model_num_labels != len(task.answer_options):
                warnings.warn(f"Model has {model.config.num_labels} labels, but task has {len(task.answer_options)} possible answers.")

            tokenizer = cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_model_name_or_path)
            return self._predict_classification(classification_instances, model, tokenizer, batch_size=batch_size)

        raise UnsupportedTaskError(self, task)

    @classmethod
    def _convert_instances(self, instances: Sequence[Dict[str, Any]], instance_format, task) -> MappedSequence:
        return MappedSequence(task.instance_conversions[instance_format], instances)
    
    @classmethod
    def _predict_qa(
        cls,
        instances: Sequence[HFQAInstance],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        pipe = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=model.device.index)
        
        contexts = [instance.context for instance in instances]
        questions = [instance.question for instance in instances]

        pipe_results = pipe(context=contexts, question=questions, batch_size=batch_size)
        with Tqdm.tqdm(pipe_results, desc="Processing instances") as instances_tqdm:
            for instance, prediction in zip(instances, instances_tqdm):
                yield {
                    "squad_metrics": (
                        {"id": instance.id, "prediction_text": prediction["answer"]},
                        {"id": instance.id, "answers": instance.answers}
                    )
                }
                
    @classmethod
    def _predict_mc(
        cls,
        instances: Sequence[HFMCInstance],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        # There is no Huggingface pipeline for this.

        model.eval()
        with Tqdm.tqdm(instances, desc="Processing instances") as instances:
            with torch.inference_mode():
                for batch in more_itertools.chunked(instances, batch_size):
                    number_of_choices = max(len(instance.answer_choices) for instance in batch)
                    texts: List[Tuple[str, str]] = []
                    labels = []
                    for instance in batch:
                        texts.extend(
                            (instance.question, choice)
                            for choice in instance.answer_choices
                        )
                        while len(texts) % number_of_choices != 0:
                            texts.append(("", ""))  # padding in the choices dimension
                        labels.append(instance.correct_answer_index)
                    tensors = tokenizer.batch_encode_plus(
                        texts,
                        padding=True,
                        truncation="only_first",
                        return_tensors="pt",
                        pad_to_multiple_of=8,
                    )
                    results = model(
                        return_dict=True,
                        **{
                            key: tensor.view(len(batch), number_of_choices, -1).to(model.device)
                            for key, tensor in tensors.items()
                        })
                    for instance, logits in zip(batch, results.logits.detach().cpu()):
                        yield {
                            "correct_answer_index": instance.correct_answer_index,
                            "logits": logits,
                            "acc": (logits, instance.correct_answer_index),
                        }

    @classmethod
    def _predict_classification(
        cls,
        instances: Sequence[HFClassificationInstance],
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        # There is no Huggingface pipeline for this.
        # HF's TextClassification pipeline only classifies single texts, not text pairs
        model.eval()
        with Tqdm.tqdm(instances, desc="Processing instances") as instances:
            with torch.inference_mode():
                for batch in more_itertools.chunked(instances, batch_size):
                    tensors = tokenizer.batch_encode_plus(
                        [instance.text for instance in batch],
                        padding=True,
                        truncation="only_first",
                        return_tensors="pt",
                        pad_to_multiple_of=8,
                    )
                    tensors = {k: v.to(model.device) for k, v in tensors.items()}
                    results = model(return_dict=True, **tensors)
                    for instance, logits in zip(batch, results.logits.detach().cpu()):
                        yield {
                            "label": instance.label,
                            "logits": logits,
                            "acc": (logits, instance.label),
                        }

    def trainable_copy(self, **kwargs) -> "TrainableHFAutoModel":
        return TrainableHFAutoModel(self.pretrained_model_name_or_path, **kwargs)


class TrainableHFAutoModel(TrainableModel):
    VERSION = "004acc"

    def __init__(self, pretrained_model_name_or_path: str, *, num_classification_labels: Optional[int] = None):
        super().__init__(None)
        self.tokenizer = cached_transformers.get_tokenizer(AutoTokenizer, pretrained_model_name_or_path)

        # This is a bit messy because huggingface doesn't support multitask training.
        # We instantiate MC, QA, and classification models, and then we set the inner modules (usually the
        # transformer itself) to be the same.
        self.mc_model = cached_transformers.get(AutoModelForMultipleChoice, pretrained_model_name_or_path, True)
        mc_modules = dict(self.mc_model.named_children())

        NEVER_SHARED_MODULE_NAMES = {'pooler', 'dropout', 'classifier'}

        self.qa_model = cached_transformers.get(AutoModelForQuestionAnswering, pretrained_model_name_or_path, True)
        qa_modules = dict(self.qa_model.named_children())
        for name in mc_modules.keys() & qa_modules.keys() - NEVER_SHARED_MODULE_NAMES:
            self.qa_model.add_module(name, mc_modules[name])  # This overwrites the existing module.

        if num_classification_labels is None:
            self.classification_model = None
        else:
            self.classification_model = cached_transformers.get(
                AutoModelForSequenceClassification,
                pretrained_model_name_or_path,
                True,
                num_labels=num_classification_labels)
            classification_modules = dict(self.classification_model.named_children())
            for name in mc_modules.keys() & classification_modules.keys() - NEVER_SHARED_MODULE_NAMES:
                self.classification_model.add_module(name, mc_modules[name])  # This overwrites the existing module.
            self.classification_num_labels = self.classification_model.num_labels
            if self.classification_num_labels == 1:
                self.classification_num_labels = 2
            self.classification_warning_shown = False

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        if task.has_instance_conversion(InstanceFormat.HF_MC):
            mc_instances = HFAutoModel._convert_instances(
                instances,
                InstanceFormat.HF_MC,
                task)
            return HFAutoModel._predict_mc(
                mc_instances,
                self.mc_model,
                self.tokenizer,
                batch_size=batch_size)
        elif task.has_instance_conversion(InstanceFormat.HF_QA):
            qa_instances = HFAutoModel._convert_instances(
                instances,
                InstanceFormat.HF_QA,
                task)
            return HFAutoModel._predict_qa(
                qa_instances,
                self.qa_model,
                self.tokenizer,
                batch_size=batch_size)
        elif task.has_instance_conversion(InstanceFormat.HF_CLASSIFICATION):
            if self.classification_model is None:
                raise ValueError("This model must be initialized with num_classification_labels to perform classification.")
            if not self.classification_warning_shown:
                assert isinstance(task, WithAnswerOptionsMixin)
                if len(task.answer_options) != self.classification_num_labels:
                    warnings.warn(f"Model has {self.classification_num_labels} labels, but task has {len(task.answer_options)} possible answers.")
                    self.classification_warning_shown = True
            classification_instances = HFAutoModel._convert_instances(
                instances,
                InstanceFormat.HF_CLASSIFICATION,
                task)
            return HFAutoModel._predict_classification(
                classification_instances,
                self.classification_model,
                self.tokenizer,
                batch_size=batch_size)
        else:
            raise UnsupportedTaskError(self, task)

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        mc_kwargs = {}
        qa_kwargs = {}
        classification_kwargs = {}
        neither_kwargs = {}
        for key in set(kwargs.keys()):
            value = kwargs.pop(key)
            if key.startswith("mc_"):
                mc_kwargs[key[3:]] = value
            elif key.startswith("qa_"):
                qa_kwargs[key[3:]] = value
            elif key.startswith("classification_"):
                classification_kwargs[key[15:]] = value
            else:
                neither_kwargs[key] = value
        mc_kwargs.update(neither_kwargs)
        qa_kwargs.update(neither_kwargs)
        classification_kwargs.update(neither_kwargs)

        if len(mc_kwargs) > len(neither_kwargs):
            mc_results = self._forward_mc(*args, **mc_kwargs)
        else:
            mc_results = {"loss": 0.0}

        if len(qa_kwargs) > len(neither_kwargs):
            qa_results = self._forward_qa(*args, **qa_kwargs)
        else:
            qa_results = {"loss": 0.0}

        if len(classification_kwargs) > len(neither_kwargs):
            classification_results = self._forward_classification(*args, **classification_kwargs)
        else:
            classification_results = {"loss": 0.0}

        acc = []
        if "acc" in mc_results:
            acc.append(mc_results["acc"])
        if "acc" in qa_results:
            acc.append(qa_results["acc"])
        if "acc" in classification_results:
            acc.append(classification_results["acc"])
        if len(acc) <= 0:
            acc = [0.0]

        results = {
            "loss": mc_results["loss"] + qa_results["loss"] + classification_results["loss"],
            "acc": sum(acc) / len(acc)
        }
        assert not isinstance(results["loss"], float), "Loss must be a tensor. Is it possible that none of the forward() functions ran?"
        results.update({"mc_" + key: value for key, value in mc_results.items()})
        results.update({"qa_" + key: value for key, value in qa_results.items()})
        results.update({"classification_" + key: value for key, value in classification_results.items()})
        return results

    def _forward_mc(self, *args, **kwargs) -> Dict[str, Any]:
        results = self.mc_model.forward(*args, **kwargs)
        results["acc"] = multiclass_accuracy(results.logits, kwargs["labels"], num_classes=results.logits.size(-1))
        return results

    def _forward_qa(self, *args, **kwargs) -> Dict[str, Any]:
        results = self.qa_model.forward(*args, **kwargs)
        return results

    def _forward_classification(self, *args, **kwargs) -> Dict[str, Any]:
        assert self.classification_model is not None
        results = self.classification_model.forward(*args, **kwargs)
        results["acc"] = multiclass_accuracy(results.logits, kwargs["labels"], num_classes=results.logits.size(-1))
        return results

    def collate_for_training(self, instances: Sequence[Tuple[Task, Instance]]) -> Any:
        mc_instances: List[HFMCInstance] = []
        qa_instances: List[HFQAInstance] = []
        classification_instances: List[HFClassificationInstance] = []
        for task, instance in instances:
            if task.has_instance_conversion(InstanceFormat.HF_MC):
                mc_instances.append(
                    cast(
                        HFMCInstance,
                        task.convert_instance(instance, InstanceFormat.HF_MC)))
            elif task.has_instance_conversion(InstanceFormat.HF_QA):
                qa_instances.append(
                    cast(
                        HFQAInstance,
                        task.convert_instance(instance, InstanceFormat.HF_QA)))
            elif task.has_instance_conversion(InstanceFormat.HF_CLASSIFICATION):
                if self.classification_model is None:
                    raise ValueError("This model must be initialized with num_classification_labels to perform classification.")
                if not self.classification_warning_shown:
                    assert isinstance(task, WithAnswerOptionsMixin)
                    if len(task.answer_options) != self.classification_num_labels:
                        warnings.warn(f"Model has {self.classification_num_labels} labels, but task has {len(task.answer_options)} possible answers.")
                        self.classification_warning_shown = True
                classification_instances.append(
                    cast(
                        HFClassificationInstance,
                        task.convert_instance(instance, InstanceFormat.HF_CLASSIFICATION)))
            else:
                raise ValueError("I don't know how to handle this instance.")

        truncation = "longest_first" if self.tokenizer.model_max_length <= LARGE_INTEGER else False

        # build MC tensors
        result: Dict[str, torch.Tensor] = {}
        if len(mc_instances) > 0:
            number_of_choices = max(len(mc_instance.answer_choices) for mc_instance in mc_instances)
            texts: List[Tuple[str, str]] = []
            labels = []
            for mc_instance in mc_instances:
                texts.extend(
                    (mc_instance.question, choice)
                    for choice in mc_instance.answer_choices
                )
                while len(texts) % number_of_choices != 0:
                    texts.append(("", ""))  # padding in the choices dimension
                labels.append(mc_instance.correct_answer_index)
            tensors = self.tokenizer.batch_encode_plus(
                texts,
                padding=True,
                truncation=truncation,
                return_tensors="pt",
                pad_to_multiple_of=8,
            )
            for key, tensor in tensors.items():
                result["mc_" + key] = tensor.view(
                    len(mc_instances),
                    number_of_choices,
                    -1
                ).to(self.mc_model.device)
            result["mc_labels"] = torch.tensor(labels, dtype=torch.long, device=self.mc_model.device)

        # build QA tensors
        if len(qa_instances) > 0:
            raise NotImplementedError("Sorry, training for QA is not implemented yet. Please make a PR!")

        # build classification tensors
        if len(classification_instances) > 0:
            assert self.classification_model is not None
            tensors = self.tokenizer.batch_encode_plus(
                [instance.text for instance in classification_instances],
                padding=True,
                truncation=truncation,
                return_tensors="pt",
                pad_to_multiple_of=8,
            )
            for key, tensor in tensors.items():
                result["classification_" + key] = tensor.to(self.classification_model.device)
            result["classification_labels"] = torch.tensor(
                [instance.label for instance in classification_instances],
                dtype=torch.long,
                device=self.classification_model.device)

        return result

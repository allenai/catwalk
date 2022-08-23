from typing import Any, Iterator, Dict, Sequence, Tuple, List, cast

import more_itertools
import torch
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from tango.integrations.torch.util import resolve_device
from transformers import (AutoModelForMultipleChoice,
                          AutoTokenizer,
                          AutoModelForQuestionAnswering,
                          QuestionAnsweringPipeline, PreTrainedModel, PreTrainedTokenizer)

from catwalk import cached_transformers
from catwalk.model import Model, UnsupportedTaskError, TrainableModel, Instance
from catwalk.task import Task, InstanceFormat
from catwalk.tasks.huggingface import HFQAInstance, HFMCInstance


@Model.register("catwalk::hf")
class HFAutoModel(Model):
    VERSION = "002var"

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
        for instance, prediction in zip(instances, Tqdm.tqdm(pipe_results, desc="Processing instances")):
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

        instances = Tqdm.tqdm(instances, desc="Processing instances")
        model.eval()
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

    def trainable_copy(self) -> "TrainableHFAutoModel":
        return TrainableHFAutoModel(self.pretrained_model_name_or_path)


class TrainableHFAutoModel(TrainableModel):
    VERSION = "002var"

    def __init__(self, pretrained_model_name_or_path: str):
        super().__init__(None)
        self.tokenizer = cached_transformers.get_tokenizer(AutoTokenizer, pretrained_model_name_or_path)

        # This is a bit messy because huggingface doesn't support multi-task training.
        # We instantiate both MC and QA models, and then we set the inner modules (usually the transformer itself)
        # to be the same.
        self.mc_model = cached_transformers.get(AutoModelForMultipleChoice, pretrained_model_name_or_path, True)
        self.qa_model = cached_transformers.get(AutoModelForQuestionAnswering, pretrained_model_name_or_path, True)
        mc_modules = dict(self.mc_model.named_children())
        qa_modules = dict(self.qa_model.named_children())
        for name in mc_modules.keys() & qa_modules.keys():
            self.qa_model.add_module(name, mc_modules[name])  # This overwrites the existing module.

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        if task.has_instance_conversion(InstanceFormat.HF_MC):
            mc_instances = HFAutoModel._convert_instances(instances, InstanceFormat.HF_MC, task)
            return HFAutoModel._predict_mc(mc_instances, self.mc_model, self.tokenizer, batch_size=batch_size)
        elif task.has_instance_conversion(InstanceFormat.HF_QA):
            qa_instances = HFAutoModel._convert_instances(instances, InstanceFormat.HF_QA, task)
            return HFAutoModel._predict_qa(qa_instances, self.qa_model, self.tokenizer, batch_size=batch_size)

        raise UnsupportedTaskError(self, task)

    def forward(self, *args, **kwargs) -> Dict[str, Any]:
        mc_kwargs = {}
        qa_kwargs = {}
        neither_kwargs = {}
        for key in set(kwargs.keys()):
            value = kwargs.pop(key)
            if key.startswith("mc_"):
                mc_kwargs[key[3:]] = value
            elif key.startswith("qa_"):
                qa_kwargs[key[3:]] = value
            else:
                neither_kwargs[key] = value
        mc_kwargs.update(neither_kwargs)
        qa_kwargs.update(neither_kwargs)

        if len(mc_kwargs) > len(neither_kwargs):
            mc_results = self._forward_mc(*args, **mc_kwargs)
        else:
            mc_results = {"loss": 0.0}

        if len(qa_kwargs) > len(neither_kwargs):
            qa_results = self._forward_qa(*args, **qa_kwargs)
        else:
            qa_results = {"loss": 0.0}

        results = {"loss": mc_results["loss"] + qa_results["loss"]}
        assert not isinstance(results["loss"], float), "Loss must be a tensor. Is it possible that none of the forward() functions ran?"
        results.update({"mc_" + key: value for key, value in mc_results.items()})
        results.update({"qa_" + key: value for key, value in qa_results.items()})
        return results

    def _forward_mc(self, *args, **kwargs) -> Dict[str, Any]:
        results = self.mc_model.forward(*args, **kwargs)
        return results

    def _forward_qa(self, *args, **kwargs) -> Dict[str, Any]:
        results = self.qa_model.forward(*args, **kwargs)
        return results

    def collate_for_training(self, instances: Sequence[Tuple[Task, Instance]]) -> Any:
        mc_instances: List[HFMCInstance] = []
        qa_instances: List[HFQAInstance] = []
        for task, instance in instances:
            if task.has_instance_conversion(InstanceFormat.HF_MC):
                mc_instances.append(cast(HFMCInstance, task.convert_instance(instance, InstanceFormat.HF_MC)))
            elif task.has_instance_conversion(InstanceFormat.HF_QA):
                qa_instances.append(cast(HFQAInstance, task.convert_instance(instance, InstanceFormat.HF_QA)))
            else:
                raise ValueError("I don't know how to handle this instance.")

        # build MC instances
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
            truncation="longest_first",
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
        mc_tensors = {
            "mc_" + key: tensor.view(len(mc_instances), number_of_choices, -1).to(self.mc_model.device)
            for key, tensor in tensors.items()
        }
        mc_tensors["mc_labels"] = torch.tensor(labels, dtype=torch.long, device=self.mc_model.device)

        # build qa tensors
        if len(qa_instances) > 0:
            raise NotImplementedError("Sorry, training for QA is not implemented yet. Please make a PR!")

        return mc_tensors

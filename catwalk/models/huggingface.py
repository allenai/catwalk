from typing import Any, Iterator, Dict, Sequence, Tuple, List

import more_itertools
import torch
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from tango.integrations.torch.util import resolve_device
from transformers import AutoModelForMultipleChoice, AutoTokenizer, AutoModelForQuestionAnswering

from catwalk import cached_transformers
from catwalk.model import Model, UnsupportedTaskError
from catwalk.task import Task, InstanceFormat
from catwalk.models.utils_qa import postprocess_qa_predictions


@Model.register("catwalk::hf")
class HFAutoModel(Model):
    VERSION = "001"

    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        
    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        if task.has_instance_conversion(InstanceFormat.HF_MC):
            return self._predict_mc(task, instances, batch_size=batch_size)
        elif task.has_instance_conversion(InstanceFormat.HF_QA):
            return self._predict_qa(task, instances, batch_size=batch_size)
        
        raise UnsupportedTaskError(self, task)

    
    def _convert_instances(self, instances: Sequence[Dict[str, Any]], instance_format, task) -> MappedSequence:
        return MappedSequence(lambda instance: task.convert_instance(instance, instance_format), instances)
    
    def _predict_qa(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        instances = self._convert_instances(instances, InstanceFormat.HF_QA, task)
        
        device = resolve_device()
        model = cached_transformers.get(AutoModelForQuestionAnswering, self.pretrained_model_name_or_path, False).eval().to(device)
        tokenizer = cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_model_name_or_path)
        pad_on_right = tokenizer.padding_side == "right"

        instances = Tqdm.tqdm(instances, desc="Processing instances")
        with torch.inference_mode():
            for batch in more_itertools.chunked(instances, batch_size):
                texts: List[Tuple[str, str]] = [(instance.question, instance.context) if pad_on_right else (instance.context, instance.question) for instance in batch]

                tensors = tokenizer.batch_encode_plus(
                    texts,
                    padding=True,
                    truncation="only_second" if pad_on_right else "only_first", 
                    pad_to_multiple_of=8,
                    stride=128,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True
                )
                
                sample_mapping = tensors.pop("overflow_to_sample_mapping")           
                offsets_mapping = tensors.pop("offset_mapping")
                
                results = model(
                    **{key: torch.tensor(tensor).to(device) for key, tensor in tensors.items()},
                    return_dict=True
                )
                
                tensors["offset_mapping"] = offsets_mapping
                tensors["example_id"] = [batch[sample_mapping[i]].id for i in range(len(tensors["input_ids"]))]  
                
                predictions = postprocess_qa_predictions(batch, tensors, (results.start_logits.cpu().numpy(), results.end_logits.cpu().numpy()))
                
                for instance in batch:
                    yield {
                        "correct_answers": instance.answers,
                        "predicted_answer": predictions[instance.id],
                        "squad_metrics": ({"id": instance.id, "prediction_text": predictions[instance.id]}, {"id": instance.id, "answers": instance.answers})
                    }

    def _predict_mc(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        instances = self._convert_instances(instances, InstanceFormat.HF_MC, task)

        # There is no Huggingface pipeline for this.
        device = resolve_device()
        model = cached_transformers.get(AutoModelForMultipleChoice, self.pretrained_model_name_or_path, False).eval().to(device)
        tokenizer = cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_model_name_or_path)

        instances = Tqdm.tqdm(instances, desc="Processing instances")
        with torch.inference_mode():
            for batch in more_itertools.chunked(instances, batch_size):
                number_of_choices = None
                texts: List[Tuple[str, str]] = []
                labels = []
                for instance in batch:
                    if number_of_choices is None:
                        number_of_choices = len(instance.answer_choices)
                    else:
                        assert len(instance.answer_choices) == number_of_choices
                    texts.extend(
                        (instance.question, choice)
                        for choice in instance.answer_choices
                    )
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
                    **{key: tensor.view(len(batch), number_of_choices, -1).to(device) for key, tensor in tensors.items()})
                for instance, logits in zip(batch, results.logits.detach().cpu()):
                    yield {
                        "correct_answer_index": instance.correct_answer_index,
                        "logits": logits,
                        "acc": (logits, instance.correct_answer_index),
                    }
                    
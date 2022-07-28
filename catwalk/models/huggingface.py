from typing import Any, Iterator, Dict, Sequence, Tuple, List

import more_itertools
import torch
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from tango.integrations.torch.util import resolve_device
from transformers import (AutoModelForMultipleChoice, 
                          AutoTokenizer, 
                          AutoModelForQuestionAnswering, 
                          QuestionAnsweringPipeline)

from catwalk import cached_transformers
from catwalk.model import Model, UnsupportedTaskError
from catwalk.task import Task, InstanceFormat

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
        converted_instances = self._convert_instances(instances, InstanceFormat.HF_QA, task)
        
        device = resolve_device()
        model = cached_transformers.get(AutoModelForQuestionAnswering, self.pretrained_model_name_or_path, False)
        tokenizer = cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_model_name_or_path)
        pipe = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer, device=device.index or torch.cuda.current_device() if device.type == "cuda" else -1)
        
        converted_instances = Tqdm.tqdm(converted_instances, desc="Processing instances")
        for batch in more_itertools.chunked(converted_instances, batch_size):
            contexts_batch = [instance.context for instance in batch]
            questions_batch = [instance.question for instance in batch]
            outputs = pipe(context=contexts_batch, question=questions_batch)
            # make outputs a list in the case where there is only one instance
            if isinstance(outputs, dict):
                outputs = [outputs]
            for instance, prediction in zip(batch, outputs):
                yield {
                    "squad_metrics": ({"id": instance.id, "prediction_text": prediction["answer"]}, {"id": instance.id, "answers": instance.answers})
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
                    
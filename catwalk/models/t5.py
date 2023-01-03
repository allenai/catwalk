from abc import ABC
from typing import Sequence, Dict, Any, Iterator, Optional

import more_itertools
import torch
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5TokenizerFast

from catwalk import cached_transformers
from catwalk.task import Task, InstanceFormat
from catwalk.model import Model, UnsupportedTaskError

_true_tensor = torch.tensor([True])
_false_tensor = torch.tensor([False])


class T5Model(Model, ABC):
    def get_model(self) -> T5ForConditionalGeneration:
        raise NotImplementedError

    def get_tokenizer(self) -> T5TokenizerFast:
        raise NotImplementedError
    
    def _predict_qa(self, task: Task, instances: Sequence[Dict[str, Any]], batch_size: int = 32) -> Iterator[Dict[str, Any]]:
        qas = MappedSequence(task.instance_conversions[InstanceFormat.HF_QA], instances)

        model = self.get_model().eval()
        tokenizer = self.get_tokenizer()
        tokenizer.model_max_length = model.config.n_positions
        
        with torch.inference_mode():
            with Tqdm.tqdm(qas, desc="Processing instances") as qas_tqdm:
                for batch in more_itertools.chunked(qas_tqdm, batch_size):
                    model_input = tokenizer([f"question:{i.question}" for i in batch],
                                            [f"context:{i.context}" for i in batch],
                                            truncation="only_second",
                                            padding="longest",
                                            return_tensors="pt")

                    model_output = model.generate(**model_input, max_new_tokens=50) # 50 new tokens is also same as GPT evaluation
                    model_output = tokenizer.batch_decode(model_output, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                    for instance, prediction in zip(batch, model_output):
                        yield {
                            "squad_metrics": ({"id": instance.id, "prediction_text": prediction}, {"id": instance.id, "answers": instance.answers})
                        }
    
    def _predict_prompt(self, task: Task, instances: Sequence[Dict[str, Any]], batch_size: int = 32) -> Iterator[Dict[str, Any]]:
        prompts = MappedSequence(task.instance_conversions[InstanceFormat.T5_PROMPT], instances)

        model = self.get_model().eval()
        tokenizer = self.get_tokenizer()

        with torch.inference_mode():
            with Tqdm.tqdm(prompts, desc="Processing instances") as prompts_tqdm:
                for batch in more_itertools.chunked(prompts_tqdm, batch_size):
                    model_input = tokenizer(
                        [i[0] for i in batch],
                        padding=True,
                        truncation="only_first",
                        return_tensors="pt",
                        pad_to_multiple_of=8)
                    model_output = model.generate(**model_input)
                    model_output = tokenizer.batch_decode(model_output, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                    for target, prediction in zip(batch, model_output):
                        target = target[1]
                        yield {
                            "acc": (torch.Tensor([target == prediction]), _true_tensor),
                            "bleu": ([prediction], [[target]]),
                            "rouge": ([prediction], [[target]])
                        }

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        if task.has_instance_conversion(InstanceFormat.T5_PROMPT):
            return self._predict_prompt(task, instances, batch_size=batch_size)
        elif task.has_instance_conversion(InstanceFormat.HF_QA):
            return self._predict_qa(task, instances, batch_size=batch_size)
        
        raise UnsupportedTaskError(self, task)
        


@Model.register("catwalk::t5_from_pretrained")
class T5ModelFromPretrained(T5Model):
    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def get_model(self) -> T5ForConditionalGeneration:
        return cached_transformers.get(AutoModelForSeq2SeqLM, self.pretrained_model_name_or_path, False)

    def get_tokenizer(self) -> T5TokenizerFast:
        return cached_transformers.get_tokenizer(T5TokenizerFast, self.pretrained_model_name_or_path)


@Model.register("catwalk::t5_from_model")
class T5ModelFromModel(T5Model):
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: Optional[T5TokenizerFast] = None
    ):
        self.model = model
        self.tokenizer = tokenizer

    def get_model(self) -> T5ForConditionalGeneration:
        return self.model

    def get_tokenizer(self) -> T5TokenizerFast:
        if self.tokenizer is None:
            return cached_transformers.get_tokenizer(T5TokenizerFast, self.get_model().name_or_path)
        else:
            return self.tokenizer

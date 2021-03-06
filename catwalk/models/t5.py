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

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        if not task.has_instance_conversion(InstanceFormat.T5_PROMPT):
            raise UnsupportedTaskError(self, task)
        prompts = MappedSequence(task.instance_conversions[InstanceFormat.T5_PROMPT], instances)

        model = self.get_model().eval()
        tokenizer = self.get_tokenizer()

        def strip_special_tokens(t: torch.Tensor) -> torch.Tensor:
            # amazing that torch has no capability for this
            start = 0
            while start < len(t) and int(t[start]) in {0, tokenizer.eos_token_id, tokenizer.pad_token_id}:
                start += 1
            end = len(t)
            while int(t[end - 1]) in {0, tokenizer.eos_token_id, tokenizer.pad_token_id} and end > start:
                end -= 1
            return t[start:end]

        with torch.inference_mode():
            for batch in more_itertools.chunked(Tqdm.tqdm(prompts, desc="Processing instances"), batch_size):
                model_input = tokenizer(
                    [i[0] for i in batch],
                    padding=True,
                    truncation="only_first",
                    return_tensors="pt",
                    pad_to_multiple_of=8)
                model_output = model.generate(**model_input)
                model_output = [strip_special_tokens(t) for t in model_output]
                model_output = tokenizer.batch_decode(model_output, clean_up_tokenization_spaces=True)
                for target, prediction in zip(batch, model_output):
                    target = target[1]
                    yield {
                        "acc": (torch.Tensor([target == prediction]), _true_tensor),
                        "bleu": ([prediction], [[target]]),
                        "rouge": ([prediction], [[target]])
                    }


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

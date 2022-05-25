from typing import Any, Iterator, Dict, Sequence, Tuple, List

import more_itertools
import torch
from tango.common import Tqdm
from tango.common.sequences import MappedSequence
from transformers import AutoModelForMultipleChoice, AutoTokenizer

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
        if not task.has_instance_conversion(InstanceFormat.HF_MC):
            raise UnsupportedTaskError(self, task)
        instances = MappedSequence(
            lambda instance: task.convert_instance(instance, InstanceFormat.HF_MC),
            instances)

        # There is no Huggingface pipeline for this.
        model = AutoModelForMultipleChoice.from_pretrained(self.pretrained_model_name_or_path).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
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
                    **{key: tensor.view(len(batch), number_of_choices, -1) for key, tensor in tensors.items()})
                for instance, logits in zip(batch, results.logits.detach().cpu()):
                    yield {
                        "correct_answer_index": instance.correct_answer_index,
                        "logits": logits,
                        "acc": (logits, instance.correct_answer_index),
                    }

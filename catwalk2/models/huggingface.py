from typing import Any, Iterator, Dict, Sequence

import more_itertools
import torch
from tango.common import Tqdm
from transformers import AutoModelForMultipleChoice, AutoTokenizer

from catwalk2.model import Model, TaskTypeModel, UnsupportedTaskError
from catwalk2.task import Task
from catwalk2.tasks import TaskWithHFMCConversion


@Model.register("hf")
class HFAutoModel(TaskTypeModel):
    VERSION = "001"

    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def predict_multiple_choice(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        if not isinstance(task, TaskWithHFMCConversion):
            raise UnsupportedTaskError(self, task)

        # There is no Huggingface pipeline for this.
        model = AutoModelForMultipleChoice.from_pretrained(self.pretrained_model_name_or_path).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        instances = Tqdm.tqdm(instances, desc="Processing instances")
        with torch.inference_mode():
            for batch in more_itertools.chunked(instances, batch_size):
                number_of_choices = None
                texts = []
                labels = []
                for instance in batch:
                    instance = task.instance_as_hf_mc(instance)
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
                        "id": instance.id,
                        "correct_answer_index": instance.correct_answer_index,
                        "logits": logits,
                        "acc": (logits, instance.correct_answer_index),
                        "f1": (logits, instance.correct_answer_index),
                        "precision": (logits, instance.correct_answer_index),
                        "recall": (logits, instance.correct_answer_index)
                    }

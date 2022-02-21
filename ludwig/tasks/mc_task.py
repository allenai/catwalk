from abc import ABC
from dataclasses import dataclass
from typing import Optional, List, Union, Any, Iterator

import datasets
import torch
from torchmetrics.functional import precision_recall, accuracy, f1_score

from ludwig.models.model import ModelForEvaluation
from ludwig.tasks.task import Task, Metrics
from ludwig.utilities import get_from_dict


class MCTask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        context: Optional[str]
        question: str
        answer_choices: List[str]
        correct_answer_index: int

    @dataclass
    class InstanceResult(Task.InstanceResult):
        label: int
        logits: torch.Tensor

    number_of_choices: int = NotImplemented

    def evaluate_model(self, model: ModelForEvaluation, **kwargs) -> Metrics:
        results = model.do_multiple_choice(self, **kwargs)
        logits = torch.stack([r.logits for r in results])
        targets = torch.tensor([r.label for r in results], dtype=torch.int32)
        p, r = precision_recall(logits, targets)
        return {
            "accuracy": float(accuracy(logits, targets)),
            "f1": float(f1_score(logits, targets)),
            "precision": float(p),
            "recall": float(r),
        }


class MCTaskFromDataset(MCTask):
    def __init__(
            self,
            name: str,
            dataset: str,
            *,
            dataset_config: Optional[str] = None,
            context_field: Optional[str],
            question_field: str,
            answer_choices_fields: Union[str, List[str]],
            correct_answer_index_field: str,
            id_field: Optional[str] = None,
            number_of_choices: int
    ):
        super().__init__(name)
        # We want this lazy, so it's a lambda.
        self.get_dataset = lambda split: datasets.load_dataset(dataset, dataset_config, split=split)
        self.context_field = context_field
        self.question_field = question_field
        self.answer_choices_fields = answer_choices_fields
        self.correct_answer_index_field = correct_answer_index_field
        self.id_field = id_field
        self.number_of_choices = number_of_choices

    @classmethod
    def _normalize_answers(cls, answer: Any) -> int:
        if isinstance(answer, int):
            return answer
        if isinstance(answer, str):
            try:
                return int(answer)
            except ValueError:
                if len(answer) == 1:
                    answer = answer.lower()
                    return ord(answer[0]) - ord('a')
                else:
                    raise

    def get_instances(self, split: str) -> Iterator[MCTask.Instance]:
        for i, instance in enumerate(self.get_dataset(split)):
            id = str(get_from_dict(instance, self.id_field)) if self.id_field else str(i)
            if isinstance(self.answer_choices_fields, str):
                answer_choices = get_from_dict(instance, self.answer_choices_fields)
            else:
                answer_choices = [get_from_dict(instance, field) for field in self.answer_choices_fields]
            yield MCTask.Instance(
                id=id,
                context=instance[self.context_field] if self.context_field else None,
                question=instance[self.question_field],
                answer_choices=answer_choices,
                correct_answer_index=self._normalize_answers(instance[self.correct_answer_index_field]))

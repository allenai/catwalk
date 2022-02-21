from abc import ABC
from dataclasses import dataclass
from typing import List

from ludwig.models.model import ModelForEvaluation
from ludwig.tasks.task import Task, Metrics


class ClassificationTask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        text: str
        label: str  # Has to be one of the labels the task specifies.

    @property
    def labels(self) -> List[str]:
        raise NotImplementedError

    def evaluate_model(self, model: ModelForEvaluation, **kwargs) -> Metrics:
        results = model.do_classification(self, **kwargs)
        metrics = ...
        return metrics

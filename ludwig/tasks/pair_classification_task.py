from abc import ABC
from dataclasses import dataclass
from typing import List

from ludwig.models.model import ModelForEvaluation
from ludwig.tasks.task import Task, Metrics


class PairClassificationTask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        text1: str
        text2: str
        label: str  # Has to be one of the labels the task specifies.

    @property
    def labels(self) -> List[str]:
        raise NotImplementedError

    def evaluate_model(self, model: ModelForEvaluation, **kwargs) -> Metrics:
        results = model.do_pair_classification(self, **kwargs)
        metrics = ...
        return metrics

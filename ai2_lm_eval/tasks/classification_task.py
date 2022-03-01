from abc import ABC
from dataclasses import dataclass
from typing import List, Sequence, Iterator

from ai2_lm_eval.models.model import ModelForEvaluation
from ai2_lm_eval.tasks.task import Task, Metrics


class ClassificationTask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        text: str
        label: str  # Has to be one of the labels the task specifies.

    @property
    def labels(self) -> List[str]:
        raise NotImplementedError

    def run_inference(
        self,
        model: ModelForEvaluation,
        instances: Sequence[Instance],
        **kwargs
    ) -> Iterator['ClassificationTask.InstanceResult']:
        return model.do_classification(self, instances, **kwargs)

    def calculate_metrics(self, results: Iterator['ClassificationTask.InstanceResult']) -> Metrics:
        raise NotImplementedError

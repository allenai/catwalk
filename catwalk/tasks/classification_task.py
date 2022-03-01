from abc import ABC
from dataclasses import dataclass
from typing import List, Sequence, Iterator

from catwalk.models.model import ModelForEvaluation
from catwalk.tasks.task import Task, Metrics


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

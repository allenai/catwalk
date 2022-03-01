from abc import ABC
from dataclasses import dataclass
from typing import Sequence, Iterator

from catwalk.models.model import ModelForEvaluation
from catwalk.tasks.task import Task, Metrics


class GenerationTask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        prompt: str
        expected: str

    @dataclass
    class InstanceResult(Task.InstanceResult):
        label: str
        predicted: str

    def run_inference(
        self,
        model: ModelForEvaluation,
        instances: Sequence[Instance],
        **kwargs
    ) -> Iterator['GenerationTask.InstanceResult']:
        return model.do_generation(self, instances, **kwargs)

    def calculate_metrics(self, results: Iterator['GenerationTask.InstanceResult']) -> Metrics:
        raise NotImplementedError

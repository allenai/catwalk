from abc import ABC
from dataclasses import dataclass
from typing import Sequence, Iterator

from ludwig.models.model import ModelForEvaluation
from ludwig.tasks.task import Task, Metrics


class SummarizationTask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        input: str
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
    ) -> Iterator['SummarizationTask.InstanceResult']:
        return model.do_summarization(self, instances, **kwargs)

    def calculate_metrics(self, results: Iterator['SummarizationTask.InstanceResult']) -> Metrics:
        raise NotImplementedError

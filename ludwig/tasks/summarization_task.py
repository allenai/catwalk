from abc import ABC
from dataclasses import dataclass

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

    def evaluate_model(self, model: ModelForEvaluation, **kwargs) -> Metrics:
        results = model.do_summarization(self, **kwargs)
        metrics = ...   # TODO
        return metrics

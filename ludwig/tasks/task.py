from abc import ABC
from dataclasses import dataclass
from typing import Iterator, Dict, TypeVar, Sequence, Any

from tango.common.det_hash import DetHashWithVersion
from tango.common.registrable import Registrable

from ludwig.models.model import ModelForEvaluation

Metrics = Dict[str, float]
InstanceT = TypeVar('InstanceT')


class Task(ABC, Registrable, DetHashWithVersion):
    @dataclass
    class Instance:
        id: str
        metadata: Dict[str, Any]

    @dataclass
    class InstanceResult:
        instance: 'Task.Instance'

    def __init__(self, name: str):
        self.name = name

    def get_instances(self, split: str) -> Sequence[Instance]:
        """
        Return the instances for the given split.

        Split names sometimes vary between datasets. It is expected that the test set lives under the name "test"
        and the validation set lives under the name "validation".

        :param split: the name of the split
        :return: a sequence of instances
        """
        raise NotImplementedError

    def run_inference(
        self,
        model: ModelForEvaluation,
        instances: Sequence[Instance],
        **kwargs
    ) -> Iterator[InstanceResult]:
        """This method runs inference on a model to produce results given instances."""
        raise NotImplementedError

    def calculate_metrics(self, results: Iterator[InstanceResult]) -> Metrics:
        """This method calculates metrics from the given results."""
        raise NotImplementedError

    def evaluate_model(self, model: ModelForEvaluation, **kwargs) -> Metrics:
        """This method evaluates a task on the test set end-to-end."""
        instances = self.get_instances("test")
        results = self.run_inference(model, instances)
        return self.calculate_metrics(results)

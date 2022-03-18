from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Dict, TypeVar, Sequence, Any, Optional, Union, List, Callable, Iterable

import datasets
from tango.common.det_hash import DetHashWithVersion
from tango.common.registrable import Registrable
from tango.common.sequences import MappedSequence, ConcatenatedSequence

from catwalk.models.model import ModelForEvaluation

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

    @abstractmethod
    def get_instances(self, split: str) -> Sequence[Instance]:
        """
        Return the instances for the given split.

        Split names sometimes vary between datasets. It is expected that the test set lives under the name "test"
        and the validation set lives under the name "validation".

        :param split: the name of the split
        :return: a sequence of instances
        """
        raise NotImplementedError

    @abstractmethod
    def run_inference(
        self,
        model: ModelForEvaluation,
        instances: Sequence[Instance],
        **kwargs
    ) -> Iterator[InstanceResult]:
        """This method runs inference on a model to produce results given instances."""
        raise NotImplementedError

    MetricFn = Callable[[Iterable[InstanceResult]], Metrics]
    calculate_metrics: MetricFn = NotImplemented

    def evaluate_model(self, model: ModelForEvaluation, **kwargs) -> Metrics:
        """This method evaluates a task on the test set end-to-end."""
        instances = self.get_instances("test")
        results = self.run_inference(model, instances, **kwargs)
        return self.calculate_metrics(results)


class FromDatasetMixin(ABC):
    def __init__(
        self,
        *,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        id_field: Optional[str] = None,
        split_mappings: Optional[Dict[str, Union[str, List[str]]]] = None
    ):
        # We want this lazy, so it's a lambda.
        self.get_dataset = lambda split: datasets.load_dataset(dataset_path, dataset_name, split=split)
        self.id_field = id_field
        self.split_mappings = split_mappings

    @abstractmethod
    def _instance_from_json(self, instance: Dict[str, Any]) -> Task.Instance:
        raise NotImplementedError

    def get_instances(self, split: str) -> Sequence[Task.Instance]:
        if self.split_mappings is None:
            splits = [split]
        else:
            try:
                splits = self.split_mappings[split]
                if isinstance(splits, str):
                    splits = [splits]
            except KeyError:
                splits = [split]

        sequences = []
        for split in splits:
            dataset = self.get_dataset(split)
            if self.id_field is None:
                # TODO: This way of adding ids duplicates IDs if there is more than one split
                dataset = dataset.add_column("__default_id", range(len(dataset)))
            sequences.append(MappedSequence(self._instance_from_json, dataset))
        if len(sequences) == 1:
            return sequences[0]
        else:
            return ConcatenatedSequence(*sequences)


class FromCrossfitMixin(ABC):
    def __init__(self, crossfit_task: str):
        from catwalk.tasks import crossfit
        self.crossfit_task = crossfit.TASKS[crossfit_task]

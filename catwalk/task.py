from abc import ABC
from dataclasses import dataclass
from enum import Enum
from functools import partial
from random import Random
from typing import Dict, Any, Optional, Sequence, Union, List, Callable, Mapping, Tuple, Set, cast, FrozenSet, \
    Iterable

import torchmetrics
from mypy_extensions import KwArg
from tango.common import Registrable, det_hash

from catwalk.metrics.entropy import EntropyMetric
from catwalk.metrics.perplexity import PerplexityMetric


MC_METRICS = {
    "acc": torchmetrics.Accuracy,
}

ENTAILMENT_METRICS = {
    "acc": torchmetrics.Accuracy,
    "f1": torchmetrics.F1Score,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall
}

BINARY_CLASSIFICATION_METRICS = ENTAILMENT_METRICS

PERPLEXITY_METRICS = {
    "word_perplexity": PerplexityMetric,
    "byte_perplexity": PerplexityMetric,
    "bits_per_byte": EntropyMetric,
}

QA_METRICS = {
    "squad_metrics": torchmetrics.SQuAD,
}


def classification_metrics(num_classes: int):
    return {
        "acc": torchmetrics.Accuracy,
        "f1": partial(torchmetrics.F1Score, num_classes=num_classes, average=None),
        "precision": partial(torchmetrics.Precision, num_classes=num_classes, average=None),
        "recall": partial(torchmetrics.Recall, num_classes=num_classes, average=None)
    }


class InstanceFormat(Enum):
    HF_DICT = 1
    ELEUTHER_DOC = 3
    ELEUTHER_CONTEXT = 4
    ELEUTHER_REQUESTS = 5
    HF_MC = 2
    T5_PROMPT = 6
    RANK_CLASSIFICATION = 7
    HF_QA = 8


@dataclass
class RankClassificationInstance:
    choices: List[Tuple[str, str]]
    correct_choice: Optional[int]


InstanceConversion = Union[Callable[[Dict[str, Any]], Any], Callable[[Dict[str, Any], KwArg()], Any]]


class Task(Registrable, ABC):
    """
    Base class for tasks in Catwalk
    """

    def __init__(self, *, version_override: Optional[str] = None):
        if version_override is not None:
            self.VERSION = version_override
        self.metrics: Dict[str, Callable[[], torchmetrics.Metric]] = {}
        self.instance_conversions: Dict[InstanceFormat, InstanceConversion] = {}

    def det_hash_object(self) -> Any:
        # It sucks that we need this. We have to fix this in Tango.
        if hasattr(self, "__getstate__"):
            return self.__getstate__()
        else:
            return self.__dict__

    def has_split(self, split: str) -> bool:
        """Returns ``True`` if a split with the given name exists. ``False`` otherwise."""
        raise NotImplementedError

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        """Returns the split with the given name."""
        raise NotImplementedError

    @property
    def default_split(self) -> str:
        """Returns the name of the default split to run evaluations on."""
        return "test"

    @property
    def fewshot_instances_split(self) -> str:
        """Returns the name of the split to use to find few-shot instances in."""
        for split_name in ["train", "training", "validation"]:
            if self.has_split(split_name):
                return split_name
        raise ValueError("This task has no split to take fewshot instances from.")

    def make_metrics(
            self,
            *,
            disable_torchmetrics_distributed_sync: bool = False
        ) -> Dict[str, torchmetrics.Metric]:
        return {
            name: metric_fn(sync_on_compute= not disable_torchmetrics_distributed_sync)
            for name, metric_fn in self.metrics.items()
        }

    def has_instance_conversion(self, format: InstanceFormat) -> bool:
        return format in self.instance_conversions

    def convert_instance(self, instance: Dict[str, Any], format: InstanceFormat, **kwargs):
        return self.instance_conversions[format](instance, **kwargs)

    def get_fewshot_instances(
        self,
        num_shots: int,
        *,
        exceptions: Union[None, Dict[str, Any], Iterable[Dict[str, Any]]] = None,
        random_seed: int = 18830087
    ) -> Sequence[Dict[str, Any]]:
        if exceptions is None:
            exceptions = []
        elif isinstance(exceptions, Dict):
            exceptions = [exceptions]
        exceptions = frozenset(det_hash(e) for e in exceptions)

        r = Random(random_seed)
        instances = self.get_split(self.fewshot_instances_split)
        sampled_instances = [
            instance
            for instance in r.sample(instances, num_shots + len(exceptions))
            if det_hash(instance) not in exceptions
        ]
        return sampled_instances[:num_shots]

    #
    # builder-style methods
    #

    def add_metric(self, name: str, metric: Callable[[], torchmetrics.Metric]):
        if isinstance(metric, torchmetrics.Metric):
            # Users should not do this, but they will, so we try to handle it.
            metric_object = metric.clone()
            metric_object.reset()
            self.metrics[name] = metric_object.clone
        else:
            self.metrics[name] = metric
        return self

    def add_metrics(self, metrics: Mapping[str, Callable[[], torchmetrics.Metric]]):
        for name, metric_fn in metrics.items():
            self.add_metric(name, metric_fn)
        return self

    def add_instance_conversion(self, format: InstanceFormat, conversion: InstanceConversion):
        self.instance_conversions[format] = conversion
        return self

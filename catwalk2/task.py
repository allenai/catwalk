from abc import ABC
from enum import Enum
from typing import Dict, Any, Optional, Sequence, TYPE_CHECKING, Union, List, Callable

import torchmetrics
from tango.common import Registrable

from catwalk2.metrics.entropy import EntropyMetric
from catwalk2.metrics.perplexity import PerplexityMetric


MC_METRICS = {
    "acc": torchmetrics.Accuracy,
    "f1": torchmetrics.F1,
    "precision": torchmetrics.Precision,
    "recall": torchmetrics.Recall
}

PERPLEXITY_METRICS = {
    "word_perplexity": PerplexityMetric,
    "byte_perplexity": PerplexityMetric,
    "bits_per_byte": EntropyMetric,
}


class InstanceFormat(Enum):
    HF_DICT = 1
    ELEUTHER_DOC = 3
    ELEUTHER_CONTEXT = 4
    ELEUTHER_REQUESTS = 5
    HF_MC = 2


InstanceConversion = Callable[[Dict[str, Any], ...], Any]


class Task(Registrable, ABC):
    def __init__(self, *, version_override: Optional[str] = None):
        if version_override is not None:
            self.VERSION = version_override
        self.metrics: Dict[str, Callable[[], torchmetrics.Metric]] = {}
        self.instance_conversions: Dict[InstanceFormat, InstanceConversion] = {}

    def has_split(self, split: str) -> bool:
        raise NotImplementedError

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError

    def make_metrics(self) -> Dict[str, torchmetrics.Metric]:
        return {
            name: metric_fn()
            for name, metric_fn in self.metrics.items()
        }

    def has_instance_conversion(self, format: InstanceFormat) -> bool:
        return format in self.instance_conversions

    def convert_instance(self, instance: Dict[str, Any], format: InstanceFormat, **kwargs):
        return self.instance_conversions[format](instance, **kwargs)

    #
    # builder-style methods
    #

    def add_metric(self, name: str, metric: Callable[[], torchmetrics.Metric]):
        if isinstance(metric, torchmetrics.Metric):
            # Users should not do this, but they will, so we try to handle it.
            metric = metric.clone()
            metric.reset()
            self.metrics[name] = metric.clone()
        else:
            self.metrics[name] = metric
        return self

    def add_metrics(self, metrics: Dict[str, Callable[[], torchmetrics.Metric]]):
        for name, metric_fn in metrics.items():
            self.add_metric(name, metric_fn)
        return self

    def add_instance_conversion(self, format: InstanceFormat, conversion: InstanceConversion):
        self.instance_conversions[format] = conversion
        return self

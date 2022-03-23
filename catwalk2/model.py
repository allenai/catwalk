from abc import ABC
from typing import Sequence, Dict, Any, Iterator

from tango.common import Registrable

from catwalk2.task import Task


class Model(Registrable, ABC):
    def predict(self, task: Task, instances: Sequence[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        raise NotImplementedError()

    def calculate_metrics(self, task: Task, predictions: Iterator[Any]) -> Dict[str, float]:
        raise NotImplementedError()

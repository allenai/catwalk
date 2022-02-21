from abc import ABC
from dataclasses import dataclass
from typing import Iterator, Dict, TypeVar

from ludwig.models.model import ModelForEvaluation

Metrics = Dict[str, float]
InstanceT = TypeVar('InstanceT')


class Task(ABC):
    @dataclass
    class Instance:
        id: str

    @dataclass
    class InstanceResult:
        id: str

    def __init__(self, name: str):
        self.name = name

    def get_instances(self, split: str) -> Iterator[InstanceT]:
        raise NotImplementedError

    def evaluate_model(self, model: ModelForEvaluation, **kwargs) -> Metrics:
        raise NotImplementedError

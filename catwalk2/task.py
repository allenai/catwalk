from abc import ABC
from enum import Enum
from typing import Dict, Any, Optional, Sequence

from tango.common import Registrable


class TaskType(Enum):
    CLASSIFICATION = 1
    QA = 2
    PERPLEXITY = 3
    GENERATION = 4
    MULTIPLE_CHOICE = 5


class Task(Registrable, ABC):
    def __init__(self, task_type: TaskType, *, version_override: Optional[str] = None):
        self.task_type = task_type
        if version_override is not None:
            self.VERSION = version_override

    def has_split(self, split: str) -> bool:
        raise NotImplementedError

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError

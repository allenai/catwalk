from abc import ABC
from enum import Enum
from typing import Dict, Any, Optional, Sequence, TYPE_CHECKING, Union, List

import torchmetrics
from tango.common import Registrable

if TYPE_CHECKING:
    from catwalk2.tasks import TaskWithHFMCConversion


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

    def get_metrics(self) -> Dict[str, torchmetrics.Metric]:
        raise NotImplementedError

    def with_hf_mc_conversion(
        self,
        *,
        context_field: Optional[str] = None,
        question_field: str,
        answer_choices_fields: Union[str, List[str]],
        correct_answer_index_field: str,
        id_field: Optional[str] = None,
    ) -> 'TaskWithHFMCConversion':
        from catwalk2.tasks import TaskWithHFMCConversion
        return TaskWithHFMCConversion(
            self,
            context_field=context_field,
            question_field=question_field,
            answer_choices_fields=answer_choices_fields,
            correct_answer_index_field=correct_answer_index_field,
            id_field=id_field)

    def with_mc_metrics(self):
        return TaskWithMCMetrics(self)


class TaskWithMCMetrics(Task):
    def __init__(
        self,
        inner_task: Task,
    ):
        assert inner_task.task_type == TaskType.MULTIPLE_CHOICE
        super().__init__(inner_task.task_type)
        self.inner_task = inner_task

    def __getattr__(self, item):
        return getattr(self.inner_task, item)

    def get_metrics(self) -> Dict[str, torchmetrics.Metric]:
        return {
            "acc": torchmetrics.Accuracy(),
            "f1": torchmetrics.F1(),
            "precision": torchmetrics.Precision(),
            "recall": torchmetrics.Recall()
        }
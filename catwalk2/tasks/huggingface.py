from abc import ABC
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, List, Union, Mapping

import datasets

from catwalk2.task import Task, TaskType


def get_from_dict(d: Union[Mapping[str, Any], Sequence[Any]], field: str) -> Any:
    components = field.split(".", 1)
    if len(components) == 0:
        raise ValueError("get_from_dict() called with empty string.")
    elif isinstance(d, Mapping) and len(components) == 1:
        return d[components[0]]
    elif isinstance(d, Sequence) and len(components) == 1:
        return d[int(components[0])]
    elif isinstance(d, Mapping):
        first, rest = components
        return get_from_dict(d[first], rest)
    elif isinstance(d, Sequence):
        first, rest = components
        return get_from_dict(d[int(first)], rest)
    else:
        raise ValueError()


class HFDatasetsTask(Task):
    def __init__(
        self,
        task_type: TaskType,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        *,
        version_override: Optional[str] = None
    ):
        super().__init__(task_type, version_override=version_override)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

    def has_split(self, split: str) -> bool:
        return split in datasets.get_dataset_split_names(self.dataset_path, self.dataset_name)

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        return datasets.load_dataset(self.dataset_path, self.dataset_name, split=split)


@dataclass
class HFMCInstance:
    id: Optional[str]
    question: str
    answer_choices: List[str]
    correct_answer_index: int


class WithHFMCConversion(ABC):
    def instance_as_hf_mc(self, instance: Dict[str, Any]) -> HFMCInstance:
        raise NotImplementedError


class TaskWithHFMCConversion(Task, WithHFMCConversion):
    def __init__(
        self,
        inner_task: Task,
        *,
        context_field: Optional[str] = None,
        question_field: str,
        answer_choices_fields: Union[str, List[str]],
        correct_answer_index_field: str,
        id_field: Optional[str] = None,
        version_override: Optional[str] = None
    ):
        assert inner_task.task_type == TaskType.MULTIPLE_CHOICE
        super().__init__(inner_task.task_type, version_override=version_override)
        self.inner_task = inner_task
        self.context_field = context_field
        self.question_field = question_field
        self.answer_choices_fields = answer_choices_fields
        self.correct_answer_index_field = correct_answer_index_field
        self.id_field = id_field

    def __getattr__(self, item):
        return getattr(self.inner_task, item)

    @classmethod
    def _normalize_answers(cls, answer: Any) -> int:
        if isinstance(answer, int):
            return answer
        if isinstance(answer, str):
            try:
                return int(answer)
            except ValueError:
                if len(answer) == 1:
                    answer = answer.lower()
                    return ord(answer[0]) - ord('a')
                else:
                    raise

    def instance_as_hf_mc(self, instance: Dict[str, Any]) -> HFMCInstance:
        if isinstance(self.answer_choices_fields, str):
            answer_choices = get_from_dict(instance, self.answer_choices_fields)
        else:
            answer_choices = [get_from_dict(instance, field) for field in self.answer_choices_fields]
        answer_choices = [a.strip() for a in answer_choices]

        question = get_from_dict(instance, self.question_field).strip()
        if self.context_field is not None:
            question = get_from_dict(instance, self.context_field).strip() + " " + question

        return HFMCInstance(
            id=str(get_from_dict(instance, self.id_field)) if self.id_field else None,
            question=question,
            answer_choices=answer_choices,
            correct_answer_index=self._normalize_answers(instance[self.correct_answer_index_field]))

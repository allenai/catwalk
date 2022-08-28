import functools
from abc import ABC
from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, List, Union, Mapping

import datasets
from tango.common.sequences import MappedSequence

from catwalk.task import Task, InstanceFormat, InstanceConversion


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
        dataset_path: str,
        dataset_name: Optional[str] = None,
        *,
        version_override: Optional[str] = None
    ):
        super().__init__(version_override=version_override)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.add_instance_conversion(InstanceFormat.HF_DICT, lambda x: x)

    @functools.lru_cache
    def has_split(self, split: str) -> bool:
        return split in datasets.get_dataset_split_names(self.dataset_path, self.dataset_name)

    @functools.lru_cache
    def dataset(self, split: str):
        return datasets.load_dataset(self.dataset_path, self.dataset_name, split=split)

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        ds = self.dataset(split=split)
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(lambda x: x, ds)
        return ds

@dataclass
class HFQAInstance:
    id: str
    question: str
    context: str
    answers: List[str]

def hfqa_conversion(
    *,
    context_field: str,
    question_field: str,
    answers_field: str,
    id_field: str,
) -> InstanceConversion:
    def convert(instance: Dict[str, Any]) -> HFQAInstance:
        return HFQAInstance(
            id=get_from_dict(instance, id_field),
            context=get_from_dict(instance, context_field),
            question=get_from_dict(instance, question_field).strip(),
            answers=get_from_dict(instance, answers_field))
        
    return convert

@dataclass
class HFMCInstance:
    id: Optional[str]
    question: str
    answer_choices: List[str]
    correct_answer_index: Optional[int]


def normalize_answers(answer: Any, answer_mappings: Optional[Dict[str, int]] = None) -> int:
    if answer_mappings is None:
        if isinstance(answer, int):
            return answer
        if isinstance(answer, str):
            if len(answer) == 1:
                answer = answer.lower()
                answer_index = ord(answer[0])
                if ord('a') <= answer_index <= ord('z'):
                    return answer_index - ord('a')
            raise ValueError(f"Don't know how to make an index from answer '{answer}'.")
        raise ValueError(f"Don't know how to make an index from answer of type {answer.__class__}.")
    else:
        return answer_mappings[answer]


def hfmc_convert(
    instance: Dict[str, Any],
    *,
    context_field: Optional[str] = None,
    question_field: str,
    answer_choices_fields: Union[str, List[str]],
    correct_answer_index_field: str,
    id_field: Optional[str] = None,
    answer_mappings: Optional[Dict[str, int]] = None
) -> HFMCInstance:
    if isinstance(answer_choices_fields, str):
        answer_choices = get_from_dict(instance, answer_choices_fields)
    else:
        answer_choices = [get_from_dict(instance, field) for field in answer_choices_fields]
    answer_choices = [a.strip() for a in answer_choices]

    question = get_from_dict(instance, question_field).strip()
    if context_field is not None:
        question = get_from_dict(instance, context_field).strip() + " " + question

    correct_answer_index: Optional[int] = normalize_answers(instance[correct_answer_index_field], answer_mappings)
    if correct_answer_index == -1:
        correct_answer_index = None

    return HFMCInstance(
        id=str(get_from_dict(instance, id_field)) if id_field else None,
        question=question,
        answer_choices=answer_choices,
        correct_answer_index=correct_answer_index)


def hfmc_conversion(
    **kwargs,
) -> InstanceConversion:
    # We're doing this in this stupid way because this makes the conversion function picklable.
    return functools.partial(hfmc_convert, **kwargs)

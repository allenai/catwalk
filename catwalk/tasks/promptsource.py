import collections
import functools
from typing import Dict, Any, Optional, Sequence, List

from catwalk.dependencies.promptsource.templates import (
    DatasetTemplates,
    TemplateCollection,
)
from catwalk.task import InstanceConversion, RankClassificationInstance, Task, InstanceFormat

_promptsource_template_collection = TemplateCollection()


def _index_case_insensitive(sequence: Sequence[str], value: str) -> Optional[int]:
    sequence = [s.lower() for s in sequence]
    try:
        return sequence.index(value.lower())
    except ValueError:
        return None


def promptsource_conversion(
    dataset_templates: DatasetTemplates,
) -> InstanceConversion:
    return functools.partial(promptsource_convert, dataset_templates=dataset_templates)


def promptsource_convert(
    instance: Dict[str, Any],
    *,
    dataset_templates: DatasetTemplates,
    fewshot_instances: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, RankClassificationInstance]:
    if fewshot_instances is None:
        fewshot_instances = []

    prefixes: Dict[str, str] = collections.defaultdict(str)
    for fewshot_instance in fewshot_instances:
        converted_fewshot_instances = promptsource_convert(
            fewshot_instance,
            dataset_templates=dataset_templates)
        for prompt_name, rc_instance in converted_fewshot_instances.items():
            if rc_instance.correct_choice is None:
                continue
            correct_choice = rc_instance.choices[rc_instance.correct_choice]
            prefixes[prompt_name] += f"{correct_choice[0].strip()}\n{correct_choice[1].strip()}\n\n"

    result: Dict[str, RankClassificationInstance] = {}
    for template_name in dataset_templates.all_template_names:
        template = dataset_templates[template_name]

        prompt_really_special_just_for_mypy = template.apply(instance)
        if prompt_really_special_just_for_mypy is None:
            continue
        prompt, correct_answer_really_special_just_for_mypy = prompt_really_special_just_for_mypy
        if correct_answer_really_special_just_for_mypy is None:
            correct_answer = None
        else:
            assert len(correct_answer_really_special_just_for_mypy) == 1
            correct_answer = correct_answer_really_special_just_for_mypy[0]

        answer_choices = template.get_answer_choices_list(instance)
        correct_choice_index: Optional[int]
        if answer_choices is None:
            answer_choices = template.get_fixed_answer_choices_list()
            if answer_choices is None:
                continue
            if correct_answer is None:
                correct_choice_index = None
            else:
                correct_choice_index = _index_case_insensitive(answer_choices, correct_answer)
                assert correct_choice_index is not None
        else:
            if correct_answer is None:
                correct_choice_index = None
            else:
                # We're doing this in a convoluted way because the matching is case-insensitive, and we don't
                # want to add the correct answer choice if it is already there with a different case.
                correct_choice_index = _index_case_insensitive(answer_choices, correct_answer)
                if correct_choice_index is None:
                    answer_choices.append(correct_answer)
                    answer_choices = list(set(answer_choices))
                    correct_choice_index = _index_case_insensitive(answer_choices, correct_answer)

        result[template_name] = RankClassificationInstance(
            choices=[
                (prefixes[template_name] + prompt, choice)
                for choice in answer_choices
            ],
            correct_choice=correct_choice_index,
        )
    return result


class WithPromptsourceMixin:
    promptsource_templates: Optional[DatasetTemplates]

    def __init__(self, dataset_name: str, subset_name: Optional[str] = None):
        if (dataset_name, subset_name) in _promptsource_template_collection.keys:
            self.promptsource_templates = _promptsource_template_collection.get_dataset(dataset_name, subset_name)
            if isinstance(self, Task):
                self.add_instance_conversion(
                    InstanceFormat.PROMPTSOURCE,
                    promptsource_conversion(dataset_templates=self.promptsource_templates))
        else:
            self.promptsource_templates = None

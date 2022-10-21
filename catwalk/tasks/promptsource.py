import collections
import functools
from typing import Dict, Any, Optional, Sequence, List, Tuple

from tango.common import det_hash

from catwalk.dependencies.promptsource.templates import DatasetTemplates, TemplateCollection

from catwalk.dependencies.promptsource.templates import (
    DatasetTemplates,
    TemplateCollection,
)
from catwalk.task import InstanceConversion, RankClassificationInstance, Task

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

    prompts = {
        template_name: (
            dataset_templates[template_name].apply(instance),
            dataset_templates[template_name].get_answer_choices_list(instance),
        )
        for template_name in dataset_templates.all_template_names
    }
    # filter out invalid prompts
    prompts = {
        template_name: (prompt, answer_choices)
        for template_name, (prompt, answer_choices) in prompts.items()
        if prompt is not None
    }
    # assert that there is only one answer
    assert all(  # type: ignore
        (
            (answer_choices is None)
            or (correct_answer is None)
            or (len(correct_answer) == 1)
        )
        for (prompt, correct_answer), answer_choices in prompts.values()
    )
    # package up as RankClassificationInstances
    result = {
        template_name: RankClassificationInstance(
            [(prefixes[template_name] + prompt, choice) for choice in answer_choices],
            _index_case_insensitive(answer_choices, correct_answer[0]) if correct_answer is not None else None
        ) for template_name, ((prompt, correct_answer), answer_choices) in prompts.items() if answer_choices is not None
    }
    return result


def promptsource_templates_for_task(task: Task) -> Optional[DatasetTemplates]:
    from catwalk.tasks.eleuther import EleutherTask
    from catwalk.tasks.huggingface import HFDatasetsTask

    if isinstance(task, EleutherTask) or isinstance(task, HFDatasetsTask):
        if (
            task.dataset_path,
            task.dataset_name,
        ) in _promptsource_template_collection.keys:
            return _promptsource_template_collection.get_dataset(
                task.dataset_path, task.dataset_name
            )
    return None

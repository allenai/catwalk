from typing import Dict, Any, Optional, Sequence

from promptsource.templates import DatasetTemplates, TemplateCollection

from catwalk.task import InstanceConversion, RankClassificationInstance, Task


_promptsource_template_collection = TemplateCollection()


def _index_case_insensitive(sequence: Sequence[str], value: str) -> int:
    sequence = [s.lower() for s in sequence]
    return sequence.index(value.lower())


def promptsource_conversion(
    *,
    dataset_templates: DatasetTemplates,
) -> InstanceConversion:
    def convert(instance: Dict[str, Any]) -> Dict[str, RankClassificationInstance]:
        prompts = {
            template_name: (
                dataset_templates[template_name].apply(instance),
                dataset_templates[template_name].get_answer_choices_list(instance)
            ) for template_name in dataset_templates.all_template_names
        }
        # filter out invalid prompts
        prompts = {
            template_name: (prompt, answer_choices)
            for template_name, (prompt, answer_choices) in prompts.items()
            if len(prompt) == 2
        }
        # assert that there is only one answer
        assert all(
            (
                (answer_choices is None) or
                (correct_answer is None) or
                (len(correct_answer) == 1)
            ) for (prompt, correct_answer), answer_choices in prompts.values()
        )
        # package up as a RankClassificationInstance
        return {
            template_name: RankClassificationInstance(
                [(prompt, choice) for choice in answer_choices],
                _index_case_insensitive(answer_choices, correct_answer[0]) if correct_answer is not None else None
            ) for template_name, ((prompt, correct_answer), answer_choices) in prompts.items() if answer_choices is not None
        }

    return convert


def promptsource_templates_for_task(task: Task) -> Optional[DatasetTemplates]:
    from catwalk.tasks.eleuther import EleutherTask
    from catwalk.tasks.huggingface import HFDatasetsTask
    if isinstance(task, EleutherTask) or isinstance(task, HFDatasetsTask):
        if (task.dataset_path, task.dataset_name) in _promptsource_template_collection.keys:
            return _promptsource_template_collection.get_dataset(
                task.dataset_path,
                task.dataset_name)
    return None
from typing import Dict, Any, Optional

from promptsource.templates import DatasetTemplates, TemplateCollection

from catwalk.task import InstanceConversion, RankClassificationInstance, Task


_promptsource_template_collection = TemplateCollection()


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
        assert all(len(correct_answer) == 1 for (prompt, correct_answer), answer_choices in prompts.values())
        return {
            template_name: RankClassificationInstance(
                [(prompt, choice) for choice in answer_choices],
                answer_choices.index(correct_answer[0]) if answer_choices is not None else None
            ) for template_name, ((prompt, correct_answer), answer_choices) in prompts.items()
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
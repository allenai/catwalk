import inspect
from typing import Any, Dict

import pytest

import catwalk.tasks
import catwalk.models

# There are too many P3 tasks, so we just pick one.
task_names = [task for task in catwalk.tasks.TASKS.keys() if not task.startswith("p3::")]
task_names.insert(0, "p3::wiki_qa_Is_This_True_")


@pytest.mark.parametrize("task_name", task_names)
def test_task(task_name: str):
    task = catwalk.tasks.TASKS[task_name]
    
    # get the first available split for this task
    instances = None
    for split in ["train", "validation", "test"]:
        if task.has_split(split):
            instances = task.get_instances(split)
            break
        
    if not instances:
        raise KeyError("No split found for task {}".format(task_name) + " in splits {}".format(splits))

    for conversion in task.instance_conversions.values():
        signature = inspect.signature(conversion)
        for instance in instances[:10]:
            kwargs: Dict[str, Any] = {}
            if "num_fewshot" in signature.parameters:
                kwargs["num_fewshot"] = 0
            if "fewshot_instances" in signature.parameters:
                kwargs["fewshot_instances"] = task.get_fewshot_instances(2, exceptions=instance)
            assert conversion(instance, **kwargs) is not None

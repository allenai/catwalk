import inspect
from typing import Any, Dict

import pytest

import catwalk.tasks
import catwalk.models

task_names = list(catwalk.tasks.TASKS.keys())


@pytest.mark.parametrize("task_name", task_names)
@pytest.mark.parametrize("split", ["train"])
def test_task(task_name: str, split: str):
    task = catwalk.tasks.TASKS[task_name]
    try:
        instances = task.get_split(split)
    except KeyError:
        return
    for conversion in task.instance_conversions.values():
        signature = inspect.signature(conversion)
        for instance in instances[:10]:
            kwargs: Dict[str, Any] = {}
            if "num_fewshot" in signature.parameters:
                kwargs["num_fewshot"] = 0
            if "fewshot_instances" in signature.parameters:
                kwargs["fewshot_instances"] = task.get_fewshot_instances(2, exceptions=instance)
            assert conversion(instance, **kwargs) is not None

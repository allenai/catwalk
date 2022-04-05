import inspect

import pytest

import catwalk.tasks


@pytest.mark.parametrize("task_name", catwalk.tasks.TASKS.keys())
@pytest.mark.parametrize("split", ["train", "validation"])
def test_task(task_name: str, split: str):
    task = catwalk.tasks.TASKS[task_name]
    for conversion in task.instance_conversions.values():
        signature = inspect.signature(conversion)
        for instance in task.get_split(split):
            args = (instance,)
            if "num_fewshot" in signature.parameters:
                kwargs = {"num_fewshot": 0}
            else:
                kwargs = {}
            assert conversion(*args, **kwargs) is not None

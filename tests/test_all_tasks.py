import inspect

import pytest

import catwalk.tasks

task_names = list(catwalk.tasks.TASKS.keys())

task_names.remove("race")
task_names.append(
    pytest.param(
        "race",
        marks=pytest.mark.xfail(reason="This task is broken in the latest version of the Eleuther framework.")))


@pytest.mark.parametrize("task_name", task_names)
@pytest.mark.parametrize("split", ["train", "validation"])
def test_task(task_name: str, split: str):
    task = catwalk.tasks.TASKS[task_name]
    try:
        split = task.get_split(split)
    except KeyError:
        return
    for conversion in task.instance_conversions.values():
        signature = inspect.signature(conversion)
        for instance in split[:10]:
            args = (instance,)
            if "num_fewshot" in signature.parameters:
                kwargs = {"num_fewshot": 0}
            else:
                kwargs = {}
            assert conversion(*args, **kwargs) is not None

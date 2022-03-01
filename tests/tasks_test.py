import pytest

from catwalk import TASKS


@pytest.mark.parametrize("task", TASKS.keys())
def test_task(task: str):
    task = TASKS[task]
    assert task.get_instances("validation")[0] is not None

import inspect
from typing import Any, Dict, cast

import pytest

import catwalk.models
import catwalk.tasks
from catwalk.task import InstanceFormat
from catwalk.tasks.huggingface import HFMCInstance

from .util import suite_B, suite_C

# These are tasks are known to fail for now due to an unreachable server.
known_failures = {
    "lambada_mt_en",
    "lambada_mt_fr",
    "lambada_mt_de",
    "lambada_mt_it",
    "lambada_mt_es",
    "triviaqa",
}

# There are too many P3 tasks, so we just pick one.
# MRQA dataset takes too long to load, so we skip it.
task_names = [
    pytest.param(
        task,
        id=task,
        marks=pytest.mark.xfail if task in known_failures else (),
    )
    for task in catwalk.tasks.TASKS.keys()
    if not task.startswith("p3::") and not task.startswith("mrqa::")
]
task_names.insert(0, "p3::wiki_qa_Is_This_True_")


@pytest.mark.parametrize("task_name", task_names)
@suite_B
def test_task(task_name: str):
    task = catwalk.tasks.TASKS[task_name]
    instances = next(
        (
            task.get_split(split)
            for split in ["train", "validation", "test"]
            if task.has_split(split)
        ),
        None,
    )
    if not instances:
        return
    for conversion in task.instance_conversions.values():
        signature = inspect.signature(conversion)
        for instance in instances[:10]:
            kwargs: Dict[str, Any] = {}
            if "num_fewshot" in signature.parameters:
                kwargs["num_fewshot"] = 0
            try:
                if "fewshot_instances" in signature.parameters:
                    kwargs["fewshot_instances"] = task.get_fewshot_instances(
                        2, exceptions=instance
                    )
            except (
                ValueError
            ):  # This task doesn't support fewshot for the chosen split.
                kwargs = {}
            assert conversion(instance, **kwargs) is not None


mc_tasks = [
    task_name
    for task_name, task in catwalk.tasks.TASKS.items()
    if task.has_instance_conversion(InstanceFormat.HF_MC)
]


@pytest.mark.parametrize("task_name", mc_tasks)
@suite_C
def test_mc_tasks(task_name):
    task = catwalk.tasks.TASKS[task_name]
    for split in ["train", "validation", "test"]:
        if not task.has_split(split):
            continue
        for instance in task.get_split(split):
            mc_instance = cast(
                HFMCInstance, task.convert_instance(instance, InstanceFormat.HF_MC)
            )
            if mc_instance.correct_answer_index is not None:
                assert mc_instance.correct_answer_index >= 0
                assert mc_instance.correct_answer_index < len(
                    mc_instance.answer_choices
                )

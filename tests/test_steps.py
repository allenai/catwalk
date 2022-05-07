import pytest

from catwalk.steps import PredictStep, CalculateMetricsStep

task_names = [
    "arc_challenge",
    "boolq",
    "copa",
    "headqa",
    "hellaswag",
    "lambada",
    "logiqa",
    "mc_taco",
    "mrpc",
    "multirc",
    "openbookqa",
    "piqa",
    "qnli",
    "qqp",
    "rte",
    "sciq",
    "sst",
    "webqs",
    "wic",
    "winogrande",
    "wsc",
]


@pytest.mark.parametrize("task_name", task_names)
@pytest.mark.parametrize("model_name", ["eai::tiny-gpt2", "eai::t5-very-small-random"])
def test_task_eval(task_name: str, model_name: str):
    predict_step = PredictStep(model=model_name, task=task_name, limit=10)
    metrics_step = CalculateMetricsStep(model=model_name, task=task_name, predictions=predict_step)
    result = metrics_step.result()
    assert result is not None

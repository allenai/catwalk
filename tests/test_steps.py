import pytest

from catwalk.steps import PredictStep, CalculateMetricsStep

# This is a list of Tasks Iz has deemed important.
task_names = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "copa",
    "headqa",
    "hellaswag",
    "lambada",
    "logiqa",
    "mathqa",
    "mc_taco",
    "mrpc",
    "multirc",
    "openbookqa",
    "piqa",
    "piqa",
    "prost",
    "pubmedqa",
    "qnli",
    "qqp",
    "race",
    "rte",
    "sciq",
    "sst",
    "triviaqa",
    "webqs",
    "wic",
    "winogrande",
    "wnli",
    "wsc",
]


@pytest.mark.parametrize("task_name", task_names)
@pytest.mark.parametrize("model_name", ["eai::tiny-gpt2", "eai::t5-very-small-random"])
def test_task_eval(task_name: str, model_name: str):
    predict_step = PredictStep(model=model_name, task=task_name, limit=10)
    metrics_step = CalculateMetricsStep(model=model_name, task=task_name, predictions=predict_step)
    result = metrics_step.result()
    assert result is not None

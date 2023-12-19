import pytest

from catwalk import MODELS
from catwalk.steps import CalculateMetricsStep, PredictStep

from .util import suite_A

task_names = [
    "arc_challenge",
    "boolq",
    "copa",
    "headqa_en",
    "hellaswag",
    "lambada",
    "mc_taco",
    "mrpc",
    "eai::multirc",
    "openbookqa",
    "qnli",
    "qqp",
    "rte",
    "webqs",
    "wic",
    "wsc",
]
model_names = ["eai::tiny-gpt2", "eai::t5-very-small-random"]
params = [(t, m) for t in task_names for m in model_names]


generation_task_names = [
    "squad2",
    "drop",
]
generation_model_names = ["eai::tiny-gpt2"]
generation_params = [
    (t, m) for t in generation_task_names for m in generation_model_names
]

params = params + generation_params


@pytest.mark.parametrize("task_name,model_name", params)
@suite_A
def test_task_eval(task_name: str, model_name: str):
    if MODELS[model_name].supports_fewshot:
        predict_kwargs = {"num_shots": 3}
    else:
        predict_kwargs = {}

    predict_step = PredictStep(
        model=model_name, task=task_name, limit=10, **predict_kwargs
    )
    metrics_step = CalculateMetricsStep(
        model=model_name, task=task_name, predictions=predict_step
    )
    result = metrics_step.result()
    assert result is not None

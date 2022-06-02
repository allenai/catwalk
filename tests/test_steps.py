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
model_names = [
    "eai::tiny-gpt2",
    "eai::t5-very-small-random"
]
params = [(t, m) for t in task_names for m in model_names]


generation_task_names = [
    "squad2",
    "drop",
    "truthfulqa_gen"
]
generation_model_names = [
    "eai::tiny-gpt2"
]
generation_params = [(t, m) for t in generation_task_names for m in generation_model_names]

params = params + generation_params

@pytest.mark.parametrize("task_name,model_name", params)
def test_task_eval(task_name: str, model_name: str):
    predict_step = PredictStep(model=model_name, task=task_name, limit=10)
    metrics_step = CalculateMetricsStep(model=model_name, task=task_name, predictions=predict_step)
    result = metrics_step.result()
    assert result is not None

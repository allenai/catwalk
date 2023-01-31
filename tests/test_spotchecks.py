import sys

import pytest

import catwalk.__main__
from catwalk.steps import PredictStep, CalculateMetricsStep


def test_squad():
    args = catwalk.__main__._parser.parse_args([
        "--model", "bert-base-uncased",
        "--task", "squad",
        "--split", "validation",
        "--limit", "100"
    ])
    catwalk.__main__.main(args)


@pytest.mark.parametrize("task", ["mnli", "cola", "rte"])
def test_gpt2_performance(task: str):
    model = "rc::gpt2"
    predictions = PredictStep(model=model, task=task, limit=100)
    metrics = CalculateMetricsStep(model=model, task=task, predictions=predictions)
    results = metrics.result()
    assert results['relative_improvement'] > 0
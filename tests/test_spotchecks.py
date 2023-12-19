import pytest

import catwalk.__main__
from catwalk.steps import CalculateMetricsStep, PredictStep

from .util import suite_C


@suite_C
def test_squad():
    args = catwalk.__main__._parser.parse_args(
        [
            "--model",
            "bert-base-uncased",
            "--task",
            "squad",
            "--split",
            "validation",
            "--limit",
            "100",
        ]
    )
    catwalk.__main__.main(args)


@pytest.mark.parametrize("task", ["mnli", "cola", "rte", "eai::multirc"])
@suite_C
def test_gpt2_performance(task: str):
    model = "rc::gpt2"
    predictions = PredictStep(model=model, task=task, limit=100)
    metrics = CalculateMetricsStep(model=model, task=task, predictions=predictions)
    results = metrics.result()
    assert results["relative_improvement"] > 0


def test_lambada_gpt():
    model = "gpt2"
    task = "lambada"
    predictions = PredictStep(model=model, task=task, limit=10)
    metrics = CalculateMetricsStep(model=model, task=task, predictions=predictions)
    results = metrics.result()
    assert results["acc"] >= 0.4


def test_perplexity_gpt():
    model = "gpt2"
    task = "wikitext"
    predictions = PredictStep(model=model, task=task, limit=10)
    metrics = CalculateMetricsStep(model=model, task=task, predictions=predictions)
    results = metrics.result()
    assert results["word_perplexity"] < 40
    assert results["byte_perplexity"] < 2.5
    assert results["bits_per_byte"] < 1.5

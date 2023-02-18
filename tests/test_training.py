import torch
from transformers import AdamW

from catwalk import MODELS, TASKS
from catwalk.steps import CalculateMetricsStep, FinetuneStep, PredictStep

from .util import suite_D


@suite_D
def test_training():
    model = MODELS["rc::tiny-gpt2"]
    task = TASKS["piqa"]
    instances = task.get_split("train")[:16]
    predictions_before = list(model.predict(task, instances))
    metrics_before = model.calculate_metrics(task, predictions_before)

    model = model.trainable_copy()
    batch = model.collate_for_training([(task, instance) for instance in instances])

    # The smallest training loop in the world.
    optimizer = AdamW(model.parameters())
    first_loss = None
    loss = None
    for _ in range(100):
        optimizer.zero_grad()
        loss = model.forward(**batch)["loss"]
        loss.backward()
        loss = float(loss)
        if first_loss is None:
            first_loss = loss
        optimizer.step()

    assert first_loss > loss

    predictions_after = list(model.predict(task, instances))
    metrics_after = model.calculate_metrics(task, list(predictions_after))
    for prediction_before, prediction_after in zip(
        predictions_before, predictions_after
    ):
        assert not torch.allclose(
            prediction_before["acc"][0], prediction_after["acc"][0]
        )
    assert metrics_before != metrics_after


@suite_D
def test_training_step_gpt():
    finetune_step = FinetuneStep(
        model="rc::tiny-gpt2",
        tasks=["piqa", "sst"],
        train_steps=10,
        validation_steps=10,
    )
    predict_step = PredictStep(model=finetune_step, task="piqa", limit=10)
    metrics_step = CalculateMetricsStep(
        model=finetune_step, task="piqa", predictions=predict_step
    )
    metrics_step.result()


@suite_D
def test_training_step_t5():
    finetune_step = FinetuneStep(
        model="rc::t5-very-small-random",
        tasks=["rte", "boolq"],
        train_steps=10,
        validation_steps=10,
    )
    predict_step = PredictStep(model=finetune_step, task="rte", limit=10)
    metrics_step = CalculateMetricsStep(
        model=finetune_step, task="rte", predictions=predict_step
    )
    metrics_step.result()


@suite_D
def test_training_step_hf():
    finetune_step = FinetuneStep(
        model="tiny-bert",
        tasks=["piqa"],
        train_steps=10,
        validation_steps=10,
    )
    predict_step = PredictStep(model=finetune_step, task="piqa", limit=10)
    metrics_step = CalculateMetricsStep(
        model=finetune_step, task="piqa", predictions=predict_step
    )
    metrics_step.result()

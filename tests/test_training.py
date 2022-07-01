from tqdm import trange
from transformers import AdamW

from catwalk import MODELS, TASKS


def test_training():
    model = MODELS['rc::tiny-gpt2'].trainable_copy()
    task = TASKS['piqa']

    instances = task.get_split("train")[:16]
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

    assert first_loss > float(loss)

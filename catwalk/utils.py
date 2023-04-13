from typing import Any
import torch

def sanitize(x: Any) -> Any:
    """
    Sanitize turns PyTorch types into basic Python types so they can be serialized into JSON.
    """

    if isinstance(x, (str, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.Tensor):
        # tensor needs to be converted to a list (and moved to cpu if necessary)
        return x.cpu().tolist()
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return {key: sanitize(value) for key, value in x.items()}
    elif isinstance(x, (list, tuple, set)):
        # Lists, tuples, and sets need their values sanitized
        return [sanitize(x_i) for x_i in x]
    elif x is None:
        return "None"
    elif hasattr(x, "to_json"):
        return x.to_json()
    else:
        raise ValueError(
            f"Cannot sanitize {x} of type {type(x)}. "
            "If this is your own custom class, add a `to_json(self)` method "
            "that returns a JSON-like object.")


def guess_instance_id(instance, max_val_length=30):
    for key in ['id', 'qid', 'q_id', 'line', 'identifier']:
        if key in instance:
            return {key: instance[key]}
    for key, value in instance.items():
        if isinstance(value, str) and len(value) <= max_val_length:
            return {key: value}
    return {"id": "NA"}
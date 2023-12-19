import dataclasses
from typing import Any

import numpy
import torch


def sanitize(x: Any) -> Any:
    """
    Sanitize turns PyTorch types into basic Python types so they can be serialized into JSON.
    """

    if isinstance(x, float) and numpy.isnan(x):
        # NaN should turn into string "NaN"
        return "NaN"
    if isinstance(x, (str, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.Tensor):
        # tensor needs to be converted to a list (and moved to cpu if necessary)
        return x.cpu().tolist()
    elif isinstance(x, numpy.ndarray):
        # array needs to be converted to a list
        return x.tolist()
    elif isinstance(x, numpy.number):
        # NumPy numbers need to be converted to Python numbers
        return x.item()
    elif isinstance(x, numpy.bool_):
        # Numpy bool_ need to be converted to python bool.
        return bool(x)
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
    elif dataclasses.is_dataclass(x):
        return sanitize(dataclasses.asdict(x))
    else:
        raise ValueError(
            f"Cannot sanitize {x} of type {type(x)}. "
            "If this is your own custom class, add a `to_json(self)` method "
            "that returns a JSON-like object."
        )


def guess_instance_id(instance, max_val_length=30, idx=None):
    ids = {}
    for key in ["id", "qid", "q_id", "line", "identifier"]:
        if key in instance:
            ids[key] = instance[key]
    for key, value in instance.items():
        if key.endswith("id") or "_id" in key:
            ids[key] = value
    if not ids:
        ids["instance_idx"] = idx
    return ids


# Filter dict on list of keys
def filter_dict_keys(dictionary, key_list, remove_none=False):
    res = {k: v for k, v in dictionary.items() if k in key_list}
    if remove_none:
        res = {k: v for k, v in res.items() if v is not None}
    return res

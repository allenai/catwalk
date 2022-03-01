from typing import Union, Mapping, Any, Sequence

import torch


def get_best_spans(span_start_logits: torch.Tensor, span_end_logits: torch.Tensor) -> torch.Tensor:
    # TODO: There is a function like this in huggingface. We don't need to have our own.
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(torch.ones((passage_length, passage_length), device=device)).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    best_spans = valid_span_log_probs.view(batch_size, -1).argmax(-1)
    span_start_indices = torch.div(best_spans, passage_length, rounding_mode="trunc")
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)


def get_from_dict(d: Union[Mapping[str, Any], Sequence[Any]], field: str) -> Any:
    components = field.split(".", 1)
    if len(components) == 0:
        raise ValueError("get_from_dict() called with empty string.")
    elif isinstance(d, Mapping) and len(components) == 1:
        return d[components[0]]
    elif isinstance(d, Sequence) and len(components) == 1:
        return d[int(components[0])]
    elif isinstance(d, Mapping):
        first, rest = components
        return get_from_dict(d[first], rest)
    elif isinstance(d, Sequence):
        first, rest = components
        return get_from_dict(d[int(first)], rest)
    else:
        raise ValueError()

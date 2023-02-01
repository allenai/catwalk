from typing import Union, List

import torch
from torchmetrics.aggregation import BaseAggregator


class AccuracyMetric(BaseAggregator):
    """
    Unfortunately torchmetrics' multilabel accuracy makes you decide on the number of possible labels
    beforehand. We need a metric that does not require this.
    """

    def __iter__(self):
        pass

    def __init__(self):
        super().__init__(fn="mean", default_value=torch.tensor(0.0, dtype=torch.float), nan_strategy="error")
        self.add_state("correct_count", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, logits: Union[List[float], torch.Tensor], label: Union[int, torch.Tensor]) -> None:  # type: ignore
        if isinstance(logits, List):
            logits = torch.tensor(logits)
        if logits.argmax() == label:
            self.correct_count += 1
        self.total_count += 1

    def compute(self) -> torch.Tensor:
        return torch.tensor(self.correct_count / self.total_count, dtype=torch.float)


class RelativeAccuracyImprovementMetric(AccuracyMetric):
    def __init__(self, num_classes: int):
        super().__init__()
        self.baseline = 1 / num_classes

    def compute(self) -> torch.Tensor:
        return (super().compute() - self.baseline) / self.baseline

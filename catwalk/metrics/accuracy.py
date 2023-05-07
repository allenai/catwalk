from typing import Union, List

import math
import numpy as np
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


class MultipleChoiceMetrics():

    def __init__(self, primary_metric="acc_per_token"):
        self.primary_metric = primary_metric
        self.state = {"count": 0, "total_raw": 0, "total_per_token": 0,
                      "total_per_char": 0, "total_uncond_norm": 0,
                      "predicted_indices_raw": {},
                      "predicted_indices_per_token": {},
                      "predicted_indices_per_char": {},
                      "total_probability_mass": 0}

    def get_metrics(self, data):
        logits = {"raw": [], "per_token": [], "per_char": []}
        for prediction in data['model_output']:
            logits["raw"].append(prediction['sum_logits'])
            logits["per_token"].append(prediction['sum_logits'] / prediction['num_tokens'])
            logits["per_char"].append(prediction['sum_logits'] / prediction['num_chars'])
        gold_index = data["correct_choice"]
        metrics = {}
        for scoring in ["raw", "per_token", "per_char"]:
            pred_index = int(np.argmax(logits[scoring]))  # int to avoid numpy numbers and keep tango happy
            acc = 1 if pred_index == gold_index else 0
            metrics[f"acc_{scoring}"] = acc
            metrics[f"predicted_index_{scoring}"] = pred_index
        metrics['probability_mass'] = sum(math.exp(logit) for logit in logits['raw'])
        if self.primary_metric in metrics:
            metrics['acc'] = metrics[self.primary_metric]
        return metrics

    def update(self, data):
        metrics = data.get("metrics", self.get_metrics(data))
        self.state['count'] += 1
        for scoring in ["raw", "per_token", "per_char"]:
            self.state[f'total_{scoring}'] += metrics[f'acc_{scoring}']
            pred_index = metrics[f"predicted_index_{scoring}"]
            index_tally = self.state[f'predicted_indices_{scoring}']
            index_tally[pred_index] = index_tally.get(pred_index, 0) + 1
        self.state["total_probability_mass"] += metrics['probability_mass']


    def compute(self):
        count = self.state['count']
        if count == 0:
            return {}
        final_metrics = {}
        for scoring in ["raw", "per_token", "per_char"]:
            final_metrics[f"acc_{scoring}"] = self.state[f'total_{scoring}'] / count
            index_tallies = []
            for index, tally in self.state[f'predicted_indices_{scoring}'].items():
                index_tallies.append((index, tally / count))
            index_tallies.sort(key=lambda x: -x[1])
            final_metrics[f'predicted_indices_{scoring}'] = index_tallies
        final_metrics['avg_total_probability_mass'] = self.state["total_probability_mass"] / count
        if self.primary_metric in final_metrics:
            final_metrics['acc'] = final_metrics[self.primary_metric]
        return final_metrics







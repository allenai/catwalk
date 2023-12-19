import math
from typing import Any, Dict, List, Union

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
        super().__init__(
            fn="mean",
            default_value=torch.tensor(0.0, dtype=torch.float),
            nan_strategy="error",
        )
        self.add_state(
            "correct_count",
            default=torch.tensor(0, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_count",
            default=torch.tensor(0, dtype=torch.int),
            dist_reduce_fx="sum",
        )

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


class RankedClassificationMetrics:
    # Tracks 4 different scoring possibilities for multiple-choice continuations:
    #   raw: Total probability of continuation
    #   per_token: Average probability per token
    #   per_char: Average probability per character (for comparing to certain Eleuther/LLaMA numbers)
    #   uncond: Probability divided by "unconditional" probability  (optional)

    def __init__(self, primary_metric="acc_per_token"):
        self.primary_metric = primary_metric
        self.scoring_types = ["raw", "per_token", "per_char", "uncond"]
        self.state: Dict[str, Any] = {
            "count": 0,
            "total_probability_mass": 0,
            "total_token_count": 0,
            "max_token_count": 0,
            "count_inputs": 0,
        }
        for scoring in self.scoring_types:
            self.state[f"total_{scoring}"] = 0
            self.state[f"predicted_indices_{scoring}"] = {}

    def get_metrics(self, data):
        # Try to avoid double processing data (shouldn't happen)
        if "metrics" in data and "probability_mass" in data["metrics"]:
            return data["metrics"]
        logits: Dict = {}
        for scoring in self.scoring_types:
            logits[scoring] = []
        for prediction in data["model_output"]:
            token_count = prediction.get("num_tokens_all", 0)
            self.state["total_token_count"] += token_count
            self.state["count_inputs"] += 1
            if token_count > self.state["max_token_count"]:
                self.state["max_token_count"] = token_count
            logits["raw"].append(prediction["sum_logits"])
            logits["per_token"].append(
                prediction["sum_logits"] / prediction["num_tokens"]
            )
            logits["per_char"].append(
                prediction["sum_logits"] / prediction["num_chars"]
            )
            if "sum_logits_uncond" in prediction:
                logits["uncond"].append(
                    prediction["sum_logits"] - prediction["sum_logits_uncond"]
                )
        gold_index = data["correct_choice"]
        metrics = {}
        if len(logits["uncond"]) == 0:
            del logits["uncond"]
        for scoring in logits.keys():
            pred_index = int(
                np.argmax(logits[scoring])
            )  # int to avoid numpy numbers and keep tango happy
            acc = 1 if pred_index == gold_index else 0
            metrics[f"acc_{scoring}"] = acc
            metrics[f"predicted_index_{scoring}"] = pred_index
        metrics["probability_mass"] = sum(math.exp(logit) for logit in logits["raw"])  # type: ignore
        if self.primary_metric in metrics:
            metrics["acc"] = metrics[self.primary_metric]
        return metrics

    def update(self, data):
        metrics = data.get("metrics", self.get_metrics(data))
        self.state["count"] += 1
        for scoring in self.scoring_types:
            key = f"acc_{scoring}"
            if key not in metrics:
                continue
            self.state[f"total_{scoring}"] += metrics[key]
            pred_index = metrics[f"predicted_index_{scoring}"]
            index_tally = self.state[f"predicted_indices_{scoring}"]
            index_tally[pred_index] = index_tally.get(pred_index, 0) + 1
        self.state["total_probability_mass"] += metrics["probability_mass"]

    def compute(self):
        count = self.state["count"]
        if count == 0:
            return {}
        final_metrics = {}
        for scoring in self.scoring_types:
            key = f"total_{scoring}"
            # hack to not score "uncond" accuracy if there isn't any total for it
            if key not in self.state or (self.state[key] == 0 and scoring == "uncond"):
                continue
            final_metrics[f"acc_{scoring}"] = self.state[key] / count
            index_tallies = []
            for index, tally in self.state[f"predicted_indices_{scoring}"].items():
                index_tallies.append((index, tally / count))
            index_tallies.sort(key=lambda x: -x[1])
            final_metrics[f"predicted_indices_{scoring}"] = index_tallies
        final_metrics["avg_total_probability_mass"] = (
            self.state["total_probability_mass"] / count
        )
        final_metrics["primary_metric"] = self.primary_metric
        final_metrics["total_token_count"] = self.state["total_token_count"]
        final_metrics["max_token_count"] = self.state["max_token_count"]
        if self.primary_metric in final_metrics:
            final_metrics["acc"] = final_metrics[self.primary_metric]
        return final_metrics

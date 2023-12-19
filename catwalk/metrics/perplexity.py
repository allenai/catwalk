import math
from typing import Any, Dict, Optional, Union

import torch
from torchmetrics.aggregation import BaseAggregator


class PerplexityMetric(BaseAggregator):
    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Dict[str, Any],
    ):
        super().__init__("sum", [], nan_strategy, **kwargs)
        self.add_state(
            "loglikelihood",
            default=torch.tensor(0.0, dtype=torch.float),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_tokens", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum"
        )

    def update(
        self,
        loglikelihood: Union[float, torch.Tensor],
        num_tokens: Union[int, torch.Tensor],
    ) -> None:  # type: ignore
        loglikelihood = self._cast_and_nan_check_input(loglikelihood)
        if not isinstance(num_tokens, torch.Tensor):
            num_tokens = torch.tensor(num_tokens)
        self.loglikelihood += loglikelihood.sum()
        self.num_tokens += num_tokens.sum()

    def compute(self) -> torch.Tensor:
        return torch.exp(-self.loglikelihood / self.num_tokens)


class PerplexityMetrics:
    # Tracks different perplexity metrics, such as:
    #   ppl_token: perplexity per token
    #   ppl_char: perplexity per character
    #   ppl_byte: perplexity per byte
    #   ppl_word: perplexity per word
    # It computes both per-doc perplexities and overall perplexity averaged over all tokens/chars/etc

    def __init__(self, primary_metric="ppl_token"):
        self.primary_metric = primary_metric
        self.scoring_types = ["token", "char", "word", "byte"]
        self.state = {
            "count": 0,
            "total_tokens_input": 0,
            "max_token_count": 0,
            "total_sum_logits": 0,
        }
        for scoring in self.scoring_types:
            self.state[f"total_{scoring}s"] = 0

    def get_metrics(self, data):
        # Try to avoid double processing data
        if "metrics" in data and "ppl_token" in data["metrics"]:
            return data["metrics"]
        logits: Dict[str, Any] = {}
        for scoring in self.scoring_types:
            logits[scoring] = []
        prediction = data["model_output"]
        token_count = prediction.get("num_tokens_all", 0)
        self.state["total_tokens_input"] += token_count
        if token_count > self.state["max_token_count"]:
            self.state["max_token_count"] = token_count
        self.state["total_sum_logits"] += prediction["sum_logits"]
        metrics = {
            "bits_per_byte": -prediction["sum_logits"]
            / (prediction["num_bytes"] * math.log(2))
        }
        for scoring in self.scoring_types:
            self.state[f"total_{scoring}s"] += prediction[f"num_{scoring}s"]
            metrics[f"ppl_{scoring}"] = safe_exp(
                -prediction["sum_logits"] / max(prediction[f"num_{scoring}s"], 1)
            )
        if self.primary_metric in metrics:
            metrics["ppl_primary"] = metrics[self.primary_metric]
        return metrics

    def update(self, data):
        _ = data.get("metrics", self.get_metrics(data))
        self.state["count"] += 1

    def compute(self):
        count = self.state["count"]
        total_logits = self.state["total_sum_logits"]
        final_metrics = {
            "bits_per_byte": -total_logits / (self.state["total_bytes"] * math.log(2))
        }
        for scoring in self.scoring_types:
            final_metrics[f"ppl_{scoring}"] = safe_exp(
                -total_logits / max(self.state[f"total_{scoring}s"], 1)
            )
        final_metrics["primary_metric"] = self.primary_metric
        final_metrics["total_tokens_input"] = self.state["total_tokens_input"]
        final_metrics["max_token_count"] = self.state["max_token_count"]
        final_metrics["num_instances"] = count
        final_metrics["total_tokens"] = self.state["total_tokens"]
        if self.primary_metric in final_metrics:
            final_metrics["ppl_primary"] = final_metrics[self.primary_metric]
        return final_metrics


def safe_exp(x):
    try:
        ans = math.exp(x)
    except OverflowError:
        ans = 1e30
    return ans

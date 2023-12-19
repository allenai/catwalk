from random import Random
from typing import Any, Dict, Optional, Sequence, Tuple

from catwalk.model import Model
from catwalk.task import Task


def get_instances(
    task: Task,
    split: str,
    limit: Optional[int] = None,
    random_subsample_seed: Optional[int] = None,
) -> Sequence[Dict[str, Any]]:
    instances = task.get_split(split)
    if limit is not None and len(instances) > limit:
        instances = (
            instances[:limit]
            if random_subsample_seed is None
            else Random(random_subsample_seed).sample(instances, limit)
        )
    return instances


class PredictStep:
    def run(
        self,
        model: Model,
        task: Task,
        split: str,
        limit: Optional[int] = None,
        random_subsample_seed: Optional[int] = None,
        **kwargs
    ) -> Sequence[Any]:
        results = []
        instances = get_instances(task, split, limit, random_subsample_seed)
        for result in model.predict(task, instances, **kwargs):
            results.append(result)
        return results


class CalculateMetricsStep:
    def run(
        self, model: Model, task: Task, predictions: Sequence[Any]
    ) -> Tuple[Dict[str, float], Sequence[Any]]:
        metrics = model.calculate_metrics(task, predictions)
        return metrics, predictions

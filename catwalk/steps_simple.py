from typing import (
    Dict,
    Any,
    Optional,
    Sequence,
)

from catwalk.task import Task
from catwalk.tasks import get_instances
from catwalk.model import Model

class PredictStep():
    def run(
        self,
        model: Model,
        task: Task,
        split: Optional[str] = None,
        limit: Optional[int] = None,
        random_subsample_seed: Optional[int] = None,
        **kwargs
    ) -> Sequence[Any]:
        results = []
        instances = get_instances(task, split, limit, random_subsample_seed)
        for result in model.predict(task, instances, **kwargs):
            results.append(result)
        return results


class CalculateMetricsStep():
    def run(
        self,
        model: Model,
        task: Task,
        predictions: Sequence[Any]
    ) -> Dict[str, float]:
        metrics = model.calculate_metrics(task, predictions)
        return metrics, predictions

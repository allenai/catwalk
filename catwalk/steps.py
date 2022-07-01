from typing import Union, Dict, Any, Optional, Sequence, Iterable
from collections import defaultdict

from tango import Step, JsonFormat
from tango.common.sequences import SqliteSparseSequence
from tango.format import SqliteSequenceFormat, TextFormat
import torch

from catwalk.task import Task
from catwalk.tasks import TASKS
from catwalk.model import Model
from catwalk.models import MODELS


@Step.register("catwalk::predict")
class PredictStep(Step):
    VERSION = "001"
    SKIP_ID_ARGUMENTS = {"batch_size"}
    FORMAT = SqliteSequenceFormat

    def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(kwargs["model"], str):
            kwargs["model"] = MODELS[kwargs["model"]]
        if isinstance(kwargs["task"], str):
            kwargs["task"] = TASKS[kwargs["task"]]
        if kwargs["split"] is None:
            kwargs["split"] = kwargs["task"].default_split
        return kwargs

    def run(
        self,
        model: Union[str, Model],
        task: Union[str, Task],
        split: Optional[str] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> Sequence[Any]:
        if isinstance(model, str):
            model = MODELS[model]
        if isinstance(task, str):
            task = TASKS[task]
        if split is None:
            split = task.default_split

        results = SqliteSparseSequence(self.work_dir_for_run / "result.sqlite")
        instances = task.get_split(split)
        if limit is not None:
            instances = instances[:limit]
        instances = instances[len(results):]
        for result in model.predict(task, instances, **kwargs):
            results.append(result)
        return results


@Step.register("catwalk::calculate_metrics")
class CalculateMetricsStep(Step):
    VERSION = "001"
    FORMAT = JsonFormat

    def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(kwargs["model"], str):
            kwargs["model"] = MODELS[kwargs["model"]]
        if isinstance(kwargs["task"], str):
            kwargs["task"] = TASKS[kwargs["task"]]
        return kwargs

    def run(
        self,
        model: Union[str, Model],
        task: Union[str, Task],
        predictions: Sequence[Any]
    ) -> Dict[str, float]:
        if isinstance(model, str):
            model = MODELS[model]
        if isinstance(task, str):
            task = TASKS[task]

        return model.calculate_metrics(task, predictions)


@Step.register("catwalk::tabulate_metrics")
class TabulateMetricsStep(Step):
    VERSION = "001"
    FORMAT = TextFormat

    def run(self, metrics: Dict[str, Dict[str, float]], format: str = "text") -> Iterable[str]:
        flattend_metrics = defaultdict(dict)
        for task_name, task_metrics in metrics.items():
            for metric_name, metric_value in task_metrics.items():
                # if metric_value is a dict, then it's a nested metric
                if isinstance(metric_value, dict):
                    for nested_metric_name, nested_metric_value in metric_value.items():
                        flattend_metrics[task_name][f"{metric_name}.{nested_metric_name}"] = nested_metric_value.item() if isinstance(nested_metric_value, torch.Tensor) else nested_metric_value
                else:
                    flattend_metrics[task_name][metric_name] = metric_value
            
        if format == "text":
            for task_name, task_metrics in flattend_metrics.items():
                for metric_name, metric_value in task_metrics.items():
                    yield f"{task_name}\t{metric_name}\t{metric_value}"
        elif format == "latex":
            raise NotImplementedError()
        else:
            raise AttributeError("At the moment, only the 'text' format is supported.")
        
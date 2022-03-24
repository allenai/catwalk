from abc import ABC
from typing import Sequence, Dict, Any, Iterator

from tango.common import Registrable

from catwalk2.task import Task, TaskType


class Model(Registrable, ABC):
    def predict(self, task: Task, instances: Sequence[Dict[str, Any]], **kwargs) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError()

    def calculate_metrics(self, task: Task, predictions: Iterator[Dict[str, Any]]) -> Dict[str, float]:
        metrics = task.get_metrics()
        for prediction in predictions:
            for metric_name, metric_args in prediction.items():
                try:
                    metric = metrics[metric_name]
                except KeyError:
                    continue
                metric.update(*metric_args)
        return {
            metric_name: metric.compute()
            for metric_name, metric in metrics.items()
        }


class UnsupportedTaskError(Exception):
    """Thrown when the model doesn't support the task it's given."""

    def __init__(self, model: Model, task: Task):
        super().__init__(f"Model {model} does not support task {task}.")
        self.model = model
        self.task = task


class TaskTypeModel(Model, ABC):
    """
    Helper class in case you want to run a different method for each task type.
    """

    # TODO: Maybe we can remove this and, in the model, just check whether the instances are convertable into the
    # right format.

    def predict(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        **kwargs,
    ) -> Iterator[Any]:
        if task.task_type == TaskType.MULTIPLE_CHOICE:
            return self.predict_multiple_choice(task, instances, **kwargs)
        elif task.task_type == TaskType.QA:
            return self.predict_qa(task, instances, **kwargs)
        elif task.task_type == TaskType.PERPLEXITY:
            return self.predict_perplexity(task, instances, **kwargs)
        elif task.task_type == TaskType.CLASSIFICATION:
            return self.predict_classification(task, instances, **kwargs)
        elif task.task_type == TaskType.GENERATION:
            return self.predict_generation(task, instances, **kwargs)
        else:
            raise UnsupportedTaskError(self, task)

    def predict_multiple_choice(self, task: Task, instances: Sequence[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        raise UnsupportedTaskError(self, task)

    def predict_qa(self, task: Task, instances: Sequence[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        raise UnsupportedTaskError(self, task)

    def predict_perplexity(self, task: Task, instances: Sequence[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        raise UnsupportedTaskError(self, task)

    def predict_classification(self, task: Task, instances: Sequence[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        raise UnsupportedTaskError(self, task)

    def predict_generation(self, task: Task, instances: Sequence[Dict[str, Any]], **kwargs) -> Iterator[Any]:
        raise UnsupportedTaskError(self, task)

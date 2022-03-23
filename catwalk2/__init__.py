from typing import Union, Dict, Any, Optional, Sequence

from tango import Step, JsonFormat, SqliteDictFormat
from tango.common.sequences import SqliteSparseSequence

from catwalk2.model import Model, MODELS
from catwalk2.task import TASKS, Task


@Step.register("catwalk::predict")
class PredictStep(Step):
    VERSION = "001"
    SKIP_ID_ARGUMENTS = {"batch_size"}
    FORMAT = SqliteDictFormat

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
        split: str = "validation",
        batch_size: int = 32,
        limit: Optional[int] = None
    ) -> Sequence[Any]:
        if isinstance(model, str):
            model = MODELS[model]
        if isinstance(task, str):
            task = TASKS[task]

        results = SqliteSparseSequence(self.work_dir_for_run / "result.sqlite")
        instances = task.get_split(split)
        if limit is not None:
            instances = instances[:limit]
        instances = instances[len(results):]
        for result in model.predict(task, instances, batch_size=batch_size):
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
        predictions: Any
    ) -> Dict[str, float]:
        return model.calculate_metrics(task, predictions)

from typing import Union, Dict, Any, Optional

from tango import Step, JsonFormat

from catwalk2.model import Model, MODELS
from catwalk2.task import TASKS, Task


@Step.register("catwalk::predict")
class PredictStep(Step):
    VERSION = "001"
    SKIP_ID_ARGUMENTS = {"batch_size"}

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
        batch_size: int = 32,
        limit: Optional[int] = None
    ) -> Any:
        if isinstance(model, str):
            model = MODELS[model]

        return model.predict(task)


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

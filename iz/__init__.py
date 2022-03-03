from typing import Union, Dict, Any, Optional

from tango import Step, JsonFormat

from iz.model import IzModel, MODELS
from iz.task import TASKS, IzTask


@Step.register("iz::predict")
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
        model: Union[str, IzModel],
        task: Union[str, IzTask],
        batch_size: int = 32,
        limit: Optional[int] = None
    ) -> Any:
        if isinstance(model, str):
            model = MODELS[model]

        return model.predict(task)


@Step.register("iz::calculate_metrics")
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
        model: Union[str, IzModel],
        task: Union[str, IzTask],
        predictions: Any
    ) -> Dict[str, float]:
        return model.calculate_metrics(task, predictions)

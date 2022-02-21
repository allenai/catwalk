from typing import Union

from tango import Step
from tango.format import JsonFormat

from ludwig.models import ModelForEvaluation, MODELS
from ludwig.tasks import TASKS
from ludwig.tasks.task import Task, Metrics


class EvalModelOnTaskStep(Step):
    VERSION = "001"
    FORMAT = JsonFormat
    SKIP_ID_ARGUMENTS = {"batch_size"}

    def run(
        self,
        model: Union[str, ModelForEvaluation],
        task: Union[str, Task],
        batch_size: int = 32
    ) -> Metrics:
        if isinstance(model, str):
            model = MODELS[model]
        if isinstance(task, str):
            task = TASKS[task]
        return task.evaluate_model(model, batch_size=batch_size)
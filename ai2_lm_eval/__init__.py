from typing import Union, Sequence, Optional

from tango import Step
from tango.format import JsonFormat
from tango.format import SqliteSequenceFormat
from tango.common.sequences import SqliteSparseSequence

from ai2_lm_eval.models import ModelForEvaluation, MODELS
from ai2_lm_eval.tasks import TASKS
from ai2_lm_eval.tasks.task import Task, Metrics


@Step.register("run_model_on_task")
class RunModelOnTaskStep(Step):
    VERSION = "002"
    FORMAT = SqliteSequenceFormat
    SKIP_ID_ARGUMENTS = {"batch_size"}

    def run(
        self,
        model: Union[str, ModelForEvaluation],
        task: Union[str, Task],
        split: str = "validation",
        batch_size: int = 32,
        limit: Optional[int] = None
    ) -> Sequence[Task.InstanceResult]:
        if isinstance(model, str):
            model = MODELS[model]
        if isinstance(task, str):
            task = TASKS[task]

        results = SqliteSparseSequence(self.work_dir_for_run / "result.sqlite")
        instances = task.get_instances(split)
        if limit is not None:
            instances = instances[:limit]
        instances = instances[len(results):]
        for result in task.run_inference(model, instances, batch_size=batch_size):
            results.append(result)
        return results


@Step.register("calculate_metrics")
class CalculateMetricsStep(Step):
    VERSION = "001"
    FORMAT = JsonFormat

    def run(
        self,
        task: Union[str, Task],
        results: Sequence[Task.InstanceResult]
    ) -> Metrics:
        if isinstance(task, str):
            task = TASKS[task]
        return task.calculate_metrics(results)

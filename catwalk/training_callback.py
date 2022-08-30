from typing import Optional, cast, List

from tango.integrations.torch import TrainCallback

from catwalk import Task, Model
from catwalk.tasks import short_name_for_task_object


class CatwalkEvaluationCallback(TrainCallback):
    def __init__(
        self,
        *args,
        tasks: List[Task],
        eval_limit: Optional[int],
        eval_split: str = "validation",
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tasks = tasks
        self.eval_limit = eval_limit
        self.eval_split = eval_split

    def post_val_loop(
        self, step: int, epoch: int, val_metric: float, best_val_metric: float
    ) -> None:
        if not self.is_local_main_process:
            return

        model_was_training = self.model.training
        self.model.eval()
        try:
            catwalk_model = cast(Model, self.model.module) if hasattr(self.model, "module") else cast(Model, self.model) 
            for task in self.tasks:
                instances = task.get_split(self.eval_split)
                if self.eval_limit is not None:
                    instances = instances[:self.eval_limit]
                predictions = catwalk_model.predict(task, instances)
                metrics = catwalk_model.calculate_metrics(task, list(predictions), disable_torchmetrics_distributed_sync=True)
                metrics_string = []
                for metric_name, metric_value in metrics.items():
                    if len(metric_value.shape) > 0:
                        metric_value_string = ", ".join(f"{v:.3f}" for v in metric_value)
                    else:
                        metric_value_string = f"{metric_value:.3f}"
                    metrics_string.append(f"{metric_name}: {metric_value_string}")
                task_name = short_name_for_task_object(task) or str(task)
                print(f"Metrics for {task_name}: {' '.join(metrics_string)}")
        finally:
            self.model.train(model_was_training)
from typing import (
    Union,
    Dict,
    Any,
    Optional,
    Sequence,
    Iterable,
    List,
)
from collections import defaultdict
from random import Random

import tango
import transformers.optimization
from tango import Step, JsonFormat
from tango.common import Lazy, DatasetDict
from tango.common.sequences import SqliteSparseSequence
from tango.format import SqliteSequenceFormat, TextFormat
from tango.integrations.torch import (
    TorchFormat,
    TorchTrainingEngine,
    DataLoader, TrainingEngine, TrainConfig,
)
from tango.integrations.torch.model import Model as TangoModel
import torch

from catwalk.task import Task
from catwalk.tasks import TASKS
from catwalk.model import Model
from catwalk.models import MODELS
from catwalk.training_callback import CatwalkEvaluationCallback


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
        random_subsample_seed: Optional[int] = None,
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
        if limit is not None and len(instances) > limit:
            instances = instances[:limit] if random_subsample_seed is None else Random(random_subsample_seed).sample(instances, limit)
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


@Step.register("catwalk::finetune")
class FinetuneStep(Step):
    VERSION = "001"
    FORMAT = TorchFormat

    def massage_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(kwargs["model"], str):
            kwargs["model"] = MODELS[kwargs["model"]]
        kwargs["tasks"] = [TASKS[task] if isinstance(task, str) else task for task in kwargs["tasks"]]

        return kwargs

    def run(
        self,
        model: Union[str, Model],
        tasks: List[Union[str, Task]],
        train_steps: int = 2600,
        validation_steps: int = 1000,
        validate_every: int = 200,
        training_engine: Lazy[TrainingEngine] = Lazy(
            TorchTrainingEngine,
            lr_scheduler=Lazy(
                transformers.optimization.get_linear_schedule_with_warmup,
                num_warmup_steps=200,
                num_training_steps=2600
            ),
            optimizer=Lazy(
                torch.optim.AdamW,
                lr=1e-5,
            )
        ),
        model_wrapper: Optional[Lazy[TangoModel]] = None,
        random_seed: int = 42,
        batch_size: int = 16,
        grad_accum: int = 1,
        device_count: int = 1,
        distributed_port: int = 54761,
        train_split: str = "train",
        validation_split: Optional[str] = "validation",
    ) -> Model:  # type: ignore
        if isinstance(model, str):
            model = MODELS[model]
        tasks_in_a_special_variable_because_mypy_is_insane = [
            TASKS[task] if isinstance(task, str) else task for task in tasks
        ]

        devices: List[int]
        if torch.cuda.is_available() and torch.cuda.device_count() >= device_count:
            devices = list(range(device_count))
            self.logger.info("Training on %d GPU%s", device_count, "s" if device_count > 1 else "")
        else:
            devices = [-1] * device_count
            self.logger.info(
                "Training on CPU with %d worker%s", device_count, "s" if device_count > 1 else ""
            )

        if devices and len(devices) > 1:
            is_distributed = True
            num_workers = len(devices)
        else:
            is_distributed = False
            num_workers = 1

        train_config = TrainConfig(
            self.unique_id,
            self.work_dir,
            step_name=self.name,
            seed=random_seed,
            train_steps=train_steps,
            validation_steps=validation_steps,
            val_metric_name="acc",
            minimize_val_metric=False,
            train_split="train",
            validation_split=None if validation_split is None else "validation",
            validate_every=validate_every,
            checkpoint_every=validate_every,
            grad_accum=grad_accum,
            is_distributed=is_distributed,
            world_size=num_workers,
            distributed_port=distributed_port,
            devices=devices
        )

        # construct dataset from the tasks
        splits = {
            "train": [
                (task, i)
                for task in tasks_in_a_special_variable_because_mypy_is_insane
                for i in task.get_split(train_split)
            ],
        }
        if validation_split is not None:
            splits["validation"] = [
                (task, i)
                for task in tasks_in_a_special_variable_because_mypy_is_insane
                for i in task.get_split(validation_split)
            ]
        dataset_dict = DatasetDict(splits=splits, metadata={})

        trainable_model = model.trainable_copy()
        data_loader = Lazy(
            DataLoader,
            collate_fn=trainable_model.collate_for_training,
            batch_size=batch_size,
            shuffle=True
        )

        if model_wrapper is None:
            wrapped_model = trainable_model
        else:
            wrapped_model = Lazy(model_wrapper.construct, model=trainable_model)

        if validation_split is None:
            # No point in stopping early when we don't have a validation set.
            callbacks = []
        else:
            callbacks = [
                Lazy(
                    CatwalkEvaluationCallback,
                    tasks=tasks_in_a_special_variable_because_mypy_is_insane,
                    eval_limit=validation_steps
                )
            ]

        if is_distributed:
            import torch.multiprocessing as mp
            from tango.common.util import get_extra_imported_modules
            mp.spawn(
                tango.integrations.torch.train._train,
                args=(
                    self.workspace,
                    train_config,
                    wrapped_model,
                    training_engine,
                    dataset_dict,
                    data_loader,
                    None,
                    callbacks,
                    get_extra_imported_modules(),
                ),
                nprocs=num_workers,
            )
        else:
            tango.integrations.torch.train._train(
                0,
                self.workspace,
                train_config,
                wrapped_model,
                training_engine,
                dataset_dict,
                data_loader,
                callbacks=callbacks
            )

        # Load best checkpoint before returning model.
        if train_config.final_weights_path.is_file():
            self.logger.info(
                f"Loading best weights from {str(train_config.final_weights_path.resolve())}"
            )
            state = torch.load(train_config.final_weights_path, map_location="cpu")
            # We use `strict=False` because there might be missing keys due to weight tying.
            trainable_model.load_state_dict(state, strict=False)

        return trainable_model


@Step.register("catwalk::tabulate_metrics")
class TabulateMetricsStep(Step):
    VERSION = "001"
    FORMAT = TextFormat

    def run(self, metrics: Dict[str, Dict[str, float]], format: str = "text") -> Iterable[str]:
        flattend_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
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
        
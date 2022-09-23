import inspect
from abc import ABC
from copy import deepcopy
from typing import Sequence, Dict, Any, Iterator, Tuple, List, Optional

import torch
from tango.common import Registrable, Tqdm
from tango.common.det_hash import DetHashWithVersion

from catwalk.task import Task


Instance = Dict[str, Any]


def tensor_args(args: Tuple[Any]) -> Tuple[Any, ...]:
    """
    Annoyingly, torchmetrics only supports tensors as input, not raw values. So we have to convert raw values
    into tensors.
    """
    fixed_args: List[Any] = []
    for arg in args:
        if isinstance(arg, (float, int)):
            fixed_args.append(torch.tensor(arg))
        else:
            fixed_args.append(arg)
    return tuple(fixed_args)


def unsqueeze_args(args: Tuple[Any]) -> Tuple[Any, ...]:
    """
    Further, torchmetrics can't handle single-instance calls when given tensors. It always needs the first
    dimension of the tensors to be the instance dimension. So we add one.
    """
    fixed_args: List[Any] = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            fixed_args.append(arg.unsqueeze(0))
        else:
            fixed_args.append(arg)
    return tuple(fixed_args)


class Model(Registrable, DetHashWithVersion, ABC):
    def predict(self, task: Task, instances: Sequence[Dict[str, Any]], **kwargs) -> Iterator[Dict[str, Any]]:
        raise NotImplementedError()

    def calculate_metrics(self, task: Task, predictions: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        metrics = task.make_metrics()
        for prediction in Tqdm.tqdm(predictions, desc="Calculating metrics"):
            for metric_name, metric_args in prediction.items():
                try:
                    metric = metrics[metric_name]
                except KeyError:
                    continue
                metric_args = tensor_args(metric_args)
                metric_args = unsqueeze_args(metric_args)
                metric.update(*metric_args)
        return {
            metric_name: metric.compute()
            for metric_name, metric in metrics.items()
        }

    @property
    def supports_fewshot(self) -> bool:
        return "num_shots" in inspect.signature(self.predict).parameters

    def trainable_copy(self) -> "TrainableModel":
        """Returns a trainable version of this model.

        Catwalk models by default are immutable. Trainable models are not, because they can be trained.

        This is an optional method. Only implement it if you want to train your model through catwalk.
        """
        raise NotImplementedError("This model does not support training.")


class TrainableModel(Model, torch.nn.Module, ABC):
    """
    This is a catwalk model that also supports utility functions to make it possible to train.
    """

    def __init__(self, inner_module: Optional[torch.nn.Module]):
        super().__init__()
        self.inner_module = inner_module

    def forward(self, *args, **kwargs):
        """
        This method takes the input created by the :meth:`collate()` method and returns a dictionary that contains
        the loss under the key ``"loss"``.
        """
        return self.inner_module.forward(*args, **kwargs)

    def collate_for_training(self, instances: Sequence[Tuple[Task, Instance]]) -> Any:
        """
        Takes a batch of instances and turns them into inputs to the forward method (usually tensors).

        Usually you would call this method from a PyTorch DataLoader. If you don't use PyTorch, you might have to
        do something else.

        :param instances: The instances to turn into tensors. Note that the input includes the task. Instances
                          could come from different tasks.
        :return: Input suitable for the trainable model's ``forward()`` method.
        """
        raise NotImplementedError

    def trainable_copy(self) -> "TrainableModel":
        return deepcopy(self)


class UnsupportedTaskError(Exception):
    """Thrown when the model doesn't support the task it's given."""

    def __init__(self, model: Model, task: Task):
        super().__init__(f"Model {model} does not support task {task}.")
        self.model = model
        self.task = task

import argparse

from tango import Workspace
from tango.common.logging import initialize_logging
from tango.common import Lazy
from tango.integrations.fairscale import FairScaleTrainingEngine, FSDPConfig, with_wrapped_modules
from tango.integrations.torch import TorchTrainingEngine

from catwalk.steps import TabulateMetricsStep, FinetuneStep
from catwalk.tasks import TASK_SETS

import transformers

import torch


def main():
    initialize_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, nargs="+")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--device_count", type=int, default=1)
    parser.add_argument("--use_fairscale", action="store_true")
    parser.add_argument("--modules_to_wrap", type=str, nargs="+")
    parser.add_argument(
        "-d",
        "-w",
        type=str,
        default=None,
        metavar="workspace",
        dest="workspace",
        help="the Tango workspace with the cache",
    )
    args = parser.parse_args()

    
    assert not args.use_fairscale or (args.device_count >= 1), "use_fairscale is for distributed use only."
    assert args.modules_to_wrap is None or args.use_fairscale, "modules_to_wrap requires use_fairscale."

    if args.workspace is None:
        workspace = None
    else:
        workspace = Workspace.from_url(args.workspace)

    from catwalk.steps import CalculateMetricsStep
    from catwalk.steps import PredictStep

    tasks = set()
    for task in args.task:
        try:
            tasks |= TASK_SETS[task]
        except KeyError:
            tasks.add(task)

    lr_scheduler = Lazy(
        transformers.optimization.get_linear_schedule_with_warmup,
        num_warmup_steps=200,
        num_training_steps=10000
    )

    optimizer = Lazy(
        torch.optim.AdamW,
        lr=1e-5,
    )

    fsdp_config = FSDPConfig(
        reshard_after_forward=True,
        move_params_to_cpu=True,
        move_grads_to_cpu=True,
        mixed_precision=False,
    )

    training_engine = Lazy(
        FairScaleTrainingEngine,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        # amp=True, # causes NaN
        fsdp_config=fsdp_config
    ) if args.use_fairscale else Lazy(
        TorchTrainingEngine,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer
    )

    model_wrapper = Lazy(
        with_wrapped_modules,
        modules_to_wrap=args.modules_to_wrap,
        fsdp_config=fsdp_config,
        activation_checkpointing=True
    ) if args.use_fairscale and args.modules_to_wrap is not None else None

    model_step = FinetuneStep(
        model=args.model,
        tasks=tasks,
        batch_size=args.batch_size,
        grad_accum=args.grad_acc,
        device_count=args.device_count,
        training_engine=training_engine,
        model_wrapper=model_wrapper
    )

    metric_task_dict = {}
    for task in tasks:
        predictions = PredictStep(model=model_step, task=task, batch_size=args.batch_size)
        metrics = CalculateMetricsStep(
            model=model_step, task=task, predictions=predictions
        )
        metric_task_dict[task] = metrics

    table_step = TabulateMetricsStep(metrics=metric_task_dict)
    table_step_result = table_step.result(workspace)
    print("\n".join(table_step_result))


if __name__ == "__main__":
    main()

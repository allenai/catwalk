import argparse

from tango import Workspace
from tango.common.logging import initialize_logging

from catwalk.steps import TabulateMetricsStep, FinetuneStep
from catwalk.tasks import TASK_SETS


def main():
    initialize_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, nargs="+")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--val_metric", type=str, default="acc")
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

    model_step = FinetuneStep(
        model=args.model,
        tasks=tasks,
        batch_size=args.batch_size,
        grad_accum=args.grad_acc,
        val_metric_name=args.val_metric
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

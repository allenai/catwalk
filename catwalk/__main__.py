import argparse

from tango import Workspace
from tango.common.logging import initialize_logging

from catwalk.steps import TabulateMetricsStep
from catwalk.tasks import TASK_SETS


def main():
    initialize_logging(log_level="WARNING")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--task', type=str, nargs="+")
    parser.add_argument('--split', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_shots', type=int)
    parser.add_argument('--fewshot_seed', type=int)
    parser.add_argument('--limit', type=int)
    parser.add_argument(
        '-d', '-w',
        type=str,
        default=None,
        metavar="workspace",
        dest="workspace",
        help="the Tango workspace with the cache")
    args = parser.parse_args()

    if args.workspace is None:
        workspace = None
    else:
        workspace = Workspace.from_url(args.workspace)

    limit = args.limit if hasattr(args, "limit") else None

    from catwalk.steps import CalculateMetricsStep
    from catwalk.steps import PredictStep

    tasks = set()
    for task in args.task:
        try:
            tasks |= TASK_SETS[task]
        except KeyError:
            tasks.add(task)

    kwargs = {}
    if args.num_shots is not None:
        kwargs["num_shots"] = args.num_shots
    if args.fewshot_seed is not None:
        kwargs["fewshot_seed"] = args.fewshot_seed

    metric_task_dict = {}
    for task in tasks:
        predictions = PredictStep(
            model=args.model,
            task=task,
            split=args.split,
            batch_size=args.batch_size,
            limit=limit,
            **kwargs)
        metrics = CalculateMetricsStep(
            model=args.model,
            task=task,
            predictions=predictions)
        metric_task_dict[task] = metrics

    table_step = TabulateMetricsStep(metrics=metric_task_dict)
    table_step_result = table_step.result(workspace)
    print("\n".join(table_step_result))


if __name__ == "__main__":
    main()

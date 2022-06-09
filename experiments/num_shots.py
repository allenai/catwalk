import argparse

from tango import Workspace
from tango.common.logging import initialize_logging

from catwalk.steps import PredictStep, CalculateMetricsStep
from catwalk.tasks import TASK_SETS

SHOTS = [0, 1, 2, 4, 8, 16, 32]

DEFAULT_TASKS = {
    "arc_challenge",
    "arc_easy",
    "boolq",
    "copa",
    #"headqa_en",       # Headqa is broken as of 2022-05-05
    "hellaswag",
    "lambada",
    "logiqa",
    "mathqa",
    "mc_taco",
    "mrpc",
    "multirc",
    "openbookqa",
    "piqa",
    "pubmedqa",
    "qnli",
    "qqp",
    "race",
    "rte",
    "sciq",
    "sst",
    "triviaqa",
    "webqs",
    "wic",
    "winogrande",
    "wnli",
    "wsc",
}


def main():
    initialize_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="eai::gpt2")
    parser.add_argument('--task', type=str, nargs="+", default=DEFAULT_TASKS)
    parser.add_argument('--split', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--metric', type=str, nargs="+", default=['acc', 'f1'])
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

    tasks = set()
    for task in args.task:
        try:
            tasks |= TASK_SETS[task]
        except KeyError:
            tasks.add(task)

    results = {}
    for task in tasks:
        for num_shots in SHOTS:
            predictions = PredictStep(
                model=args.model,
                task=task,
                batch_size=args.batch_size,
                limit=limit,
                num_shots=num_shots
            )
            metrics = CalculateMetricsStep(
                model=args.model,
                task=task,
                predictions=predictions)

            result = metrics.result(workspace)
            for metric_name in args.metric:
                metric_value = result.get(metric_name)
                if metric_value is not None:
                    break

            results[(task, num_shots)] = metric_value

    for key, value in results.items():
        task, num_shots = key
        print(f"{task}\t{num_shots}\t{value}")


if __name__ == "__main__":
    main()

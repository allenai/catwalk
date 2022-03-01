import argparse
import json

from catwalk import RunModelOnTaskStep, CalculateMetricsStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--task', type=str, nargs="+")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--limit', type=int)
    parser.add_argument(
        '-d',
        type=str,
        default=None,
        metavar="workspace",
        dest="workspace",
        help="the Tango workspace with the cache")
    args = parser.parse_args()

    if args.workspace is None:
        workspace = None
    else:
        from tango.workspaces import LocalWorkspace
        workspace = LocalWorkspace(args.workspace)

    limit = args.limit if hasattr(args, "limit") else None

    for task in args.task:
        results = RunModelOnTaskStep(
            model=args.model,
            task=task,
            batch_size=args.batch_size,
            limit=limit)
        metrics = CalculateMetricsStep(
            task=task,
            results=results)

        result = metrics.result(workspace)
        print(json.dumps(result, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()

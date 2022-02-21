import argparse
import json

from ludwig import EvalModelOnTaskStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
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
        from tango import LocalWorkspace
        workspace = LocalWorkspace(args.workspace)

    step = EvalModelOnTaskStep(
        model=args.model,
        task=args.task,
        batch_size=args.batch_size)

    result = step.result(workspace)
    print(json.dumps(result, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()

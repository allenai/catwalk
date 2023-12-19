import argparse

from tango import Workspace
from tango.common.logging import initialize_logging


def main():
    initialize_logging(log_level="WARNING")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", "-r", type=str, required=True)
    parser.add_argument("--step", "-s", type=str, default="tabulate")
    parser.add_argument(
        "-d",
        "-w",
        type=str,
        default="beaker://ai2/task-complexity",
        metavar="workspace",
        dest="workspace",
        help="the Tango workspace with the cache",
    )
    args = parser.parse_args()

    workspace = Workspace.from_url(args.workspace)
    r = workspace.step_result_for_run(args.run, args.step)
    print("\n".join(r))


if __name__ == "__main__":
    main()

import argparse

from tango import StepGraph
from tango.common import Params
from tango.common.exceptions import ConfigurationError
from tango.common.logging import initialize_logging
from tango.common.util import import_extra_module

from lm_eval_step import LMEvalStep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_config', type=str, required=True)
    parser.add_argument('--eval_tasks', type=str, default="all_tasks")
    parser.add_argument(
        '-d',
        type=str,
        default=None,
        metavar="workspace",
        dest="workspace",
        help="the Tango workspace with the cache")
    args = parser.parse_args()

    initialize_logging()

    if args.workspace is None:
        workspace = None
    else:
        from tango import LocalWorkspace
        workspace = LocalWorkspace(args.workspace)

    # TODO: dirkgr: Replace this with StepGraph.from_file() in the next version of Tango
    #params = Params.from_file(args.training_config)
    #for package_name in params.pop("include_package", []):
    #    import_extra_module(package_name)
    #step_graph = StepGraph(params.pop("steps", keep_as_dict=True))
    step_graph = StepGraph.from_file(args.training_config)

    try:
        training_step = step_graph["trained_model"]
    except KeyError:
        raise ConfigurationError("The configuration file you specify must contain a step named 'trained_model' "
                                 "that returns the trained model.")

    step = LMEvalStep(
        model=training_step,
        tasks=args.eval_tasks.split(","),
        batch_size=32)

    result = step.result(workspace)
    from lm_eval.evaluator import make_table
    print(make_table(result))


if __name__ == "__main__":
    main()
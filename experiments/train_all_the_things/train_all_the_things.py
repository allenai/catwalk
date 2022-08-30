import argparse

from tango import Workspace, StepGraph, StepResources
from tango.common.logging import initialize_logging
from tango.integrations.beaker import BeakerExecutor, BeakerWorkspace
from tango.step_info import StepState

from catwalk.tasks import TASK_SETS

RANDOM_SEEDS = [42, 1337, 2147483647, 1, 1985]


def main():
    initialize_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+")
    parser.add_argument("--task", type=str, nargs="+")
    parser.add_argument(
        "-d",
        "-w",
        type=str,
        default=None,
        metavar="workspace",
        dest="workspace",
        help="the Beaker workspace with the cache",
        required=True
    )
    args = parser.parse_args()

    workspace = BeakerWorkspace(args.workspace)

    tasks = set()
    for task in args.task:
        try:
            tasks |= TASK_SETS[task]
        except KeyError:
            tasks.add(task)

    step_graph = {}
    metric_steps = {}

    from catwalk.steps import CalculateMetricsStep
    from catwalk.steps import PredictStep
    from catwalk.steps import FinetuneStep
    from catwalk.steps import TabulateMetricsStep

    step_resources_with_gpu = StepResources(gpu_count=1)
    step_resources_without_gpu = StepResources(gpu_count=0)

    for model in args.model:
        batch_size = 16
        grad_acc = 1
        if "xlarge" in model:
            batch_size /= 2
            grad_acc *= 2
        if "xxlarge" in model:
            batch_size /= 2
            grad_acc *= 2

        for task in tasks:
            for seed in RANDOM_SEEDS:
                step_graph_name_suffix = f"{model}_{task}_{seed}"

                model_step = FinetuneStep(
                    model=model,
                    tasks=tasks,
                    batch_size=batch_size,
                    grad_accum=grad_acc,
                    random_seed=seed,
                    step_resources=step_resources_with_gpu
                )
                step_graph["finetune_" + step_graph_name_suffix] = model_step

                pred_step = PredictStep(
                    model=model_step,
                    task=task,
                    batch_size=batch_size * 2,
                    step_resources=step_resources_with_gpu
                )
                step_graph["predict_" + step_graph_name_suffix] = pred_step

                metrics_step = CalculateMetricsStep(
                    model=model_step,
                    task=task,
                    predictions=pred_step,
                    step_resources=step_resources_without_gpu
                )
                step_graph["metrics_" + step_graph_name_suffix] = metrics_step
                metric_steps[step_graph_name_suffix] = metrics_step

    table_step = TabulateMetricsStep(metrics=metric_steps)
    step_graph["make_table"] = table_step

    executor = BeakerExecutor(
        workspace=workspace,
        clusters=["ai2/allennlp-cirrascale", "ai2/general-cirrascale", "ai2/general-gcp"],
        beaker_workspace=args.workspace,
        venv_name="base",
        docker_image="ghcr.io/allenai/pytorch:1.12.0-cuda11.3-python3.9",
        parallelism=32
    )
    executor.execute_step_graph(StepGraph(step_graph))

    assert workspace.step_info(table_step).state == StepState.COMPLETED
    table_step_result = table_step.result(workspace)
    print("\n".join(table_step_result))


if __name__ == "__main__":
    main()

import argparse
import json
import logging

from tango import Workspace
from tango.common.logging import initialize_logging

from catwalk.dependencies.lm_eval.utils import simple_parse_args_string
from catwalk.models import MODELS, add_decoder_only_model
from catwalk.steps import TabulateMetricsStep
from catwalk.tasks import TASKS, TASK_SETS, get_instances
from catwalk.utils import guess_instance_id, sanitize


_parser = argparse.ArgumentParser()
_parser.add_argument('--model', type=str, required=True)
_parser.add_argument('--task', type=str, nargs="+")
_parser.add_argument('--split', type=str)
_parser.add_argument('--batch_size', type=int, default=32)
_parser.add_argument('--num_shots', type=int)
_parser.add_argument('--fewshot_seed', type=int)
_parser.add_argument('--limit', type=int)
_parser.add_argument('--full_output_file', type=str, default=None, help="Filename for verbose output")
_parser.add_argument('--metrics_file', type=str, default=None, help="Filename for metrics output")
_parser.add_argument('-d', '-w', type=str, default=None, metavar="workspace", dest="workspace", help="the Tango workspace with the cache")


def main(args: argparse.Namespace):
    initialize_logging(log_level="WARNING")
    logger = logging.getLogger()

    if args.workspace is None:
        workspace = None
    else:
        workspace = Workspace.from_url(args.workspace)

    # Add arbitrary pretrained Huggingface models to MODELS on the fly
    # TODO add support for model types other than decoder_only
    if args.model not in MODELS:
        prefix_split = args.model.split("::", 1)
        model_name = prefix_split[-1]
        # prefix = "" if len(prefix_split) == 1 else prefix_split[0]+"::"
        model_args = simple_parse_args_string(model_name)
        if 'pretrained' not in model_args:
            raise ValueError(f"Unknown model {args.model}")
        hf_name = model_args['pretrained']
        del model_args['pretrained']
        logger.info(f"Dynamically adding decoder-only models for: {model_name}")
        add_decoder_only_model(model_name, hf_name, **model_args)
        if args.model not in MODELS:
            # Happens if prefix not present
            raise ValueError(f"Unknown model {args.model}")


    limit = args.limit if hasattr(args, "limit") else None
    save_output = args.full_output_file or args.metrics_file

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
    random_subsample_seed = None

    metric_task_dict = {}
    if save_output:
        verbose_output = []
    for task in tasks:
        logger.info(f"Processing task: {task}")
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
        if save_output:
            task_obj = task
            if isinstance(task_obj, str):
                task_obj = TASKS[task]
            split = args.split
            if split is None:
                split = task_obj.default_split
            instances = get_instances(task_obj, split, limit, random_subsample_seed)
            predictions_explicit = list(predictions.result(workspace))
            metrics_explicit = metrics.result(workspace)
            output = {"task": task, "model": args.model, "split": split, "limit": limit, "metrics": metrics_explicit}
            output["per_instance"] = [{"instance": guess_instance_id(inst), "prediction": prediction} for \
                                        inst, prediction in zip(instances, predictions_explicit)]
            verbose_output.append(output)

    if args.full_output_file:
        logger.info(f"Saving verbose output in {args.full_output_file}...")
        with open(args.full_output_file, 'w') as file:
            for d in verbose_output:
                file.write(json.dumps(sanitize(d)) + "\n")
    if args.metrics_file:
        logger.info(f"Saving metrics in {args.metrics_file}...")
        with open(args.metrics_file, 'w') as file:
            for d in verbose_output:
                del d['per_instance']  # Destructive
            file.write(json.dumps(sanitize(verbose_output)))
    table_step = TabulateMetricsStep(metrics=metric_task_dict)
    table_step_result = table_step.result(workspace)
    logger.info("Overall metrics:")
    logger.info("\n  " + "\n  ".join(table_step_result))


if __name__ == "__main__":
    main(_parser.parse_args())

import argparse
import json
import logging
import time
from pydoc import locate
from typing import Dict, List

import torch
from tango.common.logging import initialize_logging

from catwalk.dependencies.lm_eval.utils import simple_parse_args_string
from catwalk.models import MODELS, add_decoder_only_model
from catwalk.steps_simple import CalculateMetricsStep, PredictStep, get_instances
from catwalk.task import rc_metrics
from catwalk.tasks import TASK_SETS, TASKS
from catwalk.tasks.tasks_lm import TASKS_LM
from catwalk.utils import filter_dict_keys, guess_instance_id, sanitize

# Catwalk eval script which is focused on LM models referenced on the fly

_parser = argparse.ArgumentParser()
_parser.add_argument("--model", type=str, required=True, help="Name of model")
_parser.add_argument("--task", type=str, nargs="+")
_parser.add_argument("--task_file", type=str, help="Jsonl file with task specs")
_parser.add_argument("--split", type=str, default="validation")
_parser.add_argument("--batch_size", type=int, default=32)
_parser.add_argument(
    "--max_batch_tokens", type=int, help="Limit batch size to max tokens"
)
_parser.add_argument(
    "--model_max_length", type=int, help="Max input length the model should accept"
)
_parser.add_argument("--num_shots", type=int, help="Number of examples in prompt")
_parser.add_argument(
    "--fewshot_seed",
    type=int,
    help="Random seed for picking fixed prompt examples, leave out for varied examples",
)
_parser.add_argument("--limit", type=int, help="Max number of instances for a task")
_parser.add_argument(
    "--full_output_file", type=str, default=None, help="Filename for verbose output"
)
_parser.add_argument(
    "--metrics_file", type=str, default=None, help="Filename for metrics output"
)
_parser.add_argument(
    "--num_recorded_inputs",
    type=int,
    default=0,
    help="Number of sample model inputs in full output, for sanity checks",
)
_parser.add_argument("--model_path", type=str, help="Explicit path to load model from")
_parser.add_argument(
    "--model_class", type=str, help="Custom Python class for loading model"
)
_parser.add_argument(
    "--random_subsample_seed",
    type=int,
    help="Random seed for subsampling task instances using limit",
)


def main(args: argparse.Namespace):
    initialize_logging(log_level="INFO")
    logger = logging.getLogger()

    # if args.workspace is None:
    #    workspace = None
    # else:
    #    workspace = Workspace.from_url(args.workspace)

    # Add arbitrary pretrained Huggingface models to MODELS on the fly
    # TODO add support for model types other than decoder_only
    if args.model not in MODELS:
        prefix_split = args.model.split("::", 1)
        model_name = prefix_split[-1]
        # prefix = "" if len(prefix_split) == 1 else prefix_split[0]+"::"
        model_args = simple_parse_args_string(model_name)
        if "pretrained" not in model_args:
            raise ValueError(f"Unknown model {args.model}")
        hf_name = model_args["pretrained"]
        del model_args["pretrained"]
        if args.model_path:
            hf_name = args.model_path
        if args.model_class:
            model_args["model_class"] = locate(args.model_class)
            # Assuming tokenizer will be loaded with model, so fail if trying to load it otherwise
            model_args["pretrained_tokenizer_name_or_path"] = "UnknownTokenizer"

        logger.info(f"Dynamically adding decoder-only models for: {model_name}")
        add_decoder_only_model(model_name, hf_name, **model_args)
        if args.model not in MODELS:
            # Happens if prefix not present
            raise ValueError(f"Unknown model {args.model}")

    default_task_args = {"limit": args.limit if hasattr(args, "limit") else None}
    default_task_args["split"] = args.split
    default_task_args["batch_size"] = args.batch_size

    if args.model_max_length is not None:
        default_task_args["model_max_length"] = args.model_max_length
    if args.max_batch_tokens is not None:
        default_task_args["max_batch_tokens"] = args.max_batch_tokens
    if args.num_shots is not None:
        default_task_args["num_shots"] = args.num_shots
    if args.fewshot_seed is not None:
        default_task_args["fewshot_seed"] = args.fewshot_seed
    if args.num_recorded_inputs:
        default_task_args["num_recorded_inputs"] = args.num_recorded_inputs
    if args.random_subsample_seed:
        default_task_args["random_subsample_seed"] = args.random_subsample_seed

    tasks = []
    task_names = set()
    if args.task_file:
        with open(args.task_file, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):
                    task_spec = json.loads(line)
                    tasks.append(task_spec)
                    task_names.add(task_spec["name"])

    if args.task:
        for task in args.task:
            if task in TASK_SETS:
                raise ValueError("Task sets not supported!")
            if task in task_names:
                continue
            task_names.add(task)
            tasks.append({"name": task})

    if not tasks:
        raise ValueError("No tasks specified!")

    # Normalize the tasks, check that they exist, etc
    for task in tasks:
        task_name = task["name"]
        if task_name in TASKS_LM:
            task_obj = TASKS_LM[task_name]
        elif task_name in TASKS:
            task_obj = TASKS[task_name]
        else:
            raise ValueError(f"Task name {task_name} not known!")
        if "task_options" in task:
            if not hasattr(task_obj, "clone"):
                raise ValueError("Cannot specify task_options for this task")
            task_obj = task_obj.clone(**task["task_options"])
        if "task_rename" in task:
            task["name"] = task_name = task["task_rename"]
        task["task_obj"] = task_obj
        if "split" not in task and not default_task_args["split"]:
            task["split"] = task_obj.default_split
        # TODO support various task construction overrides here?
        # Hack to change MC accuracy metrics TODO Fix this!
        if "relative_improvement" in task_obj.metrics or "primary_metric" in task:
            kwargs = {}
            if "primary_metric" in task:
                kwargs["primary"] = task["primary_metric"]
                logger.info(
                    f"Overriding metric for {task_name} with rc_metrics ({kwargs})"
                )
            else:
                logger.warning(
                    f"Overriding 'acc' metric for {task_name} with rc_metrics"
                )
            task_obj.metrics = {}
            task_obj.add_metrics(rc_metrics(**kwargs))
        if "unconditioned_prompt" not in task:
            if hasattr(task_obj, "inner_task") and hasattr(
                task_obj.inner_task, "unconditioned_prompt"
            ):
                prompt = task_obj.inner_task.unconditioned_prompt()
                logger.info(f"Using unconditioned prompt for {task_name}: '{prompt}'")
                task["unconditioned_prompt"] = prompt

    verbose_output = []

    # Initial loading of model done here for early failures and overrides if needed
    model_obj = MODELS[args.model]
    if hasattr(model_obj, "_make_model"):
        logger.info("Loading model...")
        model_cached = model_obj._make_model(
            model_obj.pretrained_model_name_or_path,
            device_map="auto" if torch.cuda.device_count() > 0 else None,
            **model_obj.model_kwargs,
        ).eval()
        if not hasattr(model_cached, "tokenizer"):
            tokenizer_cached = model_obj._make_tokenizer()

    valid_model_args = [
        "split",
        "limit",
        "batch_size",
        "max_batch_tokens",
        "num_shots",
        "model_max_length",
        "fewshot_seed",
        "num_recorded_inputs",
        "unconditioned_prompt",
        "random_subsample_seed",
    ]
    for task in tasks:
        start_time = time.time()
        task_name = task["name"]
        task_obj = task["task_obj"]
        logger.info(f"Processing task: {task_name}")
        task_dict = task.copy()
        task_dict.update(default_task_args)
        predictions = PredictStep().run(
            model=model_obj,
            task=task_obj,
            **filter_dict_keys(task_dict, valid_model_args),
        )
        metrics_calculated, predictions_updated = CalculateMetricsStep().run(
            model=model_obj, task=task_obj, predictions=predictions
        )
        instances = get_instances(
            task_obj,
            **filter_dict_keys(task_dict, ["split", "limit", "random_subsample_seed"]),
        )
        output = {
            "task": task_name,
            "model": args.model,
            "task_options": filter_dict_keys(
                task_dict, valid_model_args, remove_none=True
            ),
            "metrics": metrics_calculated,
            "num_instances": len(instances),
            "processing_time_seconds": time.time() - start_time,
        }
        if "task_options" in task_dict:
            output["custom_task_options"] = task_dict["task_options"]
        logger.info(f"Results from task {task_name}: {output}")
        per_instance: List = []
        pred: Dict
        for instance, pred in zip(instances, predictions_updated):  # type: ignore
            instance_id = guess_instance_id(instance, idx=len(per_instance))
            if "keep_instance_fields" in task_dict:
                for field in task_dict["keep_instance_fields"]:
                    if field in instance:
                        instance_id[field] = instance[field]
            prediction = pred.get("prediction", pred)
            model_input = None
            # Move model_input from prediction if need be
            if "model_input" in pred:
                model_input = pred["model_input"]
                if "model_input" in prediction:
                    del prediction["model_input"]
            res1 = {"instance": instance_id, "prediction": prediction}
            if model_input is not None:
                res1["model_input"] = model_input
            per_instance.append(res1)
        output["per_instance"] = per_instance
        if hasattr(task_obj, "process_extra_output"):
            output = task_obj.process_extra_output(output)
        if per_instance:
            logger.info(
                f"First instance details for task {task_name}: {per_instance[0]}"
            )
        verbose_output.append(output)
        if args.full_output_file:
            logger.info(f"Saving full output in {args.full_output_file}...")
            with open(args.full_output_file, "w") as file:
                for d in verbose_output:
                    file.write(json.dumps(sanitize(d)) + "\n")

    if args.metrics_file:
        logger.info(f"Saving metrics in {args.metrics_file}...")
        with open(args.metrics_file, "w") as file:
            for d in verbose_output:
                del d["per_instance"]  # Destructive
            file.write(json.dumps(sanitize({"metrics": verbose_output})))

    metrics_printed = []
    for d in verbose_output:
        metrics_printed.append(
            f" *** {d['task']} ***  (n = {d['num_instances']})  [{d['task_options']}]"
        )
        metrics: Dict = {}
        # Code is a bit confused about nestedness of metrics
        for metric_name, metric in d["metrics"].items():
            if isinstance(metric, dict):
                metrics.update(metric)
            else:
                metrics[metric_name] = metric
        for metric_name, metric in metrics.items():
            metrics_printed.append(f"    {metric_name}: {metric}")
        metrics_printed.append("-----------------")
    logger.info("Overall metrics:\n  " + "\n".join(metrics_printed))


if __name__ == "__main__":
    main(_parser.parse_args())


import argparse

from tango import Workspace
from tango.common.logging import initialize_logging
from tango.integrations.transformers.ia3 import modify_with_ia3, MODEL_NAME_TO_CONFIG

from catwalk import cached_transformers
from catwalk.models.rank_classification import DecoderOnlyRCModel
from catwalk.steps import TabulateMetricsStep, FinetuneStep
from catwalk.tasks import TASK_SETS

import torch

from transformers import AutoModelForCausalLM, GPT2LMHeadModel


class DecoderOnlyIA3Mixin:
    @classmethod
    def _make_model(self, pretrained_model_name_or_path: str, *, ia3_weights_file: str = None, make_copy: bool = True, **kwargs) -> GPT2LMHeadModel:
        model = cached_transformers.get(AutoModelForCausalLM, pretrained_model_name_or_path, make_copy=make_copy, **kwargs)
        assert pretrained_model_name_or_path in MODEL_NAME_TO_CONFIG, f"{pretrained_model_name_or_path} not in built in IA3 configs. Please add your own IA3ForGPT2Config."
        config = MODEL_NAME_TO_CONFIG[pretrained_model_name_or_path]
        model = modify_with_ia3(model, config)
        if ia3_weights_file is not None:
            state_dict = torch.load(ia3_weights_file)
            model.load_state_dict(state_dict, strict=False)
        
        return model


class IA3DecoderOnlyRCModel(DecoderOnlyIA3Mixin, DecoderOnlyRCModel):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        likelihood_averaging: str = 'char',
        ia3_weights_file: str = None,
        **model_kwargs
    ):
        super().__init__(
            pretrained_model_name_or_path,
            likelihood_averaging=likelihood_averaging,
            ia3_weights_file=ia3_weights_file,
            **model_kwargs
        )


def main():
    initialize_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, nargs="+")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_acc", type=int, default=1)
    parser.add_argument("--device_count", type=int, default=1)
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

    assert args.model in MODEL_NAME_TO_CONFIG, f'no default IA3 config for {args.model}'
    model = IA3DecoderOnlyRCModel(args.model)

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
            model=model,
            tasks=tasks,
            batch_size=args.batch_size,
            grad_accum=args.grad_acc,
            device_count=args.device_count
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

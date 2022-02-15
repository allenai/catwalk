import argparse
from typing import Optional, List, Union

from tango import Step, JsonFormat
from tango.integrations.torch.util import resolve_device
from transformers import GPT2LMHeadModel, AutoModelForCausalLM


class LMEvalStep(Step):
    VERSION = "001"

    SKIP_ID_ARGUMENTS = {"batch_size"}
    FORMAT = JsonFormat()

    def run(
        self,
        model: Union[str, GPT2LMHeadModel],     # TODO: Allow more types of model
        tasks: List[str],
        model_args: str = "",
        num_fewshot: int = 0,
        batch_size: int = None,
        limit: int = None,
        bootstrap_iters: int = 100000,
    ):
        from lm_eval.evaluator import simple_evaluate

        tasks = set(tasks)
        if "all_tasks" in tasks:
            from lm_eval.tasks import ALL_TASKS
            tasks = ALL_TASKS

        device = resolve_device()

        if isinstance(model, str):
            model_name = model
        else:
            # Because lm_eval can't handle being passed a model directly, we hack it up.
            from lm_eval.models.gpt2 import HFLM
            hflm = HFLM(batch_size=batch_size, device=str(device))
            hflm.gpt2 = model
            hflm.gpt2.to(hflm.device)
            hflm.gpt2.eval()
            # TODO: When we allow model types other than GPT2, we have to handle the tokenizer here.
            model = hflm

            if isinstance(self.kwargs["model"], Step):
                model_name = self.kwargs["model"].unique_id
            else:
                model_name = None       # We have nothing to go on ...

        result = simple_evaluate(
            model=model,
            model_args=model_args,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=str(device),
            limit=limit,
            bootstrap_iters=bootstrap_iters,
            no_cache=True   # Caching in lm_eval is broken when you pass in a model, and Tango will cache anyways.
        )

        # lm_eval just sticks the input model into the result, making it unserializable if we pass in a model
        # instance instead of a huggingface model string. So we replace it.
        result['config']['model'] = model_name

        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_args', type=str, default="")
    parser.add_argument('--tasks', type=str, default="all_tasks")
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--limit', type=int, default=None)
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

    step = LMEvalStep(
        model=args.model,
        model_args=args.model_args,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit)

    result = step.result(workspace)
    from lm_eval.evaluator import make_table
    print(make_table(result))


if __name__ == "__main__":
    main()

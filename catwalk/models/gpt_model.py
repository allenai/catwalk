from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import more_itertools
import torch
from tango.common import Tqdm
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

from catwalk.models.model import ModelForEvaluation
from catwalk.tasks import GenerationTask, PairClassificationTask, ClassificationTask, QATask, MCTask
from catwalk.tasks.perplexity_task import PerplexityTask


@ModelForEvaluation.register("gpt")
class GPTModel(ModelForEvaluation):
    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def do_generation(
        self,
        task: GenerationTask,
        instances: Iterable[GenerationTask.Instance],
        **kwargs
    ) -> Iterator[GenerationTask.InstanceResult]:
        raise NotImplementedError

    def do_multiple_choice(
        self,
        task: MCTask,
        instances: Iterable[MCTask.Instance],
        **kwargs
    ) -> Iterator[MCTask.InstanceResult]:
        raise NotImplementedError

    def do_qa(
        self,
        task: QATask,
        instances: Iterable[QATask.Instance],
        **kwargs
    ) -> Iterator[QATask.InstanceResult]:
        raise NotImplementedError

    def do_classification(
        self,
        task: ClassificationTask,
        instances: Iterable[ClassificationTask.Instance],
        **kwargs
    ) -> Iterator[ClassificationTask.InstanceResult]:
        raise NotImplementedError

    def do_pair_classification(
        self,
        task: PairClassificationTask,
        instances: Iterable[PairClassificationTask.Instance],
        **kwargs
    ) -> Iterator[PairClassificationTask.InstanceResult]:
        raise NotImplementedError

    def do_perplexity(
        self,
        task: PerplexityTask,
        instances: Iterable[PerplexityTask.Instance],
        *,
        batch_size: int = 32,
    ) -> Iterator[PerplexityTask.InstanceResult]:
        model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)

        @dataclass
        class ModelInstance:
            instance: PerplexityTask.Instance
            num_context_tokens: int
            input_ids: torch.Tensor
            targets: torch.Tensor

        def make_model_instances(
            instances: Iterable[PerplexityTask.Instance],
            overlap: int = 1
        ) -> Iterator[ModelInstance]:
            for instance in instances:
                token_ids = [tokenizer.eos_token_id] + tokenizer.encode(instance.text)
                token_ids = torch.tensor(token_ids, dtype=torch.long)
                window_start = 0
                while True:
                    window_end = window_start + tokenizer.model_max_length
                    if window_end > len(token_ids) - 1:
                        break
                    yield ModelInstance(
                        instance,
                        1 if window_start == 0 else overlap,
                        token_ids[window_start:window_end],
                        token_ids[window_start+1:window_end+1])
                    window_start += tokenizer.model_max_length
                    window_start -= overlap
                window_end = len(token_ids) - 1
                if window_start == 0:
                    last_batch_context_tokens = 1
                else:
                    new_window_start = window_end - tokenizer.model_max_length
                    last_batch_context_tokens = window_start - new_window_start + overlap
                    window_start = new_window_start
                    del new_window_start
                yield ModelInstance(
                    instance,
                    last_batch_context_tokens,
                    token_ids[window_start:window_end],
                    token_ids[window_start+1:window_end+1])

        def make_model_predictions(model_instances: Iterator[ModelInstance]) -> Iterator[Tuple[PerplexityTask.Instance, torch.Tensor]]:
            for batch in more_itertools.chunked(model_instances, batch_size):
                batch_results = []
                with torch.inference_mode():
                    inputs = pad_sequence([mi.input_ids for mi in batch], batch_first=True)
                    outputs = model(inputs)
                    outputs = log_softmax(outputs.logits, dim=-1).cpu()
                    for mi, output in zip(batch, outputs):
                        output = output[:len(mi.targets)]   # gets rid of padding
                        logprobs = torch.gather(
                            output[mi.num_context_tokens:],
                            1,
                            mi.targets[mi.num_context_tokens:].unsqueeze(-1)).squeeze(-1)
                        batch_results.append((mi.instance, logprobs))
                yield from batch_results

        model_instances = make_model_instances(Tqdm.tqdm(instances, desc="Calculating log probabilities"))
        model_predictions = make_model_predictions(model_instances)
        last_instance = None
        summed_logprobs = 0.0
        for instance, logprobs in model_predictions:
            if last_instance is not None and instance != last_instance:
                yield PerplexityTask.InstanceResult(last_instance, float(summed_logprobs))
                summed_logprobs = 0.0
            summed_logprobs += logprobs.sum()
            last_instance = instance
        yield PerplexityTask.InstanceResult(last_instance, float(summed_logprobs))

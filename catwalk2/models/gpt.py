from dataclasses import dataclass
from typing import Sequence, Dict, Any, Iterator, Tuple

import more_itertools
import torch
from tango.common import Tqdm
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from catwalk2.task import Task
from catwalk2.model import Model


@Model.register("catwalk::gpt")
class GPTModel(Model):
    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def predict(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)

        @dataclass
        class ModelInstance:
            text: str
            num_context_tokens: int
            input_ids: torch.Tensor
            targets: torch.Tensor

        def make_model_instances(
            texts: Iterator[str],
            overlap: int = 1
        ) -> Iterator[ModelInstance]:
            for text in texts:
                token_ids = [tokenizer.eos_token_id] + tokenizer.encode(text)
                token_ids = torch.tensor(token_ids, dtype=torch.long)
                window_start = 0
                while True:
                    window_end = window_start + tokenizer.model_max_length
                    if window_end > len(token_ids) - 1:
                        break
                    yield ModelInstance(
                        text,
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
                    text,
                    last_batch_context_tokens,
                    token_ids[window_start:window_end],
                    token_ids[window_start+1:window_end+1])

        def make_model_predictions(model_instances: Iterator[ModelInstance]) -> Iterator[Tuple[str, torch.Tensor]]:
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
                        batch_results.append((mi.text, logprobs))
                yield from batch_results

        def group_model_predictions(model_predictions: Iterator[Tuple[str, torch.Tensor]]) -> Iterator[Tuple[str, float]]:
            last_text = None
            summed_logprobs = 0.0
            for text, logprobs in model_predictions:
                if last_text is not None and text != last_text:
                    yield last_text, float(summed_logprobs)
                    summed_logprobs = 0.0
                summed_logprobs += logprobs.sum()
                last_text = text
            yield last_text, float(summed_logprobs)

        model_instances = make_model_instances(
            instance["text"] for instance in Tqdm.tqdm(
                instances,
                desc="Calculating log probabilities")
        )
        model_predictions = make_model_predictions(model_instances)
        grouped_predictions = group_model_predictions(model_predictions)

        from spacy.lang.en import English
        spacy_tokenizer = English().tokenizer
        for text, logprob in grouped_predictions:
            yield {
                "text": text,
                "word_perplexity": (logprob, len(spacy_tokenizer(text))),
                "byte_perplexity": (logprob, len(text)),        # bytes aren't characters, but this is what Eleuther calls it
                "bits_per_byte": (logprob, len(text))
            }

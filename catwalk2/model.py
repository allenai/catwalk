import math
import re
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any, Iterator, Tuple

import more_itertools
import torch
from tango.common import Tqdm, Registrable
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from catwalk2.task import Task


class Model(Registrable, ABC):
    def predict(self, task: Task, **kwargs) -> Any:
        raise NotImplementedError()

    def calculate_metrics(self, task: Task, predictions: Any) -> Dict[str, float]:
        raise NotImplementedError()


@Model.register("catwalk::gpt")
class GPTModel(Model):
    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def predict(self, task: Task, *, batch_size: int = 32) -> Iterator[Tuple[str, float]]:
        model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)

        def wikitext_detokenizer(string: str) -> str:
            # contractions
            string = string.replace("s '", "s'")
            string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
            # number separators
            string = string.replace(" @-@ ", "-")
            string = string.replace(" @,@ ", ",")
            string = string.replace(" @.@ ", ".")
            # punctuation
            string = string.replace(" : ", ": ")
            string = string.replace(" ; ", "; ")
            string = string.replace(" . ", ". ")
            string = string.replace(" ! ", "! ")
            string = string.replace(" ? ", "? ")
            string = string.replace(" , ", ", ")
            # double brackets
            string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
            string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
            string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
            string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
            string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
            # miscellaneous
            string = string.replace("= = = =", "====")
            string = string.replace("= = =", "===")
            string = string.replace("= =", "==")
            string = string.replace(" " + chr(176) + " ", chr(176))
            string = string.replace(" \n", "\n")
            string = string.replace("\n ", "\n")
            string = string.replace(" N ", " 1 ")
            string = string.replace(" 's", "'s")

            return string

        def make_texts(ds_instances: Iterator[Dict[str, Any]]) -> Iterator[str]:
            ret = []
            for line in ds_instances:
                # Stolen from Eleuther
                line = line["text"]
                rline = line.replace("= = =", "===").replace("= =", "==").strip()
                if rline.startswith('= ') and rline.strip().endswith(' ='):
                    s = '\n'.join(ret)
                    if s.strip():
                        yield wikitext_detokenizer(s)
                    ret = []
                ret.append(line)
            yield wikitext_detokenizer('\n'.join(ret))

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

        texts = make_texts(Tqdm.tqdm(
            task.get_split("validation"),
            desc="Calculating log probabilities"))
        model_instances = make_model_instances(texts)
        model_predictions = make_model_predictions(model_instances)
        last_text = None
        summed_logprobs = 0.0
        for text, logprobs in model_predictions:
            if last_text is not None and text != last_text:
                yield last_text, float(summed_logprobs)
                summed_logprobs = 0.0
            summed_logprobs += logprobs.sum()
            last_text = text
        yield last_text, float(summed_logprobs)

    def calculate_metrics(self, task: Task, predictions: Iterator[Tuple[str, float]]) -> Dict[str, float]:
        from spacy.lang.en import English
        tokenizer = English().tokenizer

        logprob_sum = 0.0
        characters = 0
        words = 0
        for text, logprob in predictions:
            logprob_sum += logprob
            words += len(tokenizer(text))
            characters += len(text)
        return {
            "word_perplexity": math.exp(-logprob_sum / words),
            "byte_perplexity": math.exp(-logprob_sum / characters),        # bytes aren't characters, but this is what Eleuther calls it
            "bits_per_byte": -(logprob_sum / characters) / math.log(2)
        }


MODELS = {
    "gpt2": GPTModel("gpt2"),
}
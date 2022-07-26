from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import more_itertools
import torch
from catwalk import cached_transformers
from catwalk.model import Model
from catwalk.task import InstanceFormat, Task
from tango.common import Tqdm
from tango.integrations.torch.util import resolve_device
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer


@Model.register("catwalk::gpt")
class GPTModel(Model):
    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32,
    ) -> Iterator[Dict[str, Any]]:
        if task.has_instance_conversion(InstanceFormat.HF_QA):
            return self._predict_qa(task, instances, batch_size=batch_size)

        raise self._predict_preplixity(task, instances, batch_size=batch_size)

    def _predict_qa(
            self,
            task: Task,
            instances: Sequence[Dict[str, Any]],
            *,
            batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        device = resolve_device()
        model = cached_transformers.get(
            AutoModelForCausalLM, self.pretrained_model_name_or_path, False).eval().to(device)
        tokenizer = cached_transformers.get_tokenizer(
            AutoTokenizer, self.pretrained_model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        def format_instance(instance: Dict[str, Any]) -> Tuple[str, str]:
            # TODO: Use promptsource to add more prompt options?
            return instance['context'], f"\nQuestion:{instance['question']}\nAnswer:"

        instances = Tqdm.tqdm(instances, desc="Processing instances")
        for batch in more_itertools.chunked(instances, batch_size):
            formatted_batch = [format_instance(instance) for instance in batch]
            encodings = tokenizer(formatted_batch,
                                  padding="max_length",
                                  truncation="only_first",
                                  max_length=384,  # TODO: Make this configurable?
                                  stride=128,
                                  return_overflowing_tokens=True,
                                  return_offsets_mapping=True,
                                  return_token_type_ids=True,
                                  return_tensors="pt"
                                  )
            sample_map = encodings.pop("overflow_to_sample_mapping")
            offset_mapping = encodings.pop("offset_mapping")
            token_type_map = encodings.pop("token_type_ids")

            sample_to_example_idx = defaultdict(list)
            for i, sample in enumerate(sample_map):
                sample_to_example_idx[sample.item()].append(i)

            filtered_encodings = {"input_ids": [], "attention_mask": []}

            # We need to use the offset mapping to get the actual start and end indices of the context.
            # To do this we need to find the index of the first/last token in the context.
            # For example,
            # Tokens:         PADDING CONTEXT QUESTION
            # Token Type IDs:    0        0       1 (Used to find the last token of the context)
            # Attention Mask:    0        1       1 (Used to find the first token of the context)

            def get_token_bounds_for_context(sample_idx: int) -> Tuple[int, int]:
                attention_mask = encodings["attention_mask"][sample_idx]
                sequence_ids = token_type_map[sample_idx]

                start_index = 0
                while (attention_mask[start_index] == 0):
                    start_index += 1

                end_index = len(sequence_ids) - 1
                while sequence_ids[end_index] != 0:
                    end_index -= 1

                return start_index, end_index

            def contains_answer(sample_idx: int, answer_indices: Tuple[int, int]) -> bool:
                start_index, end_index = get_token_bounds_for_context(
                    sample_idx)
                offset = offset_mapping[sample_idx]
                return offset[start_index][0] <= answer_indices[0] and answer_indices[1] <= offset[end_index][1]

            # Keep only the first sample for each instance that contains the answer.
            for example_idx in sample_to_example_idx.keys():
                answers = batch[example_idx]["answers"]
                for sample_idx in sample_to_example_idx[example_idx]:
                    for answer_idx in range(len(answers["text"])):
                        start_char = answers["answer_start"][answer_idx]
                        end_char = start_char + \
                            len(answers["text"][answer_idx].strip())
                        if contains_answer(sample_idx, (start_char, end_char)):
                            filtered_encodings["input_ids"].append(
                                encodings["input_ids"][sample_idx])
                            filtered_encodings["attention_mask"].append(
                                encodings["attention_mask"][sample_idx])
                            break # No need to check other answers for this example

                    # check if we found a sample that contains the answer and break out of the loop
                    if len(filtered_encodings["input_ids"])-1 == example_idx:
                        break

            with torch.inference_mode():
                outputs = model.generate(input_ids=torch.stack(filtered_encodings["input_ids"]).to(device),
                                         attention_mask=torch.stack(
                                             filtered_encodings["attention_mask"]).to(device),
                                         max_new_tokens=20,
                                         pad_token_id=tokenizer.pad_token_id,
                                         num_beams=1)
            outputs = [output[384 + 1:] for output in outputs]
            outputs = tokenizer.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            for instance, prediction in zip(batch, outputs):
                yield {
                    "squad_metrics": ({"id": instance["id"], "prediction_text": prediction}, {"id": instance["id"], "answers": instance["answers"]})
                }

    def _predict_preplixity(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32
    ) -> Iterator[Dict[str, Any]]:
        device = resolve_device()
        model = cached_transformers.get(
            AutoModelForCausalLM, self.pretrained_model_name_or_path, False).eval().to(device)
        tokenizer = cached_transformers.get_tokenizer(
            AutoTokenizer, self.pretrained_model_name_or_path)

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
                    inputs = pad_sequence(
                        [mi.input_ids for mi in batch], batch_first=True)
                    outputs = model(inputs)
                    outputs = log_softmax(outputs.logits, dim=-1).cpu()
                    for mi, output in zip(batch, outputs):
                        # gets rid of padding
                        output = output[:len(mi.targets)]
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
            if last_text is not None:
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
                # bytes aren't characters, but this is what Eleuther calls it
                "byte_perplexity": (logprob, len(text)),
                "bits_per_byte": (logprob, len(text))
            }

import collections
from typing import Sequence, Dict, Any, Iterator, Callable, Mapping, List, Tuple

import more_itertools
import torch
from lm_eval.base import Request
from tango.common import Tqdm
from tango.integrations.torch.util import resolve_device
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, \
    AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5TokenizerFast

from catwalk.task import Task, InstanceFormat
from catwalk.model import Model
from catwalk.tasks.eleuther import EleutherTask


@Model.register("eai::gpt")
class EAIGPT(Model):
    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32,
        max_instances_in_memory: int = 16 * 1024
    ) -> Iterator[Dict[str, Any]]:
        device = resolve_device()
        model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)

        for instance_chunk in more_itertools.chunked(instances, max_instances_in_memory):
            yield from self.predict_chunk(task, instance_chunk, model, tokenizer, batch_size=batch_size)

    def predict_chunk(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        batch_size: int = 32,
    ) -> Iterator[Dict[str, Any]]:
        instance_index_to_request_indices: Mapping[int, Mapping[str, List[int]]] = \
            collections.defaultdict(lambda: collections.defaultdict(list))
        requests: Mapping[str, List[Request]] = collections.defaultdict(list)

        # get all the requests
        for instance_index, instance in enumerate(instances):
            instance_requests = task.convert_instance(instance, InstanceFormat.ELEUTHER_REQUESTS)
            if not isinstance(instance_requests, (list, tuple)):
                instance_requests = [instance_requests]
            for instance_request in instance_requests:
                request_type = instance_request.request_type
                instance_index_to_request_indices[instance_index][request_type].append(len(requests[request_type]))
                requests[request_type].append(instance_request)

        # run the requests
        results: Dict[str, Sequence] = {}
        request_type_to_fn: Mapping[str, Callable[[Sequence[Request], GPT2LMHeadModel, GPT2Tokenizer, int], Sequence]] = {
            "loglikelihood": self._run_loglikelihood,
            "loglikelihood_rolling": self._run_loglikelihood_rolling,
            "greedy_until": self._run_greedy_until
        }
        for request_type, requests_per_type in requests.items():
            results[request_type] = request_type_to_fn[request_type](
                requests_per_type,
                model,
                tokenizer,
                batch_size
            )

        assert isinstance(task, EleutherTask), "We can only calculate metrics for EleutherTasks."
        for instance_index, instance in enumerate(instances):
            doc = task.convert_instance(instance, InstanceFormat.ELEUTHER_DOC)

            results_for_instance: List = []
            for request_type, request_indices in instance_index_to_request_indices[instance_index].items():
                results_for_instance.extend(results[request_type][i] for i in request_indices)

            yield task.inner_task.process_results(doc, results_for_instance)

    def _run_loglikelihood(
        self,
        requests: Sequence[Request],
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        batch_size: int = 32,
    ) -> Sequence:
        tokenized_contexts = tokenizer([r.args[0] for r in requests])
        tokenized_continuations = tokenizer([r.args[1] for r in requests])

        # transpose the token ids so we can access them one instance at a time
        cc_pairs: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = []
        assert tokenized_contexts.keys() == tokenized_continuations.keys()
        for field_name in tokenized_contexts.keys():
            contexts = tokenized_contexts[field_name]
            continuations = tokenized_continuations[field_name]
            assert len(contexts) == len(continuations)
            for i, (context, continuation) in enumerate(zip(contexts, continuations)):
                if len(cc_pairs) <= i:
                    cc_pairs.append({})
                if len(context) == 0:
                    context = [tokenizer.eos_token_id]
                cc_pairs[i][field_name] = (
                    torch.tensor(context, dtype=torch.long),
                    torch.tensor(continuation, dtype=torch.long)
                )

        # find out the order to process sequences in
        lengths = torch.tensor([
            len(cc_pair["input_ids"][0]) + len(cc_pair["input_ids"][1])
            for cc_pair in cc_pairs
        ], dtype=torch.int)
        ordered_indices = torch.argsort(lengths, descending=True)
        del lengths

        # actually do the processing
        results = [None] * len(ordered_indices)
        with torch.inference_mode():
            batches_of_indices = more_itertools.chunked(
                Tqdm.tqdm(ordered_indices, desc="Running log-likelihood queries"),
                batch_size)
            for batch_of_indices in batches_of_indices:
                unpadded_batch = collections.defaultdict(list)
                input_lengths = []
                batch_contexts = []
                batch_continuations = []
                for index in batch_of_indices:
                    for field_name, (context_ids, continuation_ids) in cc_pairs[index].items():
                        ids = torch.cat([context_ids, continuation_ids])
                        ids = ids[-(tokenizer.model_max_length+1):][:-1]
                        unpadded_batch[field_name].append(ids)

                    input_lengths.append(len(unpadded_batch["input_ids"][-1]))
                    batch_contexts.append(cc_pairs[index]["input_ids"][0])
                    batch_continuations.append(cc_pairs[index]["input_ids"][1])

                padded_batch = {
                    field_name: pad_sequence(tensors, batch_first=True).to(model.device)
                    for field_name, tensors in unpadded_batch.items()
                }

                batch_logits = log_softmax(model(**padded_batch)[0], dim=-1).cpu()
                z = zip(batch_of_indices, batch_logits, input_lengths, batch_contexts, batch_continuations)
                for i, instance_logits, input_length, instance_context, instance_continuation in z:
                    instance_logits = instance_logits[input_length-len(instance_continuation):input_length].unsqueeze(0)

                    greedy_tokens = instance_logits.argmax(dim=-1)
                    instance_continuation = instance_continuation.unsqueeze(0)
                    max_equal = (greedy_tokens == instance_continuation).all()

                    instance_logits = torch.gather(instance_logits, 2, instance_continuation.unsqueeze(-1)).squeeze(-1)

                    instance_result: Any = (float(instance_logits.sum()), bool(max_equal))
                    if requests[i].index is not None:
                        instance_result = instance_result[requests[i].index]
                    results[i] = instance_result

        assert None not in results
        return results

    def _run_loglikelihood_rolling(
        self,
        requests: Sequence[Request],
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        batch_size: int = 32,
    ) -> Sequence:
        raise NotImplementedError

    def _run_greedy_until(
        self,
        requests: Sequence[Request],
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        batch_size: int = 32,
    ) -> Sequence:
        raise NotImplementedError

    def calculate_metrics(self, task: Task, predictions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        assert isinstance(task, EleutherTask), "We can only calculate metrics for EleutherTasks."
        return {
            key: fn([p[key] for p in predictions])
            for key, fn in task.inner_task.aggregation().items()
        }


@Model.register("eai::t5")
class EAIT5(Model):
    VERSION = "003gat"

    def __init__(self, pretrained_model_name_or_path: str):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32,
        max_instances_in_memory: int = 16 * 1024
    ) -> Iterator[Dict[str, Any]]:
        device = resolve_device()
        model = AutoModelForSeq2SeqLM.from_pretrained(self.pretrained_model_name_or_path).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)

        for instance_chunk in more_itertools.chunked(instances, max_instances_in_memory):
            yield from self.predict_chunk(task, instance_chunk, model, tokenizer, batch_size=batch_size)

    def predict_chunk(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        batch_size: int = 32,
    ) -> Iterator[Dict[str, Any]]:
        instance_index_to_request_indices: Mapping[int, Mapping[str, List[int]]] = \
            collections.defaultdict(lambda: collections.defaultdict(list))
        requests: Mapping[str, List[Request]] = collections.defaultdict(list)

        # get all the requests
        for instance_index, instance in enumerate(instances):
            instance_requests = task.convert_instance(instance, InstanceFormat.ELEUTHER_REQUESTS)
            if not isinstance(instance_requests, (list, tuple)):
                instance_requests = [instance_requests]
            for instance_request in instance_requests:
                request_type = instance_request.request_type
                instance_index_to_request_indices[instance_index][request_type].append(len(requests[request_type]))
                requests[request_type].append(instance_request)

        # run the requests
        results: Dict[str, Sequence] = {}
        request_type_to_fn: Mapping[str, Callable[[Sequence[Request], T5ForConditionalGeneration, T5TokenizerFast, int], Sequence]] = {
            "loglikelihood": self._run_loglikelihood,
            "loglikelihood_rolling": self._run_loglikelihood_rolling,
            "greedy_until": self._run_greedy_until
        }
        for request_type, requests_per_type in requests.items():
            results[request_type] = request_type_to_fn[request_type](
                requests_per_type,
                model,
                tokenizer,
                batch_size
            )

        assert isinstance(task, EleutherTask), "We can only calculate metrics for EleutherTasks."
        for instance_index, instance in enumerate(instances):
            doc = task.convert_instance(instance, InstanceFormat.ELEUTHER_DOC)

            results_for_instance: List = []
            for request_type, request_indices in instance_index_to_request_indices[instance_index].items():
                results_for_instance.extend(results[request_type][i] for i in request_indices)

            yield task.inner_task.process_results(doc, results_for_instance)

    def _run_loglikelihood(
        self,
        requests: Sequence[Request],
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        batch_size: int = 32,
    ) -> Sequence:
        encoder_inputs = tokenizer([r.args[0] for r in requests])
        model_inputs: List[Dict[str, torch.Tensor]] = []
        for field_name, encoder_input in encoder_inputs.items():
            for i, input_as_list in enumerate(encoder_input):
                if len(model_inputs) <= i:
                    model_inputs.append({})
                model_inputs[i][field_name] = torch.tensor(input_as_list, dtype=torch.long)
        del encoder_inputs

        with tokenizer.as_target_tokenizer():
            decoder_inputs = tokenizer([r.args[1] for r in requests], return_attention_mask=False)
        for i, input_as_list in enumerate(decoder_inputs['input_ids']):
            model_inputs[i]["labels"] = torch.tensor(input_as_list, dtype=torch.long)
        del decoder_inputs

        # find out the order to process sequences in
        lengths = torch.tensor([
            len(model_input["input_ids"])
            for model_input in model_inputs
        ], dtype=torch.int)
        ordered_indices = torch.argsort(lengths, descending=True)
        del lengths

        # actually do the processing
        results = [None] * len(ordered_indices)
        with torch.inference_mode():
            batches_of_indices = more_itertools.chunked(
                Tqdm.tqdm(ordered_indices, desc="Running log-likelihood queries"),
                batch_size)
            for batch_of_indices in batches_of_indices:
                unpadded_batch = collections.defaultdict(list)
                for index in batch_of_indices:
                    for field_name, model_input in model_inputs[index].items():
                        unpadded_batch[field_name].append(model_input)
                padded_batch = {
                    field_name: pad_sequence(tensors, batch_first=True).to(model.device)
                    for field_name, tensors in unpadded_batch.items()
                }

                batch_logits = log_softmax(model(**padded_batch).logits, dim=-1).cpu()

                for i, instance_logits, decoder_input_ids in zip(batch_of_indices, batch_logits, unpadded_batch["labels"]):
                    instance_logits = instance_logits[:len(decoder_input_ids)]      # TODO: make sure this works with uneven decoder lengths
                    greedy_tokens = instance_logits.argmax(dim=-1)
                    max_equal = (greedy_tokens == decoder_input_ids).all()

                    instance_logits = torch.gather(instance_logits, 1, decoder_input_ids.unsqueeze(-1))
                    instance_result: Any = (float(instance_logits.sum()), bool(max_equal))
                    if requests[i].index is not None:
                        instance_result = instance_result[requests[i].index]
                    results[i] = instance_result

        assert None not in results
        return results

    def _run_loglikelihood_rolling(
        self,
        requests: Sequence[Request],
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        batch_size: int = 32,
    ) -> Sequence:
        raise NotImplementedError

    def _run_greedy_until(
        self,
        requests: Sequence[Request],
        model: T5ForConditionalGeneration,
        tokenizer: T5TokenizerFast,
        batch_size: int = 32,
    ) -> Sequence:
        raise NotImplementedError

    def calculate_metrics(self, task: Task, predictions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        assert isinstance(task, EleutherTask), "We can only calculate metrics for EleutherTasks."
        return {
            key: fn([p[key] for p in predictions])
            for key, fn in task.inner_task.aggregation().items()
        }

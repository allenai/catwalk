import collections
import re
from typing import Dict, Any, List, Tuple, Sequence, Iterator, Union, Mapping, Optional

import more_itertools
import torch
from tango.common import Tqdm
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, T5ForConditionalGeneration, GPT2LMHeadModel, \
    AutoTokenizer, GPT2Tokenizer, T5TokenizerFast

from catwalk import cached_transformers
from catwalk.model import Model
from catwalk.task import Task, InstanceFormat, RankClassificationInstance
from catwalk.dependencies.lm_eval.utils import get_rolling_token_windows, make_disjoint_window

_Model = Union[T5ForConditionalGeneration, GPT2LMHeadModel]
_Tokenizer = Union[T5TokenizerFast, GPT2Tokenizer]


class LanguageModel(Model):
    VERSION = "002met"

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        pretrained_tokenizer_name_or_path: Optional[str] = None,
        likelihood_averaging: str = 'token',
        **model_kwargs
    ):
        """
        # Parameters

        pretrained_model_name_or_path : `str`
            The name of the transformer, for example `"gpt2-large"`
        likelihood_averaging : `str`, optional (default = `token`)
            The method for averaging the sum likelihood of the continuation. 'char' averages by 
            character length, 'token' averages by token length.
        model_kwargs:
            Additional kwargs passed to the `_make_model` method.
        """
        assert likelihood_averaging in {'char', 'token'}
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_tokenizer_name_or_path is None:
            pretrained_tokenizer_name_or_path = pretrained_model_name_or_path
        self.pretrained_tokenizer_name_or_path = pretrained_tokenizer_name_or_path

        self.likelihood_averaging = likelihood_averaging
        self.model_kwargs = model_kwargs

    @classmethod
    def _make_model(cls, pretrained_model_name_or_path: str, *, make_copy: bool = False, **kwargs) -> _Model:
        raise NotImplementedError

    def _make_tokenizer(self) -> AutoTokenizer:
        return cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_tokenizer_name_or_path)

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32,
        max_batch_tokens: int = None, # If set, max number of tokens in a batch
        max_instances_in_memory: int = 32 * 1024,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None,
        model_max_length: Optional[int] = None, # Max input length model should support
        num_recorded_inputs: Optional[int] = 0,  # Number of instances to log in detail
        unconditioned_prompt: Optional[str] = None, # Optional unconditioned prompt, e.g., "Answer:"
    ) -> Iterator[Dict[str, Any]]:
        model = self._make_model(
            self.pretrained_model_name_or_path,
            device_map="auto" if torch.cuda.device_count() > 0 else None,
            **self.model_kwargs).eval()
        if hasattr(model, "tokenizer"):
            tokenizer = model.tokenizer
        else:
            tokenizer = self._make_tokenizer()

        if task.has_instance_conversion(InstanceFormat.RANK_CLASSIFICATION):
            predictor = self.predict_chunk_rank_classification
        elif task.has_instance_conversion(InstanceFormat.ELEUTHER_DOC):
            # Assume perplexity tasks here, only for Eleuther tasks for now, but easy to add others
            predictor = self.predict_chunk_perplexity
        else:
            raise ValueError("Unknown task type for LM model")

        for instance_chunk in more_itertools.chunked(instances, max_instances_in_memory):
            yield from predictor(
                task,
                instance_chunk,
                model,
                tokenizer,
                batch_size=batch_size,
                max_batch_tokens=max_batch_tokens,
                num_shots=num_shots,
                fewshot_seed=fewshot_seed,
                model_max_length=model_max_length,
                num_recorded_inputs=num_recorded_inputs,
                unconditioned_prompt=unconditioned_prompt
            )

    def predict_chunk_rank_classification(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: int = None,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None,
        model_max_length: Optional[int] = None,
        num_recorded_inputs: Optional[int] = 0,
        unconditioned_prompt: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        instance_index_to_tuple_indices: Mapping[int, List[int]] = collections.defaultdict(list)
        tuples: List[Tuple[str, str]] = []

        rc_instances: List[RankClassificationInstance] = [
            task.convert_instance(
                instance,
                InstanceFormat.RANK_CLASSIFICATION,
                fewshot_instances=task.get_fewshot_instances(
                    num_shots,
                    random_seed=fewshot_seed if fewshot_seed is not None else i,
                    exceptions=instance))
            for i, instance in enumerate(instances)
        ]

        # get all the tuples
        for instance_index, instance in enumerate(rc_instances):
            for instance_request in instance.choices:
                instance_index_to_tuple_indices[instance_index].append(len(tuples))
                tuples.append(instance_request)
        unconditioned_offset = len(tuples)
        if unconditioned_prompt:
            for instance_index, instance in enumerate(rc_instances):
                for instance_request in instance.choices:
                    tuples.append((unconditioned_prompt, instance_request[1]))

        # run the requests
        results = self._run_loglikelihood(tuples, model, tokenizer, batch_size,
                                          max_batch_tokens=max_batch_tokens,
                                          model_max_length=model_max_length)

        # collect the results
        for instance_index, instance in enumerate(rc_instances):
            tuple_indices = instance_index_to_tuple_indices[instance_index]
            results_for_instance = [results[i] for i in tuple_indices]
            if unconditioned_prompt:
                for idx, i in enumerate(tuple_indices):
                    unconditioned_results = results[i + unconditioned_offset]
                    results_for_instance[idx]['sum_logits_uncond'] = unconditioned_results['sum_logits']
            res = {"model_output": results_for_instance, "correct_choice": instance.correct_choice}
            if instance_index < num_recorded_inputs:
                res["model_input"] = [tuples[i] for i in tuple_indices]
                if unconditioned_prompt:
                    res["unconditioned_input"] = [tuples[i + unconditioned_offset] for i in tuple_indices]
            yield res

    def predict_chunk_perplexity(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: int = None,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None,
        model_max_length: Optional[int] = None,
        num_recorded_inputs: Optional[int] = 0,
        unconditioned_prompt: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:

        doc_instances: List[str] = [
            task.convert_instance(instance, InstanceFormat.ELEUTHER_DOC)
            for i, instance in enumerate(instances)
        ]
        truncation_length = tokenizer.model_max_length
        if model_max_length:
            truncation_length = min(truncation_length, model_max_length)
        instance_index_to_cc_indices: Mapping[int, List[int]] = collections.defaultdict(list)
        cc_pairs = []

        for instance_index, doc_instance in enumerate(doc_instances):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=tokenizer.encode(doc_instance, add_special_tokens=False),
                        prefix_token=tokenizer.eos_token_id,
                        max_seq_len=truncation_length,
                        context_len=1,
                    ),
                )
            )
            for context, continuation in rolling_token_windows:
                instance_index_to_cc_indices[instance_index].append(len(cc_pairs))
                cc_pairs.append({"input_ids": (
                    torch.tensor(context, dtype=torch.long),
                    torch.tensor(continuation, dtype=torch.long))})

        results = self._run_loglikelihood_tokens(cc_pairs, model, tokenizer, batch_size,
                                                 max_batch_tokens=max_batch_tokens,
                                                 model_max_length=model_max_length)

        # collect the results
        for instance_index, doc in enumerate(doc_instances):
            cc_indices = instance_index_to_cc_indices[instance_index]
            results_for_instance = [results[i] for i in cc_indices]
            model_output = {"sum_logits": 0, "num_tokens": 0, "num_tokens_all": 0}
            model_output["num_chars"] = len(doc)
            model_output["num_words"] = len(re.split(r"\s+", doc))
            model_output["num_bytes"] = len(doc.encode("utf-8"))
            for result in results_for_instance:
                model_output["sum_logits"] += result["sum_logits"]
                model_output["num_tokens"] += result["num_tokens"]
                model_output["num_tokens_all"] += result["num_tokens_all"]

            res = {"model_output": model_output}
            if instance_index < num_recorded_inputs:
                res["model_input"] = []
                for index in cc_indices:
                    inp = [tokenizer.decode(x) for x in cc_pairs[index]['input_ids']]
                    res["model_input"].append(inp)
            yield res


    def _run_loglikelihood(
        self,
        tuples: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: int = None,
        model_max_length: Optional[int] = None
    ) -> Sequence[float]:
        raise NotImplementedError


@Model.register("lm::decoder_only")
class DecoderOnlyLanguageModel(LanguageModel):
    @classmethod
    def _make_model(
        cls,
        pretrained_model_name_or_path: str,
        *,
        make_copy: bool = False,
        model_class: Any = AutoModelForCausalLM,
        **kwargs
    ) -> GPT2LMHeadModel:
        return cached_transformers.get(
            model_class,
            pretrained_model_name_or_path,
            make_copy=make_copy,
            **kwargs)

    @staticmethod
    def _prefix_with_space(s: str) -> str:
        if not s.startswith(' '):
            return f" {s}"
        else:
            return s

    def _run_loglikelihood(
        self,
        tuples: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: int = None,
        model_max_length: Optional[int] = None
    ) -> Sequence[Dict]:

        tokenized_contexts = tokenizer([t[0] for t in tuples], add_special_tokens=False)
        tokenized_continuations = tokenizer([self._prefix_with_space(t[1]) for t in tuples], add_special_tokens=False)

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
        results = self._run_loglikelihood_tokens(cc_pairs, model, tokenizer, batch_size,
                                                 max_batch_tokens=max_batch_tokens,
                                                 model_max_length=model_max_length)
        for i, result in enumerate(results):
            result['num_chars'] = len(tuples[i][1])

        return results

    def _run_loglikelihood_tokens(
        self,
        cc_pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: int = None,
        model_max_length: Optional[int] = None
    ) -> Sequence[Dict]:

        truncation_length = tokenizer.model_max_length
        if model_max_length:
            truncation_length = min(truncation_length, model_max_length)

        # find out the order to process sequences in
        lengths = torch.tensor([
            len(cc_pair["input_ids"][0]) + len(cc_pair["input_ids"][1])
            for cc_pair in cc_pairs
        ], dtype=torch.int)
        ordered_indices = torch.argsort(lengths, descending=True)

        # actually do the processing
        results: List[Optional[float]] = [None] * len(ordered_indices)
        last_index = 0
        with torch.inference_mode():
            with Tqdm.tqdm(ordered_indices, desc="Running log-likelihood queries") as ordered_indices_tqdm:
                while last_index < len(ordered_indices):
                    next_batch_size = batch_size
                    if max_batch_tokens:
                        current_length = min(lengths[ordered_indices[last_index]].item(), truncation_length)
                        next_batch_size = min(next_batch_size, max_batch_tokens // current_length)
                        if next_batch_size < 1:
                            next_batch_size = 1
                    first_index = last_index
                    last_index = min(first_index + next_batch_size, len(ordered_indices))
                    ordered_indices_tqdm.update(last_index - first_index)
                    unpadded_batch = collections.defaultdict(list)
                    input_lengths = []
                    batch_contexts = []
                    batch_continuations = []
                    batch_of_indices = ordered_indices[first_index:last_index]
                    for index in batch_of_indices:
                        for field_name, (context_ids, continuation_ids) in cc_pairs[index].items():
                            ids = torch.cat([context_ids, continuation_ids])
                            # Use truncation_length+1 since the last token is not in the input
                            if len(ids) > (truncation_length+1):
                                ids = ids[-(truncation_length+1):]
                            ids = ids[:-1]
                            unpadded_batch[field_name].append(ids)

                        input_lengths.append(len(unpadded_batch["input_ids"][-1]))
                        batch_contexts.append(cc_pairs[index]["input_ids"][0])
                        batch_continuations.append(cc_pairs[index]["input_ids"][1])

                    padded_batch = {
                        field_name: pad_sequence(tensors, batch_first=True).to(model.device)
                        for field_name, tensors in unpadded_batch.items()
                    }

                    batch_logits = log_softmax(model(**padded_batch)[0], dim=-1)
                    z = zip(batch_of_indices, batch_logits, input_lengths, batch_contexts, batch_continuations)
                    for i, instance_logits, input_length, instance_context, instance_continuation in z:
                        instance_logits = instance_logits[input_length-len(instance_continuation):input_length]
                        instance_logits = torch.gather(instance_logits, 1, instance_continuation.unsqueeze(-1).to(model.device))
                        results[i] = {"sum_logits": float(instance_logits.sum()), "num_tokens": len(instance_continuation),
                                     "num_tokens_all": input_length + 1}
        del lengths
        assert None not in results
        return results
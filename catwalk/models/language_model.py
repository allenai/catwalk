import collections
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import more_itertools
import torch
from tango.common import Tqdm
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

from catwalk import cached_transformers
from catwalk.dependencies.lm_eval.utils import (
    get_rolling_token_windows,
    make_disjoint_window,
)
from catwalk.model import Model
from catwalk.task import InstanceFormat, RankClassificationInstance, Task

_Model = Union[T5ForConditionalGeneration, GPT2LMHeadModel]
_Tokenizer = Union[T5TokenizerFast, GPT2Tokenizer]


class LanguageModel(Model):
    VERSION = "002met"

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        pretrained_tokenizer_name_or_path: Optional[str] = None,
        likelihood_averaging: str = "token",
        **model_kwargs,
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
        assert likelihood_averaging in {"char", "token"}
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        if pretrained_tokenizer_name_or_path is None:
            pretrained_tokenizer_name_or_path = pretrained_model_name_or_path
        self.pretrained_tokenizer_name_or_path = pretrained_tokenizer_name_or_path

        self.likelihood_averaging = likelihood_averaging
        self.model_kwargs = model_kwargs

    @classmethod
    def _make_model(
        cls, pretrained_model_name_or_path: str, *, make_copy: bool = False, **kwargs
    ) -> _Model:
        raise NotImplementedError

    def _make_tokenizer(self) -> AutoTokenizer:
        return cached_transformers.get_tokenizer(
            AutoTokenizer, self.pretrained_tokenizer_name_or_path
        )

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32,
        max_batch_tokens: Optional[
            int
        ] = None,  # If set, max number of tokens in a batch
        max_instances_in_memory: int = 32 * 1024,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None,
        model_max_length: Optional[int] = None,  # Max input length model should support
        num_recorded_inputs: int = 0,  # Number of instances to log in detail
        unconditioned_prompt: Optional[
            str
        ] = None,  # Optional unconditioned prompt, e.g., "Answer:"
    ) -> Iterator[Dict[str, Any]]:
        model = self._make_model(
            self.pretrained_model_name_or_path,
            device_map="auto" if torch.cuda.device_count() > 0 else None,
            **self.model_kwargs,
        ).eval()
        if hasattr(model, "tokenizer"):
            tokenizer = model.tokenizer
        else:
            tokenizer = self._make_tokenizer()

        if "eleuther_metrics" in task.metrics:
            predictor = self.predict_chunk_eleuther
        elif task.has_instance_conversion(InstanceFormat.RANK_CLASSIFICATION):
            predictor = self.predict_chunk_rank_classification  # type: ignore
        elif task.has_instance_conversion(InstanceFormat.ELEUTHER_DOC):
            # Assume perplexity tasks here, only for Eleuther tasks for now, but easy to add others
            predictor = self.predict_chunk_perplexity  # type: ignore
        else:
            raise ValueError("Unknown task type for LM model")

        for instance_chunk in more_itertools.chunked(
            instances, max_instances_in_memory
        ):
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
                unconditioned_prompt=unconditioned_prompt,
            )

    def predict_chunk_rank_classification(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: Optional[int] = None,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None,
        model_max_length: Optional[int] = None,
        num_recorded_inputs: int = 0,
        unconditioned_prompt: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        instance_index_to_tuple_indices: Mapping[
            int, List[int]
        ] = collections.defaultdict(list)
        tuples: List[Tuple[str, str]] = []

        rc_instances: List[RankClassificationInstance] = [
            task.convert_instance(
                instance,
                InstanceFormat.RANK_CLASSIFICATION,
                fewshot_instances=task.get_fewshot_instances(
                    num_shots,
                    random_seed=fewshot_seed if fewshot_seed is not None else i,
                    exceptions=instance,
                ),
            )
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
        results = self._run_loglikelihood(
            tuples,
            model,
            tokenizer,
            batch_size,
            max_batch_tokens=max_batch_tokens,
            model_max_length=model_max_length,
        )

        # collect the results
        for instance_index, instance in enumerate(rc_instances):
            tuple_indices = instance_index_to_tuple_indices[instance_index]
            results_for_instance = [results[i] for i in tuple_indices]
            if unconditioned_prompt:
                for idx, i in enumerate(tuple_indices):
                    unconditioned_results = results[i + unconditioned_offset]
                    results_for_instance[idx][
                        "sum_logits_uncond"
                    ] = unconditioned_results["sum_logits"]
            res = {
                "model_output": results_for_instance,
                "correct_choice": instance.correct_choice,
            }
            if instance_index < num_recorded_inputs:
                res["model_input"] = [tuples[i] for i in tuple_indices]  # type: ignore
                if unconditioned_prompt:
                    res["unconditioned_input"] = [tuples[i + unconditioned_offset] for i in tuple_indices]  # type: ignore
            yield res

    def predict_chunk_perplexity(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: Optional[int] = None,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None,
        model_max_length: Optional[int] = None,
        num_recorded_inputs: int = 0,
        unconditioned_prompt: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        doc_instances: List[str] = [
            task.convert_instance(instance, InstanceFormat.ELEUTHER_DOC)
            for i, instance in enumerate(instances)
        ]
        truncation_length = tokenizer.model_max_length
        if model_max_length:
            truncation_length = min(truncation_length, model_max_length)
        instance_index_to_cc_indices: Mapping[int, List[int]] = collections.defaultdict(
            list
        )
        cc_pairs: List = []

        for instance_index, doc_instance in enumerate(doc_instances):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=tokenizer.encode(
                            doc_instance, add_special_tokens=False
                        ),
                        prefix_token=tokenizer.eos_token_id,
                        max_seq_len=truncation_length,
                        context_len=1,
                    ),
                )
            )
            for context, continuation in rolling_token_windows:
                instance_index_to_cc_indices[instance_index].append(len(cc_pairs))
                cc_pairs.append(
                    {
                        "input_ids": (
                            torch.tensor(context, dtype=torch.long),
                            torch.tensor(continuation, dtype=torch.long),
                        )
                    }
                )

        verbose = hasattr(task, "detailed_output") and task.detailed_output
        results = self._run_loglikelihood_tokens(
            cc_pairs,
            model,
            tokenizer,
            batch_size,
            max_batch_tokens=max_batch_tokens,
            model_max_length=model_max_length,
            verbose=verbose,
        )

        # collect the results
        for instance_index, doc in enumerate(doc_instances):
            cc_indices = instance_index_to_cc_indices[instance_index]
            results_for_instance = [results[i] for i in cc_indices]
            model_output: Dict[str, Any] = {
                "sum_logits": 0,
                "num_tokens": 0,
                "num_tokens_all": 0,
            }
            if verbose:
                model_output["tokens"] = []
                model_output["logits"] = []
            model_output["num_chars"] = len(doc)
            model_output["num_words"] = len(re.split(r"\s+", doc))
            model_output["num_bytes"] = len(doc.encode("utf-8"))
            for result in results_for_instance:
                model_output["sum_logits"] += result["sum_logits"]
                model_output["num_tokens"] += result["num_tokens"]
                model_output["num_tokens_all"] += result["num_tokens_all"]
                if verbose:
                    model_output["tokens"] += result["tokens"]
                    model_output["logits"] += result["logits"]

            res: Dict[str, Any] = {"model_output": model_output}
            if instance_index < num_recorded_inputs:
                res["model_input"] = []
                for index in cc_indices:
                    inp = [tokenizer.decode(x) for x in cc_pairs[index]["input_ids"]]
                    res["model_input"].append(inp)
            yield res

    # For tasks we're coopting directly from Eleuther
    def predict_chunk_eleuther(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        *,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None,
        model_max_length: Optional[int] = None,
        num_recorded_inputs: int = 0,
        unconditioned_prompt: Optional[str] = None,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        instance_index_to_request_indices: collections.defaultdict = (
            collections.defaultdict(lambda: collections.defaultdict(list))
        )
        requests: collections.defaultdict = collections.defaultdict(list)

        # get all the requests
        for instance_index, instance in enumerate(instances):
            instance_requests = task.convert_instance(
                instance,
                InstanceFormat.ELEUTHER_REQUESTS,
                num_fewshot=num_shots,
                fewshot_seed=fewshot_seed
                if fewshot_seed is not None
                else instance_index,
            )
            if not isinstance(instance_requests, (list, tuple)):
                instance_requests = [instance_requests]
            for instance_request in instance_requests:
                request_type = instance_request.request_type
                instance_index_to_request_indices[instance_index][request_type].append(
                    len(requests[request_type])
                )
                requests[request_type].append(instance_request)

        # run the requests
        results: Dict[str, Sequence] = {}
        request_type_to_fn = {
            "loglikelihood": (
                self._run_loglikelihood,
                lambda x: (x["sum_logits"], x["is_greedy"]),
            ),
            "loglikelihood_rolling": (
                self._run_loglikelihood_rolling,
                lambda x: [x["sum_logits"]],
            ),
            "greedy_until": (self._run_greedy_until, lambda x: x["text"]),
        }
        extra_kw_args = {}
        if hasattr(task, "model_args"):
            extra_kw_args = task.model_args
        for request_type, requests_per_type in requests.items():
            results[request_type] = request_type_to_fn[request_type][0](
                [tuple(r.args) for r in requests_per_type],  # type: ignore
                model,
                tokenizer,
                model_max_length=model_max_length,
                **extra_kw_args,
                **kwargs,
            )
        for instance_index, instance in enumerate(instances):
            doc = task.convert_instance(instance, InstanceFormat.ELEUTHER_DOC)
            results_for_instance: List = []
            model_outputs_for_instance = []
            model_inputs: Dict[str, Any] = {}
            for request_type, request_indices in instance_index_to_request_indices[
                instance_index
            ].items():
                for i in request_indices:
                    result = results[request_type][i]
                    # Look up the appropriate key
                    eleuther_result = request_type_to_fn[request_type][1](result)
                    results_for_instance.append(eleuther_result)
                    model_outputs_for_instance.append(result)
                    if instance_index < num_recorded_inputs:
                        if request_type not in model_inputs:
                            model_inputs[request_type] = []
                        model_inputs[request_type].append(
                            requests[request_type][i].args
                        )

            metrics = task.inner_task.process_results(doc, results_for_instance)
            res = {"model_output": model_outputs_for_instance, "metrics": metrics}
            if instance_index < num_recorded_inputs:
                res["model_input"] = model_inputs

            yield res

    def _run_loglikelihood(
        self,
        tuples: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: Optional[int] = None,
        model_max_length: Optional[int] = None,
    ) -> Sequence[Dict]:
        raise NotImplementedError

    def _run_loglikelihood_rolling(
        self,
        tuples: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: Optional[int] = None,
        model_max_length: Optional[int] = None,
    ) -> Sequence[Dict]:
        raise NotImplementedError

    def _run_greedy_until(
        self,
        requests: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        **kwargs,
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
        **kwargs,
    ) -> GPT2LMHeadModel:
        return cached_transformers.get(
            model_class, pretrained_model_name_or_path, make_copy=make_copy, **kwargs
        )

    @staticmethod
    def _prefix_with_space(s: str, needed=True) -> str:
        if not needed and s.startswith(" "):
            return s[1:]
        elif not s.startswith(" ") and needed:
            return f" {s}"
        else:
            return s

    def _run_loglikelihood(
        self,
        tuples: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: Optional[int] = None,
        model_max_length: Optional[int] = None,
    ) -> Sequence[Dict]:
        prefix_space_needed = True
        # Hack to detect tokenizer which treats words at start of sentences as having a prefix space,
        # e.g., Llama tokenizers
        if tokenizer.tokenize("A")[0] == tokenizer.tokenize(" A")[-1]:
            prefix_space_needed = False
        tokenized_contexts = tokenizer([t[0] for t in tuples], add_special_tokens=False)
        tokenized_continuations = tokenizer(
            [self._prefix_with_space(t[1], prefix_space_needed) for t in tuples],
            add_special_tokens=False,
        )

        # We don't need token_type_ids, and it trips up some models apparently (like LLaMA)
        if "token_type_ids" in tokenized_contexts:
            del tokenized_contexts["token_type_ids"]
            del tokenized_continuations["token_type_ids"]
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
                    torch.tensor(continuation, dtype=torch.long),
                )
        results = self._run_loglikelihood_tokens(
            cc_pairs,
            model,
            tokenizer,
            batch_size,
            max_batch_tokens=max_batch_tokens,
            model_max_length=model_max_length,
        )
        for i, result in enumerate(results):
            result["num_chars"] = len(tuples[i][1])

        return results

    def _run_loglikelihood_tokens(
        self,
        cc_pairs: Sequence[Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        max_batch_tokens: Optional[int] = None,
        model_max_length: Optional[int] = None,
        verbose=False,
    ) -> Sequence[Dict]:
        truncation_length = tokenizer.model_max_length
        if model_max_length:
            truncation_length = min(truncation_length, model_max_length)

        # find out the order to process sequences in
        lengths = torch.tensor(
            [
                len(cc_pair["input_ids"][0]) + len(cc_pair["input_ids"][1])
                for cc_pair in cc_pairs
            ],
            dtype=torch.int,
        )
        ordered_indices = torch.argsort(lengths, descending=True)

        # actually do the processing
        results: List = [None] * len(ordered_indices)
        last_index = 0
        with torch.inference_mode():
            with Tqdm.tqdm(
                ordered_indices, desc="Running log-likelihood queries"
            ) as ordered_indices_tqdm:
                while last_index < len(ordered_indices):
                    next_batch_size = batch_size
                    if max_batch_tokens:
                        current_length = min(
                            lengths[ordered_indices[last_index]].item(),
                            truncation_length,
                        )
                        next_batch_size = min(
                            next_batch_size, max_batch_tokens // current_length
                        )
                        if next_batch_size < 1:
                            next_batch_size = 1
                    first_index = last_index
                    last_index = min(
                        first_index + next_batch_size, len(ordered_indices)
                    )
                    ordered_indices_tqdm.update(last_index - first_index)
                    unpadded_batch = collections.defaultdict(list)
                    input_lengths = []
                    batch_contexts = []
                    batch_continuations = []
                    batch_of_indices = ordered_indices[first_index:last_index]
                    for index in batch_of_indices:
                        for field_name, (context_ids, continuation_ids) in cc_pairs[
                            index
                        ].items():
                            ids = torch.cat([context_ids, continuation_ids])
                            # Use truncation_length+1 since the last token is not in the input
                            if len(ids) > (truncation_length + 1):
                                ids = ids[-(truncation_length + 1) :]
                            ids = ids[:-1]
                            unpadded_batch[field_name].append(ids)

                        input_lengths.append(len(unpadded_batch["input_ids"][-1]))
                        batch_contexts.append(cc_pairs[index]["input_ids"][0])
                        batch_continuations.append(cc_pairs[index]["input_ids"][1])

                    padded_batch = {
                        field_name: pad_sequence(tensors, batch_first=True).to(
                            model.device
                        )
                        for field_name, tensors in unpadded_batch.items()
                    }

                    batch_logits = log_softmax(model(**padded_batch)[0], dim=-1)
                    z = zip(
                        batch_of_indices,
                        batch_logits,
                        input_lengths,
                        batch_contexts,
                        batch_continuations,
                    )
                    for (
                        i,
                        instance_logits,
                        input_length,
                        instance_context,
                        instance_continuation,
                    ) in z:
                        instance_logits = instance_logits[
                            input_length - len(instance_continuation) : input_length
                        ]
                        instance_logits = torch.gather(
                            instance_logits,
                            1,
                            instance_continuation.unsqueeze(-1).to(model.device),
                        )
                        greedy_tokens = instance_logits.argmax(dim=-1)
                        is_greedy = bool(
                            (
                                greedy_tokens
                                == instance_continuation.unsqueeze(0).to(model.device)
                            ).all()
                        )
                        results[i] = {
                            "sum_logits": float(instance_logits.sum()),
                            "num_tokens": len(instance_continuation),
                            "num_tokens_all": input_length + 1,
                            "is_greedy": is_greedy,
                        }
                        if verbose:
                            instance_tokens = [
                                tokenizer.decode(x) for x in instance_continuation
                            ]
                            results[i]["tokens"] = instance_tokens
                            results[i]["logits"] = instance_logits.squeeze(-1).tolist()
        del lengths
        assert None not in results
        return results

    def _run_greedy_until(
        self,
        requests,
        model: GPT2LMHeadModel,
        tokenizer: GPT2Tokenizer,
        max_gen_toks: int = 100,
        **kwargs,
    ) -> Sequence:
        tokenized_contexts = tokenizer([r[0] for r in requests])["input_ids"]
        # the stop generation phrases
        untils_per_instance = [r[1] for r in requests]
        if hasattr(model, "config") and hasattr(model.config, "n_positions"):
            model_max_length = model.config.n_positions
        else:
            model_max_length = 2048
        model_max_length = kwargs.get("model_max_length", model_max_length)
        assert model_max_length is not None
        assert model_max_length > max_gen_toks

        results = []
        for tokenized_context, untils in Tqdm.tqdm(
            zip(tokenized_contexts, untils_per_instance),
            desc="Running greedy_until queries",
            total=len(tokenized_contexts),
        ):
            # there can be multiple stop phrases with multiple tokens
            if isinstance(untils, str):
                untils = [untils]
            # if any of the stop phrases are single tokens we can use that for early termination
            primary_until = None
            for tokenized_until in tokenizer(untils)["input_ids"]:
                if len(tokenized_until) == 1:
                    primary_until = tokenized_until[0]

            # truncate from left if no room for generation
            context_tensor = torch.tensor(
                [tokenized_context[max_gen_toks - model_max_length :]]
            ).to(model.device)

            full_text_tensor = model.generate(
                context_tensor,
                max_length=context_tensor.shape[1] + max_gen_toks,
                eos_token_id=primary_until,
                do_sample=False,
                pad_token_id=primary_until,  # temporary hack to suppress irrelevant warning until batch processing is added
            )
            continuation_tensor = full_text_tensor[0, context_tensor.shape[1] :]
            continuation = tokenizer.decode(continuation_tensor.tolist())
            raw_continuation = continuation
            # truncate by all the additional until phrases
            for term in untils:
                continuation = continuation.split(term)[0]
            results.append(
                {
                    "text": continuation,
                    "raw_text": raw_continuation,
                    "num_input_tokens": context_tensor.shape[1],
                    "num_generated_tokens": len(continuation_tensor),
                }
            )

        return results

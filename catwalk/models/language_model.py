import collections
import functools
from typing import Dict, Any, List, Tuple, Sequence, Iterator, Union, Mapping, Optional, cast, Callable

import more_itertools
import torch
from tango.common import Tqdm
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, GPT2LMHeadModel, \
    AutoTokenizer, GPT2Tokenizer, T5TokenizerFast

from catwalk import cached_transformers
from catwalk.model import Model, TrainableModel, Instance
from catwalk.task import Task, InstanceFormat, RankClassificationInstance

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
        max_instances_in_memory: int = 32 * 1024,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None,
        num_recorded_inputs: Optional[int] = 0,  # Number of instances to log in detail
        unconditioned_prompt: Optional[str] = None, # Optional unconditioned prompt, e.g., "Answer:"
    ) -> Iterator[Dict[str, Any]]:
        model = self._make_model(
            self.pretrained_model_name_or_path,
            device_map="auto" if torch.cuda.device_count() > 0 else None,
            **self.model_kwargs).eval()
        tokenizer = self._make_tokenizer()

        for instance_chunk in more_itertools.chunked(instances, max_instances_in_memory):
            yield from self.predict_chunk(
                task,
                instance_chunk,
                model,
                tokenizer,
                batch_size=batch_size,
                num_shots=num_shots,
                fewshot_seed=fewshot_seed,
                num_recorded_inputs=num_recorded_inputs,
                unconditioned_prompt=unconditioned_prompt
            )

    def predict_chunk(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None,
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
        results = self._run_loglikelihood(tuples, model, tokenizer, batch_size)

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
                    res["unconditoned_input"] = [tuples[i + unconditioned_offset] for i in tuple_indices]
            yield res

    def _run_loglikelihood(
        self,
        tuples: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
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

        # find out the order to process sequences in
        lengths = torch.tensor([
            len(cc_pair["input_ids"][0]) + len(cc_pair["input_ids"][1])
            for cc_pair in cc_pairs
        ], dtype=torch.int)
        ordered_indices = torch.argsort(lengths, descending=True)
        del lengths

        # actually do the processing
        results: List[Optional[float]] = [None] * len(ordered_indices)
        with torch.inference_mode():
            with Tqdm.tqdm(ordered_indices, desc="Running log-likelihood queries") as ordered_indices_tqdm:
                batches_of_indices = more_itertools.chunked(ordered_indices_tqdm, batch_size)
                for batch_of_indices in batches_of_indices:
                    unpadded_batch = collections.defaultdict(list)
                    input_lengths = []
                    batch_contexts = []
                    batch_continuations = []
                    for index in batch_of_indices:
                        for field_name, (context_ids, continuation_ids) in cc_pairs[index].items():
                            ids = torch.cat([context_ids, continuation_ids])
                            if len(ids) > tokenizer.model_max_length+1:
                                ids = ids[-(tokenizer.model_max_length+1):]
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
                        # denom = len(tuples[i][1]) if self.likelihood_averaging == 'char' else len(instance_continuation)
                        results[i] = {"sum_logits": float(instance_logits.sum()), "num_tokens": len(tuples[i][1]),
                                      "num_chars": len(instance_continuation)}
                        # results[i] = float(instance_logits.sum()) / denom

        assert None not in results
        return results

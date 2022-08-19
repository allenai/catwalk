import collections
from dataclasses import dataclass
from typing import Dict, Any, List, OrderedDict, Tuple, Sequence, Iterator, Union, Mapping, Optional, cast, Callable

import more_itertools
import torch
from tango.common import Tqdm
from tango.integrations.torch.util import resolve_device
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, GPT2LMHeadModel, \
    AutoTokenizer, GPT2Tokenizer, T5TokenizerFast, BatchEncoding

from catwalk import cached_transformers
from catwalk.model import Model, TrainableModel, Instance
from catwalk.task import Task, InstanceFormat, RankClassificationInstance
from catwalk.utils import PrefixTrie

_Model = Union[T5ForConditionalGeneration, GPT2LMHeadModel]
_Tokenizer = Union[T5TokenizerFast, GPT2Tokenizer]


class RankClassificationModel(Model):
    VERSION = "001nul"

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        likelihood_averaging: str = 'char',
        override_weights_file: str = None,
        **model_kwargs
    ):
        """
        # Parameters

        pretrained_model_name_or_path : `str`
            The name of the transformer, for example `"gpt2-large"`
        likelihood_averaging : `str`, optional (default = `char`)
            The method for averaging the sum likelihood of the continuation. 'char' averages by 
            character length, 'token' averages by token length.
        override_weights_file : `str`, optional (default = `None`)
            If set, this specifies a file from which to load alternate weights that override the
            weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
            with `torch.save()`.
        model_kwargs:
            Additional kwargs passed to the `_make_model` method.
        """
        assert likelihood_averaging in {'char', 'token'}
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.likelihood_averaging = likelihood_averaging
        self.override_weights_file = override_weights_file
        self.model_kwargs = model_kwargs

    def _make_model(cls, pretrained_model_name_or_path: str, *, override_weights_file: str = None, **kwargs) -> _Model:
        raise NotImplementedError

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32,
        max_instances_in_memory: int = 32 * 1024,
        num_shots: int = 0,
        fewshot_seed: int = None
    ) -> Iterator[Dict[str, Any]]:
        device = resolve_device()
        model = self._make_model(self.pretrained_model_name_or_path, override_weights_file=self.override_weights_file, **self.model_kwargs).to(device).eval()
        tokenizer = cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_model_name_or_path)

        for instance_chunk in more_itertools.chunked(instances, max_instances_in_memory):
            yield from self.predict_chunk(
                task,
                instance_chunk,
                model,
                tokenizer,
                batch_size=batch_size,
                num_shots=num_shots,
                fewshot_seed=fewshot_seed
            )

    def predict_chunk(
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        num_shots: int = 0,
        fewshot_seed: int = None
    ) -> Iterator[Dict[str, Any]]:
        instance_index_to_tuple_indices: Mapping[int, List[int]] = collections.defaultdict(list)
        tuples: List[Tuple[str, str]] = []
        rc_instances: List[RankClassificationInstance] = [
            task.convert_instance(
                instance,
                InstanceFormat.RANK_CLASSIFICATION,
                fewshot_instances=task.get_fewshot_instances(num_shots, random_seed=fewshot_seed if fewshot_seed is not None else i, exceptions=instance))
            for i, instance in enumerate(instances)
        ]

        # get all the tuples
        for instance_index, instance in enumerate(rc_instances):
            for instance_request in instance.choices:
                instance_index_to_tuple_indices[instance_index].append(len(tuples))
                tuples.append(instance_request)

        # run the requests
        results = self._run_loglikelihood(tuples, model, tokenizer, batch_size)

        # collect the results
        for instance_index, instance in enumerate(rc_instances):
            tuple_indices = instance_index_to_tuple_indices[instance_index]
            results_for_instance = [results[i] for i in tuple_indices]
            result_tensor = torch.tensor(results_for_instance)
            metric_args = (result_tensor, instance.correct_choice)
            yield {
                "acc": metric_args,
                "f1": metric_args,
                "precision": metric_args,
                "recall": metric_args,
            }

    def _run_loglikelihood(
        self,
        tuples: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
    ) -> Sequence[float]:
        raise NotImplementedError

    def trainable_copy(self) -> TrainableModel:
        return TrainableRankClassificationModel(
            self._make_model(self.pretrained_model_name_or_path),
            cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_model_name_or_path),
            self.predict_chunk
        )


class TrainableRankClassificationModel(TrainableModel):
    def __init__(self, model: _Model, tokenizer: _Tokenizer, predict_chunk_fn: Callable):
        super().__init__(model)
        self.model = model
        self.tokenizer = tokenizer
        self.predict_chunk = predict_chunk_fn
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32,
        max_instances_in_memory: int = 32 * 1024,
        num_shots: int = 0
    ) -> Iterator[Dict[str, Any]]:
        training_mode = self.model.training
        try:
            self.model.eval()
            for instance_chunk in more_itertools.chunked(instances, max_instances_in_memory):
                yield from self.predict_chunk(
                    task,
                    instance_chunk,
                    self.model,
                    self.tokenizer,
                    batch_size=batch_size,
                    num_shots=num_shots)
        finally:
            self.model.train(training_mode)

    def collate_for_training(self, instances: Sequence[Tuple[Task, Instance]]) -> Any:
        rc_instances = (
            task.convert_instance(instance, InstanceFormat.RANK_CLASSIFICATION)
            for task, instance in instances
        )
        correct_strings = [
            rc.choices[rc.correct_choice]
            for rc in rc_instances
        ]
        tokenized_strings = self.tokenizer(
            correct_strings,
            padding=True,
            truncation=True,
            pad_to_multiple_of=8,
            return_tensors='pt',
            is_split_into_words=False)
        tokenized_strings['labels'] = torch.full_like(tokenized_strings.input_ids, -100)
        for i, label in enumerate(tokenized_strings.labels):
            mask = [s == 1 for s in tokenized_strings.sequence_ids(i)]
            label[mask] = tokenized_strings.input_ids[i, mask]
        return {
            key: tensor.to(self.model.device)
            for key, tensor in tokenized_strings.items()
        }


@Model.register("rc::encoder_decoder")
class EncoderDecoderRCModel(RankClassificationModel):
    @classmethod
    def _make_model(cls, pretrained_model_name_or_path: str, *, override_weights_file: str = None, **kwargs) -> T5ForConditionalGeneration:
        return cached_transformers.get(AutoModelForSeq2SeqLM, pretrained_model_name_or_path, False, override_weights_file=override_weights_file)

    def _run_loglikelihood(
        self,
        tuples: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
    ) -> Sequence[float]:
        encoder_inputs = tokenizer([t[0] for t in tuples])
        model_inputs: List[Dict[str, torch.Tensor]] = []
        for field_name, encoder_input in encoder_inputs.items():
            for i, input_as_list in enumerate(encoder_input):
                if len(model_inputs) <= i:
                    model_inputs.append({})
                model_inputs[i][field_name] = torch.tensor(input_as_list, dtype=torch.long)
        del encoder_inputs

        with tokenizer.as_target_tokenizer():
            decoder_inputs = tokenizer([r[1] for r in tuples], return_attention_mask=False)
        for i, input_as_list in enumerate(decoder_inputs['input_ids']):
            input_as_list = input_as_list[:-1]      # remove the EOS token
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
        results: List[Optional[float]] = [None] * len(ordered_indices)
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

                batch_logits = log_softmax(model(**padded_batch).logits, dim=-1)

                for i, instance_logits, decoder_input_ids in zip(batch_of_indices, batch_logits, unpadded_batch["labels"]):
                    instance_logits = instance_logits[:len(decoder_input_ids)]
                    instance_logits = torch.gather(instance_logits, 1, decoder_input_ids.unsqueeze(-1).to(model.device))
                    denom = len(tuples[i][1]) if self.likelihood_averaging == 'char' else len(decoder_input_ids)
                    results[i] = float(instance_logits.sum()) / denom

        assert None not in results
        return cast(Sequence[float], results)


@dataclass
class CacheData:
    cached_sequence: Optional[Sequence[Optional[int]]] = None
    cached_past_key_values: torch.Tensor = None
    longest_prefix_to_indices: Optional[Dict[Sequence[Optional[int]],Sequence[int]]] = None
    indices_to_longest_prefix: Optional[OrderedDict[int,Sequence[Optional[int]]]] = None

@Model.register("rc::decoder_only")
class DecoderOnlyRCModel(RankClassificationModel):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        likelihood_averaging: str = 'char',
        override_weights_file: str = None,
        prefix_caching: bool = False,
        **model_kwargs
    ):
        """
        # Parameters

        pretrained_model_name_or_path : `str`
            The name of the transformer, for example `"gpt2-large"`
        likelihood_averaging : `str`, optional (default = `char`)
            The method for averaging the sum likelihood of the continuation. 'char' averages by 
            character length, 'token' averages by token length.
        override_weights_file : `str`, optional (default = `None`)
            If set, this specifies a file from which to load alternate weights that override the
            weights from huggingface. The file is expected to contain a PyTorch `state_dict`, created
            with `torch.save()`.
        prefix_caching : `bool`, optional (default = `False`)
            If set to True uses a caching strategy that improves performance when many inputs in a task 
            share prefixes. This orders the dataset by common prefixes and caches the current shared prefix.
        model_kwargs:
            Additional kwargs passed to the `_make_model` method.
        """
        super().__init__(pretrained_model_name_or_path, likelihood_averaging=likelihood_averaging, override_weights_file=override_weights_file, **model_kwargs)
        self.prefix_caching = prefix_caching

    @classmethod
    def _make_model(cls, pretrained_model_name_or_path: str, *, override_weights_file: str = None, **kwargs) -> GPT2LMHeadModel:
        return cached_transformers.get(AutoModelForCausalLM, pretrained_model_name_or_path, False, override_weights_file=override_weights_file)

    def _run_loglikelihood(
        self,
        tuples: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
    ) -> Sequence[float]:
        cache = CacheData() if self.prefix_caching else None

        tokenized_contexts = tokenizer([t[0] for t in tuples])
        tokenized_continuations = tokenizer([t[1] for t in tuples])

        self._final_truncatation(
            tokenized_contexts, tokenized_continuations, tokenizer.model_max_length
        )

        ordered_indices = self._reorder_instances(
            tokenized_contexts, tokenized_continuations, cache
        )

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

        # actually do the processing
        results: List[Optional[float]] = [None] * len(ordered_indices)
        with torch.inference_mode():
            batches_of_indices = more_itertools.chunked(
                Tqdm.tqdm(ordered_indices, desc="Running log-likelihood queries"),
                batch_size)
            for batch_of_indices in batches_of_indices:
                inputs, input_lengths, batch_contexts, batch_continuations = self._get_inputs(batch_of_indices, cc_pairs, model, cache)
                batch_logits = log_softmax(model(**inputs)[0], dim=-1)
                z = zip(batch_of_indices, batch_logits, input_lengths, batch_contexts, batch_continuations)
                for i, instance_logits, input_length, instance_context, instance_continuation in z:
                    assert input_length-len(instance_continuation) >=0
                    instance_logits = instance_logits[input_length-len(instance_continuation):input_length]
                    instance_logits = torch.gather(instance_logits, 1, instance_continuation.unsqueeze(-1).to(model.device))
                    denom = len(tuples[i][1]) if self.likelihood_averaging == 'char' else len(instance_continuation)
                    results[i] = float(instance_logits.sum()) / denom

        assert None not in results
        return cast(Sequence[float], results)

    def _final_truncatation(self, tokenized_contexts: BatchEncoding, tokenized_continuations: BatchEncoding, model_max_length: int):
        """ Apply a last pass of truncation on the concatenated inputs to make sure it fits in the model_max_length"""
        assert len(tokenized_contexts['input_ids']) == len(tokenized_continuations['input_ids'])
        for i in range(len(tokenized_contexts['input_ids'])):
            context_len = len(tokenized_contexts['input_ids'][i])
            cont_len = len(tokenized_continuations['input_ids'][i])
            assert cont_len < model_max_length
            if context_len +  cont_len > model_max_length:
                tokenized_contexts['input_ids'][i] = tokenized_contexts['input_ids'][i][-model_max_length + cont_len:]
                tokenized_contexts['attention_mask'][i] = tokenized_contexts['attention_mask'][i][-model_max_length + cont_len:]
    
    def _reorder_instances(self, tokenized_contexts: BatchEncoding, tokenized_continuations: BatchEncoding, cache: CacheData = None) -> Sequence[int]:
        if self.prefix_caching:
            assert cache is not None, 'prefix reordering requires a CacheData object'
            return self._reorder_by_prefix(tokenized_contexts, tokenized_continuations, cache)
        else:
            return self._reorder_by_longest(tokenized_contexts, tokenized_continuations)
    
    def _reorder_by_prefix(self, tokenized_contexts: BatchEncoding, tokenized_continuations: BatchEncoding, cache: CacheData) -> Sequence[int]:
        combined_ids = [context + continuation for context, continuation in zip(tokenized_contexts['input_ids'], tokenized_continuations['input_ids'])]
        cache.longest_prefix_to_indices = self._greedy_assign_prefix_by_total_coverage(combined_ids)
        cache.indices_to_longest_prefix = OrderedDict()
        # secondarily sort by length so that largest batches that may cause memory overflow are likely come early
        for prefix in sorted(cache.longest_prefix_to_indices.keys(), key = lambda x : -len(x)):
            # sequence indices for each prefix are already sorted by length from reading trie from leaf to root
            for index in cache.longest_prefix_to_indices[prefix]:
                cache.indices_to_longest_prefix[index] = prefix
        return list(cache.indices_to_longest_prefix.keys())
    
    def _greedy_assign_prefix_by_total_coverage(self, sequences: Sequence[Sequence[int]]) -> Dict[Sequence[Optional[int]],Sequence[int]]:
        """Returns a Dict of prefixes and the sequence indices assigned to them. Sorts possible prefixes by total tokens covered in 
        subsequences and assigns sequences to the first prefix they appear in. PrefixTrie only tracks subsequences after a minimum 
        track_after_depth so short coincidental overlaps are be ignored."""
        longest_prefix_to_indices: Dict[Sequence[Optional[int]],Sequence[int]] = {}
        trie = PrefixTrie(sequences)
        leaves = trie.get_leaf_nodes()
        leaves_sequences = [tuple(leaf.get_sequence()) for leaf in leaves]
        leaves_and_sequences = Tqdm.tqdm(zip(leaves_sequences, leaves), desc="Finding prefixes", total=len(leaves))
        leaves2prefixes = {leaf_sequence:leaf.get_subsequences() for leaf_sequence, leaf in leaves_and_sequences}

        # greedily assign sequences to prefixes with top total coverage
        indices_already_assigned = set()
        for leaf_sequence in sorted(leaves_sequences, key=lambda leaf_sequence : -leaves2prefixes[leaf_sequence][1]):
            prefix_indices, _ = leaves2prefixes[leaf_sequence]
            prefix_indices = [prefix_index for prefix_index in prefix_indices if prefix_index not in indices_already_assigned]
            indices_already_assigned.update(prefix_indices)
            if len(prefix_indices) > 0:
                longest_prefix_to_indices[leaf_sequence] = tuple(prefix_indices)

        return longest_prefix_to_indices

    def _reorder_by_longest(self, tokenized_contexts: BatchEncoding, tokenized_continuations: BatchEncoding) -> Sequence[int]:
        assert len(tokenized_contexts['input_ids']) == len(tokenized_continuations['input_ids'])
        lengths = torch.tensor([
            len(tokenized_contexts["input_ids"][i]) + len(tokenized_continuations["input_ids"][i])
            for i in range(len(tokenized_contexts['input_ids']))
        ], dtype=torch.int)
        return torch.argsort(lengths, descending=True).tolist()

    def _get_inputs(self, batch_of_indices: Sequence[int], cc_pairs: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], model: _Model, cache: CacheData = None):
        if self.prefix_caching:
            assert cache is not None
            return self._get_inputs_with_cache(batch_of_indices, cc_pairs, model, cache)
        else:
            return self._get_inputs_without_cache(batch_of_indices, cc_pairs, model)
            

    def _get_inputs_with_cache(self, batch_of_indices: Sequence[int], cc_pairs: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], model: _Model, cache: CacheData):
        assert cache.indices_to_longest_prefix is not None
        prefixes = [cache.indices_to_longest_prefix[index] for index in batch_of_indices]
        prefix2cache = OrderedDict()

        # compute prefixes
        if prefixes[0] == cache.cached_sequence:
            prefix2cache[prefixes[0]] = cache.cached_past_key_values

        uncached_prefixes = list(set(prefix for prefix in prefixes if prefix not in prefix2cache)) # ordering must be fixed
        if len(uncached_prefixes) > 0:
            unpadded_prefixes = [torch.tensor(prefix) for prefix in uncached_prefixes]
            unpadded_prefix_mask = [torch.ones_like(prefix) for prefix in unpadded_prefixes]
            padded_prefixes = pad_sequence(unpadded_prefixes, batch_first=True).to(model.device)
            padded_prefix_masks = pad_sequence(unpadded_prefix_mask, batch_first=True, padding_value=0.0).to(model.device)
            past_key_values = model(input_ids=padded_prefixes, attention_mask=padded_prefix_masks).past_key_values
            # tensor(layers, keys/values, batch_size, num_heads, sequence_len, embed_size_per_head)
            past_key_values = torch.stack(tuple(torch.stack(past_key_values[i]) for i in range(len(past_key_values))))
            for i, prefix in enumerate(uncached_prefixes):
                # tensor(layers, keys/values, num_heads, sequence_len, embed_size_per_head)
                prefix2cache[prefix] = past_key_values[:,:,i,:,:len(prefix),:]

        # update cache with last one retrieved since instances come in order by common prefix
        cache.cached_sequence = prefixes[-1]
        cache.cached_past_key_values = prefix2cache[prefixes[-1]]

        # pad and mask batched past_key_values
        unpadded_past_keys_values = [prefix2cache[prefix] for prefix in prefixes]
        unpadded_past_keys_values_attn_mask = []
        # only use the prefixed part of past_key_values that is present in the instance
        for prefix_idx, cc_pairs_idx in enumerate(batch_of_indices):
            is_identical = True
            for tok_idx, tok in enumerate(cc_pairs[cc_pairs_idx]['input_ids'][0]):
                if tok.item() != prefixes[prefix_idx][tok_idx]:
                    unpadded_past_keys_values_attn_mask.append(torch.tensor([1] * tok_idx, dtype=torch.int64))
                    is_identical = False
                    break
            if is_identical:
                # Avoid empty input by leaving last token of context for input because continuations drop one token for right shift
                max_prefix_len = len(cc_pairs[cc_pairs_idx]['input_ids'][0]) - 1
                unpadded_past_keys_values_attn_mask.append(torch.tensor([1] * max_prefix_len, dtype=torch.int64))
        
        # past_keys_values needs its own attention mask
        padded_past_keys_values_attn_mask = pad_sequence(unpadded_past_keys_values_attn_mask, batch_first=True, padding_value=0)
        cache_lengths = [mask.sum().item() for mask in padded_past_keys_values_attn_mask]
        max_past_key_value_len = max(cache_lengths)

        # pad and truncate past_keys_values to longest actually used
        unpadded_past_keys_values = [t.transpose(0,-2) for t in unpadded_past_keys_values]
        padded_past_keys_values = pad_sequence(unpadded_past_keys_values, batch_first=True)
        padded_past_keys_values = padded_past_keys_values.permute((4, 2, 0, 3, 1, 5))
        # tensor(layers, keys/values, batch_size, num_heads, sequence_len, embed_size_per_head)
        padded_past_keys_values = padded_past_keys_values[:,:,:,:,:max_past_key_value_len]

        # make input_ids by removing whatever parts of past_key_values are present
        unpadded_input_ids = []
        input_lengths = []
        batch_contexts = []
        batch_continuations = []

        for prefix_idx, cc_pairs_idx in enumerate(batch_of_indices):
            context_ids, continuation_ids = cc_pairs[cc_pairs_idx]['input_ids']
            ids = torch.cat([context_ids, continuation_ids])[:-1]
            ids = ids[cache_lengths[prefix_idx]:]

            # used to find logits specifically for continuation
            input_lengths.append(len(ids))
            batch_contexts.append(cc_pairs[cc_pairs_idx]["input_ids"][0])
            batch_continuations.append(cc_pairs[cc_pairs_idx]["input_ids"][1])

            unpadded_input_ids.append(ids)

        # batch and pad and make attention mask
        unpadded_attn_mask = [torch.ones_like(t) for t in unpadded_input_ids]
        padded_attn_mask = pad_sequence(unpadded_attn_mask, batch_first=True, padding_value=0)
        padded_input_ids = pad_sequence(unpadded_input_ids, batch_first=True)

        # combine the attention masks
        full_attn_mask = torch.cat((padded_past_keys_values_attn_mask, padded_attn_mask), dim=1)
        assert full_attn_mask.shape[1] <= model.config.n_positions, "Presently batches with wide range of prefix and input lengths are not supported due overrun of max model size"

        # make position_ids
        max_input_len = padded_input_ids.shape[-1]
        position_ids = torch.stack([torch.arange(cache_length, cache_length + max_input_len) for cache_length in cache_lengths], dim=0)
        position_ids = position_ids * padded_attn_mask
        assert (position_ids < model.config.n_positions).all()

        inputs = {
            'input_ids': padded_input_ids.to(model.device),
            'past_key_values': padded_past_keys_values,
            'attention_mask': full_attn_mask.to(model.device),
            'position_ids': position_ids.to(model.device)
        }

        return inputs, input_lengths, batch_contexts, batch_continuations
    
    def _get_inputs_without_cache(self, batch_of_indices: Sequence[int], cc_pairs: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], model: _Model):
        unpadded_batch = collections.defaultdict(list)
        input_lengths = []
        batch_contexts = []
        batch_continuations = []

        for index in batch_of_indices:
            for field_name, (context_ids, continuation_ids) in cc_pairs[index].items():
                ids = torch.cat([context_ids, continuation_ids])[:-1]
                unpadded_batch[field_name].append(ids)

            input_lengths.append(len(unpadded_batch["input_ids"][-1]))
            batch_contexts.append(cc_pairs[index]["input_ids"][0])
            batch_continuations.append(cc_pairs[index]["input_ids"][1])

        padded_batch = {
            field_name: pad_sequence(tensors, batch_first=True).to(model.device)
            for field_name, tensors in unpadded_batch.items()
        }
        return padded_batch, input_lengths, batch_contexts, batch_continuations
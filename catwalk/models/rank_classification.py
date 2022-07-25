import collections
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

    def __init__(self, pretrained_model_name_or_path: str, override_weights_file: str = None):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.override_weights_file = override_weights_file

    @classmethod
    def _make_model(cls, pretrained_model_name_or_path: str, override_weights_file: str = None) -> _Model:
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
        model = self._make_model(self.pretrained_model_name_or_path, self.override_weights_file).to(device).eval()
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
    def _make_model(cls, pretrained_model_name_or_path: str, override_weights_file: str = None) -> T5ForConditionalGeneration:
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

                batch_logits = log_softmax(model(**padded_batch).logits, dim=-1).cpu()

                for i, instance_logits, decoder_input_ids in zip(batch_of_indices, batch_logits, unpadded_batch["labels"]):
                    instance_logits = instance_logits[:len(decoder_input_ids)]
                    instance_logits = torch.gather(instance_logits, 1, decoder_input_ids.unsqueeze(-1))
                    results[i] = float(instance_logits.sum()) / len(tuples[i][1])

        assert None not in results
        return cast(Sequence[float], results)


@Model.register("rc::decoder_only")
class DecoderOnlyRCModel(RankClassificationModel):
    def __init__(self, pretrained_model_name_or_path: str, *, override_weights_file: str = None, prefix_caching: bool = False):
        super().__init__(pretrained_model_name_or_path, override_weights_file=override_weights_file)
        self.prefix_caching = prefix_caching
        self._reset_cache_variables()

    @classmethod
    def _make_model(cls, pretrained_model_name_or_path: str, override_weights_file: str = None) -> GPT2LMHeadModel:
        return cached_transformers.get(AutoModelForCausalLM, pretrained_model_name_or_path, False, override_weights_file=override_weights_file)

    def _run_loglikelihood(
        self,
        tuples: Sequence[Tuple[str, str]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
    ) -> Sequence[float]:
        tokenized_contexts = tokenizer([t[0] for t in tuples])
        tokenized_continuations = tokenizer([t[1] for t in tuples])

        self._final_truncatation(
            tokenized_contexts, tokenized_continuations, tokenizer.model_max_length
        )

        ordered_indices = self._reorder_instances(
            tokenized_contexts, tokenized_continuations
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
                inputs, input_lengths, batch_contexts, batch_continuations = self._get_inputs(batch_of_indices, cc_pairs, model)
                batch_logits = log_softmax(model(**inputs)[0], dim=-1).cpu()
                z = zip(batch_of_indices, batch_logits, input_lengths, batch_contexts, batch_continuations)
                for i, instance_logits, input_length, instance_context, instance_continuation in z:
                    assert input_length-len(instance_continuation) >=0
                    instance_logits = instance_logits[input_length-len(instance_continuation):input_length]
                    instance_logits = torch.gather(instance_logits, 1, instance_continuation.unsqueeze(-1))
                    results[i] = float(instance_logits.sum()) / len(tokenized_continuations.input_ids[i])

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
    
    def _reorder_instances(self, tokenized_contexts: BatchEncoding, tokenized_continuations: BatchEncoding) -> Sequence[int]:
        if self.prefix_caching:
            return self._reorder_by_prefix(tokenized_contexts, tokenized_continuations)
        else:
            return self._reorder_by_longest(tokenized_contexts, tokenized_continuations)
    
    def _reorder_by_prefix(self, tokenized_contexts: BatchEncoding, tokenized_continuations: BatchEncoding) -> Sequence[int]:
        self._reset_cache_variables()
        combined_ids = [context + continuation for context, continuation in zip(tokenized_contexts['input_ids'], tokenized_continuations['input_ids'])]
        self.longest_prefix_to_indices = self._order_by_common_prefix(combined_ids)
        self.indices_to_longest_prefix = OrderedDict()
        for prefix in sorted(self.longest_prefix_to_indices.keys(), key = lambda x : -len(x)):
            # indices for each prefix are already sorted by trie
            for index in self.longest_prefix_to_indices[prefix]:
                self.indices_to_longest_prefix[index] = prefix
        return list(self.indices_to_longest_prefix.keys())
    
    def _order_by_common_prefix(self, sequences: Sequence[Sequence[int]]) -> Dict[Sequence[Optional[int]],Sequence[int]]:
        longest_prefix_to_indices: Dict[Sequence[Optional[int]],Sequence[int]] = {}
        trie = PrefixTrie(sequences)
        leaves = trie.get_leaf_nodes()
        leaves_sequences = [tuple(leaf.get_sequence()) for leaf in leaves]
        leaves_and_sequences = Tqdm.tqdm(zip(leaves_sequences, leaves), desc="Finding prefixes", total=len(leaves))
        leaves2prefixes = {leaf_sequence:leaf.get_prefix_indices() for leaf_sequence, leaf in leaves_and_sequences}

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

    def _reset_cache_variables(self):
        self.cached_sequence: Sequence[Optional[int]] = None
        self.cached_past_key_values: torch.Tensor = None
        self.longest_prefix_to_indices: Dict[Sequence[Optional[int]],Sequence[int]] = None
        self.indices_to_longest_prefix: OrderedDict[int,Sequence[int]] = None

    def _get_inputs(self, batch_of_indices: Sequence[int], cc_pairs: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], model: _Model):
        if self.prefix_caching:
            return self._get_inputs_with_cache(batch_of_indices, cc_pairs, model)
        else:
            return self._get_inputs_without_cache(batch_of_indices, cc_pairs, model)
            

    def _get_inputs_with_cache(self, batch_of_indices: Sequence[int], cc_pairs: List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]], model: _Model):
        prefixes = [self.indices_to_longest_prefix[index] for index in batch_of_indices]
        prefix2cache = OrderedDict()

        # compute prefixes
        for prefix in set(prefixes):
            if prefix == self.cached_sequence:
                past_key_values = self.cached_past_key_values
            else:
                past_key_values = model(input_ids=torch.tensor(prefix).to(model.device)).past_key_values
                # tensor(layers, keys/values, batch_size, num_heads, sequence_len, embed_size_per_head)
                past_key_values = torch.stack(tuple(torch.stack(past_key_values[i]) for i in range(len(past_key_values))))
                # tensor(layers, keys/values, num_heads, sequence_len, embed_size_per_head)
                past_key_values = past_key_values.squeeze(2)
            prefix2cache[prefix] = past_key_values

        # update cache with last one retrieved since instances come in order by common prefix
        self.cached_sequence, self.cached_past_key_values = list(prefix2cache.items())[-1]

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
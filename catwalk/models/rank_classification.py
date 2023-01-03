import collections
import functools
from typing import Dict, Any, List, Tuple, Sequence, Iterator, Union, Mapping, Optional, cast, Callable

import more_itertools
import torch
from tango.common import Tqdm
from tango.integrations.torch.util import resolve_device
from torch import log_softmax
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, GPT2LMHeadModel, \
    AutoTokenizer, GPT2Tokenizer, T5TokenizerFast

from catwalk import cached_transformers
from catwalk.model import Model, TrainableModel, Instance
from catwalk.task import Task, InstanceFormat, RankClassificationInstance

_Model = Union[T5ForConditionalGeneration, GPT2LMHeadModel]
_Tokenizer = Union[T5TokenizerFast, GPT2Tokenizer]


class RankClassificationModel(Model):
    VERSION = "001nul"

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        *,
        likelihood_averaging: str = 'char',
        **model_kwargs
    ):
        """
        # Parameters

        pretrained_model_name_or_path : `str`
            The name of the transformer, for example `"gpt2-large"`
        likelihood_averaging : `str`, optional (default = `char`)
            The method for averaging the sum likelihood of the continuation. 'char' averages by 
            character length, 'token' averages by token length.
        model_kwargs:
            Additional kwargs passed to the `_make_model` method.
        """
        assert likelihood_averaging in {'char', 'token'}
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.likelihood_averaging = likelihood_averaging
        self.model_kwargs = model_kwargs

    @classmethod
    def _make_model(cls, pretrained_model_name_or_path: str, *, make_copy: bool = False, **kwargs) -> _Model:
        raise NotImplementedError

    def predict(  # type: ignore
        self,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        *,
        batch_size: int = 32,
        max_instances_in_memory: int = 32 * 1024,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        model = self._make_model(
            self.pretrained_model_name_or_path,
            device_map="auto" if torch.cuda.device_count() > 0 else None,
            **self.model_kwargs).eval()
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
        fewshot_seed: Optional[int] = None
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

    def trainable_copy(self, **kwargs) -> TrainableModel:
        return TrainableRankClassificationModel(
            self._make_model(self.pretrained_model_name_or_path, make_copy=True, **self.model_kwargs),
            cached_transformers.get_tokenizer(AutoTokenizer, self.pretrained_model_name_or_path),
            self.predict_chunk)


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
        tokenize = functools.partial(
            self.tokenizer,
            return_tensors="pt",
            padding=True,
            truncation=True,
            pad_to_multiple_of=8,
            is_split_into_words=False)
        if hasattr(self.model, "encoder"):
            # Huggingface tokenizers don't do any of this correctly, so we have to do it manually.
            tokenized_strings = tokenize([s[0] for s in correct_strings])
            with self.tokenizer.as_target_tokenizer():
                tokenized_labels = self.tokenizer(
                    [s[1] for s in correct_strings],
                    return_attention_mask=False
                )['input_ids']
            # remove trailing EOS token
            for i in range(len(tokenized_labels)):
                if tokenized_labels[i][-1] == self.tokenizer.eos_token_id:
                    tokenized_labels[i] = tokenized_labels[i][:-1]
            tokenized_labels = pad_sequence(
                [torch.tensor(s, dtype=torch.long) for s in tokenized_labels],
                batch_first=True,
                padding_value=-100)
            tokenized_strings["labels"] = tokenized_labels
        else:
            tokenized_strings = tokenize(correct_strings, return_token_type_ids=True)
            tokenized_strings['labels'] = torch.full_like(tokenized_strings.input_ids, -100)
            for i, label in enumerate(tokenized_strings.labels):
                mask = [bool(s == 1) for s in tokenized_strings['token_type_ids'][i]]
                # This is a workaround because token_type_ids does not correctly account
                # for special tokens that are added by the tokenizer, causing this sequence
                # to be shorter than the actual input_ids sequence. We simply pad with false
                # at the beginning assuming that special tokens will be added at the beginning
                mask = [False] * (len(tokenized_strings.input_ids[0]) - len(mask)) + mask
                label[mask] = tokenized_strings.input_ids[i, mask]
            del tokenized_strings['token_type_ids']
        return {
            key: tensor.to(self.model.device)
            for key, tensor in tokenized_strings.items()
        }


@Model.register("rc::encoder_decoder")
class EncoderDecoderRCModel(RankClassificationModel):
    VERSION = RankClassificationModel.VERSION + "002spt"

    @classmethod
    def _make_model(
        cls,
        pretrained_model_name_or_path: str,
        *,
        make_copy: bool = False,
        **kwargs
    ) -> T5ForConditionalGeneration:
        return cached_transformers.get(
            AutoModelForSeq2SeqLM,
            pretrained_model_name_or_path,
            make_copy=make_copy,
            **kwargs)

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
            with Tqdm.tqdm(ordered_indices, desc="Running log-likelihood queries") as ordered_indices_tqdm:
                batches_of_indices = more_itertools.chunked(ordered_indices_tqdm, batch_size)
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


@Model.register("rc::decoder_only")
class DecoderOnlyRCModel(RankClassificationModel):
    @classmethod
    def _make_model(
        cls,
        pretrained_model_name_or_path: str,
        *,
        make_copy: bool = False,
        **kwargs
    ) -> GPT2LMHeadModel:
        return cached_transformers.get(
            AutoModelForCausalLM,
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
    ) -> Sequence[float]:
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
                            ids = ids[-(tokenizer.model_max_length+1):][:-1]
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
                        denom = len(tuples[i][1]) if self.likelihood_averaging == 'char' else len(instance_continuation)
                        results[i] = float(instance_logits.sum()) / denom

        assert None not in results
        return cast(Sequence[float], results)

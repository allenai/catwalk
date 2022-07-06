import collections
from typing import Dict, Any, List, Tuple, Sequence, Iterator, Union, Mapping

import torch
from transformers import T5ForConditionalGeneration, GPT2LMHeadModel, \
    GPT2Tokenizer, T5TokenizerFast
import numpy as np
import warnings

from catwalk.task import Task, InstanceFormat, RankClassificationInstance
from catwalk.models.rank_classification import DecoderOnlyRCModel


_Model = Union[T5ForConditionalGeneration, GPT2LMHeadModel]
_Tokenizer = Union[T5TokenizerFast, GPT2Tokenizer]


class MetaICLModel(DecoderOnlyRCModel):

    def __init__(self, pretrained_model_name_or_path: str, max_length_per_example : int = 256, continuation_seperator: str = '\n', example_seperatior='\n\n\n'):
        super().__init__(pretrained_model_name_or_path)
        self.max_length_per_example = max_length_per_example
        self.continuation_seperator = continuation_seperator
        self.example_seperatior = example_seperatior
        self.instances_truncated = 0
        self.instances_total = 0


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
        self.instances_truncated = 0
        self.instances_total = 0

        rc_instances: List[RankClassificationInstance] = []
        for i, instance in enumerate(instances):
            # pre-instance processsing (both fewshot and test example)
            instance['input'] = self._apply_meta_icl_per_instance_truncation(instance, tokenizer, is_icl_demonstration=False)
            fewshot_instances=task.get_fewshot_instances(num_shots, random_seed=fewshot_seed if fewshot_seed is not None else i, exceptions=instance)
            truncated_fewshot_instances = []
            for i, fewshot_instance in enumerate(fewshot_instances):
                if i == 0:
                    fewshot_instance['input'] = self._apply_meta_icl_per_instance_truncation(fewshot_instance, tokenizer, is_first=True)
                else:
                    fewshot_instance['input'] =  self._apply_meta_icl_per_instance_truncation(fewshot_instance, tokenizer)
                truncated_fewshot_instances.append(fewshot_instance)
            
            rc_instances.append(task.convert_instance(
                instance,
                InstanceFormat.RANK_CLASSIFICATION,
                fewshot_instances=truncated_fewshot_instances,
                continuation_seperator=self.continuation_seperator,
                example_seperator=self.example_seperatior,
            ))
        
        rc_instances = [self._apply_meta_icl_concatinated_input_truncation(instance, tokenizer) for instance in rc_instances]

        # get all the tuples
        for instance_index, instance in enumerate(rc_instances):
            for instance_request in instance.choices:
                instance_index_to_tuple_indices[instance_index].append(len(tuples))
                tuples.append(instance_request)

        # run the requests
        results = self._run_loglikelihood(tuples, model, tokenizer, batch_size)

        # TODO figure out why this is being suppressed
        if self.instances_truncated == self.instances_total:
            warnings.warn("All examples in this dataset chunk are being truncated after concatenation, consider using smaller max_length_per_example")

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
        
    def _apply_meta_icl_per_instance_truncation(self, instance: Dict[str, Any], tokenizer: _Tokenizer, is_icl_demonstration: bool = True, is_first: bool = False) -> str:
        """Applies identical truncation as in metaicl.data.py.MetaICLData._prepro_each_datapoint"""
        # formatting is added and then thrown away just because it is needed for truncation lengths
        input_tokens = tokenizer(('' if is_first else self.example_seperatior) + instance['input'])["input_ids"]

        if is_icl_demonstration:
            output_tokens = tokenizer(self.continuation_seperator + instance["output"])["input_ids"]
            input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]
            assert len(input_tokens)+len(output_tokens)+2<=self.max_length_per_example, \
                (instance.get("task", None), len(input_tokens), len(output_tokens), self.max_length_per_example)
            instance['output'] = tokenizer.decode(output_tokens)
        else:
            assert len(instance["options"])>=2, instance
            assert instance["output"] in instance["options"]
            option_tokens = [tokenizer(self.continuation_seperator + option)["input_ids"] for option in instance["options"]]
            option_length = np.max([len(option) for option in option_tokens])

            if len(input_tokens)>=self.max_length_per_example - 2 - option_length:
                input_tokens = input_tokens[:self.max_length_per_example - 2 - option_length]

        return tokenizer.decode(input_tokens).strip()

    def _apply_meta_icl_concatinated_input_truncation(self, instance: RankClassificationInstance, tokenizer: _Tokenizer) -> RankClassificationInstance:
        """Applies identical truncation as in metaicl.data.py.MetaICLData.prepro_sentence_pair_single"""
        new_choices = []
        for choice in instance.choices:
            ids1 = tokenizer(choice[0])['input_ids']
            ids2 = tokenizer(choice[1])['input_ids']
            self.instances_total += 1
            if len(ids1)+len(ids2) > tokenizer.model_max_length:
                self.instances_truncated += 1
                ids1 = ids1[len(ids1)+len(ids2)-tokenizer.model_max_length:] # len = max_length-len(ids2)
                assert len(ids1)+len(ids2)==tokenizer.model_max_length

            n_mask = tokenizer.model_max_length-len(ids1)-len(ids2)
            assert n_mask>=0, (tokenizer.model_max_length, len(ids1), len(ids2))
            new_choices.append((tokenizer.decode(ids1), choice[1]))

        label = instance.correct_choice
        assert label < len(new_choices)
        return RankClassificationInstance(new_choices, label)
        
import collections
import functools
import logging
from dataclasses import dataclass
from typing import Sequence, Dict, Any, Iterator, Mapping, Tuple, List, Optional

import bettermap
import torch
from tango.common import Tqdm

from catwalk.models.rank_classification import RankClassificationModel, _Model, _Tokenizer, EncoderDecoderRCModel, \
    DecoderOnlyRCModel
from catwalk.task import RankClassificationInstance, InstanceFormat, Task
from catwalk.tasks.promptsource import WithPromptsourceMixin


logger = logging.getLogger(__name__)


class PromptsourceEncoderDecoderRCModel(EncoderDecoderRCModel):
    VERSION = EncoderDecoderRCModel.VERSION + "004acc"

    def predict_chunk(
        self: RankClassificationModel,
        task: Task,
        instances: Sequence[Dict[str, Any]],
        model: _Model,
        tokenizer: _Tokenizer,
        batch_size: int = 32,
        num_shots: int = 0,
        fewshot_seed: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        instance_index_to_tuple_indices: Mapping[Tuple[int, str], List[int]] = collections.defaultdict(list)
        tuples: List[Tuple[str, str]] = []

        # Applying promptsource is slow, so we do it in parallel.
        def convert_instance(i: int) -> Dict[str, RankClassificationInstance]:
            instance = instances[i]
            return task.convert_instance(
                instance,
                InstanceFormat.PROMPTSOURCE,
                fewshot_instances=task.get_fewshot_instances(
                    num_shots,
                    random_seed=fewshot_seed if fewshot_seed is not None else i,
                    exceptions=instance))

        map_fn = functools.partial(bettermap.map_in_chunks, chunk_size=64)
        #map_fn = map        # for debugging
        with Tqdm.tqdm(
            map_fn(convert_instance, range(len(instances))),
            total=len(instances),
            desc="Converting instances"
        ) as converted_instances_tqdm:
            rc_instances: List[Dict[str, RankClassificationInstance]] = list(converted_instances_tqdm)

        # Remove prompts that have no gold answer, since we can't evaluate them.
        for instance_dict in rc_instances:
            prompt_names = list(instance_dict.keys())
            for prompt_name in prompt_names:
                if instance_dict[prompt_name].correct_choice is None:
                    del instance_dict[prompt_name]
            assert len(instance_dict) > 0

        # get all the tuples
        for instance_index, instance_dict in enumerate(rc_instances):
            for prompt_name, rc_instance in instance_dict.items():
                for instance_request in rc_instance.choices:
                    instance_index_to_tuple_indices[(instance_index, prompt_name)].append(len(tuples))
                    tuples.append(instance_request)

        # run the requests
        results = self._run_loglikelihood(tuples, model, tokenizer, batch_size)

        # collect the results
        for instance_index, instance_dict in enumerate(rc_instances):
            result = {}
            for prompt_name, rc_instance in instance_dict.items():
                tuple_indices = instance_index_to_tuple_indices[(instance_index, prompt_name)]
                results_for_instance_and_prompt = [results[i] for i in tuple_indices]
                result_tensor = torch.tensor(results_for_instance_and_prompt)
                metric_args = (result_tensor, rc_instance.correct_choice)
                prompt_name = prompt_name.strip().replace(" ", "_")
                result[prompt_name + "_acc"] = metric_args
            yield result

    def calculate_metrics(self, task: Task, predictions: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        assert isinstance(task, WithPromptsourceMixin)
        templates = task.promptsource_templates
        assert templates is not None

        @dataclass
        class AccuracyAccumulator:
            instances_seen: int
            num_correct: int

        accaccs: Dict[str, AccuracyAccumulator] = {}
        for template_name in templates.all_template_names:
            accaccs[template_name.strip().replace(" ", "_") + "_acc"] = AccuracyAccumulator(0, 0)

        with Tqdm.tqdm(predictions, desc="Calculating metrics") as predictions_tqdm:
            for prediction in predictions_tqdm:
                for metric_name, metric_args in prediction.items():
                    try:
                        accacc = accaccs[metric_name]
                    except KeyError:
                        continue

                    logits, label = metric_args
                    if logits.argmax() == label:
                        accacc.num_correct += 1
                    accacc.instances_seen += 1
        for metric_name, accacc in accaccs.items():
            if accacc.instances_seen <= 0:
                logger.warning("Metric %s was not seen in predictions.", metric_name)
        return {
            metric_name: accacc.num_correct / accacc.instances_seen
            for metric_name, accacc in accaccs.items()
            if accacc.instances_seen > 0
        }


class PromptsourceDecoderOnlyRCModel(DecoderOnlyRCModel):
    VERSION = PromptsourceEncoderDecoderRCModel.VERSION
    predict_chunk = PromptsourceEncoderDecoderRCModel.predict_chunk
    calculate_metrics = PromptsourceEncoderDecoderRCModel.calculate_metrics
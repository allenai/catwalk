from abc import ABC
from dataclasses import dataclass
from typing import Optional, Iterator, Sequence, Dict, Any, List

import datasets

from tango.common.sequences import MappedSequence

from ai2_lm_eval.models.model import ModelForEvaluation
from ai2_lm_eval.tasks.task import Task, Metrics
from ai2_lm_eval.utilities import get_from_dict


class PerplexityTask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        text: str

    @dataclass
    class InstanceResult(Task.InstanceResult):
        pass    # The instance's text is its own label for this task.

    def run_inference(
        self,
        model: ModelForEvaluation,
        instances: Sequence[Instance],
        **kwargs
    ) -> Iterator['PerplexityTask.InstanceResult']:
        return model.do_perplexity(self, instances, **kwargs)

    def calculate_metrics(self, results: Iterator['PerplexityTask.InstanceResult']) -> Metrics:
        raise NotImplementedError

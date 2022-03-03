from abc import ABC
from dataclasses import dataclass
from typing import Optional, List, Union, Any, Iterator, Sequence, Dict

import torch
from torchmetrics.functional import precision_recall, accuracy, f1_score

from catwalk.models.model import ModelForEvaluation
from catwalk.tasks.task import Task, Metrics, FromDatasetMixin
from catwalk.utilities import get_from_dict


def _mc_default_metrics(results: Iterator['MCTask.InstanceResult']) -> Metrics:
    logits_and_targets = [(r.logits, r.label) for r in results]
    logits = torch.stack([l for l, _ in logits_and_targets])
    targets = torch.tensor([t for _, t in logits_and_targets], dtype=torch.int32)
    p, r = precision_recall(logits, targets)
    return {
        "accuracy": float(accuracy(logits, targets)),
        "f1": float(f1_score(logits, targets)),
        "precision": float(p),
        "recall": float(r),
    }


class MCTask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        context: Optional[str]
        question: str
        answer_choices: List[str]
        correct_answer_index: int

    @dataclass
    class InstanceResult(Task.InstanceResult):
        label: int
        logits: torch.Tensor

    number_of_choices: int = NotImplemented

    def run_inference(
        self,
        model: ModelForEvaluation,
        instances: Sequence[Instance],
        **kwargs
    ) -> Iterator['MCTask.InstanceResult']:
        return model.do_multiple_choice(self, instances, **kwargs)

    calculate_metrics = _mc_default_metrics


class MCTaskFromDataset(FromDatasetMixin, MCTask):
    VERSION = "001"

    def __init__(
        self,
        name: str,
        dataset_path: str,
        dataset_name: Optional[str],
        *,
        number_of_choices: int,
        context_field: Optional[str] = None,
        question_field: str,
        answer_choices_fields: Union[str, List[str]],
        correct_answer_index_field: str,
        id_field: Optional[str] = None,
        split_mappings: Optional[Dict[str, Union[str, List[str]]]] = None
    ):
        MCTask.__init__(self, name)
        FromDatasetMixin.__init__(
            self,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            id_field=id_field,
            split_mappings=split_mappings)
        self.context_field = context_field
        self.question_field = question_field
        self.answer_choices_fields = answer_choices_fields
        self.correct_answer_index_field = correct_answer_index_field
        self.number_of_choices = number_of_choices

    @classmethod
    def _normalize_answers(cls, answer: Any) -> int:
        if isinstance(answer, int):
            return answer
        if isinstance(answer, str):
            try:
                return int(answer)
            except ValueError:
                if len(answer) == 1:
                    answer = answer.lower()
                    return ord(answer[0]) - ord('a')
                else:
                    raise

    def _instance_from_json(self, instance: Dict[str, Any]) -> MCTask.Instance:
        id = str(get_from_dict(instance, self.id_field)) if self.id_field else instance["__default_id"]
        if isinstance(self.answer_choices_fields, str):
            answer_choices = get_from_dict(instance, self.answer_choices_fields)
        else:
            answer_choices = [get_from_dict(instance, field) for field in self.answer_choices_fields]
        return MCTask.Instance(
            id=id,
            metadata={},
            context=instance[self.context_field] if self.context_field else None,
            question=instance[self.question_field],
            answer_choices=answer_choices,
            correct_answer_index=self._normalize_answers(instance[self.correct_answer_index_field]))


class CBTTask(FromDatasetMixin, MCTask):
    VERSION = "001"

    def __init__(self, dataset_name: str):
        MCTask.__init__(self, "cbt_" + dataset_name)
        FromDatasetMixin.__init__(
            self,
            dataset_path="cbt",
            dataset_name=dataset_name)
        self.number_of_choices = 10

    @classmethod
    def detokenize(cls, text):
        text = text.replace(" '", "'")
        text = text.replace(" \n", "\n")
        text = text.replace("\n ", "\n")
        text = text.replace(" n't", "n't")
        text = text.replace("`` ", '"')
        text = text.replace("''", '"')
        # punctuation
        text = text.replace(" :", ":")
        text = text.replace(" ;", ";")
        text = text.replace(" !", "!")
        text = text.replace(" ?", "?")
        text = text.replace(" ,", ",")
        text = text.replace(" .", ".")
        return text

    def _instance_from_json(self, instance: Dict[str, Any]) -> MCTask.Instance:
        context = self.detokenize(" ".join(instance["sentences"]))
        answer_choices = instance["options"]
        return MCTask.Instance(
            id=instance["__default_id"],
            metadata={},
            context=context,
            question=instance["question"],
            answer_choices=answer_choices,
            correct_answer_index=answer_choices.index(instance["answer"]))

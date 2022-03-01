from abc import ABC
from dataclasses import dataclass
from typing import Optional, Iterator, Sequence, Dict, Any, List, Union

from ai2_lm_eval.models.model import ModelForEvaluation
from ai2_lm_eval.tasks.task import Task, Metrics, FromDatasetMixin
from ai2_lm_eval.utilities import get_from_dict


class QATask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        context: Optional[str]
        question: str
        answers: List[str]   # Could be empty if the question is unanswerable

    @dataclass
    class InstanceResult(Task.InstanceResult):
        predicted: str

    def run_inference(
        self,
        model: ModelForEvaluation,
        instances: Sequence[Instance],
        **kwargs
    ) -> Iterator['QATask.InstanceResult']:
        return model.do_qa(self, instances, **kwargs)

    def calculate_metrics(self, results: Iterator['QATask.InstanceResult']) -> Metrics:
        from datasets import load_metric
        squad = load_metric("squad_v2")

        # Why is the squad metric so weird?
        preds = []
        target = []
        for result in results:
            preds.append({
                "prediction_text": result.predicted or "",
                "id": result.instance.id,
                "no_answer_probability": 1 if result.predicted is None else 0
            })
            target.append({
                "id": result.instance.id,
                "answers": {
                    "answer_start": [-1 for _ in result.instance.answers],
                    "text": [answer for answer in result.instance.answers]
                }
            })

        return {
            key: float(value)
            for key, value in squad.compute(predictions=preds, references=target, no_answer_threshold=0.5).items()
        }


class QATaskFromDataset(FromDatasetMixin, QATask):
    VERSION = "002"

    def __init__(
        self,
        name: str,
        dataset_path: str,
        dataset_name: Optional[str],
        *,
        context_field: Optional[str],
        question_field: str,
        answer_field: str,
        id_field: Optional[str] = None,
        split_mappings: Optional[Dict[str, Union[str, List[str]]]] = None
    ):
        QATask.__init__(self, name)
        FromDatasetMixin.__init__(
            self,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            id_field=id_field,
            split_mappings=split_mappings)

        self.context_field = context_field
        self.question_field = question_field
        self.answer_field = answer_field

    def _instance_from_json(self, instance: Dict[str, Any]) -> QATask.Instance:
        answers = get_from_dict(instance, self.answer_field)
        if isinstance(answers, str):
            answers = [answers]
        else:
            answers = answers
        return QATask.Instance(
            id=str(get_from_dict(instance, self.id_field)) if self.id_field else instance["__default_id"],
            metadata={},
            context=get_from_dict(instance, self.context_field) if self.context_field else None,
            question=get_from_dict(instance, self.question_field),
            answers=answers)

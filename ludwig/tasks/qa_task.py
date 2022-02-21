from abc import ABC
from dataclasses import dataclass
from typing import Optional, Iterator

import datasets
from torchmetrics.functional import squad

from ludwig.models.model import ModelForEvaluation
from ludwig.tasks.task import Task, Metrics
from ludwig.utilities import get_from_dict


class QATask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        context: str
        question: str
        answer: Optional[str]   # If not given, the question is unanswerable / "not enough information"

    @dataclass
    class InstanceResult(Task.InstanceResult):
        label: str
        predicted: str

    def evaluate_model(self, model: ModelForEvaluation, **kwargs) -> Metrics:
        results = model.do_qa(self, **kwargs)

        # Why is the squad metric so weird?
        preds = []
        target = []
        for result in results:
            preds.append({
                "prediction_text": result.predicted or "",  # Hack to make the squad metric work with unanswerable questions.
                "id": result.id
            })
            target.append({
                "id": result.id,
                "answers": {
                    "answer_start": None,
                    "text": [result.label]
                }
            })

        return {
            key: float(value)
            for key, value in squad(preds, target).items()
        }


class QATaskFromDataset(QATask):
    def __init__(
            self,
            name: str,
            dataset: str,
            *,
            dataset_config: Optional[str] = None,
            context_field: Optional[str],
            question_field: str,
            answer_field: str,
            id_field: Optional[str] = None,
    ):
        super().__init__(name)
        # We want this lazy, so it's a lambda.
        self.get_dataset = lambda split: datasets.load_dataset(dataset, dataset_config, split=split)
        self.context_field = context_field
        self.question_field = question_field
        self.answer_field = answer_field
        self.id_field = id_field

    def get_instances(self, split: str) -> Iterator[QATask.Instance]:
        for i, instance in enumerate(self.get_dataset(split)):
            yield QATask.Instance(
                id=str(get_from_dict(instance, self.id_field)) if self.id_field else str(i),
                context=get_from_dict(instance, self.context_field) if self.context_field else None,
                question=get_from_dict(instance, self.question_field),
                answer=get_from_dict(instance, self.answer_field))

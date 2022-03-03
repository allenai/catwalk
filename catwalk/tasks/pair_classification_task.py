from abc import ABC
from dataclasses import dataclass
from typing import List, Sequence, Iterator, Optional, Dict, Any, Union

from catwalk.models.model import ModelForEvaluation
from catwalk.tasks.task import Task, FromDatasetMixin
from catwalk.utilities import get_from_dict


class PairClassificationTask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        text1: str
        text2: str
        label: str  # Has to be one of the labels the task specifies.

    @property
    def labels(self) -> List[str]:
        raise NotImplementedError

    def run_inference(
        self,
        model: ModelForEvaluation,
        instances: Sequence[Instance],
        **kwargs
    ) -> Iterator['PairClassificationTask.InstanceResult']:
        return model.do_pair_classification(self, instances, **kwargs)


class PairClassificationTaskFromDataset(FromDatasetMixin, PairClassificationTask):
    VERSION = "001"

    def __init__(
        self,
        name: str,
        dataset_path: str,
        dataset_name: Optional[str],
        text1_field: str,
        text2_field: str,
        label_field: str,
        labels: List[str],
        *,
        id_field: Optional[str] = None,
        split_mappings: Optional[Dict[str, Union[str, List[str]]]]
    ):
        PairClassificationTask.__init__(self, name)
        FromDatasetMixin.__init__(
            self,
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            id_field=id_field,
            split_mappings=split_mappings)
        self.text1_field = text1_field
        self.text2_field = text2_field
        self.label_field = label_field
        self.labels_list = labels

    @property
    def labels(self) -> List[str]:
        return self.labels_list

    def _instance_from_json(self, instance: Dict[str, Any]) -> PairClassificationTask.Instance:
        label = get_from_dict(instance, self.label_field)
        if isinstance(label, int):
            label = self.labels[label]
        return PairClassificationTask.Instance(
            id=str(get_from_dict(instance, self.id_field)) if self.id_field else instance["__default_id"],
            metadata={},
            text1=get_from_dict(instance, self.text1_field),
            text2=get_from_dict(instance, self.text2_field),
            label=label)


class BlimpTask(FromDatasetMixin, PairClassificationTask):
    def __init__(self, dataset_name: str):
        PairClassificationTask.__init__(self, "blimp_" + dataset_name)
        FromDatasetMixin.__init__(
            self,
            dataset_path="blimp",
            dataset_name=dataset_name,
            id_field="UID",
            split_mappings={"validation": "train"}     # There is no validation set, and Eleuther does it this way.
        )

    @property
    def labels(self) -> List[str]:
        return ["True", "False"]

    def _instance_from_json(self, instance: Dict[str, Any]) -> PairClassificationTask.Instance:
        if instance['pair_id'] % 2 == 0:
            text1 = instance["sentence_good"]
            text2 = instance["sentence_bad"]
            label = self.labels[0]
        else:
            text1 = instance["sentence_bad"]
            text2 = instance["sentence_good"]
            label = self.labels[1]

        return PairClassificationTask.Instance(
            id=str(get_from_dict(instance, self.id_field)) if self.id_field else instance["__default_id"],
            metadata={},
            text1=text1,
            text2=text2,
            label=label)

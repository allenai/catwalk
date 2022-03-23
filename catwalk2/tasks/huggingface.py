from typing import Optional, Sequence, Dict, Any

import datasets

from catwalk2 import Task
from catwalk2.task import TaskType


class HFDatasetsTask(Task):
    def __init__(
        self,
        task_type: TaskType,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        *,
        version_override: Optional[str] = None
    ):
        super().__init__(task_type, version_override=version_override)
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name

    def has_split(self, split: str) -> bool:
        return split in datasets.get_dataset_split_names(self.dataset_path, self.dataset_name)

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        return datasets.load_dataset(self.dataset_path, self.dataset_name, split=split)


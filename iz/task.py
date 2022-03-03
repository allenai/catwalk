from abc import ABC
from typing import Dict, Any, Iterable, Callable, Optional

import datasets
from tango.common import Registrable

#
# These are classes (instead of functions) so we can attach a VERSION to them to invalidate results when a task
# changes.
#


class IzTask(Registrable, ABC):
    def get_split(self, split: str) -> Iterable[Dict[str, Any]]:
        raise NotImplementedError


class DatasetsTask(IzTask):
    def __init__(
        self,
        dataset_path: str,
        dataset_name: Optional[str] = None,
        *,
        version_override: Optional[str] = None
    ):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        if version_override is not None:
            self.VERSION = version_override

    def get_split(self, split: str) -> Iterable[Dict[str, Any]]:
        return datasets.load_dataset(self.dataset_path, self.dataset_name, split=split)


Task = Callable[[str], Iterable[Dict[str, Any]]]

TASKS = {
    "wikitext": DatasetsTask("wikitext", "wikitext-2-raw-v1")
}
import re
from abc import ABC
from typing import Dict, Any, Iterable, Callable, Optional

import datasets
from tango.common import Registrable

#
# These are classes (instead of functions) so we can attach a VERSION to them to invalidate results when a task
# changes.
#
from catwalk2 import crossfit


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


class CrossfitTask(IzTask):
    def __init__(self, name: str):
        self.name = name

    _split_re = re.compile(r"[^\s]+:")

    def get_split(self, split: str) -> Iterable[Dict[str, Any]]:
        crossfit_task = crossfit.TASKS[self.name]
        dataset = crossfit_task.load_dataset()
        train_lines, test_lines = crossfit_task.get_train_test_lines(dataset)

        if split == "train":
            lines = train_lines
        elif split == "validation":
            lines = test_lines  # For some reason, CrossFit calls this "test", but the code loads the validation set.
        else:
            raise KeyError()

        for input, output in lines:
            result = {
                "target": output
            }
            i = 0
            current_field = f"field_{len(result)}"
            while i < len(input):
                m = self._split_re.search(input, i)
                if not m:
                    break
                value = input[i:m.start(0)].strip()
                if len(value) > 0:
                    result[current_field] = value
                current_field = m.group(0)[:-1]
                i = m.end(0)
            value = input[i:].strip()
            if len(value) > 0:
                result[current_field] = value
            yield result


Task = Callable[[str], Iterable[Dict[str, Any]]]

TASKS = {
    "wikitext": DatasetsTask("wikitext", "wikitext-2-raw-v1")
}

for crossfit_task in crossfit.TASKS.keys():
    TASKS[crossfit_task] = CrossfitTask(crossfit_task)
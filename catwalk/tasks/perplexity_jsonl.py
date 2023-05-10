from typing import Dict, Any, Sequence
from copy import deepcopy
import gzip
import json

from catwalk.task import Task, InstanceFormat
from cached_path import cached_path

class PerplexityJsonLTask(Task):
    def __init__(
        self,
        files=None  # files (or URLs) to be used
    ):
        Task.__init__(self)
        self.files = files
        self._cached_paths = None
        self._cache_dir = None   # Can override cache dir
        self.add_instance_conversion(InstanceFormat.ELEUTHER_DOC, self.instance_as_eleuther_doc)

    def clone(self, files):
        new_task = deepcopy(self)
        new_task.files = files
        return new_task

    def has_split(self, split: str) -> bool:
        return True  # Assume the files are for the requested split

    def cached_paths(self):
        if self._cached_paths is None:
            self._cached_paths = []
            for file in self.files:
                self._cached_paths.append((file, cached_path(file, cache_dir=self._cache_dir)))
        return self._cached_paths

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        instances = []
        for (orig_file, cache_file) in self.cached_paths():
            if orig_file.endswith('.gz'):
                with gzip.open(cache_file, 'r') as file:
                    for line in file:
                        instances.append(json.loads(line.decode("utf-8").strip()))
            else:
                with open(cache_file, 'r') as file:
                    for line in file:
                        instances.append(json.loads(line.strip()))
        return instances

    def instance_as_eleuther_doc(self, instance):
        return instance.get('text', instance.get('doc'))

    @property
    def default_split(self) -> str:
        return "validation"

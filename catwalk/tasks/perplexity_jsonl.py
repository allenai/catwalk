import gzip
import json
import os
from copy import deepcopy
from typing import Any, Dict, Sequence

from cached_path import cached_path

from catwalk.task import InstanceFormat, Task

# Task for evaluating a generic set of files (or URLs) on perplexity metrics


class PerplexityJsonLTask(Task):
    def __init__(
        self,
        files=None,  # files (or URLs) to be used
        detailed_output=False,  # Whether a detailed output should be generated
    ):
        Task.__init__(self)
        self.files = files
        self.detailed_output = detailed_output
        self._cached_paths = None
        self._cache_dir = None  # Can override cache dir
        self.add_instance_conversion(
            InstanceFormat.ELEUTHER_DOC, self.instance_as_eleuther_doc
        )
        self.file_extensions = [".jsonl.gz"]

    def clone(self, files, detailed_output=False):
        new_task = deepcopy(self)
        new_task.files = files
        new_task.detailed_output = detailed_output
        return new_task

    def has_split(self, split: str) -> bool:
        return True  # Assume the files are for the requested split

    def cached_paths(self):
        if self._cached_paths is None:
            self._cached_paths = []
            for file in self.files:
                self._cached_paths.append(
                    (file, cached_path(file, cache_dir=self._cache_dir))
                )
        return self._cached_paths

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        instances = []
        all_files = []
        # expand directories if need be
        for orig_file, cache_file in self.cached_paths():
            if os.path.isdir(cache_file):
                for root, dirs, files in os.walk(cache_file):
                    for file in files:
                        for extension in self.file_extensions:
                            if file.endswith(extension):
                                all_files.append((file, os.path.join(root, file)))
            else:
                all_files.append((orig_file, cache_file))

        for orig_file, cache_file in all_files:
            if orig_file.endswith(".gz"):
                with gzip.open(cache_file, "r") as file:
                    for line in file:
                        instance = json.loads(line.decode("utf-8").strip())
                        instance["orig_file_name"] = orig_file
                        instances.append(instance)
            else:
                with open(cache_file, "r") as file:
                    for line in file:
                        instance = json.loads(line.strip())
                        instance["orig_file_name"] = orig_file
                        instances.append(instance)
        return instances

    def process_extra_output(self, output):
        # Combine (and then delete) individual token/logits stats
        if not self.detailed_output:
            return output
        token_stats: Dict = {}
        for instance in output["per_instance"]:
            tokens = instance["prediction"]["model_output"]["tokens"]
            logits = instance["prediction"]["model_output"]["logits"]
            del instance["prediction"]["model_output"]["tokens"]
            del instance["prediction"]["model_output"]["logits"]
            assert len(tokens) == len(logits)
            source = instance["instance"].get("source", "")
            subdomain = instance["instance"].get("subdomain", "")
            domain_id = f"{source}__{subdomain}"
            if domain_id not in token_stats:
                token_stats[domain_id] = {}
            for token, logit in zip(tokens, logits):
                old_count, old_avg = token_stats[domain_id].get(token, (0, 0))
                new_count = old_count + 1
                new_avg = (old_count * old_avg + logit) / new_count
                token_stats[domain_id][token] = (new_count, new_avg)
        for domain_id, stats in token_stats.items():
            # Sort by average logit
            token_stats[domain_id] = dict(sorted(stats.items(), key=lambda x: x[1][1]))
        output["extra_output"] = {"token_count_avg_logits_by_domain": token_stats}
        return output

    def instance_as_eleuther_doc(self, instance):
        return instance.get("text", instance.get("doc"))

    @property
    def default_split(self) -> str:
        return "validation"

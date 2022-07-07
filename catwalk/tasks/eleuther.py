import random
from typing import Dict, Any, Optional, Union, Callable, Sequence, List

from tango.common.sequences import MappedSequence

from catwalk.task import Task, InstanceFormat, RankClassificationInstance

import lm_eval.tasks
from lm_eval.base import Task as EAITask


@Task.register("eleuther")
class EleutherTask(Task):
    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAITask]],
        *,
        random_seed: Optional[int] = None,
        version_override: Optional[str] = None,
        ranked_classification: bool = False
    ):
        super().__init__(version_override=version_override)

        if isinstance(eleuther_task, str):
            # Eleuther tasks eagerly download their data when they are created. We don't want that, so we have to
            # make this lazy.
            self.eleuther_task_fn = lambda: lm_eval.tasks.get_task(eleuther_task)()
        else:
            self.eleuther_task_fn = eleuther_task

        self.eleuther_task: Optional[EAITask] = None

        # We maintain our own random, to help with determinism.
        if random_seed is None:
            random_seed = 57885161 + 43112609    # What could be more perfect than the sum of two perfect numbers?
        self.random = random.Random(random_seed)

        self.add_instance_conversion(InstanceFormat.HF_DICT, lambda x: x)
        self.add_instance_conversion(InstanceFormat.ELEUTHER_DOC, self.instance_as_eleuther_doc)
        self.add_instance_conversion(InstanceFormat.ELEUTHER_CONTEXT, self.instance_to_eleuther_context)
        self.add_instance_conversion(InstanceFormat.ELEUTHER_REQUESTS, self.instance_as_eleuther_requests)
        if ranked_classification:
            self.add_instance_conversion(InstanceFormat.RANK_CLASSIFICATION, self.instance_as_rank_classification)

    def __getstate__(self):
        result = self.__dict__.copy()
        result["eleuther_task"] = None  # We just cache this, so it doesn't need to be serialized.
        return result

    @property
    def inner_task(self) -> EAITask:
        if self.eleuther_task is None:
            self.eleuther_task = self.eleuther_task_fn()
        return self.eleuther_task

    def has_split(self, split: str) -> bool:
        print(self.inner_task.dataset)
        return split in self.inner_task.dataset

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        ds = self.inner_task.dataset[split]
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(lambda x: x, ds)
        return ds

    @property
    def default_split(self) -> str:
        # In EAI, this is different than `has_split`.
        if self.inner_task.has_test_docs():
            return "test"
        elif self.inner_task.has_validation_docs():
            return "validation"
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs.")

    def instance_as_eleuther_doc(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        return self.inner_task._process_doc(instance)

    def instance_to_eleuther_context(self, instance: Dict[str, Any], *, num_fewshot: int = 0) -> str:
        return self.inner_task.fewshot_context(self.instance_as_eleuther_doc(instance), num_fewshot, rnd=random)

    def instance_as_eleuther_requests(self, instance: Dict[str, Any], *, num_fewshot: int = 0):
        context = self.instance_to_eleuther_context(instance, num_fewshot=num_fewshot)
        return self.inner_task.construct_requests(self.instance_as_eleuther_doc(instance), context)

    def instance_as_rank_classification(
        self,
        instance: Dict[str, Any],
        *,
        fewshot_instances: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> RankClassificationInstance:
        if fewshot_instances is None:
            fewshot_instances = []
        prefix = ""
        for fewshot_instance in fewshot_instances:
            as_rc = self.instance_as_rank_classification(fewshot_instance)
            if as_rc.correct_choice is None:
                raise ValueError("Could not determine correct choice in ranked classification instance.")
            correct_choice = as_rc.choices[as_rc.correct_choice]
            prefix += f"{correct_choice[0].strip()} {correct_choice[1].strip()}\n\n"

        requests = self.instance_as_eleuther_requests(instance, **kwargs)
        choices = [
            (prefix + r.args[0], r.args[1])
            for r in requests
        ]

        doc = self.instance_as_eleuther_doc(instance)
        label = doc.get("label")
        if label is None:
            label = doc.get("gold")
        if label is None:
            label = doc.get("answer")
        if label is None:
            raise ValueError("Could not find label for instance.")

        if isinstance(label, str):
            label = label[0].lower()
            try:
                label = int(label) - 1
            except ValueError:
                label = ord(label) - ord('a')
        if not isinstance(label, int):
            raise ValueError("Could not find label for instance.")

        assert label < len(choices)
        return RankClassificationInstance(choices, label)


@Task.register("eleuther::race")
class RaceEleutherTask(EleutherTask):
    """This task is different because there is no 1:1 correspondence between HF instances and EAI instances."""
    def __init__(self, *, random_seed: Optional[int] = None, version_override: Optional[str] = None):
        super().__init__(
            "race",
            random_seed=random_seed,
            version_override=version_override)
        del self.instance_conversions[InstanceFormat.HF_DICT]
        self.add_instance_conversion(InstanceFormat.ELEUTHER_DOC, lambda x: x)

    def has_split(self, split: str) -> bool:
        if split == "train":
            return self.inner_task.has_training_docs()
        if split == "test":
            return self.inner_task.has_test_docs()
        if split == "validation":
            return self.inner_task.has_validation_docs()
        return False

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        if split == "train":
            return self.inner_task.training_docs()
        if split == "test":
            return self.inner_task.test_docs()
        if split == "validation":
            return self.inner_task.validation_docs()
        raise KeyError(split)


@Task.register("eleuther::pubmedqa")
class PubmedqaEleutherTask(EleutherTask):
    """This task is different because EAI relabels the datasets."""
    def __init__(self, *, random_seed: Optional[int] = None, version_override: Optional[str] = None):
        super().__init__("pubmedqa", random_seed=random_seed, version_override=version_override)

    def has_split(self, split: str) -> bool:
        if split == "train":
            return self.inner_task.has_training_docs()
        if split == "test":
            return self.inner_task.has_test_docs()
        if split == "validation":
            return self.inner_task.has_validation_docs()
        return False

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        if split == "train":
            result = self.inner_task.training_docs()
        elif split == "test":
            result = self.inner_task.test_docs()
        elif split == "validation":
            result = self.inner_task.validation_docs()
        else:
            raise KeyError(split)
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        return MappedSequence(lambda x: x, result)


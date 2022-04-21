import random
from tempfile import NamedTemporaryFile
from typing import Dict, Any, Optional, Union, Callable, Sequence

from tango.common import Tqdm
from tango.common.sequences import SqliteSparseSequence, MappedSequence

from catwalk.task import Task, InstanceFormat, PERPLEXITY_METRICS

import lm_eval.tasks
from lm_eval.base import Task as EAITask, PerplexityTask as EAIPerplexityTask
from lm_eval.tasks.common import HFTask as EAIHFTask


@Task.register("eleuther")
class EleutherTask(Task):
    """
    This is a generic Eleuther task adapter.

    Many Eleuther tasks don't return exactly their HF equivalent, which means the instances won't be convertable
    to Crossfit, Promptsource, or any other format.

    Some Eleuther tasks also disagree with HF's choice of splits.
    """

    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAITask]],
        *,
        random_seed: Optional[int] = None,
        version_override: Optional[str] = None
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

        self.add_instance_conversion(InstanceFormat.ELEUTHER_DOC, self.instance_as_eleuther_doc)
        self.add_instance_conversion(InstanceFormat.ELEUTHER_CONTEXT, self.instance_to_eleuther_context)
        self.add_instance_conversion(InstanceFormat.ELEUTHER_REQUESTS, self.instance_as_eleuther_requests)

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
        if split == "train":
            return self.inner_task.has_training_docs()
        elif split == "test":
            return self.inner_task.has_test_docs()
        elif split == "validation":
            return self.inner_task.has_validation_docs()
        else:
            return False

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        if split == "train":
            return self.inner_task.training_docs()
        elif split == "test":
            return self.inner_task.test_docs()
        elif split == "validation":
            return self.inner_task.validation_docs()
        else:
            raise KeyError(split)

    def instance_as_eleuther_doc(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        return instance

    def instance_to_eleuther_context(self, instance: Dict[str, Any], *, num_fewshot: int = 0) -> str:
        return self.inner_task.fewshot_context(self.instance_as_eleuther_doc(instance), num_fewshot, rnd=random)

    def instance_as_eleuther_requests(self, instance: Dict[str, Any], *, num_fewshot: int = 0):
        context = self.instance_to_eleuther_context(instance, num_fewshot=num_fewshot)
        return self.inner_task.construct_requests(self.instance_as_eleuther_doc(instance), context)


@Task.register("eleuther::hf")
class EleutherHFTask(EleutherTask):
    """
    This is an adapter for Eleuther tasks that are based on HF datasets.

    For most of the tasks that are based on HF datasets, we can get both the HF format and the Eleuther format of
    the instances, which means we can also apply Promptsource and Crossfit.

    Some Eleuther tasks disagree with HF's choice of splits.
    """

    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAIHFTask]],
        *,
        random_seed: Optional[int] = None,
        version_override: Optional[str] = None
    ):
        super().__init__(eleuther_task, random_seed=random_seed, version_override=version_override)

    @property
    def inner_task(self) -> EAIHFTask:
        task = super().inner_task
        assert isinstance(task, EAIHFTask)
        return task

    def has_split(self, split: str) -> bool:
        return split in self.inner_task.data

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        ds = self.inner_task.data[split]
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(lambda x: x, ds)
        return ds

    def instance_as_eleuther_doc(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        return self.inner_task._convert_standard(instance)

    def instance_to_eleuther_context(self, instance: Dict[str, Any], *, num_fewshot: int = 0) -> str:
        doc = self.instance_as_eleuther_doc(instance)
        return self.inner_task.fewshot_context(doc, num_fewshot, rnd=random)


@Task.register("eleuther::perplexity")
class EleutherPerplexityTask(EleutherTask):
    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAIPerplexityTask]],
        *,
        random_seed: Optional[int] = None,
        version_override: Optional[str] = None
    ):
        super().__init__(
            eleuther_task,
            random_seed=random_seed,
            version_override=version_override)

        self.add_metrics(PERPLEXITY_METRICS)

    @property
    def inner_task(self) -> EAIPerplexityTask:
        task = super().inner_task
        assert isinstance(task, EAIPerplexityTask)
        return task

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        # EAI Perplexity tasks have instances that are just strings. We need to wrap them in a dict.
        strings = super().get_split(split)
        if hasattr(strings, "__next__"):
            # Sometimes Eleuther returns a generator. We don't want to store the entire contents of
            # the generator in memory, so we write it to disk.
            tempfile = NamedTemporaryFile(prefix=f"{self.__class__.__name__}.", suffix=".sqlite")
            sqlite_strings = SqliteSparseSequence(tempfile.name)
            sqlite_strings.extend(Tqdm.tqdm(strings, desc=f"Reading instances for {self.__class__.__name__}"))
            strings = sqlite_strings
        return MappedSequence(lambda t: {"text": t}, strings)

    def instance_as_eleuther_doc(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        return instance["text"]

import os
import random
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

from tango.common.sequences import MappedSequence

from catwalk.dependencies.lm_eval.base import Task as EAITask
from catwalk.dependencies.lm_eval.tasks import get_task as get_eai_task
from catwalk.dependencies.lm_eval.tasks.hendrycks_test import SUBJECTS
from catwalk.metrics import EleutherMetrics
from catwalk.task import (
    InstanceFormat,
    RankClassificationInstance,
    Task,
    WithAnswerOptionsMixin,
    classification_metrics,
    rc_metrics,
)
from catwalk.tasks.promptsource import WithPromptsourceMixin

T = TypeVar("T")


def _identity(x: T) -> T:
    return x


@Task.register("eleuther")
class EleutherTask(Task, WithPromptsourceMixin):
    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAITask]],
        *,
        version_override: Optional[str] = None,
        ranked_classification: bool = False,
        promptsource_task_spec: Optional[Tuple[str, str]] = None,
        eleuther_metrics: bool = False,  # Whether to directly use Eleuther metrics
        model_args: Optional[Dict] = None,  # Extra arguments to supply to model calls
    ):
        Task.__init__(self, version_override=version_override)

        self.eleuther_task: Optional[EAITask]
        if isinstance(eleuther_task, str):
            # Eleuther tasks eagerly download their data when they are created. We don't want that, so we have to
            # make this lazy.
            self.eleuther_task_fn = get_eai_task(eleuther_task)
            self.dataset_name = self.eleuther_task_fn.DATASET_NAME
            self.dataset_path = self.eleuther_task_fn.DATASET_PATH
            self.eleuther_task = None
        else:
            self.eleuther_task_fn = eleuther_task
            self.eleuther_task = eleuther_task()
            self.dataset_name = self.eleuther_task.DATASET_NAME
            self.dataset_path = self.eleuther_task.DATASET_PATH
        if model_args:
            self.model_args = model_args
        # Sometimes the "path" is a path to a Python file. We have to fix that.
        self.dataset_path = os.path.splitext(os.path.basename(self.dataset_path))[0]

        self.add_instance_conversion(InstanceFormat.HF_DICT, _identity)
        self.add_instance_conversion(
            InstanceFormat.ELEUTHER_DOC, self.instance_as_eleuther_doc
        )
        self.add_instance_conversion(
            InstanceFormat.ELEUTHER_CONTEXT, self.instance_to_eleuther_context
        )
        self.add_instance_conversion(
            InstanceFormat.ELEUTHER_REQUESTS, self.instance_as_eleuther_requests
        )
        if ranked_classification:
            self.add_instance_conversion(
                InstanceFormat.RANK_CLASSIFICATION, self.instance_as_rank_classification
            )
        if eleuther_metrics:
            self.add_metric(
                "eleuther_metrics", partial(EleutherMetrics, inner_task=self.inner_task)
            )

        if promptsource_task_spec is None:
            WithPromptsourceMixin.__init__(self, self.dataset_path, self.dataset_name)
        else:
            WithPromptsourceMixin.__init__(self, *promptsource_task_spec)

    def __getstate__(self):
        result = self.__dict__.copy()
        result[
            "eleuther_task"
        ] = None  # We just cache this, so it doesn't need to be serialized.
        return result

    @property
    def inner_task(self) -> EAITask:
        if self.eleuther_task is None:
            self.eleuther_task = self.eleuther_task_fn()
        return self.eleuther_task

    def has_split(self, split: str) -> bool:
        return split in self.inner_task.dataset

    def get_split(self, split: str) -> Sequence[Dict[str, Any]]:
        ds = self.inner_task.dataset[split]
        # HF datasets are not sequences, even though they sometimes pretend they are. So we apply this hack
        # to make them act like sequences.
        ds = MappedSequence(_identity, ds)
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

    def instance_to_eleuther_context(
        self,
        instance: Dict[str, Any],
        *,
        num_fewshot: int = 0,
        fewshot_seed: int = 18830087,
    ) -> str:
        rnd = random.Random(fewshot_seed)
        return self.inner_task.fewshot_context(
            self.instance_as_eleuther_doc(instance), num_fewshot, rnd=rnd
        )

    def instance_as_eleuther_requests(
        self, instance: Dict[str, Any], *, num_fewshot: int = 0, fewshot_seed=None
    ):
        context = self.instance_to_eleuther_context(
            instance, num_fewshot=num_fewshot, fewshot_seed=fewshot_seed
        )
        return self.inner_task.construct_requests(
            self.instance_as_eleuther_doc(instance), context
        )

    def _guess_label(self, instance: Dict[str, Any]) -> int:
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
                label = ord(label) - ord("a")

        if not isinstance(label, int):
            raise ValueError("Could not find label for instance.")

        return label

    def instance_as_rank_classification(
        self,
        instance: Dict[str, Any],
        *,
        fewshot_instances: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> RankClassificationInstance:
        """
        Converts the given instance to an instance for performing ranked classification

        :param instance: the instance to convert
        :param fewshot_instances: the number of few-show instances to include
        :return: the instance in :class:`~catwalk.task.RankClassificationInstance` format
        """

        if fewshot_instances is None:
            fewshot_instances = []
        prefix = ""
        for fewshot_instance in fewshot_instances:
            as_rc = self.instance_as_rank_classification(fewshot_instance)
            if as_rc.correct_choice is None:
                raise ValueError(
                    "Could not determine correct choice in ranked classification instance."
                )
            correct_choice = as_rc.choices[as_rc.correct_choice]
            prefix += f"{correct_choice[0].strip()} {correct_choice[1].strip()}\n\n"

        requests = self.instance_as_eleuther_requests(instance, **kwargs)
        choices = [(prefix + r.args[0], r.args[1]) for r in requests]

        label = self._guess_label(instance)
        assert label < len(choices)
        return RankClassificationInstance(choices, label)


@Task.register("eleuther::classification")
class EleutherClassificationTask(EleutherTask, WithAnswerOptionsMixin):
    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAITask]],
        *,
        answer_options: Sequence[str],
        version_override: Optional[str] = None,
        metrics=None,
    ):
        EleutherTask.__init__(
            self,
            eleuther_task,
            version_override=version_override,
            ranked_classification=True,
        )
        WithAnswerOptionsMixin.__init__(self, answer_options)
        self.add_instance_conversion(
            InstanceFormat.RANK_CLASSIFICATION, self.instance_as_rank_classification
        )
        if not metrics:
            self.add_metrics(classification_metrics(len(answer_options)))
        else:
            self.add_metrics(metrics)

    def instance_as_rank_classification(
        self,
        instance: Dict[str, Any],
        *,
        fewshot_instances: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> RankClassificationInstance:
        """
        Converts the given instance to an instance for performing ranked classification

        :param instance: the instance to convert
        :param fewshot_instances: a list of few-shot instances to include. These instances are given in
                                  Huggingface dict format.
        :param kwargs: extra arguments that are ignored
        :return: the instance in :class:`~catwalk.task.RankClassificationInstance` format
        """

        if fewshot_instances is None:
            fewshot_instances = []
        prefix = ""
        for fewshot_instance in fewshot_instances:
            as_rc = self.instance_as_rank_classification(fewshot_instance)
            if as_rc.correct_choice is None:
                raise ValueError(
                    "Could not determine correct choice in ranked classification instance."
                )
            correct_choice = as_rc.choices[as_rc.correct_choice]
            prefix += f"{correct_choice[0].strip()} {correct_choice[1].strip()}\n\n"

        requests = self.instance_as_eleuther_requests(instance, **kwargs)
        choices = [(prefix + r.args[0], r.args[1]) for r in requests]
        assert len(choices) == len(self.answer_options)

        # Reorder the choices so they correspond to self.answer_options.
        # This is important because otherwise doc.label does not match.
        normalized_answer_to_choice = {
            continuation.strip().lower(): (context, continuation)
            for context, continuation in choices
        }
        choices = [
            normalized_answer_to_choice[answer_option.strip().lower()]
            for answer_option in self.answer_options
        ]

        label = self._guess_label(instance)
        assert label < len(choices)
        return RankClassificationInstance(choices, label)


@Task.register("eleuther::race")
class RaceEleutherTask(EleutherTask):
    """The EAI Race task is different because there is no 1:1 correspondence between HF instances and EAI
    instances. EAI chose to follow the GPT3 evaluation approach, which combines multiple questions into one.
    """

    def __init__(self, *, version_override: Optional[str] = None):
        super().__init__("race", version_override=version_override)
        del self.instance_conversions[InstanceFormat.HF_DICT]
        del self.instance_conversions[InstanceFormat.PROMPTSOURCE]
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


@Task.register("eleuther::renamed_splits")
class EleutherTaskWithRenamedSplits(EleutherTask):
    """This task is different because EAI relabels the datasets."""

    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAITask]],
        *,
        version_override: Optional[str] = None,
        ranked_classification: bool = False,
    ):
        super().__init__(
            eleuther_task,
            version_override=version_override,
            ranked_classification=ranked_classification,
        )

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


@Task.register("eleuther::classification_with_renamed_splits")
class EleutherClassificationTaskWithRenamedSplits(
    EleutherTaskWithRenamedSplits, WithAnswerOptionsMixin
):
    def __init__(
        self,
        eleuther_task: Union[str, Callable[[], EAITask]],
        *,
        answer_options: Sequence[str],
        version_override: Optional[str] = None,
    ):
        EleutherTaskWithRenamedSplits.__init__(
            self,
            eleuther_task,
            version_override=version_override,
            ranked_classification=True,
        )
        WithAnswerOptionsMixin.__init__(self, answer_options)
        self.add_instance_conversion(
            InstanceFormat.RANK_CLASSIFICATION, self.instance_as_rank_classification
        )
        self.add_metrics(classification_metrics(len(answer_options)))

    instance_as_rank_classification = (
        EleutherClassificationTask.instance_as_rank_classification
    )


@Task.register("eleuther::mmlu")
class EleutherMMLUTask(EleutherTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def instance_as_rank_classification(
        self,
        instance: Dict[str, Any],
        *,
        fewshot_seed: int = 18830087,
        fewshot_instances: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> RankClassificationInstance:
        """
        Converts the given instance to an instance for performing ranked classification

        :param instance: the instance to convert
        :param fewshot_instances: the number of few-show instances to include
        :return: the instance in :class:`~catwalk.task.RankClassificationInstance` format
        """

        if fewshot_instances is None:
            fewshot_instances = []

        # This is safe to do with MMLU since the few shot examples are the first k examples from
        # the validation set. So there is no randomness
        prefix = self.inner_task.fewshot_context(
            self.instance_as_eleuther_doc(instance),
            num_fewshot=len(fewshot_instances),
            rnd=random.Random(fewshot_seed),
            **kwargs,
        )
        # construct the answer choice requests with the prefix (which includes the task instruction)
        # as the context
        requests = self.inner_task.construct_requests(
            ctx=prefix, doc=self.instance_as_eleuther_doc(instance)
        )
        choices = [(r.args[0], r.args[1]) for r in requests]
        label = self._guess_label(instance)
        assert label < len(choices)
        return RankClassificationInstance(choices, label)

    @property
    def fewshot_instances_split(self) -> str:
        # Official MMLU eval uses the validation split
        return "validation"

    def get_fewshot_instances(
        self, num_shots: int, *args, **kwargs
    ) -> Sequence[Dict[str, Any]]:
        if num_shots <= 0:
            return []
        # Official MMLU eval uses the first k instances
        instances = self.get_split(self.fewshot_instances_split)
        return instances[:num_shots]


def create_mmlu_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {hendrycksTest-abstract_algebra: Task, hendrycksTest-anatomy: Task}
    """

    return {
        f"mmlu_{sub}": create_eleuther_mmlu_task(f"hendrycksTest-{sub}")
        for sub in SUBJECTS
    }


def create_eleuther_mmlu_task(subject):
    return EleutherMMLUTask(subject, ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    )

from abc import ABC
from typing import Iterable, TYPE_CHECKING, Iterator

from tango.common.registrable import Registrable
from tango.common.det_hash import DetHashWithVersion

if TYPE_CHECKING:
    from ludwig.tasks.classification_task import ClassificationTask
    from ludwig.tasks.mc_task import MCTask
    from ludwig.tasks.pair_classification_task import PairClassificationTask
    from ludwig.tasks.qa_task import QATask
    from ludwig.tasks.generation_task import GenerationTask


class ModelForEvaluation(ABC, Registrable, DetHashWithVersion):
    def do_generation(
        self,
        task: 'GenerationTask',
        instances: Iterable['GenerationTask.Instance'],
        **kwargs
    ) -> Iterator['GenerationTask.InstanceResult']:
        raise NotImplementedError

    def do_multiple_choice(
        self,
        task: 'MCTask',
        instances: Iterable['MCTask.Instance'],
        **kwargs
    ) -> Iterator['MCTask.InstanceResult']:
        raise NotImplementedError

    def do_qa(
        self,
        task: 'QATask',
        instances: Iterable['QATask.Instance'],
        **kwargs
    ) -> Iterator['QATask.InstanceResult']:
        raise NotImplementedError

    def do_classification(
        self,
        task: 'ClassificationTask',
        instances: Iterable['ClassificationTask.Instance'],
        **kwargs
    ) -> Iterator['ClassificationTask.InstanceResult']:
        raise NotImplementedError

    def do_pair_classification(
        self,
        task: 'PairClassificationTask',
        instances: Iterable['PairClassificationTask.Instance'],
        **kwargs
    ) -> Iterator['PairClassificationTask.InstanceResult']:
        raise NotImplementedError

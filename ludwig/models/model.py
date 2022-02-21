from typing import Iterable

from tango.common.registrable import Registrable
from tango.common.det_hash import DetHashWithVersion


class ModelForEvaluation(Registrable, DetHashWithVersion):
    def do_summarization(self, task: 'SummarizationTask', **kwargs) -> Iterable['SummarizationTask.InstanceResult']:
        raise NotImplementedError

    def do_multiple_choice(self, task: 'MCTask', **kwargs) -> Iterable['MCTask.InstanceResult']:
        raise NotImplementedError

    def do_qa(self, task: 'QATask', **kwargs) -> Iterable['QATask.InstanceResult']:
        raise NotImplementedError

    def do_classification(self, task: 'ClassificationTask', **kwargs) -> Iterable['ClassificationTask.InstanceResult']:
        raise NotImplementedError

    def do_pair_classification(self, task: 'PairClassificationTask', **kwargs) -> Iterable['PairClassificationTask.InstanceResult']:
        raise NotImplementedError

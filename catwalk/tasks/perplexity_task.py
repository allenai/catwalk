import math
import re
from abc import ABC
from dataclasses import dataclass
from typing import Iterator, Sequence, Iterable

import datasets

from catwalk.models.model import ModelForEvaluation
from catwalk.tasks.task import Task, Metrics


class PerplexityTask(Task, ABC):
    @dataclass
    class Instance(Task.Instance):
        text: str

    @dataclass
    class InstanceResult(Task.InstanceResult):
        logprob: float

    def run_inference(
        self,
        model: ModelForEvaluation,
        instances: Sequence[Instance],
        **kwargs
    ) -> Iterator['PerplexityTask.InstanceResult']:
        return model.do_perplexity(self, instances, **kwargs)

    def calculate_metrics(self, results: Iterable[InstanceResult]) -> Metrics:
        from spacy.lang.en import English
        tokenizer = English().tokenizer

        logprob = 0.0
        characters = 0
        words = 0
        for result in results:
            logprob += result.logprob
            words += len(tokenizer(result.instance.text))
            characters += len(result.instance.text)
        return {
            "word_perplexity": math.exp(-logprob / words),
            "byte_perplexity": math.exp(-logprob / characters),        # bytes aren't characters, but this is what Eleuther calls it
            "bits_per_byte": -(logprob / characters) / math.log(2)
        }


class WikitextTask(PerplexityTask):
    VERSION = "002"

    def __init__(
        self,
        name: str,
        dataset_name: str,
    ):
        super().__init__(name)
        self.dataset_name = dataset_name

    @classmethod
    def wikitext_detokenizer(cls, string: str) -> str:
        # contractions
        string = string.replace("s '", "s'")
        string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
        # number separators
        string = string.replace(" @-@ ", "-")
        string = string.replace(" @,@ ", ",")
        string = string.replace(" @.@ ", ".")
        # punctuation
        string = string.replace(" : ", ": ")
        string = string.replace(" ; ", "; ")
        string = string.replace(" . ", ". ")
        string = string.replace(" ! ", "! ")
        string = string.replace(" ? ", "? ")
        string = string.replace(" , ", ", ")
        # double brackets
        string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
        string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
        string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
        string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
        string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
        # miscellaneous
        string = string.replace("= = = =", "====")
        string = string.replace("= = =", "===")
        string = string.replace("= =", "==")
        string = string.replace(" " + chr(176) + " ", chr(176))
        string = string.replace(" \n", "\n")
        string = string.replace("\n ", "\n")
        string = string.replace(" N ", " 1 ")
        string = string.replace(" 's", "'s")

        return string

    def get_instances(self, split: str) -> Sequence[PerplexityTask.Instance]:
        instances = []
        ret = []
        id = 0
        for id, line in enumerate(datasets.load_dataset("wikitext", self.dataset_name, split=split)):
            # Stolen from Eleuther
            line = line["text"]
            rline = line.replace("= = =", "===").replace("= =", "==").strip()
            if rline.startswith('= ') and rline.strip().endswith(' ='):
                s = '\n'.join(ret)
                if s.strip():
                    instances.append(
                        PerplexityTask.Instance(
                            id=str(id),
                            metadata={},
                            text=self.wikitext_detokenizer(s)))
                ret = []
            ret.append(line)
        instances.append(
            PerplexityTask.Instance(
                id=str(id),
                metadata={},
                text=self.wikitext_detokenizer('\n'.join(ret))))
        return instances
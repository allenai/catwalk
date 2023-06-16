"""
TLDR: Extreme Summarization of Scientific Documents
https://arxiv.org/abs/2004.15011

The SciTLDR dataset is created specifically to study the task of TLDR (Too Long; Didn't Read) 
generation for scientific papers. TLDR generation is a form of extreme summarization that requires 
high source compression, expertise, and a thorough understanding of domain-specific language.

We use the Abstract only setting of TLDR (where the input is only the abstract of the paper)

Homepage: https://github.com/allenai/scitldr
"""
import re
from itertools import islice

import numpy as np
from scipy.optimize import linear_sum_assignment
import string
from catwalk.dependencies.lm_eval.base import Task, rf
from catwalk.dependencies.lm_eval.metrics import mean
from datasets import load_metric

_CITATION = """
@article{cachola2020tldr,
  title={{TLDR}: Extreme Summarization of Scientific Documents},
  author={Isabel Cachola and Kyle Lo and Arman Cohan and Daniel S. Weld},
  journal={arXiv:2004.15011},
  year={2020},
}
"""

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


class SciTLDR(Task):
    VERSION = 0
    DATASET_PATH = "scitldr"
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        self.bertscore = load_metric("bertscore")
        self.bertscore_model_type = "microsoft/deberta-xlarge-mnli"
        self.rouge = load_metric("rouge")
        self.metric_keys = {} # will be populated by get_metrics and used for aggregation

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        # Cache training for faster few-shot.
        # Data is too large to fit in memory.
        if self._training_docs is None:
            self._training_docs = list(self.dataset["train"])
        return self._training_docs

    def validation_docs(self):
        return self.dataset["validation"]

    def fewshot_examples(self, k, rnd):
        # Data is too large to fit in memory. We just sample from the first bit.
        if self._training_docs is None:
            self._training_docs = list(islice(self.training_docs(), 0, 100000))

        return rnd.sample(self._training_docs, k)

    def doc_to_text(self, doc):
        source_sentences = doc["source"]
        source = " ".join(source_sentences)
        return f"Article: {source}\nTLDR:"

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return " ".join(doc["source"])

    def doc_to_target(self, doc):
        # we take the first target. this is used for fewshot
        summary = doc["target"][0]
        return summary

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        conts = [rf.greedy_until(ctx, ["\n"])]
        return conts

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # minimal cleaning
        results = [res.replace("\n", " ").replace("\t", " ").replace("\r", " ").strip() for res in results]
        preds, golds = results, doc["target"]
        results = self.get_metrics(preds, golds)
        return results

    def get_metrics(self, predicted, gold_summaries):
        """ """
        # each summary can have multiple golds
        # rouge_score of huggingface doesn't support passing multiple summaries
        # we split it into two instances
        predictions = []
        for _ in gold_summaries:
            predictions.append(predicted)

        rouge_results = self.rouge.compute(
            predictions=predictions, references=gold_summaries, use_stemmer=True, use_aggregator=False,
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"]
        )
        for key, value in rouge_results.items():
            rouge_results[key] = {
                "precision": [score.precision * 100 for score in value],
                "recall": [score.recall * 100 for score in value],
                "fmeasure": [score.fmeasure * 100 for score in value],
                "precision_mean": np.mean([score.precision for score in value]) * 100,
                "precision_max": np.max([score.precision for score in value]) * 100,
                "recall_mean": np.mean([score.recall for score in value]) * 100,
                "recall_max": np.max([score.recall for score in value]) * 100,
                "fmeasure_mean": np.mean([score.fmeasure for score in value]) * 100,
                "fmeasure_max": np.max([score.fmeasure for score in value]) * 100,
            }

        bert_score_results = self.bertscore.compute(
            predictions=predictions,
            references=gold_summaries,
            # These are mostly based on the recommendations in https://github.com/Tiiiger/bert_score
            model_type=self.bertscore_model_type,
            lang="en",
            rescale_with_baseline=True,
            use_fast_tokenizer=True,
        )
        bert_score_results["f1_mean"] = np.mean(bert_score_results["f1"])
        bert_score_results["f1_max"] = np.max(bert_score_results["f1"])

        results = {
            "rouge1_fmeasure_mean": rouge_results["rouge1"]["fmeasure_mean"],
            "rouge1_fmeasure_max": rouge_results["rouge1"]["fmeasure_max"],
            "rouge2_fmeasure_mean": rouge_results["rouge2"]["fmeasure_mean"],
            "rouge2_fmeasure_max": rouge_results["rouge2"]["fmeasure_max"],
            "rougeL_fmeasure_mean": rouge_results["rougeL"]["fmeasure_mean"],
            "rougeL_fmeasure_max": rouge_results["rougeL"]["fmeasure_max"],
            "rougeLsum_fmeasure_mean": rouge_results["rougeLsum"]["fmeasure_mean"],
            "rougeLsum_fmeasure_max": rouge_results["rougeLsum"]["fmeasure_max"],
            "bertscore_f1_mean": bert_score_results["f1_mean"],
            "bertscore_f1_max": bert_score_results["f1_max"],
        }
        self.metric_keys = list(results.keys())
        return results

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {metric_key: mean for metric_key in self.metric_keys}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {metric_key: True for metric_key in self.metric_keys}

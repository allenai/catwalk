from typing import Any, Dict, Optional, Sequence, Union

import datasets
from random import Random
from torchmetrics import MeanMetric

from catwalk.task import InstanceFormat, ENTAILMENT_METRICS, QA_METRICS, Task, \
    classification_metrics, BINARY_CLASSIFICATION_METRICS, mc_metrics, rc_metrics, ppl_metrics, PERPLEXITY_METRICS
from catwalk.tasks.eleuther import EleutherTask, RaceEleutherTask, EleutherTaskWithRenamedSplits, \
    EleutherClassificationTask, EleutherClassificationTaskWithRenamedSplits
from catwalk.tasks.perplexity_jsonl import PerplexityJsonLTask
from catwalk.tasks.huggingface import hfmc_conversion, HFDatasetsTask, hfqa_conversion, hfclassification_conversion
from catwalk.tasks.p3 import P3Task
from catwalk.tasks.raft import RaftTask
from catwalk.tasks.metaicl import MetaICLTask
from catwalk.tasks.mrqa import MrqaTask
from catwalk.tasks.t5 import t5_prompt_conversion

# Version of __init__.py defining TASKS_LM specifically for task variants geared towards LM type models
# Usually will use TASKS from __init__.py as fallback

TASKS_LM: Dict[str, Task] = {
    "squad2": EleutherTask("squad2", eleuther_metrics=True),
    "drop": EleutherTask("drop", eleuther_metrics=True),
    "ppl_custom": PerplexityJsonLTask().add_metrics(ppl_metrics(primary="ppl_token")),
    "wikitext": EleutherTask("wikitext").add_metrics(ppl_metrics(primary="ppl_token")),
    "piqa": EleutherTask("piqa", ranked_classification=True).add_metrics(rc_metrics(primary="acc_per_token")),
    "mrpc": EleutherClassificationTask("mrpc", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")),
    "qnli": EleutherClassificationTask("qnli", answer_options=["yes", "no"], metrics=rc_metrics(primary="acc_raw")),
    "qqp": EleutherClassificationTask("qqp", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")),
    "sst": EleutherClassificationTask("sst", answer_options=["negative", "positive"], metrics=rc_metrics(primary="acc_raw")),
    "rte": EleutherClassificationTask("rte", answer_options=["True", "False"], metrics=rc_metrics(primary="acc_raw")),
    "wnli": EleutherTask("wnli", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    "boolq": EleutherTask("boolq", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    "copa": EleutherTask("copa", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    "wic": EleutherTask("wic", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    "wsc": EleutherTask(
        "wsc",
        ranked_classification=True,
        promptsource_task_spec=('super_glue', 'wsc.fixed')
    ).add_metrics(rc_metrics(primary="acc_raw")),
    # "drop": EleutherTask("drop").add_metrics(QA_METRICS),
    "naturalqs_short_open": EleutherTask("naturalqs_short_open", eleuther_metrics=True),
    # "lambada": EleutherTask("lambada_standard").add_metrics(PERPLEXITY_METRICS).add_metric("acc", MeanMetric),
    # "pubmedqa": EleutherTaskWithRenamedSplits("pubmedqa").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "sciq": EleutherTask("sciq", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    #"triviaqa": EleutherTask(
    #    "triviaqa",
    #    promptsource_task_spec=("trivia_qa", "unfiltered")
    #).add_metrics(QA_METRICS),
    "arc_easy": EleutherTask("arc_easy", ranked_classification=True).add_metrics(rc_metrics(primary="acc_uncond")),
    "arc_easy:mc": EleutherTask("arc_easy:mc", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    "arc_challenge": EleutherTask("arc_challenge", ranked_classification=True).add_metrics(rc_metrics(primary="acc_uncond")),
    "arc_challenge:mc": EleutherTask("arc_challenge:mc", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    # For logiqa the answer choices are shown, but full answer string, so trying acc_raw here
    "logiqa": EleutherTask("logiqa", ranked_classification=True).add_metrics(rc_metrics(primary="acc_raw")),
    "hellaswag": EleutherTask("hellaswag", ranked_classification=True).add_metrics(rc_metrics(primary="acc_per_token")),
    "openbookqa": EleutherTask("openbookqa", ranked_classification=True).add_metrics(rc_metrics(primary="acc_uncond")),
    "headqa_en": EleutherTask("headqa_en", ranked_classification=True)  .add_metrics(rc_metrics(primary="acc_uncond")),
    "mathqa": EleutherTask("mathqa", ranked_classification=True).add_metrics(rc_metrics(primary="acc_per_token")),
    #"wsc273": EleutherTask("wsc273", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "winogrande": EleutherTask("winogrande", ranked_classification=True).add_metrics(rc_metrics(primary="acc_per_token")),
}
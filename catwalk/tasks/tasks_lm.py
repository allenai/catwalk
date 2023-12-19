from random import Random
from typing import Any, Dict, Optional, Sequence, Union

import datasets
from torchmetrics import MeanMetric

from catwalk.task import (
    BINARY_CLASSIFICATION_METRICS,
    ENTAILMENT_METRICS,
    PERPLEXITY_METRICS,
    QA_METRICS,
    InstanceFormat,
    Task,
    classification_metrics,
    mc_metrics,
    ppl_metrics,
    rc_metrics,
)
from catwalk.tasks.eleuther import (
    EleutherClassificationTask,
    EleutherClassificationTaskWithRenamedSplits,
    EleutherTask,
    EleutherTaskWithRenamedSplits,
    RaceEleutherTask,
    create_mmlu_tasks,
)
from catwalk.tasks.huggingface import (
    HFDatasetsTask,
    hfclassification_conversion,
    hfmc_conversion,
    hfqa_conversion,
)
from catwalk.tasks.metaicl import MetaICLTask
from catwalk.tasks.mrqa import MrqaTask
from catwalk.tasks.p3 import P3Task
from catwalk.tasks.perplexity_jsonl import PerplexityJsonLTask
from catwalk.tasks.raft import RaftTask
from catwalk.tasks.t5 import t5_prompt_conversion

# Version of __init__.py defining TASKS_LM specifically for task variants geared towards LM type models
# Usually will use TASKS from __init__.py as fallback

TASKS_LM: Dict[str, Task] = {
    "squad2": EleutherTask("squad2", eleuther_metrics=True),
    "drop": EleutherTask(
        "drop", eleuther_metrics=True, model_args={"max_gen_toks": 50}
    ),
    "ppl_custom": PerplexityJsonLTask().add_metrics(ppl_metrics(primary="ppl_token")),
    "wikitext": EleutherTask("wikitext").add_metrics(ppl_metrics(primary="ppl_token")),
    "piqa": EleutherTask("piqa", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_per_token")
    ),
    "mrpc": EleutherClassificationTask(
        "mrpc", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")
    ),
    "qnli": EleutherClassificationTask(
        "qnli", answer_options=["yes", "no"], metrics=rc_metrics(primary="acc_raw")
    ),
    "qqp": EleutherClassificationTask(
        "qqp", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")
    ),
    "sst": EleutherClassificationTask(
        "sst",
        answer_options=["negative", "positive"],
        metrics=rc_metrics(primary="acc_raw"),
    ),
    "rte": EleutherClassificationTask(
        "rte", answer_options=["True", "False"], metrics=rc_metrics(primary="acc_raw")
    ),
    "wnli": EleutherClassificationTask(
        "wnli", answer_options=["False", "True"], metrics=rc_metrics(primary="acc_raw")
    ),
    "boolq": EleutherClassificationTask(
        "boolq", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")
    ),
    "copa": EleutherTask("copa", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    "wic": EleutherClassificationTask(
        "wic", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")
    ),
    "wsc": EleutherClassificationTask(
        "wsc", answer_options=["no", "yes"], metrics=rc_metrics(primary="acc_raw")
    ),
    # "drop": EleutherTask("drop").add_metrics(QA_METRICS),
    "naturalqs_short_open": EleutherTask(
        "naturalqs_short_open", eleuther_metrics=True, model_args={"max_gen_toks": 50}
    ),
    "scitldr": EleutherTask(
        "scitldr", eleuther_metrics=True, model_args={"max_gen_toks": 150}
    ),
    "xsum": EleutherTask(
        "xsum", eleuther_metrics=True, model_args={"max_gen_toks": 150}
    ),
    # hendrycksTest (MMLU) (57 tasks)
    **create_mmlu_tasks(),
    # "lambada": EleutherTask("lambada_standard").add_metrics(PERPLEXITY_METRICS).add_metric("acc", MeanMetric),
    # "pubmedqa": EleutherTaskWithRenamedSplits("pubmedqa").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "sciq": EleutherTask("sciq", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    # "triviaqa": EleutherTask(
    #    "triviaqa",
    #    promptsource_task_spec=("trivia_qa", "unfiltered")
    # ).add_metrics(QA_METRICS),
    "arc_easy": EleutherTask("arc_easy", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_uncond")
    ),
    "arc_easy:mc": EleutherTask("arc_easy:mc", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    "arc_challenge": EleutherTask(
        "arc_challenge", ranked_classification=True
    ).add_metrics(rc_metrics(primary="acc_uncond")),
    "arc_challenge:mc": EleutherTask(
        "arc_challenge:mc", ranked_classification=True
    ).add_metrics(rc_metrics(primary="acc_raw")),
    "eurlex": EleutherTask(
        "eurlex", eleuther_metrics=True, model_args={"max_gen_toks": 200}
    ),
    "unfair_tos": EleutherTask(
        "unfair_tos", eleuther_metrics=True, model_args={"max_gen_toks": 50}
    ),
    "case_hold:mc": EleutherTask(
        "case_hold:mc", ranked_classification=True
    ).add_metrics(rc_metrics(primary="acc_raw")),
    # For logiqa the answer choices are shown, but full answer string, so trying acc_raw here
    "logiqa": EleutherTask("logiqa", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_raw")
    ),
    "hellaswag": EleutherTask("hellaswag", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_per_token")
    ),
    "openbookqa": EleutherTask("openbookqa", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_uncond")
    ),
    "headqa_en": EleutherTask("headqa_en", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_uncond")
    ),
    "mathqa": EleutherTask("mathqa", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_per_token")
    ),
    # "wsc273": EleutherTask("wsc273", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "winogrande": EleutherTask("winogrande", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_per_token")
    ),
    "social_iqa": EleutherTask("social_iqa", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_uncond")
    ),
    "csqa": EleutherTask("csqa", ranked_classification=True).add_metrics(
        rc_metrics(primary="acc_uncond")
    ),
}

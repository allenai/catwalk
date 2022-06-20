from typing import Dict

import datasets

from catwalk.task import MC_METRICS, InstanceFormat, ENTAILMENT_METRICS, QA_METRICS, Task, \
    classification_metrics, BINARY_CLASSIFICATION_METRICS
from catwalk.tasks.eleuther import EleutherTask, RaceEleutherTask, PubmedqaEleutherTask
from catwalk.tasks.huggingface import hfmc_conversion, HFDatasetsTask
from catwalk.tasks.p3 import P3Task
from catwalk.tasks.raft import RaftTask
from catwalk.tasks.t5 import t5_prompt_conversion

TASKS: Dict[str, Task] = {
    "wikitext": EleutherTask("wikitext"),
    "piqa": EleutherTask("piqa", ranked_classification=True).add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="goal",
            answer_choices_fields=["sol1", "sol2"],
            correct_answer_index_field="label"
        )
    ).add_metrics(MC_METRICS),
    "squad2": EleutherTask("squad2").add_metrics(QA_METRICS),
    "rte": EleutherTask("rte", ranked_classification=True).add_instance_conversion(
        InstanceFormat.T5_PROMPT,
        t5_prompt_conversion(
            task_name="rte",
            label_map={0: "entailment", 1: "not_entailment"},
            use_fields=["sentence1", "sentence2"]
        )
    ).add_metrics(ENTAILMENT_METRICS),
    "superglue::rte": HFDatasetsTask("super_glue", "rte").add_instance_conversion(
        InstanceFormat.T5_PROMPT,
        t5_prompt_conversion(
            task_name="rte",
            label_map={0: "entailment", 1: "not_entailment"},
            use_fields=["premise", "hypothesis"]
        )
    ).add_metrics(ENTAILMENT_METRICS),
    "cola": EleutherTask("cola", ranked_classification=True).add_metrics(classification_metrics(2)),
    "mnli": EleutherTask("mnli", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "mnli_mismatched": EleutherTask("mnli_mismatched", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "mrpc": EleutherTask("mrpc", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "qnli": EleutherTask("qnli", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "qqp": EleutherTask("qqp", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "sst": EleutherTask("sst", ranked_classification=True).add_metrics(classification_metrics(5)),
    "wnli": EleutherTask("wnli", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "boolq": EleutherTask("boolq", ranked_classification=True).add_metrics(classification_metrics(2)),
    "cb": EleutherTask("cb", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "copa": EleutherTask("copa", ranked_classification=True).add_metrics(MC_METRICS),
    "multirc": EleutherTask("multirc", ranked_classification=True).add_metrics(MC_METRICS),
    #"record": EleutherTask("record"),    # record doesn't have a 1:1 correspondence between HF instances and EAI instances
    "wic": EleutherTask("wic", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "wsc": EleutherTask("wsc", ranked_classification=True).add_metrics(MC_METRICS),
    #"coqa": EleutherTask("coqa"),  # currently broken in the datasets library
    "drop": EleutherTask("drop").add_metrics(QA_METRICS),
    "lambada": EleutherTask("lambada"),
    "lambada_cloze": EleutherTask("lambada_cloze"),
    "lambada_mt_en": EleutherTask("lambada_mt_en"),
    "lambada_mt_fr": EleutherTask("lambada_mt_fr"),
    "lambada_mt_de": EleutherTask("lambada_mt_de"),
    "lambada_mt_it": EleutherTask("lambada_mt_it"),
    "lambada_mt_es": EleutherTask("lambada_mt_es"),
    "prost": EleutherTask("prost", ranked_classification=True).add_metrics(MC_METRICS),
    "mc_taco": EleutherTask("mc_taco", ranked_classification=True).add_metrics(BINARY_CLASSIFICATION_METRICS),
    "pubmedqa": PubmedqaEleutherTask().add_metrics(BINARY_CLASSIFICATION_METRICS),
    "sciq": EleutherTask("sciq", ranked_classification=True).add_metrics(MC_METRICS),
    "qa4mre_2011": EleutherTask("qa4mre_2011", ranked_classification=True).add_metrics(MC_METRICS),
    "qa4mre_2012": EleutherTask("qa4mre_2012", ranked_classification=True).add_metrics(MC_METRICS),
    "qa4mre_2013": EleutherTask("qa4mre_2013", ranked_classification=True).add_metrics(MC_METRICS),
    "triviaqa": EleutherTask("triviaqa").add_metrics(QA_METRICS),
    "arc_easy": EleutherTask("arc_easy", ranked_classification=True).add_metrics(MC_METRICS),
    "arc_challenge": EleutherTask("arc_challenge", ranked_classification=True).add_metrics(MC_METRICS),
    "logiqa": EleutherTask("logiqa", ranked_classification=True).add_metrics(MC_METRICS),
    "hellaswag": EleutherTask("hellaswag", ranked_classification=True).add_metrics(MC_METRICS),
    "openbookqa": EleutherTask("openbookqa", ranked_classification=True).add_metrics(MC_METRICS),
    "race": RaceEleutherTask().add_metrics(MC_METRICS),
    "headqa": EleutherTask("headqa", ranked_classification=True).add_metrics(MC_METRICS),
    "headqa_es": EleutherTask("headqa_es", ranked_classification=True).add_metrics(MC_METRICS),
    "headqa_en": EleutherTask("headqa_en", ranked_classification=True).add_metrics(MC_METRICS),
    "mathqa": EleutherTask("mathqa", ranked_classification=True).add_metrics(MC_METRICS),
    "webqs": EleutherTask("webqs").add_metrics(QA_METRICS),
    "wsc273": EleutherTask("wsc273", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "winogrande": EleutherTask("winogrande", ranked_classification=True).add_metrics(MC_METRICS),
    "anli_r1": EleutherTask("anli_r1", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "anli_r2": EleutherTask("anli_r2", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "anli_r3": EleutherTask("anli_r3", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "ethics_cm": EleutherTask("ethics_cm").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_deontology": EleutherTask("ethics_deontology").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_justice": EleutherTask("ethics_justice").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_utilitarianism_original": EleutherTask("ethics_utilitarianism_original").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_utilitarianism": EleutherTask("ethics_utilitarianism").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_virtue": EleutherTask("ethics_virtue").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "truthfulqa_mc": EleutherTask("truthfulqa_mc", ranked_classification=True),
    "truthfulqa_gen": EleutherTask("truthfulqa_gen"),
    "mutual": EleutherTask("mutual"),
    "mutual_plus": EleutherTask("mutual_plus"),
    "math_algebra": EleutherTask("math_algebra").add_metrics(QA_METRICS),
    "math_counting_and_prob": EleutherTask("math_counting_and_prob").add_metrics(QA_METRICS),
    "math_geometry": EleutherTask("math_geometry").add_metrics(QA_METRICS),
    "math_intermediate_algebra": EleutherTask("math_intermediate_algebra").add_metrics(QA_METRICS),
    "math_num_theory": EleutherTask("math_num_theory").add_metrics(QA_METRICS),
    "math_prealgebra": EleutherTask("math_prealgebra").add_metrics(QA_METRICS),
    "math_precalc": EleutherTask("math_precalc").add_metrics(QA_METRICS),
    "math_asdiv": EleutherTask("math_asdiv").add_metrics(QA_METRICS),
    "arithmetic_2da": EleutherTask("arithmetic_2da").add_metrics(QA_METRICS),
    "arithmetic_2ds": EleutherTask("arithmetic_2ds").add_metrics(QA_METRICS),
    "arithmetic_3da": EleutherTask("arithmetic_3da").add_metrics(QA_METRICS),
    "arithmetic_3ds": EleutherTask("arithmetic_3ds").add_metrics(QA_METRICS),
    "arithmetic_4da": EleutherTask("arithmetic_4da").add_metrics(QA_METRICS),
    "arithmetic_4ds": EleutherTask("arithmetic_4ds").add_metrics(QA_METRICS),
    "arithmetic_5da": EleutherTask("arithmetic_5da").add_metrics(QA_METRICS),
    "arithmetic_5ds": EleutherTask("arithmetic_5ds").add_metrics(QA_METRICS),
    "arithmetic_2dm": EleutherTask("arithmetic_2dm").add_metrics(QA_METRICS),
    "arithmetic_1dc": EleutherTask("arithmetic_1dc").add_metrics(QA_METRICS),
    #"iwslt17-en-ar": EleutherTask("iwslt17-en-ar"),    # no support for translations tasks for now
    #"iwslt17-ar-en": EleutherTask("iwslt17-ar-en"),    # no support for translations tasks for now
    "anagrams1": EleutherTask("anagrams1").add_metrics(QA_METRICS),
    "anagrams2": EleutherTask("anagrams2").add_metrics(QA_METRICS),
    "cycle_letters": EleutherTask("cycle_letters").add_metrics(QA_METRICS),
    "random_insertion": EleutherTask("random_insertion").add_metrics(QA_METRICS),
    "reversed_words": EleutherTask("reversed_words").add_metrics(QA_METRICS),
    # RAFT
    "raft::ade_corpus_v2": RaftTask("ade_corpus_v2"),
    "raft::banking_77": RaftTask("banking_77", 77),
    "raft::neurips_impact_statement_risks": RaftTask("neurips_impact_statement_risks"),
    "raft::one_stop_english": RaftTask("one_stop_english", 3),
    "raft::overruling": RaftTask("overruling"),
    "raft::semiconductor_org_types": RaftTask("semiconductor_org_types", 3),
    "raft::systematic_review_inclusion": RaftTask("systematic_review_inclusion"),
    "raft::tai_safety_research": RaftTask("tai_safety_research"),
    "raft::terms_of_service": RaftTask("terms_of_service"),
    "raft::tweet_eval_hate": RaftTask("tweet_eval_hate"),
    "raft::twitter_complaints": RaftTask("twitter_complaints"),
}

for config in datasets.get_dataset_config_names("bigscience/P3"):
    TASKS[f"p3::{config}"] = P3Task(config)

TASK_SETS = {
    "iz": {
        "arc_challenge",
        "arc_easy",
        "boolq",
        "copa",
        #"headqa_en",       # Headqa is broken as of 2022-05-05
        "hellaswag",
        "lambada",
        "logiqa",
        "mathqa",
        "mc_taco",
        "mrpc",
        "multirc",
        "openbookqa",
        "piqa",
        "prost",
        "pubmedqa",
        "qnli",
        "qqp",
        "race",
        "rte",
        "sciq",
        "sst",
        "triviaqa",
        "webqs",
        "wic",
        "winogrande",
        "wnli",
        "wsc",
    },
    "raft": {name for name in TASKS.keys() if name.startswith("raft::")}
}
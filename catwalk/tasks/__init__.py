from typing import Dict

from catwalk.task import MC_METRICS, InstanceFormat, ENTAILMENT_METRICS, CLASSIFICATION_METRICS, QA_METRICS, Task
from catwalk.tasks.eleuther import EleutherTask, RaceEleutherTask, PubmedqaEleutherTask
from catwalk.tasks.huggingface import hfmc_conversion, HFDatasetsTask
from catwalk.tasks.t5 import t5_prompt_conversion

TASKS: Dict[str, Task] = {
    "wikitext": EleutherTask("wikitext"),
    "piqa": EleutherTask("piqa").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="goal",
            answer_choices_fields=["sol1", "sol2"],
            correct_answer_index_field="label"
        )
    ).add_metrics(MC_METRICS),
    "squad2": EleutherTask("squad2").add_metrics(QA_METRICS),
    "rte": EleutherTask("rte").add_instance_conversion(
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
    "cola": EleutherTask("cola").add_metrics(CLASSIFICATION_METRICS),
    "mnli": EleutherTask("mnli").add_metrics(ENTAILMENT_METRICS),
    "mnli_mismatched": EleutherTask("mnli_mismatched").add_metrics(ENTAILMENT_METRICS),
    "mrpc": EleutherTask("mrpc").add_metrics(ENTAILMENT_METRICS),
    "qnli": EleutherTask("qnli").add_metrics(ENTAILMENT_METRICS),
    "qqp": EleutherTask("qqp").add_metrics(ENTAILMENT_METRICS),
    "sst": EleutherTask("sst").add_metrics(CLASSIFICATION_METRICS),
    "wnli": EleutherTask("wnli").add_metrics(ENTAILMENT_METRICS),
    "boolq": EleutherTask("boolq").add_metrics(CLASSIFICATION_METRICS),
    "cb": EleutherTask("cb").add_metrics(ENTAILMENT_METRICS),
    "copa": EleutherTask("copa").add_metrics(CLASSIFICATION_METRICS),
    "multirc": EleutherTask("multirc").add_metrics(CLASSIFICATION_METRICS),
    #"record": EleutherTask("record"),    # record doesn't have a 1:1 correspondence between HF instances and EAI instances
    "wic": EleutherTask("wic").add_metrics(ENTAILMENT_METRICS),
    "wsc": EleutherTask("wsc").add_metrics(MC_METRICS),
    #"coqa": EleutherTask("coqa"),  # currently broken in the datasets library
    "drop": EleutherTask("drop").add_metrics(QA_METRICS),
    "lambada": EleutherTask("lambada"),
    "lambada_cloze": EleutherTask("lambada_cloze"),
    "lambada_mt_en": EleutherTask("lambada_mt_en"),
    "lambada_mt_fr": EleutherTask("lambada_mt_fr"),
    "lambada_mt_de": EleutherTask("lambada_mt_de"),
    "lambada_mt_it": EleutherTask("lambada_mt_it"),
    "lambada_mt_es": EleutherTask("lambada_mt_es"),
    "prost": EleutherTask("prost").add_metrics(MC_METRICS),
    "mc_taco": EleutherTask("mc_taco").add_metrics(CLASSIFICATION_METRICS),
    "pubmedqa": PubmedqaEleutherTask().add_metrics(CLASSIFICATION_METRICS),
    "sciq": EleutherTask("sciq").add_metrics(MC_METRICS),
    "qa4mre_2011": EleutherTask("qa4mre_2011").add_metrics(MC_METRICS),
    "qa4mre_2012": EleutherTask("qa4mre_2012").add_metrics(MC_METRICS),
    "qa4mre_2013": EleutherTask("qa4mre_2013").add_metrics(MC_METRICS),
    "triviaqa": EleutherTask("triviaqa").add_metrics(QA_METRICS),
    "arc_easy": EleutherTask("arc_easy").add_metrics(MC_METRICS),
    "arc_challenge": EleutherTask("arc_challenge").add_metrics(MC_METRICS),
    "logiqa": EleutherTask("logiqa").add_metrics(MC_METRICS),
    "hellaswag": EleutherTask("hellaswag").add_metrics(MC_METRICS),
    "openbookqa": EleutherTask("openbookqa").add_metrics(MC_METRICS),
    "race": RaceEleutherTask().add_metrics(MC_METRICS),
    "headqa": EleutherTask("headqa").add_metrics(MC_METRICS),
    "headqa_es": EleutherTask("headqa_es").add_metrics(MC_METRICS),
    "headqa_en": EleutherTask("headqa_en").add_metrics(MC_METRICS),
    "mathqa": EleutherTask("mathqa").add_metrics(MC_METRICS),
    "webqs": EleutherTask("webqs").add_metrics(QA_METRICS),
    "wsc273": EleutherTask("wsc273").add_metrics(ENTAILMENT_METRICS),
    "winogrande": EleutherTask("winogrande").add_metrics(MC_METRICS),
    "anli_r1": EleutherTask("anli_r1").add_metrics(ENTAILMENT_METRICS),
    "anli_r2": EleutherTask("anli_r2").add_metrics(ENTAILMENT_METRICS),
    "anli_r3": EleutherTask("anli_r3").add_metrics(ENTAILMENT_METRICS),
    "ethics_cm": EleutherTask("ethics_cm").add_metrics(CLASSIFICATION_METRICS),
    "ethics_deontology": EleutherTask("ethics_deontology").add_metrics(CLASSIFICATION_METRICS),
    "ethics_justice": EleutherTask("ethics_justice").add_metrics(CLASSIFICATION_METRICS),
    "ethics_utilitarianism_original": EleutherTask("ethics_utilitarianism_original").add_metrics(CLASSIFICATION_METRICS),
    "ethics_utilitarianism": EleutherTask("ethics_utilitarianism").add_metrics(CLASSIFICATION_METRICS),
    "ethics_virtue": EleutherTask("ethics_virtue").add_metrics(CLASSIFICATION_METRICS),
    "truthfulqa_mc": EleutherTask("truthfulqa_mc"),
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
}

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
    }
}
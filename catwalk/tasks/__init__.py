from typing import Dict

from catwalk.task import MC_METRICS, InstanceFormat, ENTAILMENT_METRICS, CLASSIFICATION_METRICS, QA_METRICS, Task
from catwalk.tasks.eleuther import EleutherTask
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
    "cola": EleutherTask("cola"),
    "mnli": EleutherTask("mnli").add_metrics(ENTAILMENT_METRICS),
    "mnli_mismatched": EleutherTask("mnli_mismatched").add_metrics(ENTAILMENT_METRICS),
    "mrpc": EleutherTask("mrpc").add_metrics(ENTAILMENT_METRICS),
    "qnli": EleutherTask("qnli").add_metrics(ENTAILMENT_METRICS),
    "qqp": EleutherTask("qqp").add_metrics(ENTAILMENT_METRICS),
    "sst": EleutherTask("sst").add_metrics(CLASSIFICATION_METRICS),
    "wnli": EleutherTask("wnli").add_metrics(ENTAILMENT_METRICS),
    "boolq": EleutherTask("boolq").add_metrics(CLASSIFICATION_METRICS),
    "cb": EleutherTask("cb"),
    "copa": EleutherTask("copa").add_metrics(CLASSIFICATION_METRICS),
    "multirc": EleutherTask("multirc").add_metrics(CLASSIFICATION_METRICS),
    "record": EleutherTask("record"),
    "wic": EleutherTask("wic").add_metrics(ENTAILMENT_METRICS),
    "wsc": EleutherTask("wsc").add_metrics(MC_METRICS),
    #"coqa": EleutherTask("coqa"),    # not an HF Task
    #"drop": EleutherTask("drop"),    # not an HF Task
    # Lambada is not an HF Task
    #"lambada": EleutherTask("lambada"),
    #"lambada_cloze": EleutherTask("lambada_cloze"),
    #"lambada_mt_en": EleutherTask("lambada_mt_en"),
    #"lambada_mt_fr": EleutherTask("lambada_mt_fr"),
    #"lambada_mt_de": EleutherTask("lambada_mt_de"),
    #"lambada_mt_it": EleutherTask("lambada_mt_it"),
    #"lambada_mt_es": EleutherTask("lambada_mt_es"),
    "prost": EleutherTask("prost").add_metrics(MC_METRICS),
    "mc_taco": EleutherTask("mc_taco").add_metrics(CLASSIFICATION_METRICS),
    #"pubmedqa": EleutherTask("pubmedqa"),  # out of date in the hardcoded datasets version, need to wait for an update
    #"sciq": EleutherTask("sciq"),    # not an HF Task
    #"qa4mre_2011": EleutherTask("qa4mre_2011"),    # not an HF Task
    #"qa4mre_2012": EleutherTask("qa4mre_2012"),    # not an HF Task
    #"qa4mre_2013": EleutherTask("qa4mre_2013"),    # not an HF Task
    #"triviaqa": EleutherTask("triviaqa"),    # not an HF Task
    "arc_easy": EleutherTask("arc_easy").add_metrics(MC_METRICS),
    "arc_challenge": EleutherTask("arc_challenge").add_metrics(MC_METRICS),
    #"logiqa": EleutherTask("logiqa"),        # not an HF Task
    "hellaswag": EleutherTask("hellaswag").add_metrics(MC_METRICS),
    "openbookqa": EleutherTask("openbookqa").add_metrics(MC_METRICS),
    "race": EleutherTask("race").add_metrics(MC_METRICS),
    "headqa": EleutherTask("headqa").add_metrics(MC_METRICS),
    "headqa_es": EleutherTask("headqa_es").add_metrics(MC_METRICS),
    "headqa_en": EleutherTask("headqa_en").add_metrics(MC_METRICS),
    "mathqa": EleutherTask("mathqa").add_metrics(MC_METRICS),
    "webqs": EleutherTask("webqs").add_metrics(QA_METRICS),
    "wsc273": EleutherTask("wsc273"),
    "winogrande": EleutherTask("winogrande").add_metrics(MC_METRICS),
    "anli_r1": EleutherTask("anli_r1").add_metrics(ENTAILMENT_METRICS),
    "anli_r2": EleutherTask("anli_r2").add_metrics(ENTAILMENT_METRICS),
    "anli_r3": EleutherTask("anli_r3").add_metrics(ENTAILMENT_METRICS),
    #"ethics_cm": EleutherTask("ethics_cm"),    # not an HF Task
    #"ethics_deontology": EleutherTask("ethics_deontology"),    # not an HF Task
    #"ethics_justice": EleutherTask("ethics_justice"),    # not an HF Task
    #"ethics_utilitarianism_original": EleutherTask("ethics_utilitarianism_original"),    # not an HF Task
    #"ethics_utilitarianism": EleutherTask("ethics_utilitarianism"),    # not an HF Task
    #"ethics_virtue": EleutherTask("ethics_virtue"),    # not an HF Task
    #"truthfulqa_mc": EleutherTask("truthfulqa_mc"),    # not an HF Task
    #"truthfulqa_gen": EleutherTask("truthfulqa_gen"),    # not an HF Task
    #"mutual": EleutherTask("mutual"),    # not an HF Task
    #"mutual_plus": EleutherTask("mutual_plus"),    # not an HF Task
    #"math_algebra": EleutherTask("math_algebra"),    # not an HF Task
    #"math_counting_and_prob": EleutherTask("math_counting_and_prob"),    # not an HF Task
    #"math_geometry": EleutherTask("math_geometry"),    # not an HF Task
    #"math_intermediate_algebra": EleutherTask("math_intermediate_algebra"),    # not an HF Task
    #"math_num_theory": EleutherTask("math_num_theory"),    # not an HF Task
    #"math_prealgebra": EleutherTask("math_prealgebra"),    # not an HF Task
    #"math_precalc": EleutherTask("math_precalc"),    # not an HF Task
    #"math_asdiv": EleutherTask("math_asdiv"),    # not an HF Task
    #"arithmetic_2da": EleutherTask("arithmetic_2da"),    # not an HF Task
    #"arithmetic_2ds": EleutherTask("arithmetic_2ds"),    # not an HF Task
    #"arithmetic_3da": EleutherTask("arithmetic_3da"),    # not an HF Task
    #"arithmetic_3ds": EleutherTask("arithmetic_3ds"),    # not an HF Task
    #"arithmetic_4da": EleutherTask("arithmetic_4da"),    # not an HF Task
    #"arithmetic_4ds": EleutherTask("arithmetic_4ds"),    # not an HF Task
    #"arithmetic_5da": EleutherTask("arithmetic_5da"),    # not an HF Task
    #"arithmetic_5ds": EleutherTask("arithmetic_5ds"),    # not an HF Task
    #"arithmetic_2dm": EleutherTask("arithmetic_2dm"),    # not an HF Task
    #"arithmetic_1dc": EleutherTask("arithmetic_1dc"),    # not an HF Task
    #"iwslt17-en-ar": EleutherTask("iwslt17-en-ar"),    # not an HF Task
    #"iwslt17-ar-en": EleutherTask("iwslt17-ar-en"),    # not an HF Task
    #"anagrams1": EleutherTask("anagrams1"),    # not an HF Task
    #"anagrams2": EleutherTask("anagrams2"),    # not an HF Task
    #"cycle_letters": EleutherTask("cycle_letters"),    # not an HF Task
    #"random_insertion": EleutherTask("random_insertion"),    # not an HF Task
    #"reversed_words": EleutherTask("reversed_words"),    # not an HF Task
}

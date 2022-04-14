from catwalk.task import MC_METRICS, InstanceFormat, ENTAILMENT_METRICS, CLASSIFICATION_METRICS, QA_METRICS
from catwalk.tasks.eleuther import EleutherPerplexityTask, EleutherHFTask
from catwalk.tasks.huggingface import hfmc_conversion, HFDatasetsTask
from catwalk.tasks.t5 import t5_prompt_conversion

TASKS = {
    "wikitext": EleutherPerplexityTask("wikitext"),
    "piqa": EleutherHFTask("piqa").add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="goal",
            answer_choices_fields=["sol1", "sol2"],
            correct_answer_index_field="label"
        )
    ).add_metrics(MC_METRICS),
    "squad2": EleutherHFTask("squad2").add_metrics(QA_METRICS),
    "rte": EleutherHFTask("rte").add_instance_conversion(
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
    "cola": EleutherHFTask("cola"),
    "mnli": EleutherHFTask("mnli").add_metrics(ENTAILMENT_METRICS),
    "mnli_mismatched": EleutherHFTask("mnli_mismatched").add_metrics(ENTAILMENT_METRICS),
    "mrpc": EleutherHFTask("mrpc").add_metrics(ENTAILMENT_METRICS),
    "qnli": EleutherHFTask("qnli").add_metrics(ENTAILMENT_METRICS),
    "qqp": EleutherHFTask("qqp").add_metrics(ENTAILMENT_METRICS),
    "sst": EleutherHFTask("sst").add_metrics(CLASSIFICATION_METRICS),
    "wnli": EleutherHFTask("wnli").add_metrics(ENTAILMENT_METRICS),
    "boolq": EleutherHFTask("boolq").add_metrics(CLASSIFICATION_METRICS),
    "cb": EleutherHFTask("cb"),
    "copa": EleutherHFTask("copa").add_metrics(CLASSIFICATION_METRICS),
    "multirc": EleutherHFTask("multirc").add_metrics(CLASSIFICATION_METRICS),
    "record": EleutherHFTask("record"),
    "wic": EleutherHFTask("wic").add_metrics(ENTAILMENT_METRICS),
    "wsc": EleutherHFTask("wsc").add_metrics(MC_METRICS),
    #"coqa": EleutherHFTask("coqa"),    # not an HF Task
    #"drop": EleutherHFTask("drop"),    # not an HF Task
    # Lambada is not an HF Task
    #"lambada": EleutherHFTask("lambada"),
    #"lambada_cloze": EleutherHFTask("lambada_cloze"),
    #"lambada_mt_en": EleutherHFTask("lambada_mt_en"),
    #"lambada_mt_fr": EleutherHFTask("lambada_mt_fr"),
    #"lambada_mt_de": EleutherHFTask("lambada_mt_de"),
    #"lambada_mt_it": EleutherHFTask("lambada_mt_it"),
    #"lambada_mt_es": EleutherHFTask("lambada_mt_es"),
    "prost": EleutherHFTask("prost").add_metrics(MC_METRICS),
    "mc_taco": EleutherHFTask("mc_taco").add_metrics(CLASSIFICATION_METRICS),
    #"pubmedqa": EleutherHFTask("pubmedqa"),  # out of date in the hardcoded datasets version, need to wait for an update
    #"sciq": EleutherHFTask("sciq"),    # not an HF Task
    #"qa4mre_2011": EleutherHFTask("qa4mre_2011"),    # not an HF Task
    #"qa4mre_2012": EleutherHFTask("qa4mre_2012"),    # not an HF Task
    #"qa4mre_2013": EleutherHFTask("qa4mre_2013"),    # not an HF Task
    #"triviaqa": EleutherHFTask("triviaqa"),    # not an HF Task
    "arc_easy": EleutherHFTask("arc_easy").add_metrics(MC_METRICS),
    "arc_challenge": EleutherHFTask("arc_challenge").add_metrics(MC_METRICS),
    #"logiqa": EleutherHFTask("logiqa"),        # not an HF Task
    "hellaswag": EleutherHFTask("hellaswag").add_metrics(MC_METRICS),
    "openbookqa": EleutherHFTask("openbookqa").add_metrics(MC_METRICS),
    "race": EleutherHFTask("race").add_metrics(MC_METRICS),
    "headqa": EleutherHFTask("headqa").add_metrics(MC_METRICS),
    "headqa_es": EleutherHFTask("headqa_es").add_metrics(MC_METRICS),
    "headqa_en": EleutherHFTask("headqa_en").add_metrics(MC_METRICS),
    "mathqa": EleutherHFTask("mathqa").add_metrics(MC_METRICS),
    "webqs": EleutherHFTask("webqs").add_metrics(QA_METRICS),
    "wsc273": EleutherHFTask("wsc273"),
    "winogrande": EleutherHFTask("winogrande").add_metrics(MC_METRICS),
    "anli_r1": EleutherHFTask("anli_r1").add_metrics(ENTAILMENT_METRICS),
    "anli_r2": EleutherHFTask("anli_r2").add_metrics(ENTAILMENT_METRICS),
    "anli_r3": EleutherHFTask("anli_r3").add_metrics(ENTAILMENT_METRICS),
    #"ethics_cm": EleutherHFTask("ethics_cm"),    # not an HF Task
    #"ethics_deontology": EleutherHFTask("ethics_deontology"),    # not an HF Task
    #"ethics_justice": EleutherHFTask("ethics_justice"),    # not an HF Task
    #"ethics_utilitarianism_original": EleutherHFTask("ethics_utilitarianism_original"),    # not an HF Task
    #"ethics_utilitarianism": EleutherHFTask("ethics_utilitarianism"),    # not an HF Task
    #"ethics_virtue": EleutherHFTask("ethics_virtue"),    # not an HF Task
    #"truthfulqa_mc": EleutherHFTask("truthfulqa_mc"),    # not an HF Task
    #"truthfulqa_gen": EleutherHFTask("truthfulqa_gen"),    # not an HF Task
    #"mutual": EleutherHFTask("mutual"),    # not an HF Task
    #"mutual_plus": EleutherHFTask("mutual_plus"),    # not an HF Task
    #"math_algebra": EleutherHFTask("math_algebra"),    # not an HF Task
    #"math_counting_and_prob": EleutherHFTask("math_counting_and_prob"),    # not an HF Task
    #"math_geometry": EleutherHFTask("math_geometry"),    # not an HF Task
    #"math_intermediate_algebra": EleutherHFTask("math_intermediate_algebra"),    # not an HF Task
    #"math_num_theory": EleutherHFTask("math_num_theory"),    # not an HF Task
    #"math_prealgebra": EleutherHFTask("math_prealgebra"),    # not an HF Task
    #"math_precalc": EleutherHFTask("math_precalc"),    # not an HF Task
    #"math_asdiv": EleutherHFTask("math_asdiv"),    # not an HF Task
    #"arithmetic_2da": EleutherHFTask("arithmetic_2da"),    # not an HF Task
    #"arithmetic_2ds": EleutherHFTask("arithmetic_2ds"),    # not an HF Task
    #"arithmetic_3da": EleutherHFTask("arithmetic_3da"),    # not an HF Task
    #"arithmetic_3ds": EleutherHFTask("arithmetic_3ds"),    # not an HF Task
    #"arithmetic_4da": EleutherHFTask("arithmetic_4da"),    # not an HF Task
    #"arithmetic_4ds": EleutherHFTask("arithmetic_4ds"),    # not an HF Task
    #"arithmetic_5da": EleutherHFTask("arithmetic_5da"),    # not an HF Task
    #"arithmetic_5ds": EleutherHFTask("arithmetic_5ds"),    # not an HF Task
    #"arithmetic_2dm": EleutherHFTask("arithmetic_2dm"),    # not an HF Task
    #"arithmetic_1dc": EleutherHFTask("arithmetic_1dc"),    # not an HF Task
    #"iwslt17-en-ar": EleutherHFTask("iwslt17-en-ar"),    # not an HF Task
    #"iwslt17-ar-en": EleutherHFTask("iwslt17-ar-en"),    # not an HF Task
    #"anagrams1": EleutherHFTask("anagrams1"),    # not an HF Task
    #"anagrams2": EleutherHFTask("anagrams2"),    # not an HF Task
    #"cycle_letters": EleutherHFTask("cycle_letters"),    # not an HF Task
    #"random_insertion": EleutherHFTask("random_insertion"),    # not an HF Task
    #"reversed_words": EleutherHFTask("reversed_words"),    # not an HF Task
}

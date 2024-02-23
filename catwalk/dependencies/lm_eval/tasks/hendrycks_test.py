"""
Measuring Massive Multitask Language Understanding
https://arxiv.org/pdf/2009.03300.pdf

The Hendryck's Test is a benchmark that measured a text model's multitask accuracy.
The test covers 57 tasks including elementary mathematics, US history, computer
science, law, and more. To attain high accuracy on this test, models must possess
extensive world knowledge and problem solving ability. By comprehensively evaluating
the breadth and depth of a model's academic and professional understanding,
Hendryck's Test can be used to analyze models across many tasks and to identify
important shortcomings.

Homepage: https://github.com/hendrycks/test
"""
from catwalk.dependencies.lm_eval.base import MultipleChoiceTask


_CITATION = """
@article{hendryckstest2021,
    title={Measuring Massive Multitask Language Understanding},
    author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
    journal={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2021}
}
"""


SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

prompt_formats = [
    "full_answer",
    "alpha_period",
    "alpha_comma",
    "alpha_colon",
    "alpha_paren",
    "alpha",
    "num_period",
    "num_comma",
    "num_colon",
    "num_paren",
    "num",
]


def create_all_tasks():
    """Creates a dictionary of tasks from a list of subjects
    :return: {task_name: task}
        e.g. {hendrycksTest-abstract_algebra: Task, hendrycksTest-anatomy: Task}
    """
    return {f"hendrycksTest-{prompt_format}-{sub}": create_task(sub, prompt_format=prompt_format) for sub in SUBJECTS for prompt_format in prompt_formats}


def create_task(subject, prompt_format=None):
    class HendrycksTest(GeneralHendrycksTest):
        def __init__(self):
            super().__init__(subject, prompt_format=prompt_format)

    return HendrycksTest


class GeneralHendrycksTest(MultipleChoiceTask):
    VERSION = 1
    DATASET_PATH = "cais/mmlu"
    DATASET_NAME = None

    def __init__(self, subject, prompt_format=None):
        # prompt_format = full_answer: predict full answer rather than label
        # prompt_format = alpha_period: alphabetical answer choices with period (eg A.)
        # prompt_format = alpha_comma: alphabetical answer choices with comma (eg A,)
        # prompt_format = alpha_colon: alphabetical answer choices with colon (eg A:)
        # prompt_format = alpha_paren: alphabetical answer choices with close paren (eg A))
        # prompt_format = alpha: alphabetical answer choices without a delimiter (eg A)
        # prompt_format = num_period: numerical answer choices with period (eg 1.)
        # prompt_format = num_comma: numerical answer choices with comma (eg 1,)
        # prompt_format = num_colon: numerical answer choices with colon (eg 1:)
        # prompt_format = num_paren: numerical answer choices with close paren (eg 1))
        # prompt_format = num: numerical answer choices without a delimiter (eg 1)
        self.DATASET_NAME = subject
        self.prompt_format = prompt_format
        super().__init__()

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        return map(self._process_doc, self.dataset["test"])

    def _format_subject(self, subject):
        words = subject.split("_")
        return " ".join(words)

    def unconditioned_prompt(self):
        # Don't need unconditioned normalization here
        return None

    def fewshot_context(self, doc, num_fewshot, **kwargs):
        subject = self.DATASET_NAME
        description = f"The following are multiple choice questions (with answers) about {self._format_subject(subject)}."
        kwargs["description"] = description
        return super().fewshot_context(doc=doc, num_fewshot=num_fewshot, **kwargs)

    def _process_doc(self, doc):
        def format_choices(keys, choices):
            if self.prompt_format == "full_answer":
                return [f" * {choice}\n" for key, choice in zip(keys, choices)]
            if self.prompt_format == "alpha_period":
                return [f" {key}. {choice}\n" for key, choice in zip(keys, choices)]
            if self.prompt_format == "alpha_comma":
                return [f" {key}, {choice}\n" for key, choice in zip(keys, choices)]
            if self.prompt_format == "alpha_colon":
                return [f" {key}: {choice}\n" for key, choice in zip(keys, choices)]
            if self.prompt_format == "alpha_paren":
                return [f" {key}) {choice}\n" for key, choice in zip(keys, choices)]
            if self.prompt_format == "alpha":
                return [f" {key} {choice}\n" for key, choice in zip(keys, choices)]
            if self.prompt_format == "num_period":
                return [f" {key}. {choice}\n" for key, choice in zip(keys, choices)]
            if self.prompt_format == "num_comma":
                return [f" {key}, {choice}\n" for key, choice in zip(keys, choices)]
            if self.prompt_format == "num_colon":
                return [f" {key}: {choice}\n" for key, choice in zip(keys, choices)]
            if self.prompt_format == "num_paren":
                return [f" {key}) {choice}\n" for key, choice in zip(keys, choices)]
            if self.prompt_format == "num":
                return [f" {key} {choice}\n" for key, choice in zip(keys, choices)]
            raise ValueError(f"Invalid prompt format: {self.prompt_format}")
        def format_example(doc, keys):
            """
            <prompt>
             A. <choice1>
             B. <choice2>
             C. <choice3>
             D. <choice4>
            Answer:
            """
            # NOTE: We added the space before the answer key which is not present in the original
            # Eleuther code base. We made this change to ensure that the token associated with the
            # answer choices here exactly matches the answer token used for evaluation.
            question = doc["question"].strip()
            gold = doc["answer"]
            if "num" in self.prompt_format:
                keys = [str(i) for i in range(1, len(doc["choices"]) + 1)]
                gold = keys[doc["answer"]]
            choices = "".join(format_choices(keys, doc["choices"]))
            prompt = f"{question}\n{choices}Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": keys,
            "gold": doc,
        }

    def fewshot_examples(self, k, rnd):
        # fewshot_examples is not just sampling from train_docs because dev is
        # in the same distribution as val/test but auxiliary_train isn't
        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["dev"]))

        # use the unchanged order of the dev set without sampling,
        # just as in the original code https://github.com/hendrycks/test/blob/master/evaluate.py#L28
        return self._fewshot_docs[:k]

    def doc_to_text(self, doc):
        return doc["query"]

    def should_decontaminate(self):
        return True

    def doc_to_decontamination_query(self, doc):
        return doc["query"]

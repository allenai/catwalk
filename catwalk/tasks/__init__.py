import datasets

from catwalk.tasks.mc_task import MCTask, MCTaskFromDataset, CBTTask
from catwalk.tasks.perplexity_task import WikitextTask
from catwalk.tasks.qa_task import QATask, QATaskFromDataset
from catwalk.tasks.generation_task import GenerationTask
from catwalk.tasks.classification_task import ClassificationTask
from catwalk.tasks.pair_classification_task import PairClassificationTask, PairClassificationTaskFromDataset, \
    BlimpTask, PairClassificationTaskFromCrossfit
from catwalk.tasks.qa_task.drop_metric import drop_metric_fn

TASKS = [
    MCTaskFromDataset(
        "piqa",
        dataset_path="piqa",
        dataset_name=None,
        number_of_choices=2,
        context_field=None,
        question_field="goal",
        answer_choices_fields=["sol1", "sol2"],
        correct_answer_index_field="label"
    ),
    MCTaskFromDataset(
        "openbookqa",
        dataset_path="openbookqa",
        dataset_name="main",
        number_of_choices=4,
        context_field=None,
        question_field="question_stem",
        answer_choices_fields="choices.text",
        correct_answer_index_field="answerKey"
    ),
    MCTaskFromDataset(
        "arc_easy",
        dataset_path="ai2_arc",
        dataset_name="ARC-Easy",
        question_field="question",
        answer_choices_fields="choices.text",
        correct_answer_index_field="answerKey",
        number_of_choices=4
    ),
    QATaskFromDataset(
        "squad",
        dataset_path="squad",
        dataset_name=None,
        context_field="context",
        question_field="question",
        answer_field="answers.text",
        id_field="id"
    ),
    QATaskFromDataset(
        "squad_v2",
        dataset_path="squad_v2",
        dataset_name=None,
        context_field="context",
        question_field="question",
        answer_field="answers.text",
        id_field="id"
    ),
    PairClassificationTaskFromDataset(
        "anli",
        dataset_path="anli",
        dataset_name=None,
        text1_field="premise",
        text2_field="hypothesis",
        label_field="label",
        labels=["entailment", "contradiction", "neutral"],  # could be ["True", "Neither", "False"]?
        id_field="uid",
        split_mappings={
            "test": ["test_r1", "test_r2", "test_r3"],
            "validation": ["dev_r1", "dev_r2", "dev_r3"],
            "train": ["train_r1", "train_r2", "train_r3"],
        }
    ),
    CBTTask("CN"),
    CBTTask("NE"),
    QATaskFromDataset(
        "drop",
        dataset_path="drop",
        dataset_name=None,
        context_field="passage",
        question_field="question",
        answer_field="answers_spans.spans",
        id_field="query_id",
        metric_fn=drop_metric_fn
    ),
    WikitextTask("wikitext", "wikitext-2-raw-v1"),
    PairClassificationTaskFromCrossfit(
        "amazon_polarity",
        text1_field="title",
        text2_field="content"
    ),
    PairClassificationTaskFromCrossfit(
        "glue_qnli",
        text1_field="question",
        text2_field="sentence")

    # TODO: Add more tasks
]

# Blimp tasks
TASKS.extend(BlimpTask(config_name) for config_name in datasets.get_dataset_config_names('blimp'))

# convert to dictionary
TASKS = {
    task.name : task for task in TASKS
}

# Missing from Eleuther:
# Arithmetic, because it's not using HF datasets
# asdiv, because it's not using HF datasets
# coqa: Theirs is not using datasets because the datasets code is unstable.
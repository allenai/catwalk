from ludwig.tasks.mc_task import MCTask, MCTaskFromDataset
from ludwig.tasks.qa_task import QATask, QATaskFromDataset
from ludwig.tasks.summarization_task import SummarizationTask
from ludwig.tasks.classification_task import ClassificationTask
from ludwig.tasks.pair_classification_task import PairClassificationTask

TASKS = {
    "piqa": MCTaskFromDataset(
        "piqa",
        dataset="piqa",
        number_of_choices=2,
        context_field=None,
        question_field="goal",
        answer_choices_fields=["sol1", "sol2"],
        correct_answer_index_field="label"
    ),
    "openbookqa": MCTaskFromDataset(
        "openbookqa",
        dataset="openbookqa",
        dataset_config="main",
        number_of_choices=4,
        context_field=None,
        question_field="question_stem",
        answer_choices_fields="choices.text",
        correct_answer_index_field="answerKey"
    ),
    "squad": QATaskFromDataset(
        "squad",
        dataset="squad",
        context_field="context",
        question_field="question",
        answer_field="answers.text.0",
        id_field="id"
    )
    # TODO: Add more tasks
}

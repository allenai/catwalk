from catwalk2.task import TaskType
from catwalk2.tasks.eleuther import EleutherPerplexityTask, EleutherHFTask
from catwalk2.tasks.huggingface import TaskWithHFMCConversion

TASKS = {
    "wikitext": EleutherPerplexityTask("wikitext"),
    "piqa": EleutherHFTask(TaskType.MULTIPLE_CHOICE, "piqa").with_hf_mc_conversion(
        context_field=None,
        question_field="goal",
        answer_choices_fields=["sol1", "sol2"],
        correct_answer_index_field="label"
    ).with_mc_metrics(),
    "squad2": EleutherHFTask(TaskType.QA, "squad2")
}

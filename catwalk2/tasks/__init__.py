from catwalk2.task import TaskType
from catwalk2.tasks.eleuther import EleutherPerplexityTask, EleutherHFTask

TASKS = {
    "wikitext": EleutherPerplexityTask("wikitext"),
    "piqa": EleutherHFTask(TaskType.MULTIPLE_CHOICE, "piqa"),
    "squad2": EleutherHFTask(TaskType.QA, "squad2")
}

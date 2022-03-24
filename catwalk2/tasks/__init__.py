from catwalk2.task import MC_METRICS, InstanceFormat
from catwalk2.tasks.eleuther import EleutherPerplexityTask, EleutherHFTask
from catwalk2.tasks.huggingface import hfmc_conversion

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
    "squad2": EleutherHFTask("squad2")
}

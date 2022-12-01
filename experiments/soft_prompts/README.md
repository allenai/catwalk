Evaluating soft prompts
=======================

This folder contains an experiment that evaluates soft prompts against full finetuning of the same model.

This experiment is severely limited and only a proof of concept.
If you really want to evaluate soft prompts, you need to fix the following:
* The evaluation metric needs to be "accuracy", not "loss".
* Hyperparameters need to be tuned, both for the full finetuning baseline, and for the soft-prompt experiment.
* This experiment hard-codes soft-prompts of length 3, but that's just a guess.
* To make this run fast, it only runs on a few models and tasks. A real experiment needs to run on more models and
  more tasks.
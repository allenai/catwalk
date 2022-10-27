# Catwalk

Catwalk shows off models.

Catwalk contains a lot of models, and a lot of tasks. The goal is to be able to run all models on all tasks. In
practice, some combinations are not possible, but many are.

<details>
<summary>Here is the current list of tasks we have implemented.
This list is not showing the `metaicl` and `p3` categories of tasks, because those are
largely variants of the other tasks.
</summary>

```
wikitext
piqa
squad
squadshifts-reddit
squadshifts-amazon
squadshifts-nyt
squadshifts-new-wiki
mrqa::race
mrqa::newsqa
mrqa::triviaqa
mrqa::searchqa
mrqa::hotpotqa
mrqa::naturalquestions
mrqa::bioasq
mrqa::drop
mrqa::relationextraction
mrqa::textbookqa
mrqa::duorc.paraphraserc
squad2
rte
superglue::rte
cola
mnli
mnli_mismatched
mrpc
qnli
qqp
sst
wnli
boolq
cb
copa
multirc
wic
wsc
drop
lambada
lambada_cloze
lambada_mt_en
lambada_mt_fr
lambada_mt_de
lambada_mt_it
lambada_mt_es
prost
mc_taco
pubmedqa
sciq
qa4mre_2011
qa4mre_2012
qa4mre_2013
triviaqa
arc_easy
arc_challenge
logiqa
hellaswag
openbookqa
race
headqa_es
headqa_en
mathqa
webqs
wsc273
winogrande
anli_r1
anli_r2
anli_r3
ethics_cm
ethics_deontology
ethics_justice
ethics_utilitarianism_original
ethics_utilitarianism
ethics_virtue
truthfulqa_gen
mutual
mutual_plus
math_algebra
math_counting_and_prob
math_geometry
math_intermediate_algebra
math_num_theory
math_prealgebra
math_precalc
math_asdiv
arithmetic_2da
arithmetic_2ds
arithmetic_3da
arithmetic_3ds
arithmetic_4da
arithmetic_4ds
arithmetic_5da
arithmetic_5ds
arithmetic_2dm
arithmetic_1dc
anagrams1
anagrams2
cycle_letters
random_insertion
reversed_words
raft::ade_corpus_v2
raft::banking_77
raft::neurips_impact_statement_risks
raft::one_stop_english
raft::overruling
raft::semiconductor_org_types
raft::systematic_review_inclusion
raft::tai_safety_research
raft::terms_of_service
raft::tweet_eval_hate
raft::twitter_complaints
```
</details>

## Installation

<!-- start install -->

**Catwalk** requires Python 3.9 or later.

Unfortunately Catwalk cannot be installed from pypi, because it depends on other packages that are not uploaded to
pypi.

Install from source:
```shell
git clone https://github.com/allenai/catwalk.git
cd catwalk
pip install -e .
```

<!-- end install -->

## Getting started

Let's run GPT2 on PIQA:
```shell
python -m catwalk --model rc::gpt2 --task piqa
```

This will load up GPT2 and use it to perform the PIQA task with the "ranked classification" approach.

You can specify multiple tasks at once:
```shell
python -m catwalk --model rc::gpt2 --task piqa arc_easy
```

It'll print you a nice table with all tasks and the metrics for each task:
```text
arc_challenge   acc     0.22440272569656372
arc_easy        acc     0.3998316526412964
piqa    acc     0.6256800889968872
```

## Training / Finetuning

Catwalk can train models. It can train models on a single task, or on multiple tasks at once.
To train, use this command line:
```shell
python -m catwalk.train --model rc::gpt2 --task piqa
```

You can train on multiple tasks at the same time, if you want to create a multi-task model:
```shell
python -m catwalk.train --model rc::gpt2 --task piqa arc_easy
```

Note that not all models support training. If you want to train one and can't, create an issue and tag @dirkgr in
it. 

## Tango integration

Catwalk uses [Tango](https://github.com/allenai/tango) for caching and executing evaluations. The command line
interface internally constructs a Tango step graph and executes it. You can point the command line to a Tango
workspace to cache results:

```shell
python -m catwalk --model rc::gpt2 --task piqa arc_easy -w ./my-workspace/
```

The second time you run one of those tasks, it will be fast:
```shell
time python -m catwalk --model rc::gpt2 --task piqa -w ./my-workspace/
```

```text
arc_easy	acc	0.39941078424453735
piqa	acc	0.626224160194397

________________________________________________________
Executed in    9.82 secs    fish           external
   usr time    6.51 secs  208.00 micros    6.51 secs
   sys time    1.25 secs  807.00 micros    1.25 secs
```

Tango workspaces also save partial results, so if you interrupt an evaluation half-way through, your progress is
saved.

## Team

<!-- start team -->

**ai2-catwalk** is developed and maintained by the AllenNLP team, backed by [the Allen Institute for Artificial Intelligence (AI2)](https://allenai.org/).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
To learn more about who specifically contributed to this codebase, see [our contributors](https://github.com/allenai/catwalk/graphs/contributors) page.

<!-- end team -->

## License

<!-- start license -->

**ai2-catwalk** is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
A full copy of the license can be found [on GitHub](https://github.com/allenai/catwalk/blob/main/LICENSE).

<!-- end license -->

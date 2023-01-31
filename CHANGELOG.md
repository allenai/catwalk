# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- Fixed the way we compute SQuAD metrics.


## [v0.2.2](https://github.com/allenai/catwalk/releases/tag/v0.2.2) - 2023-01-27

### Changed

- Changed the package name to ai2-catwalk to avoid a name conflict on Pypi.


## [v0.2.1](https://github.com/allenai/catwalk/releases/tag/v0.2.1) - 2023-01-26

### Fixed

- Fixed the release process


## [v0.2.0](https://github.com/allenai/catwalk/releases/tag/v0.2.0) - 2022-12-02

### Changed

- MetaICLTask now supports fewshots less than 16 and only support getting the test split
- set default logging level to `"WARNING"` instead of `"ERROR"` when invoking `python -m catwalk`
- changed MetaICLModel formatting to always preserve whitespace, to reproduce MetaICL results
- improved speed of rank classification models by aggregating sequence logits on GPU rather than on CPU
- The promptsource templates now live directly inside of Catwalk. This avoids dependency issues.
- Promptsource now applies the templates in parallel across all CPUs.
- Replaced a dependency on `lmeval` with a copy of the source code

### Added

- Adds the ability to train models
- Few-shot abilities
- P3 tasks
- Encoder-only QA models
- SQuAD and SQuADShifts tasks
- Adds a new MetaICLTask that supports all evaluation tasks in that benchmark
- Adds a new MetaICLModel that replicates the formatting and truncation used by MetaICL for few shot evaluation
- An option for rank classification to average log likelihoods by token length
- Adds support for inference with IA3 adaptors loaded from a file on decoder only ranked classification models
- Add support for MetaICL's race-high and numer_sense tasks
- Adds QA task support for autoregressive (previously only available with Eleuther task format)
- Adds QA task support for T5 models
- Optional `random_subsample_seed` for PredictStep
- An option for rank classification to average log likelihoods by token length
- Added MRQA task
- Adds support for inference with IA3 adapters loaded from a file on decoder only ranked classification models
- Added the ability to train `HFAutoModel`
- Added the ability for `HFAutoModel` to run NLI tasks
- Adds ability to backoff to auto device_map on out of memory error for ranked classification models
- Format conversions for a number of multiple choice models
- Added an experiment config that trains many models on many tasks
- Added promptsource support
- Added support for soft prompts
- Added more models, T0 variants of T5 and Eleuther variants of GPT
- Added support for Huggingface's accelerate project, but only for inference
- Promptsource now supports few-shot ICL.
- Compatibility with the latest version of torchmetrics

### Fixed

- Fixed progress bar for HFAutoModel QA evaluation
- Fixed bug causing few-shot to use more than specified number of shots
- Fixed bug in cached_transformer.get() that prevented using override_weights_file arg
- Fixed the `load_weights` arg in cached_transformers.get() which was documented but not implemented
- Fixed support for training with OPT models
- Countless tweaks to `FinetuneStep`
- Some models insert special tokens where they should not. This fixes that.
- Metrics were messy for classification tasks. They are still messy, but less so.
- Applied workaround for T5 bug in huggingface tokenizers
- Fixed fine-tuning T5 ranked classification models
- Fixed the names of the T5 1.1 models
- Cached transformers now take `kwargs` into account.
- Fixed various tasks: WSC, TriviaQA, Race, HeadQA
- Fixed the case where different promptsource templates produce different numbers of answer choices
- `tqdm` has to be closed or it'll start printing a bunch of newlines.


## [v0.1.0](https://github.com/allenai/catwalk/releases/tag/v0.1.0) - 2022-06-10

### Changed

- Catwalk is now Open Source and has a release process. 


## [v0.0.0](https://github.com/allenai/catwalk/commit/7c78b9bb989685f92decef6bd0593e16ff164587)

### Added

- Catwalk

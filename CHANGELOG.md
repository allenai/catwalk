# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed
- MetaICLTask now supports fewshots less than 16 and only support getting the test split
- set default logging level to `"WARNING"` instead of `"ERROR"` when invoking `python -m catwalk`
- changed MetaICLModel formatting to always preserve whitespace, to reproduce MetaICL results
- improved speed of rank classification models by aggregating sequence logits on GPU rather than on CPU

### Added

- Adds the ability to train models
- Few-shot abilities
- P3 tasks
- Encoder-only QA models
- SQuAD and SQuADShifts tasks
- Adds a new MetaICLTask that supports all evaluation tasks in that benchmark
- Adds a new MetaICLModel that replicates the formatting and truncation used by MetaICL for few shot evaluation
- An option for rank classification to average log likelihoods by token length
- Adds support for inference with IA3 adapters loaded from a file on decoder only ranked classification models
- Add support for MetaICL's race-high and numer_sense tasks
- Adds QA task support for autoregressive (previously only available with Eleuther task format)
- Adds QA task support for T5 models
- Optional `random_subsample_seed` for PredictStep
- An option for rank classification to average log likelihoods by token length
- Added MRQA task
- Adds an example experiment that uses IA3 adapters from Tango with catwalk training
- Added the ability to train `HFAutoModel`
- Adds ability to backoff to auto device_map on out of memory error for ranked classification models

### Fixed

- Fixed progress bar for HFAutoModel QA evaluation
- Fixed bug causing few-shot to use more than specified number of shots
- Fixed bug in cached_transformer.get() that prevented using override_weights_file arg
- Fixed the `load_weights` arg in cached_transformers.get() which was documented but not implemented
- Fixed support for training with OPT models

## [v0.1.0](https://github.com/allenai/catwalk/releases/tag/v0.1.0) - 2022-06-10

### Changed

- Catwalk is now Open Source and has a release process. 


## [v0.0.0](https://github.com/allenai/catwalk/commit/7c78b9bb989685f92decef6bd0593e16ff164587)

### Added

- Catwalk

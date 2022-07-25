# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Changed
- MetaICLTask now supports fewshots less than 16 and only support getting the test split
- set default logging level to `"WARNING"` instead of `"ERROR"` when invoking `python -m catwalk`

### Added

- Adds the ability to train models
- Few-shot abilities
- P3 tasks
- Encoder-only QA models
- SQuAD and SQuADShifts tasks
- Adds a new MetaICLTask that supports the evaluation classification tasks in that benchmark
- Adds a new MetaICLModel that replicates the formatting and truncation used by MetaICL for few shot evaluation
- Prefix caching for DecoderOnlyRCModel that reuses overlapping prefixes between instances rather than recomputing them
- Option random_subsample_seed for PrefixCache

### Fixed

- Fixed bug causing few-shot to use more than specified number of shots
- Fixed bug in cached_transformer.get() that prevented using override_weights_file arg

## [v0.1.0](https://github.com/allenai/catwalk/releases/tag/v0.1.0) - 2022-06-10

### Changed

- Catwalk is now Open Source and has a release process. 


## [v0.0.0](https://github.com/allenai/catwalk/commit/7c78b9bb989685f92decef6bd0593e16ff164587)

### Added

- Catwalk

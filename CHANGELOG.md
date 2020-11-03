# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project uses [Semantic Versioning](http://semver.org/).

# [2.2.2] - 2020-10-29
 ### Fixed
  - Update creevey to new name wildebeest
  
# [2.2.1] - 2020-10-29
### Added
 - note to notebook tutorials about potential out-of-memory issue when `DataLoaders` have too high a value for `num_workers`
### Changed
 - included `Dockerfile` now installs the `octopod` library
### Fixed
 - included `Dockerfile` now installs `libgl1-mesa-glx` to avoid `ImportError: libGL.so.1: cannot open shared object file: No such file or directory` when importing `octopod`
 - batch sizes of size 1 are skipped correctly for all models

# [2.2.0] - 2020-8-20
### Changed
 - Switched to using Github Actions for CI/CD

# [2.1.0] - 2020-7-15
### Changed
 - Torch is no longer pinned to version 1.2. This allows user to use Octopod with python 3.8.

# [2.0.2] - 2020-7-14
### Fixed
 - Support for `torch.optim.lr_scheduler.ReduceLROnPlateau` for `scheduler` argument in `MultiTaskLearner.fit`
 - Learner object skips and `ResnetForMultiTaskClassification` `forward` method automatically handles batches of size 1 to avoid `ValueError` with `nn.BatchNorm1d` failing on batches of size 1

# [2.0.1] - 2020-7-14
### Added
 - logo now in octopod readme

# [2.0.0] - 2020-7-10
### Added
 - Octopod learners can now use multiple loss functions and do multi-label in addition to multi-class tasks
### Changed
 - loss functions and metrics are specified via a dictionary of tasks and corresponding loss functions

# [1.0.0] - 2020-7-09
### Changed
 - Tonks is now called Ocotopod

==== Below is Tonks development ====

# [1.0.0] - 2020-7-08
### Added
 - Warnings to switch to new name

# [0.4.4] - 2020-7-5
### Fixed
 - A few paren locations in multi_task_ensemble.py

# [0.4.3] - 2020-6-24
### Added
 - Version pin for sentencepiece

# [0.4.2] - 2020-6-23
### Added
 - Dockerfile

# [0.4.1] - 2020-6-16
### Added
 - Missing 's' in `extras_requires` in setup.py

# [0.4.0] - 2020-6-12
### Added
 - warning about upcoming rename of library

# [0.3.0] - 2020-4-14
### Added
 - read the docs fixes

# [0.2.0] - 2020-4-13
### Added
 - devs requirements in setup.py

# [0.1.0] - 2020-4-13
### Added
 - Link to docs in README

# [0.0.0] - 2020-4-13
### Added
 - Open-sourcing Tonks codebase for multi-dataset multi-task learning

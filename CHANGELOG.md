# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project uses [Semantic Versioning](http://semver.org/).


# [3.3.0] - 2022-03-08
  ### Changed
  - removed pinned transformers library and changed the way that bert is loaded in the text and ensemble modules

# [3.2.0] - 2022-03-08
  ### Added
  - Added an `import_model` method to the class BertForMultiTaskClassification in multi_task_bert.py that allows a file that was created with `export` to be easily imported

# [3.1.7] - 2022-02-18
  ### Changed
  - Added `ipywidgets` to requirements to support Jupyter out of the box

# [3.1.6] - 2021-10-22
  ### Changed
  - Use a running average for total training loss as well as individual task training loss
  
# [3.1.5] - 2021-04-14
  ### Changed
  - base `Dockerfile` image to one with `python@3.8` and `torch@1.8.1`
  - recompiled requirements
  - added `python@3.9` to GitHub workflows
  - documentation requirements has been moved to `requirements-dev.txt`

# [3.1.4] - 2021-04-02
  ### Added
  - PyPI version badge to `README.md`

# [3.1.3] - 2021-04-02
  ### Added
  - CI testing and documentation badges to `README.md`

# [3.1.2] - 2021-03-31
  ### Fixed
  - Rebuild reqirements.txt to use m2r2

# [3.1.1] - 2021-03-31
  ### Fixed
  - sphinx and m2r were no longer in requirements files so docs failed added sphinx and had to update to m2r2 because m2r is not actively supported
  - Changelog dates were still for 2020

# [3.1.0] - 2021-02-03
 ### Changed
  - Datasets can now take string or encoded labels using sklearn label encoders.
 ### Added
  - Tests for new functionality in the Dataloaders for image, text, & ensemble

# [3.0.1] - 2021-01-13
 ### Fixed
  - `transformers` was listed twice in setup.py

# [3.0.0] - 2021-01-04
 ### Added
  - Report smoothed training losses in progress bar during fitting.
 ### Changed
  - Report exponentially weighted moving average rather than simple average of training loss over batches at the end of each epoch.

# [2.2.6] - 2020-12-31
 ### Fixed
  - Added `matplotlib` dependency to display progress bars in notebooks
  - Removed parameter that no longer exists from `MultiTaskLearner.fit` docstring
 ### Changed
  - Use `pip-tools`

# [2.2.5] - 2020-12-15
 ### Fixed
  - Multi Task BERT model save works for models without new classifiers

# [2.2.4] - 2020-12-14
 ### Fixed
  - Load method in ensemble model loads to `image_dense_layers`

# [2.2.3] - 2020-12-10
 ### Fixed
  - Multi-label datasets are imported in top level init

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

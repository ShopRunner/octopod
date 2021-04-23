![octopod logo](https://net-shoprunner-scratch-data-science.s3.amazonaws.com/msugimura/octopod/octopod_small.jpg)

# Octopod

[![PyPI version](https://badge.fury.io/py/octopod.svg)](https://badge.fury.io/py/octopod)
[![Workflows Passing](https://github.com/ShopRunner/octopod/workflows/Python%20package/badge.svg)](https://github.com/ShopRunner/octopod/actions/workflows/prod.yaml)
[![Documentation Status](https://readthedocs.org/projects/octopod/badge/?version=latest)](https://octopod.readthedocs.io/en/latest/?badge=latest)

Octopod is a general purpose deep learning library developed by the ShopRunner Data Science team to train multi-task image, text, or ensemble (image + text) models.

What differentiates our library is that you can train a multi-task model with different datasets for each of your tasks. For example, you could train one model to label dress length for dresses and pants length for pants.

See the [docs](https://octopod.readthedocs.io/en/latest/) for more details.

To quickly get started, check out one of our tutorials in the `notebooks` folder. In particular, the `synthetic_data` tutorial provides a very quick example of how the code works.

Note 7/08/20: We are renaming this repository Octopod (previously called Tonks). The last version of the PyPI library under the name Tonks will not break but will warn the user to begin installing and using Octopod instead. No further development will continue under the name Tonks.

Note 6/12/20: Our team previously had a tradition of naming projects with terms or characters from the Harry Potter series, but we are disappointed by J.K. Rowling’s persistent transphobic comments. In response, we will be renaming this repository, and are working to develop an inclusive solution that minimizes disruption to our users.

## Structure
- `notebooks`
    - `fashion_data`: a set of notebooks demonstrating training Octopod models on an open source fashion dataset consisting of images and text descriptions
    - `synthetic_data`: a set of notebooks demonstrating training Octopod models on a set of generated color swatches. This is meant to be an easy fast demo of the library's capabilities that can be run on CPU's.
- `octopod`
    - `ensemble`: code for ensemble models of text and vision models
    - `text`: code for text models with a BERT architecture
    - `vision`: code for vision models with ResNet50 architectures

## Installation
```
pip install octopod
```

You may get an error from the `tokenizer` package if you do not have a Rust compiler installed; see https://github.com/huggingface/transformers/issues/2831#issuecomment-592724471.

## Notes
Currently, this library supports ResNet50 and BERT models.

In some of our documentation the terms `pretrained` and `vanilla` appear. `pretrained` is our shorthand for Octopod models that have been trained at least once already so their weights have been tuned for a specific use case. `vanilla` is our shorthand for base weights coming from `transformers` or `PyTorch` for the out-of-the-box BERT and ResNet50 models.

For our examples using text models, we use the [transformers](https://github.com/huggingface/transformers) repository managed by huggingface. The most recent version is called `transformers`. The huggingface repo is the appropriate place to check on BERT documentation and procedures.

## Development

Want to add to or fix issues in Octopod? We welcome outside input and have tried to make it easier to test. You can run everything inside a docker container with the following:

```bash
# to build the container
# NOTE: this may take a while
docker build -t octopod .
# nvidia-docker run : basic startup with nvidia docker to access gpu
# --rm : deletes container when closed
# -p : exposes ports (ex: for jupyter notebook to work)
# bash : opens bash in the container once it starts
# "pip install jupyter && bash" : install requirements-dev and bash
nvidia-docker run \
    -it \
    --rm \
    -v "${PWD}:/octopod" \
    -p 8888:8888 \
    octopod /bin/bash -c "pip install jupyter && bash"
# run jupyter notebook
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

# Tonks

Note 6/12/20: Our team previously had a tradition of naming projects with terms or characters from the Harry Potter series, but we are disappointed by J.K. Rowling’s persistent transphobic comments. In response, we will be renaming this repository, and are working to develop an inclusive solution that minimizes disruption to our users.

Tonks is a general purpose deep learning library developed by the ShopRunner Data Science team to train multi-task image, text, or ensemble (image + text) models.

What differentiates our library is that you can train a multi-task model with different datasets for each of your tasks. For example, you could train one model to label dress length for dresses and pants length for pants.

See the [docs](https://tonks.readthedocs.io/en/latest/) for more details.

To quickly get started, check out one of our tutorials in the `notebooks` folder. In particular, the `synthetic_data` tutorial provides a very quick example of how the code works.

## Structure
- `notebooks`
    - `fashion_data`: a set of notebooks demonstrating training Tonks models on an open source fashion dataset consisting of images and text descriptions
    - `synthetic_data`: a set of notebooks demonstrating training Tonks models on a set of generated color swatches. This is meant to be an easy fast demo of the library's capabilities that can be run on CPU's.
- `tonks`
    - `ensemble`: code for ensemble models of text and vision models
    - `text`: code for text models with a BERT architecture
    - `vision`: code for vision models with ResNet50 architectures

## Installation 
```
pip install tonks
```

You may get an error from the `tokenizer` package if you do not have a Rust compiler installed; see https://github.com/huggingface/transformers/issues/2831#issuecomment-592724471.

## Notes
Currently, this library supports ResNet50 and BERT models.

In some of our documentation the terms `pretrained` and `vanilla` appear. `pretrained` is our shorthand for Tonks models that have been trained at least once already so their weights have been tuned for a specific use case. `vanilla` is our shorthand for base weights coming from `transformers` or `PyTorch` for the out-of-the-box BERT and ResNet50 models.

For our examples using text models, we use the [transformers](https://github.com/huggingface/transformers) repository managed by huggingface. The most recent version is called `transformers`. The huggingface repo is the appropriate place to check on BERT documentation and procedures.

## Development

Want to add to or fix issues in Tonks? We welcome outside input and have tried to make it easier to test. You can run everything inside a docker container with the following:

```bash
# to build the container
# NOTE: this may take a while
nvidia-docker build -t tonks .
# nvidia-docker run : basic startup with nvidia docker to access gpu
# --rm : deletes container when closed
# -p : exposes ports (ex: for jupyter notebook to work)
# bash : opens bash in the container once it starts
# "pip install jupyter && bash" : install requirements-dev and bash
nvidia-docker run \
    -it \
    --rm \
    -v "${PWD}:/tonks" \
    -p 8888:8888 \
    -p 8000:8000 \
    tonks /bin/bash -c "pip install jupyter && bash"
# run jupyter notebook
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```
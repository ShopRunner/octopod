# Octopod

Note 6/12/20: Our team previously had a tradition of naming projects with terms or characters from the Harry Potter series, but we are disappointed by J.K. Rowling’s persistent transphobic comments. In response, we will be renaming this repository, and are working to develop an inclusive solution that minimizes disruption to our users.

Octopod is a general purpose deep learning library developed by the ShopRunner Data Science team to train multi-task image, text, or ensemble (image + text) models.

What differentiates our library is that you can train a multi-task model with different datasets for each of your tasks. For example, you could train one model to label dress length for dresses and pants length for pants.

See the [docs](https://octopod.readthedocs.io/en/latest/) for more details.

To quickly get started, check out one of our tutorials in the `notebooks` folder. In particular, the `synthetic_data` tutorial provides a very quick example of how the code works.

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

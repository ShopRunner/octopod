import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from octopod.vision.config import cropped_transforms, full_img_transforms
from octopod.vision.helpers import center_crop_pil_image


class OctopodEnsembleDataset(Dataset):
    """
    Load image and text data specifically for an ensemble model

    Parameters
    ----------
    text_inputs: pandas Series
        the text to be used
    img_inputs: pandas Series
        the paths to images to be used
    y: list
        A list of lists of dummy-encoded categories
    tokenizer: pretrained BERT Tokenizer
        BERT tokenizer likely from `transformers`
    max_seq_length: int (defaults to 128)
        Maximum number of tokens to allow
    transform: str or list of PyTorch transforms
        specifies how to preprocess the full image for a Octopod image model
        To use the built-in Octopod image transforms, use the strings: `train` or `val`
        To use custom transformations supply a list of PyTorch transforms.
    crop_transform: str or list of PyTorch transforms
        specifies how to preprocess the center cropped image for a Octopod image model
        To use the built-in Octopod image transforms, use strings `train` or `val`
        To use custom transformations supply a list of PyTorch transforms.
    """
    def __init__(self,
                 text_inputs,
                 img_inputs,
                 y,
                 tokenizer,
                 max_seq_length=128,
                 transform='train',
                 crop_transform='train'):
        self.text_inputs = text_inputs
        self.img_inputs = img_inputs
        self.y = y
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        if transform == 'train' or 'val':
            self.transform = full_img_transforms[transform]
        else:
            self.transform = transform

        if crop_transform == 'train' or 'val':
            self.crop_transform = cropped_transforms[crop_transform]
        else:
            self.crop_transform = crop_transform

    def __getitem__(self, index):
        """Return dict of PyTorch tensors for preprocessed images and text and tensor of labels"""
        # Text processing
        x_text = self.text_inputs[index].replace('\n', ' ').replace('\r', ' ')

        tokenized_x = (
            ['[CLS]']
            + self.tokenizer.tokenize(x_text)[:self.max_seq_length - 2]
            + ['[SEP]']
        )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_x)

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == self.max_seq_length

        bert_text = torch.from_numpy(np.array(input_ids))

        # Image processing
        full_img = Image.open(self.img_inputs[index]).convert('RGB')

        cropped_img = center_crop_pil_image(full_img)

        full_img = self.transform(full_img)
        cropped_img = self.crop_transform(cropped_img)

        y_output = torch.from_numpy(np.array(self.y[index])).long()

        return {'bert_text': bert_text,
                'full_img': full_img,
                'crop_img': cropped_img}, y_output

    def __len__(self):
        return len(self.text_inputs)


class OctopodEnsembleDatasetMultiLabel(OctopodEnsembleDataset):
    """
    Multi label subclass of OctopodEnsembleDataset

    Parameters
    ----------
    text_inputs: pandas Series
        the text to be used
    img_inputs: pandas Series
        the paths to images to be used
    y: list
        a list of binary encoded categories with length equal to number of
        classes in the multi-label task. For a 4 class multi-label task
        a sample list would be [1,0,0,1]
    tokenizer: pretrained BERT Tokenizer
        BERT tokenizer likely from `transformers`
    max_seq_length: int (defaults to 128)
        Maximum number of tokens to allow
    transform: str or list of PyTorch transforms
        specifies how to preprocess the full image for a Octopod image model
        To use the built-in Octopod image transforms, use the strings: `train` or `val`
        To use custom transformations supply a list of PyTorch transforms.
    crop_transform: str or list of PyTorch transforms
        specifies how to preprocess the center cropped image for a Octopod image model
        To use the built-in Octopod image transforms, use strings `train` or `val`
        To use custom transformations supply a list of PyTorch transforms.
    """
    def __getitem__(self, index):
        """Return dict of PyTorch tensors for preprocessed images and text and tensor of labels"""
        # Text processing
        x_text = self.text_inputs[index].replace('\n', ' ').replace('\r', ' ')

        tokenized_x = (
            ['[CLS]']
            + self.tokenizer.tokenize(x_text)[:self.max_seq_length - 2]
            + ['[SEP]']
        )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_x)

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == self.max_seq_length

        bert_text = torch.from_numpy(np.array(input_ids))

        # Image processing
        full_img = Image.open(self.img_inputs[index]).convert('RGB')

        cropped_img = center_crop_pil_image(full_img)

        full_img = self.transform(full_img)
        cropped_img = self.crop_transform(cropped_img)

        y_output = torch.FloatTensor(self.y[index])

        return {'bert_text': bert_text,
                'full_img': full_img,
                'crop_img': cropped_img}, y_output

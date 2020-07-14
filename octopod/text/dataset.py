import numpy as np
import torch
from torch.utils.data import Dataset


class OctopodTextDataset(Dataset):
    """
    Load data for use with a BERT model

    Parameters
    ----------
    x: pandas Series
        the text to be used
    y: list
        A list of lists of dummy-encoded categories
    tokenizer: pretrained BERT Tokenizer
        BERT tokenizer likely from `transformers`
    max_seq_length: int (defaults to 128)
        Maximum number of tokens to allow
    """
    def __init__(self, x, y, tokenizer, max_seq_length=128):
        self.x = x
        self.y = y
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __getitem__(self, index):
        """Return text as PyTorch tensor of token ids and tensor of labels"""
        text_x = self.x[index].replace('\n', ' ').replace('\r', ' ')

        tokenized_x = (
            ['[CLS]']
            + self.tokenizer.tokenize(text_x)[:self.max_seq_length - 2]
            + ['[SEP]']
        )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_x)

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == self.max_seq_length

        x_input = torch.from_numpy(np.array(input_ids))

        y_output = torch.from_numpy(np.array(self.y[index])).long()

        return x_input, y_output

    def __len__(self):
        return len(self.x)


class OctopodTextDatasetMultiLabel(OctopodTextDataset):
    """
    Multi label subclass of OctopodTextDataset

    Parameters
    ----------
    x: pandas Series
        the text to be used
    y: list
        a list of binary encoded categories with length equal to number of
        classes in the multi-label task. For a 4 class multi-label task
        a sample list would be [1,0,0,1]
    tokenizer: pretrained BERT Tokenizer
        BERT tokenizer likely from `transformers`
    max_seq_length: int (defaults to 128)
        Maximum number of tokens to allow
    """

    def __getitem__(self, index):
        """Return text as PyTorch tensor of token ids and tensor of labels"""
        text_x = self.x[index].replace('\n', ' ').replace('\r', ' ')

        tokenized_x = (
            ['[CLS]']
            + self.tokenizer.tokenize(text_x)[:self.max_seq_length - 2]
            + ['[SEP]']
        )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_x)

        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == self.max_seq_length

        x_input = torch.from_numpy(np.array(input_ids))

        y_output = torch.FloatTensor(self.y[index])

        return x_input, y_output

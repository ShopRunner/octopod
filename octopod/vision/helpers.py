import numpy as np
from PIL import Image
import torch.nn as nn
from wildebeest.ops.image import centercrop


def center_crop_pil_image(img):
    """
    Helper function to crop the center out of images.

    Utilizes the centercrop function from `wildebeest`

    Parameters
    ----------
    img: array
        PIL image array

    Returns
    -------
    PIL.Image: Slice of input image corresponding to a cropped area around the center
    """
    img = np.array(img)
    cropped_img = centercrop(img, reduction_factor=.4)
    return Image.fromarray(cropped_img)


class _Identity(nn.Module):
    """
    Used to pass penultimate layer features to the the ensemble

    Motivation for this is that the features from the penultimate layer
    are likely more informative than the 1000 way softmax that was used
    in the multi_output_model_v2.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def _dense_block(in_f, out_f, reg):
    return nn.Sequential(nn.Linear(in_f, out_f),
                         nn.BatchNorm1d(out_f, eps=reg),
                         nn.ReLU()
                         )

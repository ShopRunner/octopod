import numpy as np
from PIL import Image
from sklearn import preprocessing
import torch
from torch.utils.data import Dataset

from octopod.vision.config import cropped_transforms, full_img_transforms
from octopod.vision.helpers import center_crop_pil_image


class OctopodImageDataset(Dataset):
    """
    Load data specifically for use with a image models

    Parameters
    ----------
    x: pandas Series
        file paths to stored images
    y: list
        A list of dummy-encoded categories or strings
        For instance, y might be [0,1,2,0] for a 3 class problem with 4 samples,
        or strings which will be encoded using a sklearn label encoder
    transform: str or list of PyTorch transforms
        specifies how to preprocess the full image for a Octopod image model
        To use the built-in Octopod image transforms, use the strings: `train` or `val`
        To use custom transformations supply a list of PyTorch transforms
    crop_transform: str or list of PyTorch transforms
        specifies how to preprocess the center cropped image for a Octopod image model
        To use the built-in Octopod image transforms, use strings `train` or `val`
        To use custom transformations supply a list of PyTorch transforms
    """
    def __init__(self,
                 x,
                 y,
                 transform='train',
                 crop_transform='train'):
        self.x = x
        self.y = y
        self.label_encoder, self.label_mapping = self._encode_labels()

        if transform in ('train', 'val'):
            self.transform = full_img_transforms[transform]
        else:
            self.transform = transform

        if crop_transform in ('train', 'val'):
            self.crop_transform = cropped_transforms[crop_transform]
        else:
            self.crop_transform = crop_transform

    def __getitem__(self, index):
        """Return tuple of images as PyTorch tensors and and tensor of labels"""
        label = self.y[index]
        label = self.label_encoder.transform([label])[0]
        full_img = Image.open(self.x[index]).convert('RGB')

        cropped_img = center_crop_pil_image(full_img)

        full_img = self.transform(full_img)
        cropped_img = self.crop_transform(cropped_img)

        label = torch.from_numpy(np.array(label)).long()

        return {'full_img': full_img,
                'crop_img': cropped_img}, label

    def __len__(self):
        return len(self.x)

    def _encode_labels(self):
        """Encodes y labels using sklearn to create allow for string or numeric inputs"""
        le = preprocessing.LabelEncoder()
        le.fit(self.y)
        mapping_dict = dict(zip(le.transform(le.classes_), le.classes_))
        return le, mapping_dict


class OctopodImageDatasetMultiLabel(OctopodImageDataset):
    """
    Subclass of OctopodImageDataset used for multi-label tasks

    Parameters
    ----------
    x: pandas Series
        file paths to stored images
    y: list
        a list of lists of binary encoded categories or strings with length equal to number of
        classes in the multi-label task. For a 4 class multi-label task
        a sample list would be [1,0,0,1], A string example would be ['cat','dog'],
        (if the classes were ['cat','frog','rabbit','dog]), which will be encoded
        using a sklearn label encoder to [1,0,0,1].
    transform: str or list of PyTorch transforms
        specifies how to preprocess the full image for a Octopod image model
        To use the built-in Octopod image transforms, use the strings: `train` or `val`
        To use custom transformations supply a list of PyTorch transforms
    crop_transform: str or list of PyTorch transforms
        specifies how to preprocess the center cropped image for a Octopod image model
        To use the built-in Octopod image transforms, use strings `train` or `val`
        To use custom transformations supply a list of PyTorch transforms
    """

    def __getitem__(self, index):
        """Return tuple of images as PyTorch tensors and and tensor of labels"""
        label = self.y[index]
        label = list(self.label_encoder.transform([label])[0])
        full_img = Image.open(self.x[index]).convert('RGB')

        cropped_img = center_crop_pil_image(full_img)

        full_img = self.transform(full_img)
        cropped_img = self.crop_transform(cropped_img)

        label = torch.FloatTensor(label)

        return {'full_img': full_img,
                'crop_img': cropped_img}, label

    def _encode_labels(self):
        """Encodes y labels using sklearn to create allow for string or numeric inputs"""
        mlb = preprocessing.MultiLabelBinarizer()
        mlb.fit(self.y)
        mapping_dict = dict(zip(list(range(0, len(mlb.classes_))), mlb.classes_))

        return mlb, mapping_dict

import pandas as pd
import pytest
import torch.nn as nn
from torch.utils.data import DataLoader

from tonks import MultiDatasetLoader
from tonks.vision.dataset import TonksImageDataset


@pytest.fixture()
def test_no_loss_dict():
    return None


@pytest.fixture()
def test_missing_task():
    loss_dict = {'task1': 'bce_logits'}

    return loss_dict


@pytest.fixture()
def test_all_tasks_diff_losses():
    loss_dict = {'task1': 'bce_logits',
                 'task2': 'categorical_cross_entropy'}

    return loss_dict


@pytest.fixture()
def test_custom_loss_good():
    loss_dict = {'task1': {'loss': nn.CrossEntropyLoss(),
                           'preprocessing': None,
                           'final_layer': None,
                           'accuracy_pre_processing': None,
                           'is_multi_class': True},
                 'task2': 'categorical_cross_entropy'}

    return loss_dict


@pytest.fixture()
def test_custom_loss_missing_keys():
    loss_dict = {'task1': {'loss': nn.CrossEntropyLoss(),
                           'preprocessing': None},
                 'task2': 'categorical_cross_entropy'}

    return loss_dict


@pytest.fixture()
def test_custom_loss_invalid_keys():
    loss_dict = {'task1': {'INVALID_KEY_loss': nn.CrossEntropyLoss(),
                           'preprocessing': None,
                           'final_layer': None,
                           'accuracy_pre_processing': None,
                           'is_multi_class': True},
                 'task2': 'categorical_cross_entropy'}

    return loss_dict


@pytest.fixture()
def test_train_val_loaders():
    df = pd.DataFrame({'image_loc': ['sample_data/tonks.jpeg'],
                       'category': [1]})

    task1_dataset = TonksImageDataset(
        x=df['image_loc'],
        y=df['category'],
        transform='train',
        crop_transform='train'
    )

    task2_datset = TonksImageDataset(
        x=df['image_loc'],
        y=df['category'],
        transform='train',
        crop_transform='train'
    )

    dataloaders_dict = {
        'task1': DataLoader(task1_dataset, batch_size=1, shuffle=True, num_workers=2),
        'task2': DataLoader(task2_datset, batch_size=1, shuffle=True, num_workers=2),
    }

    dataset_loader = MultiDatasetLoader(loader_dict=dataloaders_dict)

    return dataset_loader

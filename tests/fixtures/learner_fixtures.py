import pandas as pd
import pytest
from torch.utils.data import DataLoader

from octopod import MultiDatasetLoader
from octopod.vision.dataset import OctopodImageDataset


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
def test_invalid_loss_name():
    loss_dict = {'task1': 'not_supported_loss_name',
                 'task2': 'categorical_cross_entropy'}

    return loss_dict


@pytest.fixture()
def test_no_acc_dict():
    return None


@pytest.fixture()
def test_acc_dict_missing_task():
    acc_dict = {'task1': 'multi_class_acc'}

    return acc_dict


@pytest.fixture()
def test_acc_dict_all_tasks_diff_losses():
    acc_dict = {'task1': 'multi_class_acc',
                'task2': 'multi_label_acc'}

    return acc_dict


@pytest.fixture()
def test_acc_dict_invalid_name():
    acc_dict = {'task1': 'not_supported_acc_name',
                'task2': 'multi_class_acc'}

    return acc_dict


@pytest.fixture()
def test_train_val_loaders():
    df = pd.DataFrame({'image_loc': ['sample_data/tonks.jpeg'],
                       'category': [1]})

    task1_dataset = OctopodImageDataset(
        x=df['image_loc'],
        y=df['category'],
        transform='train',
        crop_transform='train'
    )

    task2_datset = OctopodImageDataset(
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

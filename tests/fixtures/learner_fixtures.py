import pandas as pd
import pytest
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from octopod import MultiDatasetLoader
from octopod.text.dataset import OctopodTextDataset, OctopodTextDatasetMultiLabel
from octopod.vision.dataset import OctopodImageDataset, OctopodImageDatasetMultiLabel
from octopod.ensemble import OctopodEnsembleDataset, OctopodEnsembleDatasetMultiLabel

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

@pytest.fixture()
def mixed_labels():
    df_string = pd.DataFrame({'image_loc': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                              'category': ['cat', 'dog']})
    df_int = pd.DataFrame({'image_loc': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                           'category': [1, 2]})

    task1_dataset = OctopodImageDataset(
        x=df_string['image_loc'],
        y=df_string['category'],
        transform='train',
        crop_transform='train'
    )

    task2_datset = OctopodImageDataset(
        x=df_int['image_loc'],
        y=df_int['category'],
        transform='train',
        crop_transform='train'
    )

    dataloaders_dict = {
        'task1': DataLoader(task1_dataset, batch_size=1, shuffle=True, num_workers=2),
        'task2': DataLoader(task2_datset, batch_size=1, shuffle=True, num_workers=2),
    }

    dataset_loader = MultiDatasetLoader(loader_dict=dataloaders_dict)
    dataset_loader2 = MultiDatasetLoader(loader_dict=dataloaders_dict)

    return dataset_loader, dataset_loader2

@pytest.fixture()
def multi_label_mixed_labels():
    df_string = pd.DataFrame({'image_loc': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                              'category': [['cat', 'dog'], ['dog']]})
    df_int = pd.DataFrame({'image_loc': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                           'category': [[1, 2], [2]]})

    task1_dataset = OctopodImageDatasetMultiLabel(
        x=df_string['image_loc'],
        y=df_string['category'],
        transform='train',
        crop_transform='train'
    )

    task2_datset = OctopodImageDatasetMultiLabel(
        x=df_int['image_loc'],
        y=df_int['category'],
        transform='train',
        crop_transform='train'
    )

    dataloaders_dict = {
        'task1': DataLoader(task1_dataset, batch_size=1, shuffle=True, num_workers=2),
        'task2': DataLoader(task2_datset, batch_size=1, shuffle=True, num_workers=2),
    }

    dataset_loader = MultiDatasetLoader(loader_dict=dataloaders_dict)
    dataset_loader2 = MultiDatasetLoader(loader_dict=dataloaders_dict)

    return dataset_loader, dataset_loader2

@pytest.fixture()
def mixed_labels_text():
    df_string = pd.DataFrame({'fake_text': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                              'category': ['cat', 'dog']})
    df_int = pd.DataFrame({'fake_text': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                           'category': [1, 2]})

    max_seq_length = 128
    bert_tok = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)

    task1_dataset = OctopodTextDataset(
        x=df_string['fake_text'],
        y=df_string['category'],
        tokenizer=bert_tok,
        max_seq_length=max_seq_length
    )

    task2_datset = OctopodTextDataset(
        x=df_int['fake_text'],
        y=df_int['category'],
        tokenizer=bert_tok,
        max_seq_length=max_seq_length

    )

    dataloaders_dict = {
        'task1': DataLoader(task1_dataset, batch_size=1, shuffle=True, num_workers=2),
        'task2': DataLoader(task2_datset, batch_size=1, shuffle=True, num_workers=2),
    }

    dataset_loader = MultiDatasetLoader(loader_dict=dataloaders_dict)
    dataset_loader2 = MultiDatasetLoader(loader_dict=dataloaders_dict)

    return dataset_loader, dataset_loader2

@pytest.fixture()
def multi_label_mixed_labels_text():
    df_string = pd.DataFrame({'fake_text': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                              'category': [['cat', 'dog'], ['dog']]})
    df_int = pd.DataFrame({'fake_text': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                           'category': [[1, 2], [2]]})

    max_seq_length = 128
    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased',
                                             do_lower_case=True)

    task1_dataset = OctopodTextDatasetMultiLabel(
        x=df_string['fake_text'],
        y=df_string['category'],
        tokenizer=bert_tok,
        max_seq_length=max_seq_length
    )

    task2_datset = OctopodTextDatasetMultiLabel(
        x=df_int['fake_text'],
        y=df_int['category'],
        tokenizer=bert_tok,
        max_seq_length=max_seq_length

    )

    dataloaders_dict = {
        'task1': DataLoader(task1_dataset, batch_size=1, shuffle=True, num_workers=2),
        'task2': DataLoader(task2_datset, batch_size=1, shuffle=True, num_workers=2),
    }

    dataset_loader = MultiDatasetLoader(loader_dict=dataloaders_dict)
    dataset_loader2 = MultiDatasetLoader(loader_dict=dataloaders_dict)

    return dataset_loader, dataset_loader2

@pytest.fixture()
def mixed_labels_ensemble():
    df_string = pd.DataFrame({'image_loc': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                              'fake_text': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                              'category': ['cat', 'dog']})
    df_int = pd.DataFrame({'image_loc': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                           'fake_text': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                           'category': [1, 2]})

    max_seq_length = 128
    bert_tok = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)

    task1_dataset = OctopodEnsembleDataset(
        text_inputs=df_string['fake_text'],
        img_inputs=df_string['image_loc'],
        y=df_string['category'],
        tokenizer=bert_tok,
        max_seq_length=max_seq_length,
        transform='train',
        crop_transform='train'
    )

    task2_datset = OctopodEnsembleDataset(
        text_inputs=df_string['fake_text'],
        img_inputs=df_string['image_loc'],
        y=df_int['category'],
        tokenizer=bert_tok,
        max_seq_length=max_seq_length,
        transform='train',
        crop_transform='train'

    )

    dataloaders_dict = {
        'task1': DataLoader(task1_dataset, batch_size=1, shuffle=True, num_workers=2),
        'task2': DataLoader(task2_datset, batch_size=1, shuffle=True, num_workers=2),
    }

    dataset_loader = MultiDatasetLoader(loader_dict=dataloaders_dict)
    dataset_loader2 = MultiDatasetLoader(loader_dict=dataloaders_dict)

    return dataset_loader, dataset_loader2

@pytest.fixture()
def mixed_labels_ensemble():
    df_string = pd.DataFrame({'image_loc': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                              'fake_text': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                              'category': [['cat', 'dog'], ['dog']]})
    df_int = pd.DataFrame({'image_loc': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                           'fake_text': ['sample_data/tonks.jpeg', 'sample_data/tonks.jpeg'],
                           'category': [[1, 2], [2]]})

    max_seq_length = 128
    bert_tok = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)

    task1_dataset = OctopodEnsembleDatasetMultiLabel(
        text_inputs=df_string['fake_text'],
        img_inputs=df_string['image_loc'],
        y=df_string['category'],
        tokenizer=bert_tok,
        max_seq_length=max_seq_length,
        transform='train',
        crop_transform='train'
    )

    task2_datset = OctopodEnsembleDatasetMultiLabel(
        text_inputs=df_string['fake_text'],
        img_inputs=df_string['image_loc'],
        y=df_int['category'],
        tokenizer=bert_tok,
        max_seq_length=max_seq_length,
        transform='train',
        crop_transform='train'

    )

    dataloaders_dict = {
        'task1': DataLoader(task1_dataset, batch_size=1, shuffle=True, num_workers=2),
        'task2': DataLoader(task2_datset, batch_size=1, shuffle=True, num_workers=2),
    }

    dataset_loader = MultiDatasetLoader(loader_dict=dataloaders_dict)
    dataset_loader2 = MultiDatasetLoader(loader_dict=dataloaders_dict)

    return dataset_loader, dataset_loader2

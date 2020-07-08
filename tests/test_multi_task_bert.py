import copy
from pathlib import Path
import shutil
import tempfile

import torch

from octopod.text.models import BertForMultiTaskClassification


def test_exporting_and_loading_works_correctly():
    task_dict = {'fake_attribute': 10}

    model = BertForMultiTaskClassification.from_pretrained(
        'bert-base-uncased',
        new_task_dict=task_dict
    )

    model_id = 27

    test_dir = Path() / tempfile.mkdtemp()

    model.export(test_dir, model_id)

    new_model = BertForMultiTaskClassification.from_pretrained(
        'bert-base-uncased',
        pretrained_task_dict=task_dict
    )

    new_model.load_state_dict(torch.load(
        test_dir / f'multi_task_bert_{model_id}.pth',
        map_location=lambda storage,
        loc: storage
    ))
    shutil.rmtree(test_dir)

    for original_val, new_val in zip(model.state_dict().values(), new_model.state_dict().values()):
        assert torch.equal(original_val, new_val)


def test_exporting_does_not_modify_original():
    task_dict = {'fake_attribute': 10}

    model = BertForMultiTaskClassification.from_pretrained(
        'bert-base-uncased',
        new_task_dict=task_dict
    )

    model_copy = copy.deepcopy(model)

    model_id = 27

    test_dir = tempfile.mkdtemp()
    model.export(test_dir, model_id)
    shutil.rmtree(test_dir)

    for orig_item, copy_item in zip(model.state_dict().items(), model_copy.state_dict().items()):
        assert orig_item[0] == copy_item[0]
        assert torch.equal(orig_item[1], copy_item[1])

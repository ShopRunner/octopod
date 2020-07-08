import copy
from pathlib import Path
import shutil
import tempfile

import torch

from octopod.ensemble.models import BertResnetEnsembleForMultiTaskClassification


def test_exporting_and_loading_works_correctly():
    image_task_dict = {
        'task1_task2': {
            'fake_attribute1': 10,
            'fake_attribute2': 3
        },
        'task3': {
            'fake_attribute3': 4
        }
    }

    model = BertResnetEnsembleForMultiTaskClassification(
        image_task_dict=image_task_dict
    )

    model_id = 27

    test_dir = Path() / tempfile.mkdtemp()

    model.export(test_dir, model_id)

    new_model = BertResnetEnsembleForMultiTaskClassification(
        image_task_dict=image_task_dict
    )

    new_model.load_state_dict(torch.load(
        test_dir / f'multi_task_ensemble_{model_id}.pth',
        map_location=lambda storage,
        loc: storage
    ))
    shutil.rmtree(test_dir)

    for original_val, new_val in zip(model.state_dict().values(), new_model.state_dict().values()):
        assert torch.equal(original_val, new_val)


def test_exporting_does_not_modify_original():
    image_task_dict = {
        'task1_task2': {
            'fake_attribute1': 10,
            'fake_attribute2': 3
        },
        'task3': {
            'fake_attribute3': 4
        }
    }

    model = BertResnetEnsembleForMultiTaskClassification(
        image_task_dict=image_task_dict
    )

    model_copy = copy.deepcopy(model)

    model_id = 27

    test_dir = tempfile.mkdtemp()
    model.export(test_dir, model_id)
    shutil.rmtree(test_dir)

    for orig_item, copy_item in zip(model.state_dict().items(), model_copy.state_dict().items()):
        assert orig_item[0] == copy_item[0]
        assert torch.equal(orig_item[1], copy_item[1])

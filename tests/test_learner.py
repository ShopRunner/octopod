import pytest

from tonks import MultiInputMultiTaskLearner
from tonks.vision.models import ResnetForMultiTaskClassification


def test_learner_with_none_loss_dict(test_no_loss_dict, test_train_val_loaders):
    loss_function_dict = test_no_loss_dict
    task_dict = {'task1': 1, 'task2': 1}

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    learn = MultiInputMultiTaskLearner(model,
                                       test_train_val_loaders,
                                       test_train_val_loaders,
                                       task_dict,
                                       loss_function_dict)

    assert str(learn.loss_function_dict['task1']['loss']) == 'CrossEntropyLoss()'
    assert str(learn.loss_function_dict['task2']['loss']) == 'CrossEntropyLoss()'


def test_learner_with_missing_loss(test_missing_task,
                                   test_train_val_loaders):
    loss_function_dict = test_missing_task
    task_dict = {'task1': 1, 'task2': 1}

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    with pytest.raises(Exception) as e:
        MultiInputMultiTaskLearner(model,
                                   test_train_val_loaders,
                                   test_train_val_loaders,
                                   task_dict,
                                   loss_function_dict)
    assert ('must provide either valid loss names for ALL tasks' in str(e.value)) is True


def test_learner_with_bce_and_cce(test_all_tasks_diff_losses,
                                  test_train_val_loaders):
    loss_function_dict = test_all_tasks_diff_losses
    task_dict = {'task1': 1, 'task2': 1}

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    learn = MultiInputMultiTaskLearner(model,
                                       test_train_val_loaders,
                                       test_train_val_loaders,
                                       task_dict,
                                       loss_function_dict)

    assert str(learn.loss_function_dict['task1']['loss']) == 'BCEWithLogitsLoss()'
    assert str(learn.loss_function_dict['task2']['loss']) == 'CrossEntropyLoss()'

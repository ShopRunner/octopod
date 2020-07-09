import pytest

from octopod import MultiInputMultiTaskLearner
from octopod.vision.models import ResnetForMultiTaskClassification


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

    assert str(learn.loss_function_dict['task1']) == 'CrossEntropyLoss()'
    assert str(learn.loss_function_dict['task2']) == 'CrossEntropyLoss()'


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
    assert ('make sure all tasks are contained in the' in str(e.value)) is True


def test_learner_with_invalid_loss_name(test_invalid_loss_name,
                                        test_train_val_loaders):
    loss_function_dict = test_invalid_loss_name
    task_dict = {'task1': 1, 'task2': 1}

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    with pytest.raises(Exception) as e:
        MultiInputMultiTaskLearner(model,
                                   test_train_val_loaders,
                                   test_train_val_loaders,
                                   task_dict,
                                   loss_function_dict)
    assert ('Found invalid loss function:' in str(e.value)) is True


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

    assert str(learn.loss_function_dict['task1']) == 'BCEWithLogitsLoss()'
    assert str(learn.loss_function_dict['task2']) == 'CrossEntropyLoss()'


def test_learner_with_none_acc_dict(test_no_acc_dict, test_train_val_loaders):
    acc_function_dict = test_no_acc_dict
    task_dict = {'task1': 1, 'task2': 1}

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    learn = MultiInputMultiTaskLearner(model,
                                       test_train_val_loaders,
                                       test_train_val_loaders,
                                       task_dict,
                                       metric_function_dict=acc_function_dict)

    assert 'multi_class_accuracy' in str(learn.metric_function_dict['task1'])
    assert 'multi_class_accuracy' in str(learn.metric_function_dict['task2'])


def test_learner_with_missing_acc(test_acc_dict_missing_task,
                                  test_train_val_loaders):
    acc_function_dict = test_acc_dict_missing_task
    task_dict = {'task1': 1, 'task2': 1}

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    with pytest.raises(Exception) as e:
        MultiInputMultiTaskLearner(model,
                                   test_train_val_loaders,
                                   test_train_val_loaders,
                                   task_dict,
                                   metric_function_dict=acc_function_dict)
    assert ('make sure all tasks are contained in the' in str(e.value)) is True


def test_learner_with_invalid_acc_name(test_acc_dict_invalid_name,
                                       test_train_val_loaders):
    acc_function_dict = test_acc_dict_invalid_name
    task_dict = {'task1': 1, 'task2': 1}

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    with pytest.raises(Exception) as e:
        MultiInputMultiTaskLearner(model,
                                   test_train_val_loaders,
                                   test_train_val_loaders,
                                   task_dict,
                                   metric_function_dict=acc_function_dict)
    assert ('Found invalid metric function:' in str(e.value)) is True


def test_learner_with_multi_class_and_label_acc(test_acc_dict_all_tasks_diff_losses,
                                                test_train_val_loaders):
    acc_function_dict = test_acc_dict_all_tasks_diff_losses
    task_dict = {'task1': 1, 'task2': 1}

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    learn = MultiInputMultiTaskLearner(model,
                                       test_train_val_loaders,
                                       test_train_val_loaders,
                                       task_dict,
                                       metric_function_dict=acc_function_dict)

    assert 'multi_class_accuracy' in str(learn.metric_function_dict['task1'])
    assert 'multi_label_accuracy' in str(learn.metric_function_dict['task2'])

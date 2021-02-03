import pytest

from octopod import MultiInputMultiTaskLearner
from octopod.ensemble import BertResnetEnsembleForMultiTaskClassification
from octopod.text.models.multi_task_bert import BertForMultiTaskClassification
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


def test_learner_w_string_and_int_label_datasets_success_image(mixed_labels,
                                                               task_dicts_string_labels):

    train_dataset_loader, val_dataset_loader = mixed_labels

    task_dict, _ = task_dicts_string_labels

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    learn = MultiInputMultiTaskLearner(model,
                                       train_dataset_loader,
                                       val_dataset_loader,
                                       task_dict)

    assert learn.train_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.train_dataloader.label_mappings['task2'] == {0: 1, 1: 2}

    assert learn.val_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.val_dataloader.label_mappings['task2'] == {0: 1, 1: 2}


def test_learner_w_string_datasets_not_matching_categories_image(mixed_labels,
                                                                 task_dicts_string_labels):
    train_dataset_loader, val_dataset_loader = mixed_labels
    val_dataset_loader.label_mappings['task1'] = {0: 'blue', 3: 'green'}

    task_dict, _ = task_dicts_string_labels

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    with pytest.raises(Exception) as e:
        MultiInputMultiTaskLearner(model,
                                   train_dataset_loader,
                                   val_dataset_loader,
                                   task_dict)
    assert ('Mapping mismatch in task1 task. Check that all categories' in str(e.value)) is True


def test_learner_w_multilabel_string_and_int_label_datasets_image(multi_label_mixed_labels,
                                                                  task_dicts_string_labels,
                                                                  multi_label_loss_acc_dicts):
    train_dataset_loader, val_dataset_loader = multi_label_mixed_labels

    task_dict, _ = task_dicts_string_labels
    loss_dict, acc_dict = multi_label_loss_acc_dicts

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)
    learn = MultiInputMultiTaskLearner(model,
                                       train_dataset_loader,
                                       val_dataset_loader,
                                       task_dict,
                                       loss_function_dict=loss_dict,
                                       metric_function_dict=acc_dict)

    assert learn.train_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.train_dataloader.label_mappings['task2'] == {0: 1, 1: 2}

    assert learn.val_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.val_dataloader.label_mappings['task2'] == {0: 1, 1: 2}


def test_w_string_multilabel_not_matching_categories_multilabel_image(multi_label_mixed_labels,
                                                                      task_dicts_string_labels,
                                                                      multi_label_loss_acc_dicts):
    train_dataset_loader, val_dataset_loader = multi_label_mixed_labels
    val_dataset_loader.label_mappings['task1'] = {0: 'blue', 3: 'green'}

    task_dict, _ = task_dicts_string_labels
    loss_dict, acc_dict = multi_label_loss_acc_dicts

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    with pytest.raises(Exception) as e:
        MultiInputMultiTaskLearner(model,
                                   train_dataset_loader,
                                   val_dataset_loader,
                                   task_dict,
                                   loss_function_dict=loss_dict,
                                   metric_function_dict=acc_dict)
    assert ('Mapping mismatch in task1 task. Check that all categories' in str(e.value)) is True


def test_learner_w_string_and_int_label_datasets_success_text(mixed_labels_text,
                                                              task_dicts_string_labels):
    train_dataset_loader, val_dataset_loader = mixed_labels_text

    task_dict, _ = task_dicts_string_labels

    model = BertForMultiTaskClassification.from_pretrained('bert-base-uncased',
                                                           new_task_dict=task_dict)

    learn = MultiInputMultiTaskLearner(model,
                                       train_dataset_loader,
                                       val_dataset_loader,
                                       task_dict)

    assert learn.train_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.train_dataloader.label_mappings['task2'] == {0: 1, 1: 2}

    assert learn.val_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.val_dataloader.label_mappings['task2'] == {0: 1, 1: 2}


def test_learner_w_string_datasets_not_matching_categoies_text(mixed_labels_text,
                                                               task_dicts_string_labels):
    train_dataset_loader, val_dataset_loader = mixed_labels_text
    val_dataset_loader.label_mappings['task1'] = {0: 'blue', 3: 'green'}

    task_dict, _ = task_dicts_string_labels

    model = BertForMultiTaskClassification.from_pretrained('bert-base-uncased',
                                                           new_task_dict=task_dict)

    with pytest.raises(Exception) as e:
        MultiInputMultiTaskLearner(model,
                                   train_dataset_loader,
                                   val_dataset_loader,
                                   task_dict)
    assert ('Mapping mismatch in task1 task. Check that all categories' in str(e.value)) is True


def test_learner_w_multilabel_string_and_int_label_datasets_text(multi_label_mixed_labels_text,
                                                                 task_dicts_string_labels,
                                                                 multi_label_loss_acc_dicts):
    train_dataset_loader, val_dataset_loader = multi_label_mixed_labels_text

    task_dict, _ = task_dicts_string_labels
    loss_dict, acc_dict = multi_label_loss_acc_dicts

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)
    learn = MultiInputMultiTaskLearner(model,
                                       train_dataset_loader,
                                       val_dataset_loader,
                                       task_dict,
                                       loss_function_dict=loss_dict,
                                       metric_function_dict=acc_dict)

    assert learn.train_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.train_dataloader.label_mappings['task2'] == {0: 1, 1: 2}

    assert learn.val_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.val_dataloader.label_mappings['task2'] == {0: 1, 1: 2}


def test_w_string_multilabel_not_matching_categories_multilabel_text(multi_label_mixed_labels_text,
                                                                     task_dicts_string_labels,
                                                                     multi_label_loss_acc_dicts):
    train_dataset_loader, val_dataset_loader = multi_label_mixed_labels_text
    val_dataset_loader.label_mappings['task1'] = {0: 'blue', 3: 'green'}

    task_dict, _ = task_dicts_string_labels
    loss_dict, acc_dict = multi_label_loss_acc_dicts

    model = ResnetForMultiTaskClassification(
        new_task_dict=task_dict)

    with pytest.raises(Exception) as e:
        MultiInputMultiTaskLearner(model,
                                   train_dataset_loader,
                                   val_dataset_loader,
                                   task_dict,
                                   loss_function_dict=loss_dict,
                                   metric_function_dict=acc_dict)
    assert ('Mapping mismatch in task1 task. Check that all categories' in str(e.value)) is True


def test_learner_w_string_and_int_label_datasets_success_ensemble(mixed_labels_ensemble,
                                                                  task_dicts_string_labels):
    train_dataset_loader, val_dataset_loader = mixed_labels_ensemble
    task_dict, image_task_dict = task_dicts_string_labels

    model = BertResnetEnsembleForMultiTaskClassification(image_task_dict=image_task_dict)

    learn = MultiInputMultiTaskLearner(model,
                                       train_dataset_loader,
                                       val_dataset_loader,
                                       task_dict)

    assert learn.train_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.train_dataloader.label_mappings['task2'] == {0: 1, 1: 2}

    assert learn.val_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.val_dataloader.label_mappings['task2'] == {0: 1, 1: 2}


def test_learner_w_string_datasets_not_matching_categoies_ensemble(mixed_labels_ensemble,
                                                                   task_dicts_string_labels):
    train_dataset_loader, val_dataset_loader = mixed_labels_ensemble
    task_dict, image_task_dict = task_dicts_string_labels

    val_dataset_loader.label_mappings['task1'] = {0: 'blue', 3: 'green'}

    model = BertResnetEnsembleForMultiTaskClassification(image_task_dict=image_task_dict)

    with pytest.raises(Exception) as e:
        MultiInputMultiTaskLearner(model,
                                   train_dataset_loader,
                                   val_dataset_loader,
                                   task_dict)
    assert ('Mapping mismatch in task1 task. Check that all categories' in str(e.value)) is True


def test_learner_w_multilabel_string_and_int_label_datasets_ensemble(mixed_multi_label_ensemble,
                                                                     task_dicts_string_labels,
                                                                     multi_label_loss_acc_dicts):
    train_dataset_loader, val_dataset_loader = mixed_multi_label_ensemble
    task_dict, image_task_dict = task_dicts_string_labels
    loss_dict, acc_dict = multi_label_loss_acc_dicts

    model = BertResnetEnsembleForMultiTaskClassification(image_task_dict=image_task_dict)

    learn = MultiInputMultiTaskLearner(model,
                                       train_dataset_loader,
                                       val_dataset_loader,
                                       task_dict,
                                       loss_function_dict=loss_dict,
                                       metric_function_dict=acc_dict)

    assert learn.train_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.train_dataloader.label_mappings['task2'] == {0: 1, 1: 2}

    assert learn.val_dataloader.label_mappings['task1'] == {0: 'cat', 1: 'dog'}
    assert learn.val_dataloader.label_mappings['task2'] == {0: 1, 1: 2}


def test_w_string_multilabel_not_match_categories_multilabel_ensemble(mixed_multi_label_ensemble,
                                                                      task_dicts_string_labels,
                                                                      multi_label_loss_acc_dicts):
    train_dataset_loader, val_dataset_loader = mixed_multi_label_ensemble
    val_dataset_loader.label_mappings['task1'] = {0: 'blue', 3: 'green'}

    task_dict, image_task_dict = task_dicts_string_labels
    loss_dict, acc_dict = multi_label_loss_acc_dicts

    model = BertResnetEnsembleForMultiTaskClassification(image_task_dict=image_task_dict)

    with pytest.raises(Exception) as e:
        MultiInputMultiTaskLearner(model,
                                   train_dataset_loader,
                                   val_dataset_loader,
                                   task_dict,
                                   loss_function_dict=loss_dict,
                                   metric_function_dict=acc_dict)
    assert ('Mapping mismatch in task1 task. Check that all categories' in str(e.value)) is True

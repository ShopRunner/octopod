from tonks.learner_utils.loss_utils_config import (DEFAULT_ACC_DICT,
                                                   DEFAULT_LOSSES_DICT,
                                                   )


def _get_loss_functions(loss_function_dict, tasks):
    """
    Takes in dictionary of tasks and their loss configurations
    if it is a supported loss, CrossEntropyLoss() or BCELogits(), or
    a custom loss function.

    Parameters
    ----------
    loss_function_dict: dictionary
        keys are the tasks, values are either strings for a supported
        loss function or a custom loss function.
        If the 'loss_function_dict' is set to None, then all tasks will be set to have
        `categorical_cross_entropy` loss. Users can also provide custom loss functions.
    tasks: list
        list of tasks that that are being used in a Tonks MultiTaskLearner, this list
        of tasks is used to check that all the tasks are inside of the loss_function_dict

    """
    loss_dict = loss_function_dict
    processed_loss_function_dict = {}

    if loss_dict is None:
loss_dict = {task: 'categorical_cross_entropy' for task in tasks}

    _check_for_all_tasks(loss_dict, tasks, 'loss')

    for key, value in loss_dict.items():

        if isinstance(value, str):
            try:
                processed_loss_function_dict[key] = DEFAULT_LOSSES_DICT[val]
            except:

                raise Exception('Found invalid loss function: {}. '
                                'Valid losses are categorical_cross_entropy '
                                'or bce_logits. Check that all tasks loss names are '
                                'valid.'.format(value))
        else:
            processed_loss_function_dict[key] = value

    return processed_loss_function_dict


def _get_acc_functions(acc_function_dict, tasks):
    """
    Takes in dictionary of tasks and a string cooresponding to an accuracy
    function configuration or custom accuracy function.
    Current supported accuracy functions are `multi_class_acc`
    for multi-class tasks and an accuracy score and `multi_label_acc` for multi-label
    tasks with an accuracy score.

    Parameters
    ----------
    acc_function_dict: dictionary
        keys are the tasks, values are either strings for a supported
        loss function. Available options are `multi_class_acc` or `multi_label_acc`.
        If the 'loss_function_dict' is set to None, then all tasks will be set to have
        `multi_class_acc`. Users can also add in custom accuracy functions
    tasks: list
        list of tasks that that are being used in a Tonks MultiTaskLearner, this list
        of tasks is used to check that all the tasks are inside of the loss_function_dict

    """
    acc_dict = acc_function_dict
    processed_acc_function_dict = {}

    if acc_dict is None:
        acc_dict = {}
        for task in tasks:
            acc_dict[task] = 'multi_class_acc'

    _check_for_all_tasks(acc_dict, tasks, 'accuracy')

    for key, value in acc_dict.items():
        if isinstance(value, str) is True:
            if value == 'multi_class_acc':
                processed_acc_function_dict[key] = DEFAULT_ACC_DICT['multi_class_acc']

            elif value == 'multi_label_acc':
                processed_acc_function_dict[key] = DEFAULT_ACC_DICT['multi_label_acc']

            else:
                raise Exception('Found invalid accuracy function: {}. '
                                'Valid accuracy functions are multi_class_acc '
                                'or multi_label_acc. Check that all tasks accuracy '
                                'functions names are valid.'.format(value))
        else:
            processed_acc_function_dict[key] = value

    return processed_acc_function_dict


def _check_for_all_tasks(input_dict, tasks, input_str):
    if not all(task in input_dict for task in tasks):
        missing_tasks = set(tasks)-input_dict.keys()

        raise Exception(f'make sure all tasks are contained in the {input_str} dictionary '
                        f'missing tasks are {missing_tasks}')

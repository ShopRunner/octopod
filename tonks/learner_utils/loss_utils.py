from tonks.learner_utils.loss_utils_config import (DEFAULT_ACC_DICT,
                                                   DEFAULT_LOSSES_DICT,
                                                   )


def _get_loss_functions(loss_function_dict, tasks):
    """
    Takes in dictionary of tasks and their loss configurations
    if it is a supported loss, CrossEntropyLoss() or BCELogits()
    then it auto configures other specifications

    Parameters
    ----------
    loss_function_dict: dictionary
        keys are the tasks, values are either strings for a supported
        loss function or the parts for a custom loss function. nessessary
        keys are 'loss' and 'acc_func'.
        If the 'loss_function_dict' is set to None, then all tasks will be set to have
        `categorical_cross_entropy` loss.
    tasks: list
        list of tasks that that are being used in a Tonks MultiTaskLearner, this list
        of tasks is used to check that all the tasks are inside of the loss_function_dict

    """
    loss_dict = loss_function_dict
    processed_loss_function_dict = {}

    if loss_dict is None:
        loss_dict = {}
        for task in tasks:
            loss_dict[task] = 'categorical_cross_entropy'

    if not all(task in loss_dict for task in tasks):
        missing_tasks = set(tasks)-loss_dict.keys()

        raise Exception('must provide either valid loss names for ALL tasks '
                        f'missing tasks are {missing_tasks}')

    for key, value in loss_dict.items():
        if value == 'categorical_cross_entropy':
            processed_loss_function_dict[key] = DEFAULT_LOSSES_DICT['categorical_cross_entropy']

        elif value == 'bce_logits':
            processed_loss_function_dict[key] = DEFAULT_LOSSES_DICT['bce_logits']

        else:

            raise Exception('Found invalid loss function: {}. '
                            'Valid losses are categorical_cross_entropy '
                            'or bce_logits. Check that all tasks loss names are '
                            'valid.'.format(value))

    return processed_loss_function_dict


def _get_acc_functions(acc_function_dict, tasks):
    """
    Takes in dictionary of tasks and their loss configurations
    if it is a supported loss, CrossEntropyLoss() or BCELogits()
    then it auto configures other specifications

    Parameters
    ----------
    acc_function_dict: dictionary
        keys are the tasks, values are either strings for a supported
        loss function. Available options are `multi_class_acc` or `multi_label_acc`.
        If the 'loss_function_dict' is set to None, then all tasks will be set to have
        `multi_class_acc`.
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

    if not all(task in acc_dict for task in tasks):
        missing_tasks = set(tasks)-acc_dict.keys()

        raise Exception('must provide valid accuracy function names for ALL tasks '
                        f'missing tasks are {missing_tasks}')

    for key, value in acc_dict.items():
        if value == 'multi_class_acc':
            processed_acc_function_dict[key] = DEFAULT_ACC_DICT['multi_class_acc']

        elif value == 'multi_label_acc':
            processed_acc_function_dict[key] = DEFAULT_ACC_DICT['multi_label_acc']

        else:

            raise Exception('Found invalid accuracy function: {}. '
                            'Valid accuracy functions are multi_class_acc '
                            'or multi_label_acc. Check that all tasks accuracy '
                            'functions names are valid.'.format(value))

    return processed_acc_function_dict

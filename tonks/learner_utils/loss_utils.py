from tonks.learner_utils.loss_utils_config import DEFAULT_LOSSES_DICT, VALID_LOSS_KEYS


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
        keys are 'loss', 'is_multi_class','final_layer', and 'accuracy_pre_processing'.
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

            raise Exception(
                            'Found invalid loss function: {}. '
                            'Valid losses are categorical_cross_entropy '
                            'or bce_logits. Check that all tasks loss names are valid.'.format(value)
                        )

    return processed_loss_function_dict

from tonks.learner_utils.loss_utils_config import DEFAULT_LOSSES_DICT, VALID_LOSS_KEYS


def _get_loss_functions(loss_function_dict, tasks):
    """
    Takes in dictionary of tasks and their loss configurations
    if it is a supported loss, CrossEntropyLoss() or BCELogits()
    then it auto configures. If it is a custom loss then the custom
    loss is checked to make sure it contains nessessary keys.

    Parameters
    ----------
    loss_function_dict: dictionary
        keys are the tasks, values are either strings for a supported
        loss function or the parts for a custom loss function. nessessary
        keys are 'loss', 'is_multi_class','final_layer', and 'accuracy_pre_processing'.
    tasks: list
        list of tasks that that are being used in a Tonks MultiTaskLearner, this list
        of tasks is used to check that all the tasks are inside of the loss_function_dict

    Notes
    -----
    For tasks where there is not a final layer activation required on the final layer
    or where an accuracy score is not relevant `final_layer` and or `accuracy_pre_processing`
    can be set to None. If 'accuracy_pre_processing' is set to None than accuracy will show as
    `N/A`.
    """
    loss_dict = loss_function_dict
    processed_loss_function_dict = {}

    if loss_dict is None:
        loss_dict = {}
        for task in tasks:
            loss_dict[task] = 'categorical_cross_entropy'

    if not all(task in loss_dict for task in tasks):
        missing_tasks = set(tasks)-loss_dict.keys()

        raise Exception('must provide either valid loss names or '
                        'loss functions and preprocessing steps for ALL tasks '
                        f'missing tasks are {missing_tasks}')

    for key, value in loss_dict.items():
        if value == 'categorical_cross_entropy':
            processed_loss_function_dict[key] = DEFAULT_LOSSES_DICT['categorical_cross_entropy']

        elif value == 'bce_logits':
            processed_loss_function_dict[key] = DEFAULT_LOSSES_DICT['bce_logits']
        else:
            try:
                processed_loss_function_dict[key] = (
                    {'accuracy_pre_processing': value['accuracy_pre_processing'],
                     'final_layer': value['final_layer'],
                     'is_multi_class': value['is_multi_class'],
                     'loss': value['loss'],
                     'preprocessing': value['preprocessing'],
                     }
                )
            except KeyError:
                checking_missing_keys = set(VALID_LOSS_KEYS) - value.keys()
                raise Exception('found invalid or missing keys, check loss dictionary keys for '
                                f'{key} task, invalid or missing keys are {checking_missing_keys}')

    return processed_loss_function_dict

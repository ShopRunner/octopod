import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F


def _multi_class_accuracy_preprocess(x):
    return torch.max(torch.tensor(x), 1)[1].numpy()


def _multi_label_accuracy_preprocess(x):
    return np.round(x, 0)


def multi_class_accuracy(y_true, y_raw_preds):
    """
    Takes in raw outputs from Octopod task heads and outputs an accuracy metric
    and the processed predictions after a softmax as been applied

    Parameters
    ----------
    y_true: np.array
        Target labels for a specific task for the predicted samples in `y_raw_preds`
    y_raw_preds: np.array
        predicted values for the validation set for a specific task

    Returns
    -------
    acc: float
        Output of a sklearn accuracy score function
    y_preds: np.array
        array of predicted values where a softmax has been applied

    """
    tensor_y_pred = torch.from_numpy(y_raw_preds)
    y_preds = F.softmax(tensor_y_pred, dim=1).numpy()
    task_preds = (
        _multi_class_accuracy_preprocess(y_preds)
    )
    acc = accuracy_score(y_true, task_preds)
    return acc, y_preds


def multi_label_accuracy(y_true, y_raw_preds):
    """
    Takes in raw outputs from Octopod task heads and outputs an accuracy metric
    and the processed predictions after a sigmoid as been applied

    Parameters
    ----------
    y_true: np.array
        Target labels for a specific task for the predicted samples in `y_raw_preds`
    y_raw_preds: np.array
        predicted values for the validation set for a specific task

    Returns
    -------
    acc: float
        Output of a sklearn accuracy score function
    y_preds: np.array
        array of predicted values where a sigmoid has been applied
    """
    tensor_y_pred = torch.from_numpy(y_raw_preds)
    y_preds = torch.sigmoid(tensor_y_pred).numpy()
    task_preds = (
        _multi_label_accuracy_preprocess(y_preds)
    )
    acc = accuracy_score(y_true, task_preds)
    return acc, y_preds

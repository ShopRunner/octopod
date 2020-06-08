import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score


def _softmax_final_layer(x):
    return F.softmax(x, dim=1)


def _multi_class_accuracy_preprocess(x):
    return torch.max(torch.tensor(x), 1)[1].numpy()


def _multi_label_accuracy_preprocess(x):
    return np.round(x, 0)


def _multi_class_accuracy(y_true, preds):
    tensor_y_pred = torch.from_numpy(preds)
    y_preds = _softmax_final_layer((tensor_y_pred)).numpy()
    task_preds = (
                _multi_class_accuracy_preprocess(y_preds)
            )
    acc = accuracy_score(y_true, task_preds)
    return acc, y_preds


def _multi_label_accuracy(y_true, preds):
    tensor_y_pred = torch.from_numpy(preds)
    y_preds = torch.sigmoid(tensor_y_pred).numpy()
    task_preds = (
                _multi_label_accuracy_preprocess(y_preds)
            )
    acc = accuracy_score(y_true, task_preds)
    return acc, y_preds


DEFAULT_LOSSES_DICT = {
    'categorical_cross_entropy': {'acc_func':_multi_class_accuracy,
                                  'loss': nn.CrossEntropyLoss(),
                                  },
    'bce_logits': {'acc_func': _multi_class_accuracy,
                   'loss': nn.BCEWithLogitsLoss(),
                   }
}

VALID_LOSS_KEYS = DEFAULT_LOSSES_DICT['categorical_cross_entropy'].keys()

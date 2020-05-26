import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cross_entropy_preprocessing(x):
    return x.type(torch.LongTensor)


def _bce_logits_preprocessing(x):
    #x = x.unsqueeze(1)
    return x.type(torch.FloatTensor)


def _softmax_final_layer(x):
    return F.softmax(x, dim=1)


def _multi_class_accuracy_preprocess(x):
    return torch.max(torch.tensor(x), 1)[1].numpy()


def _multi_label_accuracy_preprocess(x):
    return np.round(x, 0)


DEFAULT_LOSSES_DICT = {
    'categorical_cross_entropy': {'accuracy_pre_processing': _multi_class_accuracy_preprocess,
                                  'final_layer': _softmax_final_layer,
                                  'is_multi_class': True,
                                  'loss': nn.CrossEntropyLoss(),
                                  'preprocessing': _cross_entropy_preprocessing,
                                  },
    'bce_logits': {'accuracy_pre_processing': _multi_label_accuracy_preprocess,
                   'final_layer': torch.sigmoid,
                   'is_multi_class': False,
                   'loss': nn.BCEWithLogitsLoss(),
                   'preprocessing': _bce_logits_preprocessing,
                   }
}

VALID_LOSS_KEYS = DEFAULT_LOSSES_DICT['categorical_cross_entropy'].keys()

import torch.nn as nn

from octopod.learner_utils.metrics_utils import multi_class_accuracy, multi_label_accuracy

DEFAULT_LOSSES_DICT = {
    'categorical_cross_entropy': nn.CrossEntropyLoss(),
    'bce_logits': nn.BCEWithLogitsLoss(),
}


DEFAULT_METRIC_DICT = {
    'multi_class_acc': multi_class_accuracy,
    'multi_label_acc': multi_label_accuracy,
}

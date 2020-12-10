import copy
from pathlib import Path

import torch
import torch.nn as nn
from transformers.modeling_bert import BertModel, BertPreTrainedModel


class BertForMultiTaskClassification(BertPreTrainedModel):
    """
    PyTorch BERT class for multitask learning. This model allows you to load
    in some pretrained tasks in addition to creating new ones.

    Examples
    --------
    To instantiate a completely new instance of BertForMultiTaskClassification
    and load the weights into this architecture you can use the `from_pretrained`
    method of the base class by specifying the name of the weights to load, e.g.::

        model = BertForMultiTaskClassification.from_pretrained(
            'bert-base-uncased',
            new_task_dict=new_task_dict
        )

        # DO SOME TRAINING

        model.save(SOME_FOLDER, SOME_MODEL_ID)

    To instantiate an instance of BertForMultiTaskClassification that has layers for
    pretrained tasks and new tasks, you would do the following::

        model = BertForMultiTaskClassification.from_pretrained(
            'bert-base-uncased',
            pretrained_task_dict=pretrained_task_dict,
            new_task_dict=new_task_dict
        )

        model.load(SOME_FOLDER, SOME_MODEL_DICT)

        # DO SOME TRAINING

    Parameters
    ----------
    config: json file
        Defines the BERT model architecture.
        Note: you will most likely be instantiating the class with the `from_pretrained` method
        so you don't need to come up with your own config.
    pretrained_task_dict: dict
        dictionary mapping each pretrained task to the number of labels it has
    new_task_dict: dict
        dictionary mapping each new task to the number of labels it has
    dropout: float
        dropout percentage for Dropout layer
    """
    def __init__(self, config, pretrained_task_dict=None, new_task_dict=None, dropout=1e-1):
        super(BertForMultiTaskClassification, self).__init__(config)
        self.bert = BertModel(config)

        self.dropout = torch.nn.Dropout(dropout)

        if pretrained_task_dict is not None:
            pretrained_layers = {}
            for key, task_size in pretrained_task_dict.items():
                pretrained_layers[key] = nn.Linear(config.hidden_size, task_size)
            self.pretrained_classifiers = nn.ModuleDict(pretrained_layers)
        if new_task_dict is not None:
            new_layers = {}
            for key, task_size in new_task_dict.items():
                new_layers[key] = nn.Linear(config.hidden_size, task_size)
            self.new_classifiers = nn.ModuleDict(new_layers)

    def forward(self, tokenized_input):
        """
        Defines forward pass for Bert model

        Parameters
        ----------
        tokenized_input: torch tensor of integers
            integers represent tokens for each word

        Returns
        ----------
        A dictionary mapping each task to its logits
        """
        outputs = self.bert(tokenized_input)

        pooled_output = self.dropout(outputs[1])

        logit_dict = {}
        if hasattr(self, 'pretrained_classifiers'):
            for key, classifier in self.pretrained_classifiers.items():
                logit_dict[key] = classifier(pooled_output)
        if hasattr(self, 'new_classifiers'):
            for key, classifier in self.new_classifiers.items():
                logit_dict[key] = classifier(pooled_output)

        return logit_dict

    def freeze_bert(self):
        """Freeze all core Bert layers"""
        for param in self.bert.parameters():
            param.requires_grad = False

    def freeze_pretrained_classifiers_and_bert(self):
        """Freeze pretrained classifier layers and core Bert layers"""
        self.freeze_bert()
        if hasattr(self, 'pretrained_classifiers'):
            for param in self.pretrained_classifiers.parameters():
                param.requires_grad = False
        else:
            print('There are no pretrained_classifier layers to be frozen.')

    def unfreeze_pretrained_classifiers(self):
        """Unfreeze pretrained classifier layers"""
        if hasattr(self, 'pretrained_classifiers'):
            for param in self.pretrained_classifiers.parameters():
                param.requires_grad = True
        else:
            print('There are no pretrained_classifier layers to be unfrozen.')

    def unfreeze_pretrained_classifiers_and_bert(self):
        """Unfreeze pretrained classifiers and core Bert layers"""
        for param in self.bert.parameters():
            param.requires_grad = True

        self.unfreeze_pretrained_classifiers()

    def save(self, folder, model_id):
        """
        Saves the model state dicts to a specific folder.
        Each part of the model is saved separately to allow for
        new classifiers to be added later.

        Note: if the model has `pretrained_classifiers` and `new_classifers`,
        they will be combined into the `pretrained_classifiers_dict`.

        Parameters
        ----------
        folder: str or Path
            place to store state dictionaries
        model_id: int
            unique id for this model

        Side Effects
        ------------
        saves three files:
            - folder / f'bert_dict_{model_id}.pth'
            - folder / f'dropout_dict_{model_id}.pth'
            - folder / f'pretrained_classifiers_dict_{model_id}.pth'
        """
        if hasattr(self, 'pretrained_classifiers'):
            # PyTorch's update method isn't working because it doesn't think ModuleDict is a Mapping
            classifiers_to_save = copy.deepcopy(self.pretrained_classifiers)
            if hasattr(self, 'new_classifiers'):
                for key, module in self.new_classifiers.items():
                    classifiers_to_save[key] = module
        else:
            classifiers_to_save = copy.deepcopy(self.new_classifiers)

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.bert.state_dict(),
            folder / f'bert_dict_{model_id}.pth'
        )
        torch.save(
            self.dropout.state_dict(),
            folder / f'dropout_dict_{model_id}.pth'
        )

        torch.save(
            classifiers_to_save.state_dict(),
            folder / f'pretrained_classifiers_dict_{model_id}.pth'
        )

    def load(self, folder, model_id):
        """
        Loads the model state dicts from a specific folder.

        Parameters
        ----------
        folder: str or Path
            place where state dictionaries are stored
        model_id: int
            unique id for this model

        Side Effects
        ------------
        loads from three files:
            - folder / f'bert_dict_{model_id}.pth'
            - folder / f'dropout_dict_{model_id}.pth'
            - folder / f'pretrained_classifiers_dict_{model_id}.pth'
        """
        folder = Path(folder)

        if torch.cuda.is_available():
            self.bert.load_state_dict(torch.load(folder / f'bert_dict_{model_id}.pth'))
            self.dropout.load_state_dict(torch.load(folder / f'dropout_dict_{model_id}.pth'))
            self.pretrained_classifiers.load_state_dict(
                torch.load(folder / f'pretrained_classifiers_dict_{model_id}.pth')
            )
        else:
            self.bert.load_state_dict(
                torch.load(
                    folder / f'bert_dict_{model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )
            self.dropout.load_state_dict(
                torch.load(
                    folder / f'dropout_dict_{model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )
            self.pretrained_classifiers.load_state_dict(
                torch.load(
                    folder / f'pretrained_classifiers_dict_{model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )

    def export(self, folder, model_id, model_name=None):
        """
        Exports the entire model state dict to a specific folder.

        Note: if the model has `pretrained_classifiers` and `new_classifers`,
        they will be combined into the `pretrained_classifiers` attribute before being saved.

        Parameters
        ----------
        folder: str or Path
            place to store state dictionaries
        model_id: int
            unique id for this model
        model_name: str (defaults to None)
            Name to store model under, if None, will default to `multi_task_bert_{model_id}.pth`

        Side Effects
        ------------
        saves one file:
            - folder / model_name
        """
        hold_new_classifiers = copy.deepcopy(self.new_classifiers)
        hold_pretrained_classifiers = None
        if not hasattr(self, 'pretrained_classifiers'):
            self.pretrained_classifiers = copy.deepcopy(self.new_classifiers)
        else:
            hold_pretrained_classifiers = copy.deepcopy(self.pretrained_classifiers)
            # PyTorch's update method isn't working because it doesn't think ModuleDict is a Mapping
            for key, module in self.new_classifiers.items():
                self.pretrained_classifiers[key] = module

        del self.new_classifiers

        if model_name is None:
            model_name = f'multi_task_bert_{model_id}.pth'

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.state_dict(),
            folder / model_name
        )
        if hold_pretrained_classifiers is not None:
            self.pretrained_classifiers = hold_pretrained_classifiers
        else:
            del self.pretrained_classifiers
        self.new_classifiers = hold_new_classifiers

from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torchvision import models as torch_models
from transformers.modeling_bert import BertConfig, BertModel

from octopod.exc import OctopodError
from octopod.vision.helpers import _dense_block, _Identity


class BertResnetEnsembleForMultiTaskClassification(nn.Module):
    """
    PyTorch ensemble class for multitask learning consisting of a text and image models

    This model is made up of multiple component models:
    - for text: Google's BERT model
    - for images: multiple ResNet50's (the exact number depends on how
    the image model tasks were split up)

    You may need to train the component image and text models first
    before combining them into an ensemble model to get good results.

    Note: For explicitness, `vanilla` refers to the
    `transformers` BERT or `PyTorch` ResNet50 weights while
    `pretrained` refers to previously trained Octopod weights.

    Examples
    --------
    The ensemble model should be used with pretrained
    BERT and ResNet50 component models.
    To initialize a model in this way::

        image_task_dict = {
            'color_pattern': {
                'color': color_train_df['labels'].nunique(),
                'pattern': pattern_train_df['labels'].nunique()
            },
            'dress_sleeve': {
                'dress_length': dl_train_df['labels'].nunique(),
                'sleeve_length': sl_train_df['labels'].nunique()
            },
            'season': {
                'season': season_train_df['labels'].nunique()
            }
        }
        model = BertResnetEnsembleForMultiTaskClassification(
            image_task_dict=image_task_dict
        )

        resnet_model_id_dict = {
            'color_pattern': 'SOME_RESNET_MODEL_ID1',
            'dress_sleeve': 'SOME_RESNET_MODEL_ID2',
            'season': 'SOME_RESNET_MODEL_ID3'
        }

        model.load_core_models(
            folder='SOME_FOLDER',
            bert_model_id='SOME_BERT_MODEL_ID',
            resnet_model_id_dict=resnet_model_id_dict
        )

        # DO SOME TRAINING

        model.save(SOME_FOLDER, SOME_MODEL_ID)

        # OR

        model.export(SOME_FOLDER, SOME_MODEL_ID)

    Parameters
    ----------
    image_task_dict: dict
        dictionary mapping each pretrained ResNet50 models to a dictionary
        of the tasks it was trained on
    dropout: float
        dropout percentage for Dropout layer
    """
    def __init__(
        self,
        image_task_dict=None,
        dropout=1e-1
    ):
        super(BertResnetEnsembleForMultiTaskClassification, self).__init__()

        # Define text architecture
        config = BertConfig()
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(dropout)

        self.image_task_dict = image_task_dict
        self.text_task_dict = self.create_text_dict(image_task_dict)

        # Define image architecture
        image_resnets = {}
        image_dense_layers = {}
        ensemble_layers = {}
        for key in self.image_task_dict.keys():
            resnet = torch_models.resnet50(pretrained=False)
            resnet.fc = _Identity()
            image_resnets[key] = resnet
            image_dense_layers[key] = nn.Sequential(
                _dense_block(2048*2, 1024, 2e-3),
                _dense_block(1024, 512, 2e-3),
                _dense_block(512, 256, 2e-3)
            )

            # Define final ensemble before classifier layers
            # The input is size 768 from BERT and 256 from ResNet50 models
            # so the total size is 1024
            ensemble_layers[key] = nn.Sequential(
                _dense_block(1024, 512, 2e-3),
                _dense_block(512, 512, 2e-3),
                _dense_block(512, 256, 2e-3),
            )

        self.image_resnets = nn.ModuleDict(image_resnets)
        self.image_dense_layers = nn.ModuleDict(image_dense_layers)
        self.ensemble_layers = nn.ModuleDict(ensemble_layers)

        pretrained_layers = {}
        for key, task_size in self.text_task_dict.items():
            pretrained_layers[key] = nn.Linear(256, task_size)
        self.classifiers = nn.ModuleDict(pretrained_layers)

    def forward(self, x):
        """
        Defines forward pass for ensemble model

        Parameters
        ----------
        x: dict
            dictionary of torch tensors with keys:
                - `bert_text`: integers mapping to BERT vocabulary
                - `full_img`: tensor of full image
                - `crop_img`: tensor of cropped image

        Returns
        ----------
        A dictionary mapping each task to its logits
        """
        bert_output = self.bert(x['bert_text'])

        pooled_output = self.dropout(bert_output[1])

        logit_dict = {}

        for key in self.image_task_dict.keys():
            full_img = self.image_resnets[key](x['full_img'])
            crop_img = self.image_resnets[key](x['crop_img'])
            full_crop_combined = torch.cat((full_img, crop_img), 1)
            dense_layer_output = self.image_dense_layers[key](full_crop_combined)
            ensemble_input = torch.cat((pooled_output, dense_layer_output), 1)
            ensemble_layer_output = self.ensemble_layers[key](ensemble_input)

            for task in self.image_task_dict[key].keys():
                classifier = self.classifiers[task]
                logit_dict[task] = classifier(ensemble_layer_output)

        return logit_dict

    def freeze_bert(self):
        """Freeze all core BERT layers"""
        for param in self.bert.parameters():
            param.requires_grad = False

    def freeze_resnets(self):
        """Freeze all core ResNet models layers"""
        for key in self.image_resnets.keys():
            for param in self.image_resnets[key].parameters():
                param.requires_grad = False
            for param in self.image_dense_layers[key].parameters():
                param.requires_grad = False

    def freeze_ensemble_layers(self):
        """Freeze all final ensemble layers"""
        for key in self.ensemble_layers.keys():
            for param in self.ensemble_layers[key].parameters():
                param.requires_grad = False

    def freeze_classifiers_and_core(self):
        """Freeze pretrained classifier layers and core BERT/ResNet layers"""
        self.freeze_bert()
        self.freeze_resnets()
        self.freeze_ensemble_layers()
        for param in self.classifiers.parameters():
            param.requires_grad = False

    def unfreeze_classifiers(self):
        """Unfreeze pretrained classifier layers"""
        for param in self.classifiers.parameters():
            param.requires_grad = True

    def unfreeze_classifiers_and_core(self):
        """Unfreeze pretrained classifiers and core BERT/ResNet layers"""
        for param in self.bert.parameters():
            param.requires_grad = True
        for key in self.image_resnets.keys():
            for param in self.image_resnets[key].parameters():
                param.requires_grad = True
            for param in self.image_dense_layers[key].parameters():
                param.requires_grad = True
            for param in self.ensemble_layers[key].parameters():
                param.requires_grad = True

        self.unfreeze_classifiers()

    def save(self, folder, model_id):
        """
        Saves the model state dicts to a specific folder.
        Each part of the model is saved separately,
        along with the image_task_dict, which is needed to reinstantiate the model.

        Parameters
        ----------
        folder: str or Path
            place to store state dictionaries
        model_id: int
            unique id for this model

        Side Effects
        ------------
        saves six files:
            - folder / f'bert_dict_{model_id}.pth'
            - folder / f'dropout_dict_{model_id}.pth'
            - folder / f'image_resnets_dict_{model_id}.pth'
            - folder / f'image_dense_layers_dict_{model_id}.pth'
            - folder / f'ensemble_layers_dict_{model_id}.pth'
            - folder / f'classifiers_dict_{model_id}.pth'
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        # BERT model
        torch.save(
            self.bert.state_dict(),
            folder / f'bert_dict_{model_id}.pth'
        )
        torch.save(
            self.dropout.state_dict(),
            folder / f'dropout_dict_{model_id}.pth'
        )

        # ResNet model(s)
        torch.save(
            self.image_resnets.state_dict(),
            folder / f'image_resnets_dict_{model_id}.pth'
        )
        torch.save(
            self.image_dense_layers.state_dict(),
            folder / f'image_dense_layers_dict_{model_id}.pth'
        )

        # Ensemble layers
        torch.save(
            self.ensemble_layers.state_dict(),
            folder / f'ensemble_layers_dict_{model_id}.pth'
        )

        # Classifier layers
        torch.save(
            self.classifiers.state_dict(),
            folder / f'classifiers_dict_{model_id}.pth'
        )

        # image_task_dict
        joblib.dump(self.image_task_dict, folder / f'image_task_dict_{model_id}.pickle')

    def load(self, folder, model_id):
        """
        Loads the model state dicts for ensemble model
        from a specific folder. This will load all the model
        components including the final ensemble and existing
        pretrained `classifiers`.

        Parameters
        ----------
        folder: str or Path
            place where state dictionaries are stored
        model_id: int
            unique id for this model

        Side Effects
        ------------
        loads from six files:
            - folder / f'bert_dict_{model_id}.pth'
            - folder / f'dropout_dict_{model_id}.pth'
            - folder / f'image_resnets_dict_{model_id}.pth'
            - folder / f'image_dense_layers_dict_{model_id}.pth'
            - folder / f'ensemble_layers_dict_{model_id}.pth'
            - folder / f'classifiers_dict_{model_id}.pth'

        """
        folder = Path(folder)

        if torch.cuda.is_available():
            self.bert.load_state_dict(
                torch.load(folder / f'bert_dict_{model_id}.pth'))
            self.dropout.load_state_dict(
                torch.load(folder / f'dropout_dict_{model_id}.pth'))

            self.image_resnets.load_state_dict(
                torch.load(folder / f'image_resnets_dict_{model_id}.pth'))
            self.image_dense_layers.load_state_dict(
                torch.load(folder / f'image_dense_layers_dict_{model_id}.pth'))

            self.ensemble_layers.load_state_dict(
                torch.load(folder / f'ensemble_layers_dict_{model_id}.pth'))
            self.classifiers.load_state_dict(
                torch.load(folder / f'classifiers_dict_{model_id}.pth')
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

            self.resnet.load_state_dict(
                torch.load(
                    folder / f'image_resnets_dict_{model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )
            self.image_dense_layers.load_state_dict(
                torch.load(
                    folder / f'image_dense_layers_dict_{model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )

            self.final_ensemble.load_state_dict(
                torch.load(
                    folder / f'ensemble_layers_dict_{model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )

            self.classifiers.load_state_dict(
                torch.load(
                    folder / f'classifiers_dict_{model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )

    def load_core_models(self, folder, bert_model_id, resnet_model_id_dict):
        """
        Loads the weights from pretrained BERT and ResNet50 Octopod models

        Does not load weights from the final ensemble and classifier layers.
        use case is for loading SR_pretrained component BERT and image model
        weights into a new ensemble model.

        Parameters
        ----------
        folder: str or Path
            place where state dictionaries are stored
        bert_model_id: int
            unique id for pretrained BERT text model
        resnet_model_id_dict: dict
            dict with unique id's for pretrained image model,
            e.g.
            ```
            resnet_model_id_dict = {
                'task1_task2': 'model_id1',
                'task3_task4': 'model_id2',
                'task5': 'model_id3'
            }
            ```

        Side Effects
        ------------
        loads from four files:
            - folder / f'bert_dict_{bert_model_id}.pth'
            - folder / f'dropout_dict_{bert_model_id}.pth'
            - folder / f'resnet_dict_{resnet_model_id}.pth'
                for each resnet_model_id in the resnet_model_id_dict
            - folder / f'dense_layers_dict_{resnet_model_id}.pth'
        """
        folder = Path(folder)

        if torch.cuda.is_available():
            self.bert.load_state_dict(
                torch.load(folder / f'bert_dict_{bert_model_id}.pth'))
            self.dropout.load_state_dict(
                torch.load(folder / f'dropout_dict_{bert_model_id}.pth'))

            for key, model_id in resnet_model_id_dict.items():
                self.image_resnets[key].load_state_dict(
                    torch.load(folder / f'resnet_dict_{model_id}.pth')
                )
                self.image_dense_layers[key].load_state_dict(
                    torch.load(folder / f'dense_layers_dict_{model_id}.pth')
                )

        else:
            self.bert.load_state_dict(
                torch.load(
                    folder / f'bert_dict_{bert_model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )
            self.dropout.load_state_dict(
                torch.load(
                    folder / f'dropout_dict_{bert_model_id}.pth',
                    map_location=lambda storage,
                    loc: storage
                )
            )

            for key, model_id in resnet_model_id_dict.items():
                self.image_resnets[key].load_state_dict(
                    torch.load(
                        folder / f'resnet_dict_{model_id}.pth',
                        map_location=lambda storage,
                        loc: storage
                    )
                )
                self.image_dense_layers[key].load_state_dict(
                    torch.load(
                        folder / f'dense_layers_dict_{model_id}.pth',
                        map_location=lambda storage,
                        loc: storage
                    )
                )

    def export(self, folder, model_id, model_name=None):
        """
        Exports the entire model state dict to a specific folder,
        along with the image_task_dict, which is needed to reinstantiate the model.

        Parameters
        ----------
        folder: str or Path
            place to store state dictionaries
        model_id: int
            unique id for this model
        model_name: str (defaults to None)
            Name to store model under, if None, will default to `multi_task_ensemble_{model_id}.pth`

        Side Effects
        ------------
        saves two files:
            - folder / f'multi_task_ensemble_{model_id}.pth'
            - folder / f'image_task_dict_{model_id}.pickle'
        """
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        if model_name is None:
            model_name = f'multi_task_ensemble_{model_id}.pth'

        torch.save(
            self.state_dict(),
            folder / model_name
        )

        joblib.dump(self.image_task_dict, folder / f'image_task_dict_{model_id}.pickle')

    @staticmethod
    def create_text_dict(image_task_dict):
        """Create a task dict for the text model from the image task dict"""
        text_task_dict = {}
        for joint_task in image_task_dict.keys():
            for task, task_size in image_task_dict[joint_task].items():
                if task in text_task_dict.keys():
                    raise OctopodError(
                        'Task {} is in multiple models. Each task can only be in one image model.'
                        .format(task)
                    )
                text_task_dict[task] = task_size

        return text_task_dict

import copy
import time

from fastprogress.fastprogress import format_time, master_bar, progress_bar
import numpy as np
import torch

from octopod.learner_utils import (DEFAULT_LOSSES_DICT,
                                   DEFAULT_METRIC_DICT,
                                   )


class MultiTaskLearner(object):
    """
    Class to encapsulate training and validation steps for a pipeline. Based off the fastai learner.

    Parameters
    ----------
    model: torch.nn.Module
        PyTorch model to use with the Learner
    train_dataloader: MultiDatasetLoader
        dataloader for all of the training data. Set of labels must match `val_dataloader`
        or a ValueError will be thrown.
    val_dataloader: MultiDatasetLoader
        dataloader for all of the validation data. Set of labels must match `train_dataloader`
        or a ValueError will be thrown.
    task_dict: dict
        dictionary with all of the tasks as keys and the number of unique labels as the values
    loss_function_dict: dict
        dictionary where keys are task names (str) and values specify loss functions.
        A loss function can be specified using the special string value 'categorical_cross_entropy'
        for a multi-class task or 'bce_logits' for a multi-label task. Alternatively,
        it can be specified using a Callable that takes the predicted values and the target values
        as positional PyTorch tensor arguments and returns a float.

        Take care to ensure that the loss function includes a final activation function if needed --
        for instance, if your model is being used for classification but does not include a softmax
        layer then calculating cross-entropy properly requires performing the softmax classification
        within the loss function (this is handled by `nn.CrossEntropyLoss()` loss function).
        For our exisiting model architecture examples we do not apply final layer activation
        functions. Instead the desired activation functions are applied when needed.
        So we use 'categorical_cross_entropy' and 'bce_logits' loss functions which apply
        softmax and sigmoid activations to their inputs before calculating the loss.
    metric_function_dict: dict
        dictionary where keys are task names and values are metric calculation functions.
        If the input is a string matching a supported metric function `multi_class_acc`
        for multi-class tasks or `multi_label_acc` for multi-label tasks the loss will be filled in.
        A user can also input a custom metric function as a function for a given task key.

        custom metric functions must take in a `y_true` and `raw_y_pred` and output some `score` and
        `y_preds`. `y_preds` are the `raw_y_pred` values after an activation function has been
        applied and `score` is the output of whatever custom metrics the user wants to calculate
        for that task.

        See `learner_utils.loss_utils_config` for examples.
    """
    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 task_dict,
                 loss_function_dict=None,
                 metric_function_dict=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.task_dict = task_dict
        self.tasks = [*task_dict]
        self.loss_function_dict = self._get_loss_functions(loss_function_dict)
        self.metric_function_dict = self._get_metric_functions(metric_function_dict)
        self._check_all_labels_present()

    def fit(
        self,
        num_epochs,
        scheduler,
        step_scheduler_on_batch,
        optimizer,
        device='cuda:0',
        best_model=False,
        smooth_loss_alpha=0.2
    ):
        """
        Fit the PyTorch model

        Parameters
        ----------
        num_epochs: int
            number of epochs to train
        scheduler: torch.optim.lr_scheduler
            PyTorch learning rate scheduler
        step_scheduler_on_batch: bool
            flag of whether to step scheduler on batch (if True) or on epoch (if False)
        optimizer: torch.optim
            PyTorch optimzer
        device: str (defaults to 'cuda:0')
            device to run calculations on
        best_model: bool (defaults to `False`)
            flag to save best model from a single `fit` training run based on validation loss
            The default is `False`, which will keep the final model from the training run.
            `True` will keep the best model from the training run instead of the model
            from the final epoch of the training cycle.
        smooth_loss_alpha: float
            Training loss values displayed during fitting and at the end of each epoch are
            exponentially weighted moving averages over batches. Using an exponentially weighted
            average over batches is a compromise between reporting the value from the most recent
            batch, which is highly relevant but noisy, and reporting a simple average over batches,
            which is more stable but reflects the value of the loss at the beginning of the epoch
            as much as at the end. `smooth_loss_alpha` controls how much weight is given to the
            current batch. It must be in the `(0, 1]` interval. Higher values are more like
            reporting only the most recent batch, while lower values are more like giving all
            batches equal weight, so this value controls the tradeoff between stability and
            relevance.
        """
        if not 0 < smooth_loss_alpha <= 1:
            raise ValueError("`smooth_loss_alpha` must be in (0, 1]")

        self.model = self.model.to(device)

        current_best_loss = np.iinfo(np.intp).max

        pbar = master_bar(range(num_epochs))
        headers = ['train_loss', 'val_loss']
        for task in self.tasks:
            headers.append(f'{task}_train_loss')
            headers.append(f'{task}_val_loss')
            headers.append(f'{task}'+'_'+self.metric_function_dict[task].__name__)
        headers.append('time')
        pbar.write(headers, table=True)

        self.smooth_training_loss_dict = {}
        for epoch in pbar:
            start_time = time.time()
            self.model.train()

            overall_training_loss = 0.0

            subpbar = progress_bar(self.train_dataloader, parent=pbar)
            for _, batch in enumerate(subpbar):
                task_type, (x, y) = batch
                x = self._return_input_on_device(x, device)
                y = y.to(device)

                num_rows = self._get_num_rows(x)

                if num_rows == 1:
                    # skip batches of size 1
                    continue

                output = self.model(x)

                current_loss = self.loss_function_dict[task_type](output[task_type], y)

                overall_training_loss += current_loss.item() * num_rows

                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                if step_scheduler_on_batch:
                    scheduler.step()

                self._update_smooth_training_loss_dict(
                    task_type, current_loss.item(), smooth_loss_alpha
                )
                subpbar.comment = self._report_smooth_training_loss()

            overall_val_loss, val_loss_dict, metrics_scores = self.validate(
                device,
                pbar
            )

            if not step_scheduler_on_batch:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(overall_val_loss)
                else:
                    scheduler.step()

            overall_training_loss = overall_training_loss/self.train_dataloader.total_samples

            stats = [overall_training_loss, overall_val_loss]
            str_stats = []
            for stat in stats:
                str_stats.append(
                    'NA' if stat is None else str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                )

            for task in self.tasks:
                str_stats.append(f'{self.smooth_training_loss_dict[task]:.6f}')
                str_stats.append(f'{val_loss_dict[task]:.6f}')
                str_stats.append(
                    f"{metrics_scores[task][self.metric_function_dict[task].__name__]:.6f}"
                )

            str_stats.append(format_time(time.time() - start_time))

            pbar.write(str_stats, table=True)

            if best_model and overall_val_loss < current_best_loss:
                current_best_loss = overall_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_model_epoch = epoch

        if best_model:
            self.model.load_state_dict(best_model_wts)
            print(f'Epoch {best_model_epoch} best model saved with loss of {current_best_loss}')

    def _update_smooth_training_loss_dict(self, task_type, current_loss, smooth_loss_alpha):
        self.smooth_training_loss_dict[task_type] = (
            smooth_loss_alpha * current_loss
            + (1 - smooth_loss_alpha) * self.smooth_training_loss_dict[task_type]
            if task_type in self.smooth_training_loss_dict
            else current_loss
        )

    def _report_smooth_training_loss(self):
        return ''.join(
            [
                f'    {task}_train_loss: {loss:.4f}'
                for task, loss in self.smooth_training_loss_dict.items()
            ]
        )

    def validate(self, device='cuda:0', pbar=None):
        """
        Evaluate the model on a validation set

        Parameters
        ----------
        loss_function: function
            function to calculate loss with in model
        device: str (defaults to 'cuda:0')
            device to run calculations on
        pbar: fast_progress progress bar (defaults to None)
            parent progress bar for all epochs

        Returns
        -------
        overall_val_loss: float
            overall validation loss for all tasks
        val_loss_dict: dict
            dictionary of validation losses for individual tasks
        metrics_scores: dict
            scores for individual tasks
        """
        preds_dict = {}

        val_loss_dict = {task: 0.0 for task in self.tasks}
        metrics_scores = (
            {task: {self.metric_function_dict[task].__name__: 0.0} for task in self.tasks}
        )

        overall_val_loss = 0.0

        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(
                progress_bar(self.val_dataloader, parent=pbar, leave=(pbar is not None))
            ):
                task_type, (x, y) = batch
                x = self._return_input_on_device(x, device)

                y = y.to(device)

                output = self.model(x)
                num_rows = self._get_num_rows(x)

                current_loss = self.loss_function_dict[task_type](output[task_type], y)

                scaled_loss = current_loss.item() * num_rows

                val_loss_dict[task_type] += scaled_loss
                overall_val_loss += scaled_loss

                y_pred = output[task_type].cpu().numpy()
                y_true = y.cpu().numpy()

                preds_dict = self._update_preds_dict(preds_dict, task_type, y_true, y_pred)

        overall_val_loss /= self.val_dataloader.total_samples

        for task in self.tasks:
            val_loss_dict[task] = (
                val_loss_dict[task]
                / len(self.val_dataloader.loader_dict[task].dataset)
            )

        for task in metrics_scores.keys():

            y_true = preds_dict[task]['y_true']
            y_raw_pred = preds_dict[task]['y_pred']

            metric_score, y_preds = self.metric_function_dict[task](y_true, y_raw_pred)

            metrics_scores[task][self.metric_function_dict[task].__name__] = metric_score

        return overall_val_loss, val_loss_dict, metrics_scores

    def get_val_preds(self, device='cuda:0'):
        """
        Return true labels and predictions for data in self.val_dataloaders

        Parameters
        ----------
        device: str (defaults to 'cuda:0')
            device to run calculations on

        Returns
        -------
        Dictionary with dictionary for each task type:
            'y_true': numpy array of true labels, shape: (num_rows,)
            'y_pred': numpy of array of predicted probabilities: shape (num_rows, num_labels)
        """
        preds_dict = {}
        self.model = self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(progress_bar(self.val_dataloader, leave=False)):
                task_type, (x, y) = batch
                x = self._return_input_on_device(x, device)

                y = y.to(device)

                output = self.model(x)

                y_pred = output[task_type].cpu().numpy()
                y_true = y.cpu().numpy()

                preds_dict = self._update_preds_dict(preds_dict, task_type, y_true, y_pred)

        for task in self.tasks:
            metrics_scores, y_preds = (
                self.metric_function_dict[task](preds_dict[task]['y_true'],
                                                preds_dict[task]['y_pred'])
            )
            preds_dict[task]['y_pred'] = y_preds

        return preds_dict

    def _return_input_on_device(self, x, device):
        return x.to(device)

    def _get_num_rows(self, x):
        return x.size(0)

    def _update_preds_dict(self, preds_dict, task_type, y_true, y_pred):
        """
        Updates prediction dictionary for a specific task with both true labels
        and predicted labels for a given batch
        """

        if task_type not in preds_dict:
            preds_dict[task_type] = {
                'y_true': y_true,
                'y_pred': y_pred
            }

        else:
            preds_dict[task_type]['y_true'] = (
                np.concatenate((preds_dict[task_type]['y_true'], y_true))
            )

            preds_dict[task_type]['y_pred'] = (
                np.concatenate((preds_dict[task_type]['y_pred'], y_pred))
            )
        return preds_dict

    def _get_loss_functions(self, loss_function_dict):
        """
        Uses dictionaries of tasks and their loss configurations to construct
        a dictionary of loss functions.
        """
        processed_loss_function_dict = {}

        if loss_function_dict is None:
            loss_function_dict = {task: 'categorical_cross_entropy' for task in self.tasks}

        self._check_for_all_tasks(loss_function_dict, self.tasks, 'loss')

        for key, value in loss_function_dict.items():

            if isinstance(value, str):
                try:
                    processed_loss_function_dict[key] = DEFAULT_LOSSES_DICT[value]
                except Exception:

                    raise Exception('Found invalid loss function: {}. '
                                    'Valid losses are categorical_cross_entropy '
                                    'or bce_logits. Check that all tasks loss names are '
                                    'valid.'.format(value))
            else:
                processed_loss_function_dict[key] = value

        return processed_loss_function_dict

    def _get_metric_functions(self, metric_function_dict):
        """
        Uses dictionaries of tasks and a string cooresponding to a metric
        function configuration or custom metric function.

        Parameters
        ----------
        metric_function_dict: dictionary
            keys are tasks and values are associated metric strings or custom metric functions.
            Current supported metric functions are `multi_class_acc`
            for multi-class tasks and an accuracy score and `multi_label_acc` for multi-label
            tasks with an accuracy score, or custom functions.
        """
        processed_metric_function_dict = {}

        if metric_function_dict is None:
            metric_function_dict = {task: 'multi_class_acc' for task in self.tasks}

        self._check_for_all_tasks(metric_function_dict, self.tasks, 'accuracy')

        for key, value in metric_function_dict.items():
            if isinstance(value, str):
                try:
                    processed_metric_function_dict[key] = DEFAULT_METRIC_DICT[value]

                except Exception:
                    raise Exception('Found invalid metric function: {}. '
                                    'Valid metric functions are multi_class_acc '
                                    'or multi_label_acc. Check that all tasks metric '
                                    'functions names are valid.'.format(value))
            else:
                processed_metric_function_dict[key] = value

        return processed_metric_function_dict

    def _check_for_all_tasks(self, input_dict, tasks, input_str):
        if not all(task in input_dict for task in tasks):
            missing_tasks = set(tasks)-input_dict.keys()

            raise Exception(f'make sure all tasks are contained in the {input_str} dictionary '
                            f'missing tasks are {missing_tasks}')

    def _check_all_labels_present(self):
        # check that all categories are in both the train and val sets for each task
        for task in self.task_dict:
            if self.train_dataloader.label_mappings[task] != self.val_dataloader.label_mappings[task]: # noqa
                raise ValueError(f'Mapping mismatch in {task} task. Check that all categories are '
                                 'represented in the train and val datasets for each task. '
                                 f'train classes {self.train_dataloader.label_mappings[task]}'
                                 f'val classes {self.val_dataloader.label_mappings[task]}')


class MultiInputMultiTaskLearner(MultiTaskLearner):
    """
    Multi Input subclass of MultiTaskLearner class

    Parameters
    ----------
    model: torch.nn.Module
        PyTorch model to use with the Learner
    train_dataloader: MultiDatasetLoader
        dataloader for all of the training data
    val_dataloader: MultiDatasetLoader
        dataloader for all of the validation data
    task_dict: dict
        dictionary with all of the tasks as keys and the number of unique labels as the values

    Notes
    -----
    Multi-input datasets should return x's as a tuple/list so that each element
    can be sent to the appropriate device before being sent to the model
    see octopod.vision.dataset's OctopodImageDataset class for an example
    """
    def _return_input_on_device(self, x, device):
        """
        Send all model inputs to the appropriate device (GPU or CPU)
        when the inputs are in a dictionary format.

        Parameters
        ----------
        x: dict
            Output of a dataloader where the dataset generator groups multiple
            model inputs (such as multiple images) into a dictionary. Example
            `{'full_img':some_tensor,'crop_img':some_tensor}`

        """
        for k, v in x.items():
            x[k] = v.to(device)
        return x

    def _get_num_rows(self, x):
        return x[next(iter(x))].size(0)

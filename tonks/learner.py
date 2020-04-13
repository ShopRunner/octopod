import copy
import time

from fastprogress.fastprogress import format_time, master_bar, progress_bar
import numpy as np
import torch
import torch.nn.functional as F


class MultiTaskLearner(object):
    """
    Class to encapsulate training and validation steps for a pipeline. Based off the fastai learner.

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
    """
    def __init__(self, model, train_dataloader, val_dataloader, task_dict):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.task_dict = task_dict
        self.tasks = [*task_dict]

    def fit(
        self,
        num_epochs,
        scheduler,
        step_scheduler_on_batch,
        loss_function,
        optimizer,
        device='cuda:0',
        best_model=False
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
        loss_function: function
            function to calculate loss with in model
        optimizer: torch.optim
            PyTorch optimzer
        device: str (defaults to 'cuda:0')
            device to run calculations on
        best_model: bool (defaults to `False`)
            flag to save best model from a single `fit` training run based on validation loss
            The default is `False`, which will keep the final model from the training run.
            `True` will keep the best model from the training run instead of the model
            from the final epoch of the training cycle.
        """
        self.model = self.model.to(device)

        current_best_loss = np.iinfo(np.intp).max

        pbar = master_bar(range(num_epochs))
        headers = ['train_loss', 'val_loss']
        for task in self.tasks:
            headers.append(f'{task}_train_loss')
            headers.append(f'{task}_val_loss')
            headers.append(f'{task}_acc')
        headers.append('time')
        pbar.write(headers, table=True)

        for epoch in pbar:
            start_time = time.time()
            self.model.train()

            training_loss_dict = {task: 0.0 for task in self.tasks}

            overall_training_loss = 0.0

            for step, batch in enumerate(progress_bar(self.train_dataloader, parent=pbar)):
                task_type, (x, y) = batch
                x = self._return_input_on_device(x, device)
                y = y.type(torch.LongTensor)
                y = y.to(device)
                num_rows = self._get_num_rows(x)

                output = self.model(x)

                current_loss = loss_function(output[task_type], y)

                scaled_loss = current_loss.item() * num_rows

                training_loss_dict[task_type] += scaled_loss

                overall_training_loss += scaled_loss

                optimizer.zero_grad()
                current_loss.backward()
                optimizer.step()
                if step_scheduler_on_batch:
                    scheduler.step()

            if not step_scheduler_on_batch:
                scheduler.step()

            overall_training_loss = overall_training_loss/self.train_dataloader.total_samples

            for task in self.tasks:
                training_loss_dict[task] = (
                    training_loss_dict[task]
                    / len(self.train_dataloader.loader_dict[task].dataset)
                )

            overall_val_loss, val_loss_dict, accuracies = self.validate(
                loss_function,
                device,
                pbar
            )

            str_stats = []
            stats = [overall_training_loss, overall_val_loss]
            for stat in stats:
                str_stats.append(
                    'NA' if stat is None else str(stat) if isinstance(stat, int) else f'{stat:.6f}'
                )

            for task in self.tasks:
                str_stats.append(f'{training_loss_dict[task]:.6f}')
                str_stats.append(f'{val_loss_dict[task]:.6f}')
                str_stats.append(f"{accuracies[task]['accuracy']:.6f}")

            str_stats.append(format_time(time.time() - start_time))

            pbar.write(str_stats, table=True)

            if best_model and overall_val_loss < current_best_loss:
                current_best_loss = overall_val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                best_model_epoch = epoch

        if best_model:
            self.model.load_state_dict(best_model_wts)
            print(f'Epoch {best_model_epoch} best model saved with loss of {current_best_loss}')

    def validate(self, loss_function, device='cuda:0', pbar=None):
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
        accuracies: dict
            accuracy measures for individual tasks
        """
        val_loss_dict = {task: 0.0 for task in self.tasks}
        accuracies = {task: {'num_items': 0.0, 'num_correct': 0.0} for task in self.tasks}

        overall_val_loss = 0.0

        self.model.eval()

        with torch.no_grad():
            for step, batch in enumerate(
                progress_bar(self.val_dataloader, parent=pbar, leave=(pbar is not None))
            ):
                task_type, (x, y) = batch
                x = self._return_input_on_device(x, device)
                y = y.type(torch.LongTensor)
                y = y.to(device)
                num_rows = self._get_num_rows(x)

                output = self.model(x)

                current_loss = loss_function(
                    output[task_type], y
                ).item() * num_rows

                val_loss_dict[task_type] += current_loss

                overall_val_loss += current_loss

                accuracies[task_type]['num_correct'] += torch.sum(
                    torch.max(output[task_type], 1)[1] == y
                ).item()

                accuracies[task_type]['num_items'] += num_rows

        overall_val_loss = overall_val_loss/self.val_dataloader.total_samples

        for task in self.tasks:
            val_loss_dict[task] = (
                val_loss_dict[task]
                / len(self.val_dataloader.loader_dict[task].dataset)
            )

        for task in accuracies.keys():
            accuracies[task]['accuracy'] = (
                accuracies[task]['num_correct'] / accuracies[task]['num_items']
            )

        return overall_val_loss, val_loss_dict, accuracies

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
        for task in self.tasks:
            current_size = len(self.val_dataloader.loader_dict[task].dataset)
            preds_dict[task] = {
                'y_true': np.zeros([current_size]),
                'y_pred': np.zeros([current_size, self.task_dict[task]])
            }

        self.model = self.model.to(device)
        self.model.eval()

        with torch.no_grad():
            index_dict = {task: 0 for task in self.tasks}
            for step, batch in enumerate(progress_bar(self.val_dataloader, leave=False)):
                task_type, (x, y) = batch
                x = self._return_input_on_device(x, device)
                y = y.type(torch.LongTensor)
                y = y.to(device)

                output = self.model(x)

                y_pred = F.softmax(output[task_type], dim=1).cpu().numpy()

                y_true = y.cpu().numpy()

                current_index = index_dict[task_type]

                num_rows = self._get_num_rows(x)

                preds_dict[task_type]['y_true'][current_index: current_index + num_rows] = y_true
                preds_dict[task_type]['y_pred'][current_index: current_index + num_rows, :] = y_pred

                index_dict[task_type] += num_rows

        return preds_dict

    def _return_input_on_device(self, x, device):
        return x.to(device)

    def _get_num_rows(self, x):
        return x.size(0)


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
    see tonks.vision.dataset's TonksImageDataset class for an example
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

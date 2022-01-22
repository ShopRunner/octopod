import random


class MultiDatasetLoader(object):
    """
    Load datasets for multiple tasks

    Parameters
    ----------
    loader_dict: dict
        dictonary of DataLoaders
    shuffle: Boolean (defaults to True)
        Flag for whether or not to shuffle the data
    """

    def __init__(self, loader_dict, shuffle=True):
        self.loader_dict = loader_dict
        self.shuffle = shuffle
        self.label_mappings = self._gen_label_mappings()

        total_samples = 0
        for key in self.loader_dict.keys():
            total_samples += len(self.loader_dict[key].dataset)

        self.total_samples = total_samples

    def __iter__(self):
        named_batches = []
        iterators = {}

        for key in self.loader_dict.keys():
            current_batches = [key] * self._get_no_of_batches(key)
            named_batches += current_batches
            iterators[key] = iter(self.loader_dict[key])

        if self.shuffle:
            random.shuffle(named_batches)

        for key in named_batches:
            yield key, next(iterators[key])

    def __len__(self):
        num_batches = 0

        for key in self.loader_dict.keys():
            num_batches += self._get_no_of_batches(key)

        return num_batches

    def _get_no_of_batches(self, key, threshold_batch_size=1):
        """
        The method ignores the batches that has number of records below the threshold_batch_size
        Say if the threshold_batch_size is set as 2,
            and total no of data is 130 and batch size 64, then 66 % 64 <= 2, and would return 130 //64 == 1
            and if batch size is 134 , then 134 % 64 > 2, then would return len(dataset) == 2
        Parameters
        ----------
        key: str
            The key for the dictionary
        threshold_batch_size: Integer
            threshold size for the batch to be considered for processing
            default 1

        Return
        ----------
            return no of batches those no of records above the threshold_batch_size
        """
        total_size = len(self.loader_dict[key].dataset)
        batch_size = self.loader_dict[key].batch_size
        if total_size % batch_size <= threshold_batch_size:
            return total_size // batch_size
        return len(self.loader_dict[key])

    def _gen_label_mappings(self):

        mapping_dict = {}
        for key, loader_dict in self.loader_dict.items():
            mapping_dict[key] = loader_dict.dataset.label_mapping
        return mapping_dict

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

        total_samples = 0
        for key in self.loader_dict.keys():
            total_samples += len(self.loader_dict[key].dataset)

        self.total_samples = total_samples

    def __iter__(self):
        named_batches = []
        iterators = {}

        for key in self.loader_dict.keys():
            current_batches = [key] * len(self.loader_dict[key])
            named_batches += current_batches
            iterators[key] = iter(self.loader_dict[key])

        if self.shuffle:
            random.shuffle(named_batches)

        for key in named_batches:
            yield key, next(iterators[key])

    def __len__(self):
        num_batches = 0

        for key in self.loader_dict.keys():
            num_batches += len(self.loader_dict[key])

        return num_batches

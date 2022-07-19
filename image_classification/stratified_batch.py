import torch
from sklearn.model_selection import StratifiedKFold


class StratifiedBatchSampler:
    """
    Custom batch sampler generates batches by preserving the percentage of samples for each target class.
    """

    def __init__(self, y, batch_size, shuffle=True, random_state=42):
        """
        :param y: target class labels
        :param batch_size: batch size
        :param shuffle: indicator of shuffling data
        :param random_state: random state
        """
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(
            n_splits=n_batches, shuffle=shuffle, random_state=random_state
        )
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self):
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y) // self.batch_size

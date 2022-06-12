import torch
from sklearn.model_selection import StratifiedKFold


class StratifiedBatchSampler:
    """
    Stratified batch sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, y, batch_size, shuffle=True, random_state=42):
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle, random_state=random_state)
        self.X = torch.randn(len(y), 1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y) // self.batch_size

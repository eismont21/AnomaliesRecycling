import numpy as np
import torch


class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve after a given patience.
    """

    def __init__(self, path, patience=3, verbose=False):
        """
        :param patience: how long to wait after last time validation accuracy improved.
        :param verbose: If True, prints a message for each validation accuracy improvement.
        :param path: Path for the checkpoint to be saved to
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.val_acc_best = 0
        self.path = path

    def __call__(self, val_acc, model):
        if val_acc <= self.val_acc_best:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        """
        Saves model when validation accuracy improves.
        :param val_acc: validation accuracy
        :param model: model
        """
        if self.verbose:
            print(f'Validation accuracy increased ({self.val_acc_best:.6f} --> {val_acc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path + 'checkpoint.pth')
        self.val_acc_best = val_acc

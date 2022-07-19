from torchmetrics import Metric
import torch


class DecisionAccuracy(Metric):
    """
    Custom metric to compute the decision accuracy.
    Correct decision means:
    1. the tray has only one lid and is accepted
    2. the tray has more than one lid or is empty and is rejected
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update the metric based on the predictions and the target labels.
        :param preds: batch of predictions
        :param target: batch of target labels
        """
        for pred, y_true in zip(preds, target):
            if (pred == y_true and y_true == 1) or (y_true != 1 and pred != 1):
                self.correct += 1

        self.total += target.numel()

    def compute(self):
        """
        Compute the metric value.
        """
        return self.correct.float() / self.total

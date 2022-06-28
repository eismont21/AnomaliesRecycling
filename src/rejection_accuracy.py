from torchmetrics import Metric
import torch

class RejectionAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for pred, y_true in zip(preds, target):
            if (pred == y_true and y_true == 1) or (y_true in (0, 2, 3, 4, 5) and pred != 1):
                self.correct += 1

        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
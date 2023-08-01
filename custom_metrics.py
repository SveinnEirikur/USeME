import torch
from torchmetrics import Metric
from torchmetrics.functional import image_gradients

class MeanAbsoluteGradientError(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent, and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, gamma=0.5):
        super().__init__()
        self.gamma = gamma
        self.gradients = image_gradients
        self.loss = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        preds_dx, preds_dy = self.gradients(preds)
        target_dx, target_dy = self.gradients(target)
        #
        grad_diff_x = torch.abs(target_dx - preds_dx)
        grad_diff_y = torch.abs(target_dy - preds_dy)

        # condense into one tensor and avg
        self.loss = self.gamma*(torch.mean(grad_diff_x) + torch.mean(grad_diff_y))

    def compute(self):
        return self.loss


class MeanSquaredGradientError(Metric):
    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better: bool = False

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent, and we will optimize the runtime of 'forward'
    full_state_update: bool = False

    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma
        self.gradients = image_gradients
        self.loss = 0

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        preds_dx, preds_dy = self.gradients(preds)
        target_dx, target_dy = self.gradients(target)
        #
        grad_diff_x = torch.square(target_dx - preds_dx)
        grad_diff_y = torch.square(target_dy - preds_dy)

        # condense into one tensor and avg
        self.loss = self.gamma * (torch.mean(grad_diff_x) + torch.mean(grad_diff_y))

    def compute(self):
        return self.loss

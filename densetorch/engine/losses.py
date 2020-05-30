import torch
import torch.nn as nn
import torch.nn.functional as F


class InvHuberLoss(nn.Module):
    """Inverse Huber Loss for depth estimation.

    The setup is taken from https://arxiv.org/abs/1606.00373

    Args:
      ignore_index (float): value to ignore in the target
                            when computing the loss.

    """

    def __init__(self, ignore_index=0):
        super(InvHuberLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, x, target):
        input = F.relu(x)  # depth predictions must be >=0
        diff = input - target
        mask = target != self.ignore_index

        err = torch.abs(diff * mask.float())
        c = 0.2 * torch.max(err)
        err2 = (diff ** 2 + c ** 2) / (2.0 * c)
        mask_err = err <= c
        mask_err2 = err > c
        cost = torch.mean(err * mask_err.float() + err2 * mask_err2.float())
        return cost

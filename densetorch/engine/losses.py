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


class CosineDistanceLoss(nn.Module):
    """Cosine Distance Loss for 3D surface normals estimation.

    Args:
      ignore_index (float): value to ignore in the target when computing the loss.

    """

    def __init__(self, ignore_index=0, dim=1):
        super(CosineDistanceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.cosine_similarity = nn.CosineSimilarity(dim=dim)
        if dim != 1:
            raise ValueError("Dim must be equal to 1. Other values are not supported.")

    def forward(self, x, target):
        if target.size(3) != 3:
            assert (
                target.size(1) == 3
            ), f"Expected surface normals target to have 3 channels either at the second or the last dimension, got shape {target.shape}"
            target = target.permute(0, 2, 3, 1)
        assert (
            x.size(1) == 3
        ), f"Expected surface normals predictions to have 3 channels, got {x.size(1):d}"
        x_bhw3 = x.permute(0, 2, 3, 1)
        # Calculate ignore region
        with torch.no_grad():
            keep_region = target.sum(dim=-1) != 3 * self.ignore_index
        x_N3 = x_bhw3[keep_region]
        target_N3 = target[keep_region]
        cosine_distance = 1.0 - self.cosine_similarity(x_N3, target_N3)
        return torch.mean(cosine_distance)

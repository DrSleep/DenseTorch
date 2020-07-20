import numpy as np
import torch

from .miou import compute_iu, fast_cm


def maybe_transpose_bchw_to_bhwc(x, c=None):
    if c is None:
        return x.transpose(0, 2, 3, 1)
    if x.shape[3] == c:
        return x
    else:
        assert (
            x.shape[1] == c
        ), f"Either the last or the second dimension must be equal to {c}, got shape {x.shape}"
        return x.transpose(0, 2, 3, 1)


class MeanIoU:
    """Mean-IoU computational block for semantic segmentation.

    Args:
      num_classes (int): number of classes to evaluate.

    Attributes:
      name (str): descriptor of the estimator.

    """

    def __init__(self, num_classes):
        if isinstance(num_classes, (list, tuple)):
            num_classes = num_classes[0]
        assert isinstance(
            num_classes, int
        ), f"Number of classes must be int, got {num_classes}"
        self.num_classes = num_classes
        self.name = "meaniou"
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.num_classes, self.num_classes), dtype=int)

    def update(self, pred, gt):
        assert isinstance(pred, torch.Tensor), "Expected a torch.Tensor as input"
        assert isinstance(gt, torch.Tensor), "Expected a torch.Tensor as input"
        pred_dims = len(pred.shape)
        assert (pred_dims - 1) == len(
            gt.shape
        ), "Prediction tensor must have 1 more dimension that ground truth"
        if pred_dims == 3:
            class_axis = 0
        elif pred_dims == 4:
            class_axis = 1
        else:
            raise ValueError("{}-dimensional input is not supported".format(pred_dims))
        assert (
            pred.shape[class_axis] == self.num_classes
        ), "Dimension {} of prediction tensor must be equal to the number of classes".format(
            class_axis
        )
        _, pred = pred.max(class_axis)
        idx = gt < self.num_classes
        self.cm += fast_cm(
            pred[idx].cpu().numpy().astype(np.uint8),
            gt[idx].cpu().numpy().astype(np.uint8),
            self.num_classes,
        )

    def val(self):
        return np.mean([iu for iu in compute_iu(self.cm) if iu <= 1.0])


class RMSE:
    """Root Mean Squared Error computational block for depth estimation.

    Args:
      ignore_index (float): value to ignore in the target
                          when computing the metric.

    Attributes:
      name (str): descriptor of the estimator.

    """

    def __init__(self, ignore_index=0):
        self.ignore_index = ignore_index
        self.name = "rmse"
        self.reset()

    def reset(self):
        self.num = 0.0
        self.den = 0.0

    def update(self, pred, gt):
        assert isinstance(pred, torch.Tensor), "Expected a torch.Tensor as input"
        assert isinstance(gt, torch.Tensor), "Expected a torch.Tensor as input"
        assert (
            pred.shape == gt.shape
        ), "Prediction tensor must have the same shape as ground truth"
        pred = pred.abs()
        idx = gt != self.ignore_index
        diff = (pred - gt)[idx]
        self.num += (diff ** 2).sum().item()
        self.den += idx.sum().item()

    def val(self):
        return np.sqrt(self.num / self.den)


class AngularError:
    """AngularError computational block for 3D surface normals estimation.

    Args:
      ignore_index (float): value to ignore in the target
                          when computing the metric.

    Attributes:
      name (str): descriptor of the estimator.

    """

    def __init__(self, ignore_index=0):
        self.ignore_index = ignore_index
        self.name = "ang_error"
        self.reset()

    def reset(self):
        self.num = 0.0
        self.den = 0.0

    def update(self, pred, gt):
        assert len(pred.shape) == len(
            gt.shape
        ), f"Prediction tensor and ground truth must have the same number of dimension, got {len(pred.shape):d} and {len(gt.shape):d}"
        if len(pred.shape) != 4:
            assert (
                len(pred.shape) == 3
            ), f"Prediction tensor must have either 3 or 4 dimensions, got {len(pred.shape):d}"
            pred = np.expand_dims(pred, axis=0)
            gt = np.expand_dims(gt, axis=0)
        gt = maybe_transpose_bchw_to_bhwc(gt, c=3)
        pred = maybe_transpose_bchw_to_bhwc(pred, c=3)
        keep_region = gt.sum(axis=-1) != 3 * self.ignore_index
        gt_N3 = gt[keep_region]
        pred_N3 = pred[keep_region]
        cos_theta = np.sum(gt_N3 * pred_N3, axis=-1) / (
            np.linalg.norm(gt_N3, axis=-1) * np.linalg.norm(pred_N3, axis=-1)
        )
        cos_theta = np.clip(cos_theta, -1, 1)
        theta = np.arccos(cos_theta)
        self.num += np.sum(theta)
        self.den += len(theta)

    def val(self):
        return (self.num / self.den) / np.pi * 180.0

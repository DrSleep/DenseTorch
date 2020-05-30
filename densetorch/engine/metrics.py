import numpy as np

from .miou import compute_iu, fast_cm

class MeanIoU:
    """Mean-IoU computational block for semantic segmentation.

    Args:
      num_classes (int): number of classes to evaluate.

    Attributes:
      name (str): descriptor of the estimator.

    """
    def __init__(self, num_classes):
        self.cm = np.zeros((num_classes, num_classes), dtype=int)
        self.num_classes = num_classes
        self.name = 'meaniou'

    def update(self, pred, gt):
        idx = gt < self.num_classes
        pred_dims = len(pred.shape)
        assert (pred_dims - 1) == len(gt.shape), \
                "Prediction tensor must have 1 more dimension that ground truth"
        if pred_dims == 3:
            class_axis = 0
        elif pred_dims == 4:
            class_axis = 1
        else:
            raise ValueError("{}-dimensional input is not supported"
                             .format(pred_dims))
        assert pred.shape[class_axis] == self.num_classes, \
                ("Dimension {} of prediction tensor must be equal to the number of classes"
                 .format(class_axis))
        pred = pred.argmax(axis=class_axis)
        self.cm += fast_cm(
            pred[idx].astype(np.uint8),
            gt[idx].astype(np.uint8),
            self.num_classes)

    def val(self):
        return np.mean([iu for iu in compute_iu(self.cm) if iu <= 1.0])

class RMSE:
    """Root Mean Squared Error computational block for depth estimation.

    Args:
      ignore_val (float): value to ignore in the target
                          when computing the metric.

    Attributes:
      name (str): descriptor of the estimator.

    """
    def __init__(self, ignore_val=0):
        self.ignore_val = ignore_val
        self.num = 0.
        self.den = 0.
        self.name = 'rmse'

    def update(self, pred, gt):
        assert pred.shape == gt.shape, \
                "Prediction tensor must have the same shape as ground truth"
        pred = np.abs(pred)
        idx = gt != self.ignore_val
        diff = (pred - gt)[idx]
        self.num += np.sum(diff**2)
        self.den += np.sum(idx)

    def val(self):
        return np.sqrt(self.num / self.den)

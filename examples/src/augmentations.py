"""Extending albumentations to support specific tasks such as depth estimation and surface normals estimation.

The source code is modified from https://github.com/albumentations-team/albumentations.
The same license rules apply.

"""

import cv2
import numpy as np

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.core.transforms_interface import DualTransform

from itertools import cycle


class MultiMaskPadIfNeeded(A.PadIfNeeded):
    def __init__(
        self,
        min_height=1024,
        min_width=1024,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        p=1.0,
    ):
        super(MultiMaskPadIfNeeded, self).__init__(
            min_height, min_width, border_mode, value, mask_value, always_apply, p
        )
        self.mask_value = cycle(mask_value)

    def apply_to_mask(
        self, img, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **params
    ):
        return F.pad_with_params(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            border_mode=self.border_mode,
            value=next(self.mask_value),
        )


class MultiMaskHorizontalFlip(A.HorizontalFlip):
    def apply(self, img, **params):
        # Inverse the x-direction for normals when horizontally flipping
        mult = [-1, 1, 1] if "normals" in params else 1
        if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
            return mult * F.hflip_cv2(img)

        return mult * F.hflip(img)

    @property
    def target_dependence(self):
        return {"normals": ["normals"]}


class MultiMaskRandomScale(A.RandomScale):
    def apply(self, img, scale=0, interpolation=cv2.INTER_LINEAR, **params):
        mult = 1
        if scale > 0 and "depth" in params:
            # Depth is inversely proportional to the scale factor
            mult = 1.0 / scale
        out = mult * F.scale(img, scale, interpolation)
        return out

    @property
    def target_dependence(self):
        return {"depth": ["depth"]}

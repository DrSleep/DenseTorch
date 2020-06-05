import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..misc.utils import broadcast, make_list

# Usual dtypes for common modalities
KEYS_TO_DTYPES = {
    "segm": torch.long,
    "mask": torch.long,
    "depth": torch.float,
    "normals": torch.float,
}


class Pad(object):
    """Pad image and mask to the desired size.

    Args:
      size (int) : minimum length/width.
      img_val (array) : image padding value.
      msk_vals (list of ints) : masks padding value.

    """

    def __init__(self, size, img_val, msk_vals):
        assert isinstance(size, int)
        self.size = size
        self.img_val = img_val
        self.msk_vals = make_list(msk_vals)

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        h, w = image.shape[:2]
        h_pad = int(np.clip(((self.size - h) + 1) // 2, 0, 1e6))
        w_pad = int(np.clip(((self.size - w) + 1) // 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        sample["image"] = np.stack(
            [
                np.pad(
                    image[:, :, c],
                    pad,
                    mode="constant",
                    constant_values=self.img_val[c],
                )
                for c in range(3)
            ],
            axis=2,
        )
        for msk_key, msk_val in zip(msk_keys, self.msk_vals):
            sample[msk_key] = np.pad(
                sample[msk_key], pad, mode="constant", constant_values=msk_val
            )
        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        crop_size (int): Desired output size.

    """

    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        sample["image"] = image[top : top + new_h, left : left + new_w]
        for msk_key in msk_keys:
            sample[msk_key] = sample[msk_key][top : top + new_h, left : left + new_w]
        return sample


class ResizeAndScale(object):
    """Resize shorter/longer side to a given value and randomly scale.

    Args:
        side (int) : shorter / longer side value.
        low_scale (float) : lower scaling bound.
        high_scale (float) : upper scaling bound.
        shorter (bool) : whether to resize shorter / longer side.

    """

    def __init__(self, side, low_scale, high_scale, shorter=True):
        assert isinstance(side, int)
        assert isinstance(low_scale, float)
        assert isinstance(high_scale, float)
        self.side = side
        self.low_scale = low_scale
        self.high_scale = high_scale
        self.shorter = shorter

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        scale = np.random.uniform(self.low_scale, self.high_scale)
        if self.shorter:
            min_side = min(image.shape[:2])
            if min_side * scale < self.side:
                scale = self.side * 1.0 / min_side
        else:
            max_side = max(image.shape[:2])
            if max_side * scale > self.side:
                scale = self.side * 1.0 / max_side
        sample["image"] = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        for msk_key in msk_keys:
            scale_mult = (1.0 / scale) if "depth" in msk_key else 1
            sample[msk_key] = scale_mult * cv2.resize(
                sample[msk_key],
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
        return sample


class RandomMirror(object):
    """Randomly flip the image and the mask"""

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        do_mirror = np.random.randint(2)
        if do_mirror:
            sample["image"] = cv2.flip(image, 1)
            for msk_key in msk_keys:
                scale_mult = [-1, 1, 1] if "normal" in msk_key else 1
                sample[msk_key] = scale_mult * cv2.flip(sample[msk_key], 1)
        return sample


class Normalise(object):
    """Normalise a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (scale * channel - mean) / std

    Args:
        scale (float): Scaling constant.
        mean (sequence): Sequence of means for R,G,B channels respecitvely.
        std (sequence): Sequence of standard deviations for R,G,B channels
            respecitvely.
        depth_scale (float): Depth divisor for depth annotations.

    """

    def __init__(self, scale, mean, std, depth_scale=1.0):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.depth_scale = depth_scale

    def __call__(self, sample):
        sample["image"] = (self.scale * sample["image"] - self.mean) / self.std
        if "depth" in sample:
            sample["depth"] = sample["depth"] / self.depth_scale
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample["image"]
        msk_keys = sample["names"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample["image"] = torch.from_numpy(image.transpose((2, 0, 1)))
        for msk_key in msk_keys:
            sample[msk_key] = torch.from_numpy(sample[msk_key]).to(
                KEYS_TO_DTYPES[msk_key]
            )
        return sample


def albumentations2densetorch(augmentation):
    """Wrapper to use Albumentations within DenseTorch dataset.

    Args:
      augmentation: either a list of augmentations or a single augmentation

    Returns:
      A composition of augmentations

    """
    from albumentations import Compose

    def wrapper_func(sample):
        if "names" in sample:
            del sample["names"]
        targets = {
            name: "image" if name == "image" else "mask" for name in sample.keys()
        }
        output = Compose(make_list(augmentation), additional_targets=targets)(**sample)
        return output

    return wrapper_func


def densetorch2torchvision(augmentation):
    """Wrapper to use DenseTorch augmentations within torchvision dataset.

    Args:
      augmentation: either a list of augmentations or a single augmentation

    Returns:
      A composition of augmentations.

    """
    from torchvision.transforms import Compose

    def wrapper_func(image, target):
        keys = ["image", "mask"]
        names = ["mask"]
        np_dtypes = [np.float32, np.uint8]
        torch_dtypes = [torch.float32, torch.long]
        sample_dict = {
            key: np.array(value, dtype=dtype)
            for key, value, dtype in zip(keys, [image, target], np_dtypes)
        }
        sample_dict["names"] = names
        output = Compose(make_list(augmentation))(sample_dict)
        return [output[key].to(dtype) for key, dtype in zip(keys, torch_dtypes)]

    return wrapper_func


def denormalise(tensor_bchw, scale, mean_c, std_c):
    """Reversed normalisation

    Args:
      tensor_bchw (torch.tensor): 4D tensor of shape BxCxHxW
      scale (float): scale value
      mean_c (np.ndarray): mean array of shape (C,)
      std_c (np.ndarray): standard deviation array of shape (C,)

    Returns:
      Un-normalised torch tensor.

    """
    mean_bchw = (
        torch.from_numpy(mean_c[None, :, None, None]).float().to(tensor_bchw.device)
    )
    std_bchw = (
        torch.from_numpy(std_c[None, :, None, None]).float().to(tensor_bchw.device)
    )
    return (tensor_bchw * std_bchw + mean_bchw) / scale


def get_loaders(
    train_batch_size,
    val_batch_size,
    train_set,
    val_set,
    num_stages=1,
    num_workers=8,
    train_shuffle=True,
    val_shuffle=False,
    train_pin_memory=False,
    val_pin_memory=False,
    train_drop_last=False,
    val_drop_last=False,
):
    """Create train and val loaders"""
    train_batch_sizes = broadcast(train_batch_size, num_stages)
    train_sets = broadcast(train_set, num_stages)
    train_loaders = [
        DataLoader(
            train_sets[i],
            batch_size=train_batch_sizes[i],
            shuffle=train_shuffle,
            num_workers=num_workers,
            pin_memory=train_pin_memory,
            drop_last=train_drop_last,
        )
        for i in range(num_stages)
    ]
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=val_shuffle,
        num_workers=num_workers,
        pin_memory=val_pin_memory,
        drop_last=val_drop_last,
    )
    return train_loaders, val_loader

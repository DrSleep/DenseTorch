import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .utils import KEYS_TO_DTYPES


def transpose_normals(normals):
    # Transpose normals: HxWxC -> CxHxW
    normals = normals.permute(2, 0, 1)
    return normals


def convert_normals_to_xzy(normals):
    normals = normals / 255.0
    normals[:, :, 0] = normals[:, :, 0] * 2.0 - 1.0  # x
    normals[:, :, 1] = normals[:, :, 1] * 2.0 - 1.0  # z
    normals[:, :, 2] = 1.0 - normals[:, :, 2] * 2.0  # y
    return normals


def set_normals_ignore_indices(
    normals, depth, normals_ignore_index, depth_ignore_index
):
    depth_ignore_region = depth == depth_ignore_index
    normals[depth_ignore_region] = normals_ignore_index
    return normals


class MMDataset(Dataset):
    """Multi-Modality dataset.

    Works with any datasets that contain image
    and any number of 2D-annotations.

    Args:
        data_file (string): Path to the data file with annotations.
        data_dir (string): Directory with all the images.
        data_list_sep (string): Separator between the columns in data_file.
        data_list_columns (list of strings): Column names in data_file.
        masks_names (list of strings): Keys for each annotation mask (e.g., 'segm', 'depth').
        ignore_indices (list of floats or ints): Ignore values for each annotation mask in the same order as `masks_names`.
        depth_scale (float): Scaling factor for depth.
        transform (callable, optional): Optional transform to be applied on a sample.

    Raises:
      ValueError: if the number of columns in data_file is not equal to `len(data_list_columns)`.

    """

    def __init__(
        self,
        data_file,
        data_dir,
        data_list_sep,
        data_list_columns,
        masks_names,
        ignore_indices,
        depth_scale=1.0,
        transform=None,
    ):
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        self.datalist = []
        for line in datalist:
            columns = line.decode("utf-8").strip("\n").split(data_list_sep)
            assert len(columns) == len(
                data_list_columns
            ), f"Inconsistent number of columns, got {len(columns):d} for the following names: {data_list_columns}"
            self.datalist.append(
                {name: path for name, path in zip(data_list_columns, columns)}
            )
        self.root_dir = data_dir
        self.transform = transform
        self.masks_names = masks_names
        self.ignore_indices = ignore_indices
        self.depth_scale = depth_scale

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        sample_paths = {
            key: os.path.join(self.root_dir, path)
            for key, path in self.datalist[idx].items()
        }
        sample = {}
        sample["image"] = self.read_image(sample_paths["image"])
        for key in self.masks_names:
            mask = np.array(Image.open(sample_paths[key]))
            if key != "normals":
                assert (
                    len(mask.shape) == 2
                ), "Segm and depth masks must be encoded without colourmap"
            else:
                mask = convert_normals_to_xzy(mask)
            sample[key] = mask
        if self.transform:
            sample["names"] = self.masks_names
            sample = self.transform(sample)
            # the names key can be removed by the transformation
            if "names" in sample:
                del sample["names"]
        for key in self.masks_names:
            if key in KEYS_TO_DTYPES:
                scale = 1
                if key == "depth":
                    scale = self.depth_scale
                sample[key] = sample[key].to(KEYS_TO_DTYPES[key]) / scale
        # The ignore regions of surface normals are dependent on the ignore regions of depth
        if "depth" in self.masks_names and "normals" in self.masks_names:
            depth_ignore_index = self.ignore_indices[self.masks_names.index("depth")]
            normals_ignore_index = self.ignore_indices[
                self.masks_names.index("normals")
            ]
            sample["normals"] = transpose_normals(
                set_normals_ignore_indices(
                    sample["normals"],
                    sample["depth"],
                    normals_ignore_index,
                    depth_ignore_index,
                )
            )
        elif "normals" in self.masks_names:
            sample["normals"] = transpose_normals(sample["normals"])
        return sample

    @staticmethod
    def read_image(x):
        """Simple image reader

        Args:
            x (str): path to image.

        Returns image as `np.array`.

        """
        img_arr = np.array(Image.open(x), dtype=np.float32)
        if len(img_arr.shape) == 2:  # grayscale
            img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
        return img_arr

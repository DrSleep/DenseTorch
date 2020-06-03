import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MMDataset(Dataset):
    """Multi-Modality dataset.

    Works with any datasets that contain image
    and any number of 2D-annotations.

    Args:
        data_file (string): Path to the data file with annotations.
        data_dir (string): Directory with all the images.
        line_to_paths_fn (callable): function to convert a line of data_file into paths
                                     (img_relpath, msk1_relpath, msk2_relpath).
        masks_names (list of strings): keys for each annotation mask (e.g., 'segm', 'depth').
                                       Must be in the same order as outputs of line_to_paths_fn
        transform (callable, optional): Optional transform to be applied on a sample.

    """

    def __init__(
        self, data_file, data_dir, line_to_paths_fn, masks_names, transform=None,
    ):
        with open(data_file, "rb") as f:
            datalist = f.readlines()
        self.datalist = [line_to_paths_fn(ll) for ll in datalist]
        self.root_dir = data_dir
        self.transform = transform
        self.masks_names = masks_names
        assert (
            len(self.datalist[0]) == len(self.masks_names) + 1
        ), f"Each line in the dataset file must have {len(self.masks_names) + 1} paths: 1 path for image and {len(self.masks_names)} for provided masks, got {len(self.datalist[0])}"

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        abs_paths = [os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]]
        sample = {}
        sample["image"] = self.read_image(abs_paths[0])
        for mask_name, mask_path in zip(self.masks_names, abs_paths[1:]):
            mask = np.array(Image.open(mask_path))
            assert len(mask.shape) == 2, "Masks must be encoded without colourmap"
            sample[mask_name] = mask
        if self.transform:
            sample["names"] = self.masks_names
            sample = self.transform(sample)
            # the names key can be removed by the transformation
            if "names" in sample:
                del sample["names"]
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

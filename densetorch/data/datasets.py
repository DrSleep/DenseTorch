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
        line_to_paths_fn (callable): function to convert a line of data_file
            into paths (img_relpath, msk_relpath, ...).
        masks_names (list of strings): keys for each annotation mask
                                        (e.g., 'segm', 'depth').
        transform_trn (callable, optional): Optional transform
            to be applied on a sample during the training stage.
        transform_val (callable, optional): Optional transform
            to be applied on a sample during the validation stage.
        stage (str): initial stage of dataset - either 'train' or 'val'.

    """
    def __init__(
            self,
            data_file,
            data_dir,
            line_to_paths_fn,
            masks_names,
            transform_trn=None,
            transform_val=None,
            stage='train'):
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = [line_to_paths_fn(ll) for ll in datalist]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = stage
        self.masks_names = masks_names

    def set_stage(self, stage):
        """Define which set of transformation to use.

        Args:
            stage (str): either 'train' or 'val'

        """
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        names = [
            os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]]
        image = self.read_image(names[0])
        masks = [np.array(Image.open(msk_name)) for msk_name in names[1:]]
        sample = {'image' : image}
        for key, mask in zip(self.masks_names, masks):
            assert len(mask.shape) == 2, \
                    'Masks must be encoded without colourmap'
            sample[key] = mask
        sample['names'] = self.masks_names
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)
        del sample['names']
        return sample

    @staticmethod
    def read_image(x):
        """Simple image reader

        Args:
            x (str): path to image.

        Returns image as `np.array`.

        """
        img_arr = np.array(Image.open(x))
        if len(img_arr.shape) == 2: # grayscale
            img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
        return img_arr

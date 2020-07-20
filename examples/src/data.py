import cv2
import numpy as np
import torch

import densetorch as dt


def get_data_loaders(
    crop_size,
    shorter_side,
    low_scale,
    high_scale,
    img_mean,
    img_std,
    img_scale,
    depth_scale,
    ignore_indices,
    num_stages,
    train_dir,
    val_dir,
    train_list_path,
    val_list_path,
    data_list_sep,
    data_list_columns,
    tasks,
    train_batch_size,
    val_batch_size,
):
    train_transforms, val_transforms = get_transforms(
        crop_size=crop_size,
        shorter_side=shorter_side,
        low_scale=low_scale,
        high_scale=high_scale,
        img_mean=img_mean,
        img_std=img_std,
        img_scale=img_scale,
        ignore_indices=ignore_indices,
        num_stages=num_stages,
    )
    train_sets, val_set = get_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        train_transforms=train_transforms,
        val_transforms=val_transforms,
        data_list_sep=data_list_sep,
        data_list_columns=data_list_columns,
        tasks=tasks,
        ignore_indices=ignore_indices,
        depth_scale=depth_scale,
    )
    train_loaders, val_loader = dt.data.get_loaders(
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        train_set=train_sets,
        val_set=val_set,
        num_stages=num_stages,
    )
    return train_loaders, val_loader


def get_transforms(
    crop_size,
    shorter_side,
    low_scale,
    high_scale,
    img_mean,
    img_std,
    img_scale,
    ignore_indices,
    num_stages,
):
    from albumentations import (
        Normalize,
        RandomCrop,
        RandomScale,
        OneOf,
    )
    from albumentations.pytorch import ToTensorV2 as ToTensor
    from augmentations import (
        MultiMaskHorizontalFlip,
        MultiMaskPadIfNeeded,
        MultiMaskRandomScale,
    )

    wrapper = dt.data.albumentations2densetorch

    common_transformations = [
        Normalize(max_pixel_value=1.0 / img_scale, mean=img_mean, std=img_std),
        ToTensor(),
    ]
    train_transforms = []
    for stage in range(num_stages):
        train_transforms.append(
            wrapper(
                [
                    MultiMaskRandomScale(
                        scale_limit=(low_scale[stage], high_scale[stage]), p=0.5,
                    ),
                    MultiMaskPadIfNeeded(
                        min_height=crop_size[stage],
                        min_width=crop_size[stage],
                        border_mode=cv2.BORDER_CONSTANT,
                        value=np.array(img_mean) / img_scale,
                        mask_value=ignore_indices,
                        p=1.0,
                    ),
                    MultiMaskHorizontalFlip(p=0.5,),
                    RandomCrop(height=crop_size[stage], width=crop_size[stage], p=1.0),
                ]
                + common_transformations
            )
        )
    val_transforms = wrapper(common_transformations)
    return train_transforms, val_transforms


def get_datasets(
    train_dir,
    val_dir,
    train_list_path,
    val_list_path,
    train_transforms,
    val_transforms,
    data_list_sep,
    data_list_columns,
    tasks,
    ignore_indices,
    depth_scale,
):
    Dataset = dt.data.MMDataset
    train_sets = [
        Dataset(
            data_file=train_list_path[i],
            data_dir=train_dir[i],
            data_list_sep=data_list_sep,
            data_list_columns=data_list_columns,
            masks_names=tasks,
            ignore_indices=ignore_indices,
            depth_scale=depth_scale,
            transform=train_transforms[i],
        )
        for i in range(len(train_transforms))
    ]
    val_set = Dataset(
        data_file=val_list_path,
        data_dir=val_dir,
        data_list_sep=data_list_sep,
        data_list_columns=data_list_columns,
        masks_names=tasks,
        ignore_indices=ignore_indices,
        depth_scale=depth_scale,
        transform=val_transforms,
    )
    return train_sets, val_set

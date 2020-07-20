import torch.nn as nn

import densetorch as dt


def get_train_loss(classes, ignore_index, task, device):
    """Creates training loss.

    Args:
      classes (int): number of classes.
      ignore_index (int or None): index to ignore when computing the task loss.
      task (str): task's name.
      device (str, torch.Device): device to use.

    Returns:
      Training loss.

    """
    if task == "segm":
        return nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
    elif task == "depth":
        return dt.engine.InvHuberLoss(ignore_index=ignore_index).to(device)
    elif task == "normals":
        return dt.engine.CosineDistanceLoss(ignore_index=ignore_index).to(device)
    else:
        raise ValueError(f"Unknown task {task}")


def get_val_loss(classes, ignore_index, task, device):
    """Creates validation loss.

    Args:
      classes (int): number of classes.
      ignore_index (int or None): index to ignore when computing the task loss.
      task (str): task's name.
      device (str, torch.Device): device to use.

    Returns:
      Validation loss.

    """
    if task == "segm":
        return dt.engine.MeanIoU(num_classes=classes)
    elif task == "depth":
        return dt.engine.RMSE(ignore_index=ignore_index)
    elif task == "normals":
        return dt.engine.AngularError(ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown task {task}")


def get_losses(
    num_classes, ignore_indices, tasks, device,
):
    """Creates training and validation losses.

    Args:
      num_classes (list of int): number of classes per each task.
      ignore_indices (list of (int, None)): indices to ignore when computing the task loss.
      tasks (list of str): tasks' names.
      device (str, torch.Device): device to use.

    Returns:
      List of training and validation losses of length equal to the number of tasks.

    """
    training_losses = []
    validation_losses = []
    for classes, ignore_index, task in zip(num_classes, ignore_indices, tasks):
        training_losses.append(
            get_train_loss(
                classes=classes, ignore_index=ignore_index, task=task, device=device
            )
        )
        validation_losses.append(
            get_val_loss(
                classes=classes, ignore_index=ignore_index, task=task, device=device
            )
        )
    return training_losses, validation_losses

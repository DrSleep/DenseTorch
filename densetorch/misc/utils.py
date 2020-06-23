import json
import os
import random
from datetime import datetime
from inspect import signature

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def broadcast(x, num_times):
    """Given an element, broadcast it number of times and return a list.
    If it is already a list, only the first element will be copied.

    Args:
      x: input.
      num_times (int): how many times to copy the element.

    Returns list of length num_times.
    """
    x_list = make_list(x)
    if len(x_list) == num_times:
        return x_list
    return x_list[:1] * num_times


def get_args(func):
    """Get function's arguments.

    Args:
      func (callable): input function.

    Returns:
      List of positional and keyword arguments.

    """
    return [
        p.name
        for p in signature(func).parameters.values()
        if p.kind == p.POSITIONAL_OR_KEYWORD
    ]


def compute_params(model):
    """Compute the total number of parameters.

    Args:
      model (nn.Module): PyTorch model.

    Returns:
      Total number of parameters - both trainable and non-trainable (int).

    """
    return sum([p.numel() for p in model.parameters()])


def create_optim(optim_type, parameters, **kwargs):
    """Initialise optimisers.

    Args:
      optim_type (string): type of optimiser - either 'SGD' or 'Adam'.
      parameters (iterable): parameters to be optimised.

    Returns:
      An instance of torch.optim.

    Raises:
      ValueError if optim_type is not either of 'SGD' or 'Adam'.

    """
    if optim_type.lower() == "sgd":
        optim = torch.optim.SGD
    elif optim_type.lower() == "adam":
        optim = torch.optim.Adam
    else:
        raise ValueError(
            "Optim {} is not supported. "
            "Only supports 'SGD' and 'Adam' for now.".format(optim_type)
        )
    args = get_args(optim)
    kwargs = {key: kwargs[key] for key in args if key in kwargs}
    return optim(parameters, **kwargs)


def create_scheduler(scheduler_type, optim, **kwargs):
    """Initialise schedulers.

    Args:
      scheduler_type (string): type of scheduler -- either 'poly' or 'multistep'.
      optim (torch.optim): optimiser to which the scheduler will be applied to.

    Returns:
      An instance of torch.optim.lr_scheduler

    Raises:
      ValueError if scheduler_type is not either of 'poly' or 'multistep'.

    """
    if scheduler_type.lower() == "poly":
        scheduler = torch.optim.lr_scheduler.LambdaLR
        lr_lambda_args = get_args(polyschedule)
        lr_lambda_kwargs = {key: kwargs[key] for key in lr_lambda_args if key in kwargs}
        kwargs["lr_lambda"] = polyschedule(**lr_lambda_kwargs)
    elif scheduler_type.lower() == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR
    else:
        raise ValueError(
            "Scheduler {} is not supported. "
            "Only supports 'poly' and 'multistep' for now.".format(scheduler_type)
        )
    args = get_args(scheduler)
    kwargs = {key: kwargs[key] for key in args if key in kwargs}
    return scheduler(optim, **kwargs)


def ctime():
    """Returns current timestamp in the format of hours-minutes-seconds."""
    return datetime.now().strftime("%H:%M:%S")


def load_state_dict(model, state_dict, strict=False):
    if state_dict is None:
        return
    logger = logging.getLogger(__name__)
    # When using dataparallel, 'module.' is prepended to the parameters' keys
    # This function handles the cases when state_dict was saved without DataParallel
    # But the user wants to load it into the model created with dataparallel, and
    # vice versa
    is_module_model_dict = list(model.state_dict().keys())[0].startswith("module")
    is_module_state_dict = list(state_dict.keys())[0].startswith("module")
    if is_module_model_dict and is_module_state_dict:
        pass
    elif is_module_model_dict:
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    elif is_module_state_dict:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    logger.info(model.load_state_dict(state_dict, strict=strict))


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


def set_seed(seed):
    """Setting the random seed across `torch`, `numpy` and `random` libraries.

    Args:
      seed (int): random seed value.

    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def polyschedule(max_epochs, gamma=0.9):
    """Poly-learning rate policy popularised by DeepLab-v2: https://arxiv.org/abs/1606.00915

    Args:
      max_epochs (int): maximum number of epochs, at which the multiplier becomes zero.
      gamma (float): decay factor.

    Returns:
      Callable that takes the current epoch as an argument and returns the learning rate multiplier.

    """

    def polypolicy(epoch):
        return (1.0 - 1.0 * epoch / max_epochs) ** gamma

    return polypolicy


class AverageMeter:
    """Simple running average estimator.

    Args:
      momentum (float): running average decay.

    """

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.avg = 0
        self.val = None

    def update(self, val):
        """Update running average given a new value.

        The new running average estimate is given as a weighted combination \
        of the previous estimate and the current value.

        Args:
          val (float): new value

        """
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1.0 - self.momentum)
        self.val = val


class Saver:
    """Saver class for checkpointing the training progress."""

    def __init__(
        self,
        args,
        ckpt_dir,
        best_val=0,
        condition=lambda x, y: x > y,
        save_interval=100,
        save_several_mode=any,
    ):
        """
        Args:
            args (dict): dictionary with arguments.
            ckpt_dir (str): path to directory in which to store the checkpoint.
            best_val (float or list of floats): initial best value.
            condition (function or list of functions): how to decide whether to save
                                                       the new checkpoint by comparing
                                                       best value and new value (x,y).
            save_interval (int): always save when the interval is triggered.
            save_several_mode (any or all): if there are multiple savers, how to trigger
                                            the saving.

        """
        if save_several_mode not in [all, any]:
            raise ValueError(
                f"save_several_mode must be either all or any, got {save_several_mode}"
            )
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        with open("{}/args.json".format(ckpt_dir), "w") as f:
            json.dump(
                {k: self.serialise(v) for k, v in args.items()},
                f,
                sort_keys=True,
                indent=4,
                ensure_ascii=False,
            )
        self.ckpt_dir = ckpt_dir
        self.best_val = make_list(best_val)
        self.condition = make_list(condition)
        self._counter = 0
        self._save_interval = save_interval
        self.save_several_mode = save_several_mode
        self.logger = logging.getLogger(__name__)

    def _do_save(self, new_val):
        """Check whether need to save"""
        do_save = [
            condition(val, best_val)
            for (condition, val, best_val) in zip(
                self.condition, new_val, self.best_val,
            )
        ]
        return self.save_several_mode(do_save)

    def maybe_save(self, new_val, dict_to_save):
        """Maybe save new checkpoint"""
        self._counter += 1
        if "epoch" not in dict_to_save:
            dict_to_save["epoch"] = self._counter
        new_val = make_list(new_val)
        if self._do_save(new_val):
            for (val, best_val) in zip(new_val, self.best_val):
                self.logger.info(
                    " New best value {:.4f}, was {:.4f}".format(val, best_val)
                )
            self.best_val = new_val
            dict_to_save["best_val"] = new_val
            torch.save(dict_to_save, "{}/checkpoint.pth.tar".format(self.ckpt_dir))
            return True
        elif self._counter % self._save_interval == 0:
            self.logger.info(" Saving at epoch {}.".format(dict_to_save["epoch"]))
            dict_to_save["best_val"] = self.best_val
            torch.save(
                dict_to_save, "{}/counter_checkpoint.pth.tar".format(self.ckpt_dir)
            )
            return False
        return False

    def maybe_load(self, ckpt_path, keys_to_load):
        """Loads existing checkpoint if exists.

        Args:
          ckpt_path (str): path to the checkpoint.
          keys_to_load (list of str): keys to load from the checkpoint.

        Returns the epoch at which the checkpoint was saved.
        """
        keys_to_load = make_list(keys_to_load)
        if not os.path.isfile(ckpt_path):
            return [None] * len(keys_to_load)
        ckpt = torch.load(ckpt_path)
        loaded = []
        for key in keys_to_load:
            val = ckpt.get(key, None)
            if key == "best_val" and val is not None:
                self.best_val = make_list(val)
                self.logger.info(f" Found checkpoint with best values {self.best_val}")
            loaded.append(val)
        return loaded

    @staticmethod
    def serialise(x):
        if isinstance(x, (list, tuple)):
            return [Saver.serialise(item) for item in x]
        elif isinstance(x, np.ndarray):
            return x.tolist()
        elif isinstance(x, (int, float, str)):
            return x
        elif x is None:
            return x
        else:
            pass


class Balancer(nn.Module):
    """Wrapper for balanced multi-GPU training.

    When forward and backward passes are fused into a single nn.Module object, \
    the multi-GPU consumption is distributed more equally across the GPUs.

    Args:
      model (nn.Module): PyTorch module.
      opts (list or single instance of torch.optim): optimisers.
      crits (list or single instance of torch.nn or nn.Module): criterions.
      loss_coeffs (list of single instance of float): loss coefficients.

    """

    def __init__(self, model, opts, crits, loss_coeffs):
        super(Balancer, self).__init__()
        self.model = model
        self.opts = make_list(opts)
        self.crits = make_list(crits)
        self.loss_coeffs = make_list(loss_coeffs)

    def forward(self, inp, targets=None):
        """Forward and (optionally) backward pass.

        When targets are provided, the backward pass is performed.
        Otherwise only the forward pass is done.

        Args:
          inp (torch.tensor): input batch.
          targets (None or torch.tensor): targets batch.

        Returns:
          Forward output if `targets=None`, else returns the loss value.

        """
        outputs = self.model(inp)
        if targets is None:
            return outputs
        outputs = make_list(outputs)
        losses = []
        for out, target, crit, loss_coeff in zip(
            outputs, targets, self.crits, self.loss_coeffs
        ):
            losses.append(
                loss_coeff
                * crit(
                    F.interpolate(
                        out,
                        size=target.size()[1:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(dim=1),
                    target.squeeze(dim=1),
                )
            )
        for opt in self.opts:
            opt.zero_grad()
        loss = torch.stack(losses).mean()
        loss.backward()
        for opt in self.opts:
            opt.step()
        return loss

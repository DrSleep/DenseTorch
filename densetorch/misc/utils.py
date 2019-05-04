import random
from datetime import datetime
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_args(func):
    """Get function's arguments.

    Args:
      func (callable): input function.

    Returns:
      List of positional and keyword arguments.

    """
    return [p.name for p in signature(func).parameters.values()
            if p.kind == p.POSITIONAL_OR_KEYWORD]

def compute_params(model):
    """Compute the total number of parameters.

    Args:
      model (nn.Module): PyTorch model.

    Returns:
      Total number of parameters - both trainable and non-trainable (int).

    """
    return sum([p.numel() for p in model.parameters()])

def create_optim(enc, parameters, **kwargs):
    """Initialise optimisers.

    Args:
      enc (string): type of optimiser - either 'SGD' or 'Adam'.
      parameters (iterable): parameters to be optimised.

    Returns:
      An instance of torch.optim.

    Raises:
      ValueError if enc is not either of 'SGD' or 'Adam'.

    """
    if enc == 'SGD':
        optim = torch.optim.SGD
    elif enc == 'Adam':
        optim = torch.optim.Adam
    else:
        raise ValueError("Optim {} is not supported. "
                         "Only supports 'SGD' and 'Adam' for now.".format(enc))
    args = get_args(optim)
    kwargs = {key : kwargs[key] for key in args if key in kwargs}
    return optim(parameters, **kwargs)

def ctime():
    """Returns current timestamp in the format of hours-minutes-seconds."""
    return datetime.now().strftime('%H:%M:%S')

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
            self.avg = self.avg * self.momentum + val * (1. - self.momentum)
        self.val = val

class Saver:
    """Saver class for monitoring the training progress.

    Given initial values and comparison functions, Saver keeps track of newly \
    added values and updates them in case all of the new values satisfy the \
    corresponding comparison functions.

    Args:
      init_vals (list): initial values. Represent lower bounds for performance
                        of each task.
      comp_fns (list): list of comparison functions.
                       Each function takes two inputs and produces
                       one boolean output.
                       Each newly provided value
                       will be compared against the initial value using
                       the corresponding comparison function.

    """
    def __init__(self, init_vals, comp_fns):
        self.vals = init_vals
        self.comp_fns = comp_fns

    def save(self, new_vals):
        """Saving criterion.

        Checks whether the saving criterion is trigerred. The saving occurs \
        when all newly added values satisfy their corresponding comparison \
        functions.

        Args:
          new_vals (list): new values for comparison.

        Returns:
          `True` if all comparison functions return `True`.
          Otherwise, returns `False`.

        """
        update_vals = []
        for (old_val, new_val, op) in zip(self.vals, new_vals, self.comp_fns):
            if op(new_val, old_val):
                update_vals.append(new_val)
            else:
                return False
        self.vals = update_vals
        return True

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
        for out, target, crit, loss_coeff in \
                zip(outputs, targets, self.crits, self.loss_coeffs):
            losses.append(loss_coeff *
                          crit(
                              F.interpolate(
                                  out,
                                  size=target.size()[1:],
                                  mode='bilinear',
                                  align_corners=False).squeeze(dim=1),
                              target.squeeze(dim=1)))
        for opt in self.opts:
            opt.zero_grad()
        loss = torch.stack(losses).mean()
        loss.backward()
        for opt in self.opts:
            opt.step()
        return loss

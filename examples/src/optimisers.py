import numpy as np
import torch.nn as nn

import densetorch as dt


def get_lr_schedulers(
    enc_optim,
    dec_optim,
    enc_lr_gamma,
    dec_lr_gamma,
    enc_scheduler_type,
    dec_scheduler_type,
    epochs_per_stage,
):
    milestones = np.cumsum(epochs_per_stage)
    max_epochs = milestones[-1]
    schedulers = [
        dt.misc.create_scheduler(
            scheduler_type=enc_scheduler_type,
            optim=enc_optim,
            gamma=enc_lr_gamma,
            milestones=milestones,
            max_epochs=max_epochs,
        ),
        dt.misc.create_scheduler(
            scheduler_type=dec_scheduler_type,
            optim=dec_optim,
            gamma=dec_lr_gamma,
            milestones=milestones,
            max_epochs=max_epochs,
        ),
    ]
    return schedulers


def get_optimisers(
    model,
    enc_optim_type,
    enc_lr,
    enc_weight_decay,
    enc_momentum,
    dec_optim_type,
    dec_lr,
    dec_weight_decay,
    dec_momentum,
):
    if isinstance(model, nn.DataParallel):
        model = model.module
    assert (
        len(model) == 2
    ), f"The network must have 2 modules for encoder and decoder, correspondingly, got {len(model.module):d}"
    optimisers = [
        dt.misc.create_optim(
            optim_type=enc_optim_type,
            parameters=model[0].parameters(),
            lr=enc_lr,
            weight_decay=enc_weight_decay,
            momentum=enc_momentum,
        ),
        dt.misc.create_optim(
            optim_type=dec_optim_type,
            parameters=model[1].parameters(),
            lr=dec_lr,
            weight_decay=dec_weight_decay,
            momentum=dec_momentum,
        ),
    ]
    return optimisers


def get_optimisers_and_schedulers(
    model,
    enc_optim_type,
    enc_lr,
    enc_weight_decay,
    enc_momentum,
    enc_lr_gamma,
    enc_scheduler_type,
    dec_optim_type,
    dec_lr,
    dec_weight_decay,
    dec_momentum,
    dec_lr_gamma,
    dec_scheduler_type,
    epochs_per_stage,
):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type=enc_optim_type,
        enc_lr=enc_lr,
        enc_weight_decay=enc_weight_decay,
        enc_momentum=enc_momentum,
        dec_optim_type=dec_optim_type,
        dec_lr=dec_lr,
        dec_weight_decay=dec_weight_decay,
        dec_momentum=dec_momentum,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=enc_lr_gamma,
        dec_lr_gamma=dec_lr_gamma,
        enc_scheduler_type=enc_scheduler_type,
        dec_scheduler_type=dec_scheduler_type,
        epochs_per_stage=epochs_per_stage,
    )
    return optimisers, schedulers

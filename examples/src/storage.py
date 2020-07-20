import numpy as np

import densetorch as dt


def parse_saving_criterions(saving_criterions):
    def parse_saving_criterion(saving_criterion):
        if saving_criterion == "up":
            return lambda x, y: x > y
        elif saving_criterion == "down":
            return lambda x, y: x < y

    return [
        parse_saving_criterion(saving_criterion)
        for saving_criterion in saving_criterions
    ]


def setup_checkpoint_and_maybe_restore(args, model, optimisers, schedulers):
    saver = dt.misc.Saver(
        args=vars(args),
        ckpt_dir=args.ckpt_dir,
        best_val=args.initial_values,
        condition=parse_saving_criterions(args.saving_criterions),
    )
    (
        restart_epoch,
        _,
        model_state_dict,
        optims_state_dict,
        scheds_state_dict,
    ) = saver.maybe_load(
        ckpt_path=args.ckpt_path,
        keys_to_load=["epoch", "best_val", "model", "optimisers", "schedulers"],
    )
    if restart_epoch is None:
        restart_epoch = 0
    dt.misc.load_state_dict(model, model_state_dict)
    if optims_state_dict is not None:
        for optim, optim_state_dict in zip(optimisers, optims_state_dict):
            optim.load_state_dict(optim_state_dict)
    if scheds_state_dict is not None:
        for sched, sched_state_dict in zip(schedulers, scheds_state_dict):
            sched.load_state_dict(sched_state_dict)
    # Calculate from which stage and which epoch to restart the training
    total_epoch = restart_epoch
    all_epochs = np.cumsum(args.epochs_per_stage)
    restart_stage = sum(restart_epoch >= all_epochs)
    if restart_stage > 0:
        restart_epoch -= all_epochs[restart_stage - 1]

    return saver, total_epoch, restart_epoch, restart_stage

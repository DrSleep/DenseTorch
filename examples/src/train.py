# general libs
import logging
import numpy as np

# pytorch libs
import torch
import torch.nn as nn

# densetorch wrapper
import densetorch as dt

# configuration for light-weight refinenet
from arguments import get_arguments
from losses import get_losses
from data import get_data_loaders
from network import get_network
from optimisers import get_optimisers_and_schedulers
from storage import setup_checkpoint_and_maybe_restore


def main():
    args = get_arguments()
    logger = logging.getLogger(__name__)
    torch.backends.cudnn.deterministic = True
    dt.misc.set_seed(args.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Network
    network = get_network(
        enc_backbone=args.enc_backbone,
        enc_pretrained=args.enc_pretrained,
        enc_return_layers=args.enc_return_layers,
        dec_backbone=args.dec_backbone,
        dec_combine_layers=args.dec_combine_layers,
        num_classes=args.num_classes,
        device=device,
    )
    # Losses
    training_losses, validation_losses = get_losses(
        num_classes=args.num_classes,
        ignore_indices=args.ignore_indices,
        tasks=args.tasks,
        device=device,
    )
    # Data
    train_loaders, val_loader = get_data_loaders(
        crop_size=args.crop_size,
        shorter_side=args.shorter_side,
        low_scale=args.low_scale,
        high_scale=args.high_scale,
        img_mean=args.img_mean,
        img_std=args.img_std,
        img_scale=args.img_scale,
        depth_scale=args.depth_scale,
        ignore_indices=args.ignore_indices,
        num_stages=args.num_stages,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        train_list_path=args.train_list_path,
        val_list_path=args.val_list_path,
        data_list_sep=args.data_list_sep,
        data_list_columns=args.data_list_columns,
        tasks=args.tasks,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
    )
    # Optimisers
    optimisers, schedulers = get_optimisers_and_schedulers(
        model=network,
        enc_optim_type=args.enc_optim_type,
        enc_lr=args.enc_lr,
        enc_weight_decay=args.enc_weight_decay,
        enc_momentum=args.enc_momentum,
        enc_lr_gamma=args.enc_lr_gamma,
        enc_scheduler_type=args.enc_scheduler_type,
        dec_optim_type=args.dec_optim_type,
        dec_lr=args.dec_lr,
        dec_weight_decay=args.dec_weight_decay,
        dec_momentum=args.dec_momentum,
        dec_lr_gamma=args.dec_lr_gamma,
        dec_scheduler_type=args.dec_scheduler_type,
        epochs_per_stage=args.epochs_per_stage,
    )
    # Checkpoint
    (
        saver,
        total_epoch,
        restart_epoch,
        restart_stage,
    ) = setup_checkpoint_and_maybe_restore(
        args, model=network, optimisers=optimisers, schedulers=schedulers,
    )
    for stage in range(restart_stage, args.num_stages):
        if stage > restart_stage:
            restart_epoch = 0
        for epoch in range(restart_epoch, args.epochs_per_stage[stage]):
            logger.info(f"Training: stage {stage} epoch {epoch}")
            dt.engine.train(
                model=network,
                opts=optimisers,
                crits=training_losses,
                dataloader=train_loaders[stage],
                freeze_bn=args.freeze_bn[stage],
                grad_norm=args.grad_norm[stage],
                loss_coeffs=args.tasks_loss_weights,
            )
            total_epoch += 1
            for scheduler in schedulers:
                scheduler.step(total_epoch)
            if (epoch + 1) % args.val_every[stage] == 0:
                logger.info(f"Validation: stage {stage} epoch {epoch}")
                vals = dt.engine.validate(
                    model=network, metrics=validation_losses, dataloader=val_loader,
                )
                saver.maybe_save(
                    new_val=vals,
                    dict_to_save={
                        "model": network.state_dict(),
                        "epoch": total_epoch,
                        "optimisers": [
                            optimiser.state_dict() for optimiser in optimisers
                        ],
                        "schedulers": [
                            scheduler.state_dict() for scheduler in schedulers
                        ],
                    },
                )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
        level=logging.INFO,
    )
    main()

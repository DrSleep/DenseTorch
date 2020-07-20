import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..misc.utils import AverageMeter, make_list


def maybe_cast_target_to_long(target):
    """Torch losses usually work on Long types"""
    if target.dtype == torch.uint8:
        return target.to(torch.long)
    return target


def get_input_and_targets(sample, dataloader, device):
    if isinstance(sample, dict):
        input = sample["image"].float().to(device)
        targets = [
            maybe_cast_target_to_long(sample[k].to(device))
            for k in dataloader.dataset.masks_names
        ]
    elif isinstance(sample, (tuple, list)):
        input, *targets = sample
        input = input.float().to(device)
        targets = [maybe_cast_target_to_long(target.to(device)) for target in targets]
    else:
        raise Exception(f"Sample type {type(sample)} is not supported.")
    return input, targets


def train(
    model, opts, crits, dataloader, loss_coeffs=(1.0,), freeze_bn=False, grad_norm=0.0
):
    """Full Training Pipeline.

    Supports multiple optimisers, multiple criteria, \
    multiple losses, multiple outputs.
    Assumes that the model.eval() property has been set up properly before the\
     function call, that the dataloader outputs have the correct type, that \
     the model outputs do not require any post-processing bar the upsampling \
     to the target size.
    Criteria, loss_coeff, and model's outputs all must have the same length, \
    and correspond to the same keys as in the ordered dict of dataloader's \
    sample.

    Args:
        model : PyTorch model object.
        opts  : list of optimisers.
        crits : list of criterions.
        dataloader : iterable over samples.
                     Each sample must contain `image` key and
                     >= 1 optional keys.
        loss_coeffs : list of coefficients for each loss term.
        freeze_bn: whether to freeze batch norm parameters in the module.
        grad_norm: if > 0, clip gradients' norm to this value.

    """
    model.train()
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    device = next(model.parameters()).device
    opts = make_list(opts)
    crits = make_list(crits)
    loss_coeffs = make_list(loss_coeffs)
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader)

    for sample in pbar:
        loss = 0.0
        input, targets = get_input_and_targets(
            sample=sample, dataloader=dataloader, device=device
        )
        outputs = model(input)
        outputs = make_list(outputs)
        for out, target, crit, loss_coeff in zip(outputs, targets, crits, loss_coeffs):
            loss += loss_coeff * crit(
                F.interpolate(
                    out, size=target.size()[-2:], mode="bilinear", align_corners=False
                ).squeeze(dim=1),
                target.squeeze(dim=1),
            )
        for opt in opts:
            opt.zero_grad()
        loss.backward()
        if grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        for opt in opts:
            opt.step()

        loss_meter.update(loss.item())
        pbar.set_description(
            "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )


def trainbal(model, dataloader):
    """Full Training Pipeline with balanced model.

    Assumes that the model.eval() property has been set up properly \
    before the function call, that the dataloader outputs have the correct type,\
    that the model outputs do not require any post-processing bar \
    the upsampling to the target size.

    Args:
        model : PyTorch model object.
        dataloader : iterable over samples.
                     Each sample must contain `image` key and
                     >= 1 optional keys.

    """
    device = next(model.parameters()).device
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader)

    for sample in pbar:
        loss = 0.0
        input, targets = get_input_and_targets(
            sample=sample, dataloader=dataloader, device=device
        )
        loss = model(input, targets)
        loss_meter.update(loss.item())
        pbar.set_description(
            "Loss {:.3f} | Avg. Loss {:.3f}".format(loss.item(), loss_meter.avg)
        )


def validate(model, metrics, dataloader):
    """Full Validation Pipeline.

    Support multiple metrics (but 1 per modality), multiple outputs.
    Assumes that the dataloader outputs have the correct type, that the model \
    outputs do not require any post-processing bar the upsampling \
    to the target size.
    Metrics and model's outputs must have the same length, and correspond to \
    the same keys as in the ordered dict of dataloader's sample.

    Args:
        model : PyTorch model object.
        metrics  : list of metric classes. Each metric class must have update
                   and val functions, and must have 'name' attribute.
        dataloader : iterable over samples.
                     Each sample must contain `image` key and
                     >= 1 optional keys.

    """
    device = next(model.parameters()).device
    model.eval()
    metrics = make_list(metrics)
    for metric in metrics:
        metric.reset()

    pbar = tqdm(dataloader)

    def get_val(metrics):
        results = [(m.name, m.val()) for m in metrics]
        names, vals = list(zip(*results))
        out = ["{} : {:4f}".format(name, val) for name, val in results]
        return vals, " | ".join(out)

    with torch.no_grad():
        for sample in pbar:
            input, targets = get_input_and_targets(
                sample=sample, dataloader=dataloader, device=device
            )
            targets = [target.squeeze(dim=1) for target in targets]
            outputs = model(input)
            outputs = make_list(outputs)
            for out, target, metric in zip(outputs, targets, metrics):
                metric.update(
                    F.interpolate(
                        out,
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(dim=1),
                    target,
                )
            pbar.set_description(get_val(metrics)[1])
    vals, _ = get_val(metrics)
    print("----" * 5)
    return vals

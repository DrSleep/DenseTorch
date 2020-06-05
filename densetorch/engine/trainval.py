import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..misc.utils import AverageMeter, make_list


def train(model, opts, crits, dataloader, loss_coeffs=(1.0,)):
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

    """
    device = next(model.parameters()).device
    opts = make_list(opts)
    crits = make_list(crits)
    loss_coeffs = make_list(loss_coeffs)
    loss_meter = AverageMeter()
    pbar = tqdm(dataloader)

    for sample in pbar:
        loss = 0.0
        input = sample["image"].float().to(device)
        targets = [sample[k].to(device) for k in dataloader.dataset.masks_names]
        outputs = model(input)
        outputs = make_list(outputs)
        for out, target, crit, loss_coeff in zip(outputs, targets, crits, loss_coeffs):
            loss += loss_coeff * crit(
                F.interpolate(
                    out, size=target.size()[1:], mode="bilinear", align_corners=False
                ).squeeze(dim=1),
                target.squeeze(dim=1),
            )
        for opt in opts:
            opt.zero_grad()
        loss.backward()
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
        input = sample["image"].float().to(device)
        targets = [sample[k].to(device) for k in dataloader.dataset.masks_names]
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
            input = sample["image"].float().to(device)
            targets = [
                sample[k].squeeze(dim=1).numpy() for k in dataloader.dataset.masks_names
            ]
            outputs = model(input)
            outputs = make_list(outputs)
            for out, target, metric in zip(outputs, targets, metrics):
                metric.update(
                    F.interpolate(
                        out, size=target.shape[1:], mode="bilinear", align_corners=False
                    )
                    .squeeze(dim=1)
                    .cpu()
                    .numpy(),
                    target,
                )
            pbar.set_description(get_val(metrics)[1])
    vals, _ = get_val(metrics)
    print("----" * 5)
    return vals

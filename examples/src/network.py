import logging
import torch.nn as nn

import densetorch as dt


def get_encoder(backbone, pretrained, return_layers):
    """Creates encoder.

    Args:
      backbone (str): type of backbone.
      pretrained (int, bool): whether to use pretrained weights.
      return_layers (list of int): indices of layers to return.

    Returns:
      Encoder network (`nn.Module`).

    """
    if backbone == "xception65":
        enc_fn = dt.nn.xception65
    elif backbone == "resnet18":
        enc_fn = dt.nn.resnet18
    elif backbone == "resnet34":
        enc_fn = dt.nn.resnet34
    elif backbone == "resnet50":
        enc_fn = dt.nn.resnet50
    elif backbone == "resnet101":
        enc_fn = dt.nn.resnet101
    elif backbone == "resnet152":
        enc_fn = dt.nn.resnet152
    elif backbone == "mobilenetv2":
        enc_fn = dt.nn.mobilenetv2
    return enc_fn(pretrained=pretrained, return_idx=return_layers)


def get_decoder(enc_info, backbone, combine_layers, num_classes):
    """Creates decoder.

    Args:
      enc_info (dict): kwargs for the decoder with the necessary encoder info.
      backbone (str): type of backbone.
      combine_layers (list of int): which input layers to combine.
      num_classes (list of int): number of classes per each task.

    Returns:
      Decoder network (`nn.Module`).

    """
    if backbone == "dlv3plus":
        dec_fn = dt.nn.MTDLv3plus
    elif backbone == "lwrefinenet":
        dec_fn = dt.nn.MTLWRefineNet
    return dec_fn(collapse_ind=combine_layers, num_classes=num_classes, **enc_info)


def get_network(
    enc_backbone,
    enc_pretrained,
    enc_return_layers,
    dec_backbone,
    dec_combine_layers,
    num_classes,
    device,
):
    """Creates encoder-decoder network.

    Args:
      enc_backbone (str): type of encoder.
      enc_pretrained (int, bool): whether to use pretrained encoder weights.
      enc_return_layers (list of int): which layers to return from the encoder.
      dec_backbone (str): type of decoder.
      dec_combine_layers (list of int): which input layers to combine in the decoder.
      num_classes (list of int): number of output classes per each task.
      device (str, torch.Device): device to use.

    Returns:
      Encoder-decoder network (`nn.Module`).

    """
    logger = logging.getLogger(__name__)
    encoder = get_encoder(
        backbone=enc_backbone,
        pretrained=enc_pretrained,
        return_layers=enc_return_layers,
    )
    decoder = get_decoder(
        enc_info=encoder.info(),
        backbone=dec_backbone,
        combine_layers=dec_combine_layers,
        num_classes=num_classes,
    )
    network = nn.Sequential(encoder, decoder).to(device)
    if device == "cuda":
        network = nn.DataParallel(network)
    logger.info(
        f" Loaded Network with encoder {enc_backbone} and decoder {dec_backbone}."
        f" Encoder Pre-Trained = {bool(enc_pretrained)}. #PARAMS = {dt.misc.compute_params(network) / 1e6:3.2f}M"
    )
    return network

from collections import namedtuple
from functools import partial

import numpy as np
import torch.nn as nn

from .inventory import Config8, Config16, model_urls
from .layer_factory import XceptionBlock, batchnorm, conv3x3, conv1x1, sepconv_bn
from .model_zoo import load_url
from ..misc.utils import make_list

__all__ = ["xception65"]


class Xception65(nn.Module):
    """Xception-65 network definition.

    More information about the model: https://arxiv.org/abs/1802.02611

    Args:
      return_idx (list or int): indices of the layers to be returned
                                during the forward pass.
      config (int): whether to use OS-16 or OS-8 setup.
                    OS-X means that the output will be of 1/X size of the input.

    Attributes:
      in_planes (int): number of channels in the stem block.

    """

    def __init__(self, return_idx=[20], config=16):
        super(Xception65, self).__init__()
        self.return_idx = make_list(return_idx)
        config = Config16 if config == 16 else Config8
        self.inplanes = 32
        self.n_layers = len(config) - 1  # for rates
        self.rates = config["rates"]
        # STEM
        self.entry_flow_conv1_1 = conv3x3(3, 32, stride=2, bias=False)
        self.entry_flow_conv1_1_BN = batchnorm(32)
        self.relu = nn.ReLU(inplace=False)
        self.pool1x1 = nn.AdaptiveAvgPool2d(1)
        self.entry_flow_conv1_2 = conv3x3(32, 64, stride=1, bias=False)
        self.entry_flow_conv1_2_BN = batchnorm(64)
        # Xception
        for i in range(self.n_layers):
            setattr(self, "layer{}".format(i + 1), self._make_layer(config[i]))
        self._out_c = [config[idx].filters[-1] for idx in self.return_idx]

    def forward(self, x):
        y = self.entry_flow_conv1_1(x)
        y = self.entry_flow_conv1_1_BN(y)
        y = self.relu(y)
        y = self.entry_flow_conv1_2(y)
        y = self.entry_flow_conv1_2_BN(y)
        y = self.relu(y)
        outs = []
        outs.append(self.layer1(y))  # 128, x / 4
        outs.append(self.layer2(outs[-1]))  # 256, x / 8
        outs.append(self.layer3(outs[-1]))
        outs.append(self.layer4(outs[-1]))
        outs.append(self.layer5(outs[-1]))
        outs.append(self.layer6(outs[-1]))
        outs.append(self.layer7(outs[-1]))
        outs.append(self.layer8(outs[-1]))
        outs.append(self.layer9(outs[-1]))
        outs.append(self.layer10(outs[-1]))
        outs.append(self.layer11(outs[-1]))
        outs.append(self.layer12(outs[-1]))
        outs.append(self.layer13(outs[-1]))
        outs.append(self.layer14(outs[-1]))
        outs.append(self.layer15(outs[-1]))
        outs.append(self.layer16(outs[-1]))
        outs.append(self.layer17(outs[-1]))
        outs.append(self.layer18(outs[-1]))
        outs.append(self.layer19(outs[-1]))
        outs.append(self.layer20(outs[-1]))  # 1024 x / 16
        outs.append(self.layer21(outs[-1]))  # 2048 x / 32
        return [outs[idx] for idx in self.return_idx]

    def _make_layer(self, config):
        """Create XceptionBlock layer.

        Args:
          config (namedtuple): defines the setup of XceptionBlock.

        Returns:
          `nn.Sequential' instance.

        """
        stride = config.stride
        in_planes = config.in_planes
        filters = config.filters
        rate = config.rate
        depth_activation = config.depth_activation
        skip_return = config.skip_return
        agg = config.agg
        layers = []
        layers.append(
            XceptionBlock(
                in_planes,
                filters,
                stride=stride,
                rate=rate,
                depth_activation=depth_activation,
                skip_return=skip_return,
                agg=agg,
            )
        )
        return nn.Sequential(*layers)


def xception65(pretrained=False, **kwargs):
    """Constructs the Xception-65 network.

    Args:
      pretrained (bool): whether to load pre-trained weights.

    Returns:
      `nn.Module` instance.

    """
    model = Xception65(**kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls["xception65"]))
    return model

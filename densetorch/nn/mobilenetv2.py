import numpy as np
import torch.nn as nn

from .inventory import model_urls
from .layer_factory import convbnrelu, InvertedResidualBlock, conv1x1
from .model_zoo import load_url
from ..misc.utils import make_list

__all__ = ["mobilenetv2"]


class MobileNetv2(nn.Module):
    """MobileNet-v2 definition.

    More information about the model: https://arxiv.org/abs/1801.04381

    Args:
      return_idx (list or int): indices of the layers to be returned
                                during the forward pass.

    Attributes:
      mobilenet_config (list): list of definitions of each layer that includes
                               expansion rate, number of output channels,
                               number of repeats, stride.
      in_planes (int): number of channels in the stem block.

    """

    # expansion rate, output channels, number of repeats, stride
    mobilenet_config = [
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    in_planes = 32  # number of input channels
    num_layers = len(mobilenet_config)

    def __init__(self, return_idx=[6]):
        super(MobileNetv2, self).__init__()
        self.return_idx = make_list(return_idx)
        self.layer1 = convbnrelu(
            3, self.in_planes, kernel_size=3, stride=2, act=nn.ReLU6(inplace=True)
        )
        c_layer = 2
        for t, c, n, s in self.mobilenet_config:
            layers = []
            for idx in range(n):
                layers.append(
                    InvertedResidualBlock(
                        self.in_planes,
                        c,
                        expansion_factor=t,
                        stride=s if idx == 0 else 1,
                    )
                )
                self.in_planes = c
            setattr(self, "layer{}".format(c_layer), nn.Sequential(*layers))
            c_layer += 1
        self._out_c = [self.mobilenet_config[idx][1] for idx in self.return_idx]

    def forward(self, x):
        outs = []
        x = self.layer1(x)
        outs.append(self.layer2(x))  # 16, x / 2
        outs.append(self.layer3(outs[-1]))  # 24, x / 4
        outs.append(self.layer4(outs[-1]))  # 32, x / 8
        outs.append(self.layer5(outs[-1]))  # 64, x / 16
        outs.append(self.layer6(outs[-1]))  # 96, x / 16
        outs.append(self.layer7(outs[-1]))  # 160, x / 32
        outs.append(self.layer8(outs[-1]))  # 320, x / 32
        return [outs[idx] for idx in self.return_idx]


def mobilenetv2(pretrained=True, **kwargs):
    """Constructs the mobilenet-v2 network.

    Args:
      pretrained (bool): whether to load pre-trained weights.

    Returns:
      `nn.Module` instance.

    """
    model = MobileNetv2(**kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls["mobilenetv2"]))
    return model

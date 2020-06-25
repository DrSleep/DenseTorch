import torch.nn as nn

from .inventory import model_urls
from .layer_factory import BasicBlock, Bottleneck
from .model_zoo import load_url
from ..misc.utils import make_list

__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]


class ResNet(nn.Module):
    """Residual network definition.

    More information about the model: https://arxiv.org/abs/1512.03385

    Args:
        block (nn.Module): type of building block (Basic or Bottleneck).
        layers (list of ints): number of blocks in each layer.
        return_idx (list or int): indices of the layers to be returned
                                  during the forward pass.

    Attributes:
      in_planes (int): number of channels in the stem block.

    """

    def __init__(self, block, layers, return_idx=[0, 1, 2, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self._out_c = []
        self.return_idx = make_list(return_idx)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self._out_c = [
            out_c for idx, out_c in enumerate(self._out_c) if idx in self.return_idx
        ]

    def _make_layer(self, block, planes, blocks, stride=1):
        """Create residual layer.

        Args:
            block (nn.Module): type of building block (Basic or Bottleneck).
            planes (int): number of input channels.
            blocks (int): number of blocks.
            stride (int): stride inside the first block.

        Returns:
            `nn.Sequential' instance of all created layers.

        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        self._out_c.append(self.inplanes)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 1/4
        outs = []
        outs.append(self.layer1(x))  # 1/4
        outs.append(self.layer2(outs[-1]))  # 1/8
        outs.append(self.layer3(outs[-1]))  # 1/16
        outs.append(self.layer4(outs[-1]))  # 1/32
        return [outs[idx] for idx in self.return_idx]


def resnet18(pretrained=False, **kwargs):
    """Constructs the ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        `nn.Module` instance.

    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet18"]))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs the ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        `nn.Module` instance.

    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet34"]))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs the ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        `nn.Module` instance.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet50"]))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs the ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        `nn.Module` instance.

    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet101"]))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs the ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        `nn.Module` instance.

    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls["resnet152"]))
    return model

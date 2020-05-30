import torch.nn as nn


def batchnorm(in_planes, affine=True, eps=1e-5, momentum=0.1):
    """2D Batch Normalisation.

    Args:
      in_planes (int): number of input channels.
      affine (bool): whether to add learnable affine parameters.
      eps (float): stability constant in the denominator.
      momentum (float): running average decay coefficient.

    Returns:
      `nn.BatchNorm2d' instance.

    """
    return nn.BatchNorm2d(in_planes, affine=affine, eps=eps, momentum=momentum)


def conv3x3(in_planes, out_planes, stride=1, dilation=1, groups=1, bias=False):
    """2D 3x3 convolution.

    Args:
      in_planes (int): number of input channels.
      out_planes (int): number of output channels.
      stride (int): stride of the operation.
      dilation (int): dilation rate of the operation.
      groups (int): number of groups in the operation.
      bias (bool): whether to add learnable bias parameter.

    Returns:
      `nn.Conv2d' instance.

    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def conv1x1(in_planes, out_planes, stride=1, groups=1, bias=False):
    """2D 1x1 convolution.

    Args:
      in_planes (int): number of input channels.
      out_planes (int): number of output channels.
      stride (int): stride of the operation.
      groups (int): number of groups in the operation.
      bias (bool): whether to add learnable bias parameter.

    Returns:
      `nn.Conv2d' instance.

    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=groups,
        bias=bias,
    )


def convbnrelu(
    in_planes, out_planes, kernel_size, stride=1, groups=1, act=nn.ReLU(inplace=False)
):
    """2D convolution => BatchNorm => activation sequence.

    Args:
      in_planes (int): number of input channels.
      out_planes (int): number of output channels.
      kernel_size (int): kernel size of the convolution.
      stride (int): stride of the convolution.
      groups (int): number of groups in the convolution.
      act (None or nn.Module): activation function.

    Returns:
      `nn.Sequential' instance.

    """
    modules = []
    modules.append(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride=stride,
            padding=int(kernel_size / 2.0),
            groups=groups,
            bias=False,
        )
    )
    modules.append(batchnorm(out_planes))
    if act is not None:
        modules.append(act)
    return nn.Sequential(*modules)


def sepconv_bn(
    in_planes,
    out_planes,
    stride=1,
    bias=False,
    rate=1,
    depth_activation=False,
    eps=1e-3,
):
    """Act. (opt.) => 2D 3x3 grouped convolution => BatchNorm =>
       Act. (opt.) => 2D 1x1 convolution => BatchNorm => Act. (opt.)

    Args:
      in_planes (int): number of input channels.
      out_planes (int): number of output channels.
      stride (int): stride of the convolution.
      bias (bool): whether to add learnable bias parameter for convolutions.
      rate (int): dilation rate in the convolution.
      depth_activation (bool): whether to use activation function (ReLU).
      eps (float): stability constant in the denominator for BatchNorm.

    Returns:
      `nn.Sequential' instance.

    """
    modules = []
    if not depth_activation:
        modules.append(nn.ReLU(inplace=False))
    modules.append(
        conv3x3(
            in_planes,
            in_planes,
            stride=stride,
            bias=bias,
            dilation=rate,
            groups=in_planes,
        )
    )
    modules.append(batchnorm(in_planes, eps=eps))
    if depth_activation:
        modules.append(nn.ReLU(inplace=False))
    modules.append(conv1x1(in_planes, out_planes, bias=bias))
    modules.append(batchnorm(out_planes, eps=eps))
    if depth_activation:
        modules.append(nn.ReLU(inplace=False))
    return nn.Sequential(*modules)


class CRPBlock(nn.Module):
    """Light-Weight Chained Residual Pooling (CRP) block.

    Residual sequence of maxpool5x5 => conv1x1.

    Args:
      in_planes (int): number of input channels.
      out_planes (int): number of output channels.
      stages (int): number of times the design is repeated
                    (with new weights)
      groups (bool): whether to do groupwise convolution.

    """

    def __init__(self, in_planes, out_planes, n_stages, groups=False):
        super(CRPBlock, self).__init__()
        for i in range(n_stages):
            setattr(
                self,
                "{}_{}".format(i + 1, "outvar_dimred"),
                conv1x1(
                    in_planes if (i == 0) else out_planes,
                    out_planes,
                    stride=1,
                    groups=in_planes if groups else 1,
                    bias=False,
                ),
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, "{}_{}".format(i + 1, "outvar_dimred"))(top)
            x = top + x
        return x


class InvertedResidualBlock(nn.Module):
    """Inverted Residual Block.

    Conv1x1-BN-ReLU6 => Conv3x3-BN-ReLU6 => Conv1x1-BN.
    Optionally, residual if the number of input and outputs channels are equal,
    and stride is 1.

    Args:
      in_planes (int): number of input channels.
      out_planes (int): number of output channels.
      expansion_factor (int): the growth factor of the bottleneck layer.
      stride (int): stride value of the bottleneck layer.

    """

    def __init__(self, in_planes, out_planes, expansion_factor, stride=1):
        super(InvertedResidualBlock, self).__init__()
        mid_planes = in_planes * expansion_factor
        self.residual = (in_planes == out_planes) and (stride == 1)
        self.output = nn.Sequential(
            convbnrelu(in_planes, mid_planes, 1, act=nn.ReLU6(inplace=True)),
            convbnrelu(
                mid_planes,
                mid_planes,
                3,
                stride=stride,
                groups=mid_planes,
                act=nn.ReLU6(inplace=True),
            ),
            convbnrelu(mid_planes, out_planes, 1, act=None),
        )

    def forward(self, x):
        residual = x
        out = self.output(x)
        if self.residual:
            return out + residual
        else:
            return out


class XceptionBlock(nn.Module):
    """Xception Block.

    SepConv-BN => SepConv-BN => SepConv-BN.

    Args:
      in_planes (int): number of input channels.
      filters (list of int): number of channels per each SepConv-BN block.
      rate (int): dilation rate in the convolution.
      depth_activation (bool): whether to use activation function (ReLU).
      stride (int): stride value of the last layer.
      skip_return (bool): whether to return residual
                          together with the output.
      agg (str): whether to apply convolution on the residual before summing up
                 with the main output.

    """

    def __init__(
        self,
        in_planes,
        filters,
        rate=1,
        depth_activation=False,
        stride=1,
        skip_return=False,
        agg="sum",
    ):
        super(XceptionBlock, self).__init__()
        self.conv1 = sepconv_bn(
            in_planes,
            filters[0],
            stride=1,
            rate=rate,
            depth_activation=depth_activation,
        )
        self.conv2 = sepconv_bn(
            filters[0],
            filters[1],
            stride=1,
            rate=rate,
            depth_activation=depth_activation,
        )
        self.conv3 = sepconv_bn(
            filters[1],
            filters[2],
            stride=stride,
            rate=rate,
            depth_activation=depth_activation,
        )
        self.skip_return = skip_return
        self.agg = agg
        if agg == "conv":
            self.skip = nn.Sequential(
                conv1x1(in_planes, filters[2], stride=stride, bias=False),
                batchnorm(filters[2]),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        skip = self.conv2(out)
        out = self.conv3(skip)

        if self.agg == "conv":
            residual = self.skip(x)
            out += residual
        elif self.agg == "sum":
            out += residual

        if self.skip_return:
            return out, skip
        else:
            return out


class BasicBlock(nn.Module):
    """Basic residual block.

    Conv-BN-ReLU => Conv-BN => Residual => ReLU.

    Args:
      inplanes (int): number of input channels.
      planes (int): number of intermediate and output channels.
      stride (int): stride of the first convolution.
      downsample (nn.Module or None): downsampling operation.

    Attributes:
      expansion (int): equals to the ratio between the numbers
                       of output and intermediate channels.

    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = batchnorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = batchnorm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block.

    Conv-BN-ReLU => Conv-BN-ReLU => Conv-BN => Residual => ReLU.

    Args:
      inplanes (int): number of input channels.
      planes (int): number of intermediate and output channels.
      stride (int): stride of the first convolution.
      downsample (nn.Module or None): downsampling operation.

    Attributes:
      expansion (int): equals to the ratio between the numbers
                       of output and intermediate channels.

    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, bias=False)
        self.bn1 = batchnorm(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = batchnorm(planes)
        self.conv3 = conv1x1(planes, planes * 4, bias=False)
        self.bn3 = batchnorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

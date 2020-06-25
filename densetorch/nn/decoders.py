import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer_factory import CRPBlock, batchnorm, conv1x1, conv3x3, sepconv_bn
from ..misc.utils import make_list


class DLv3plus(nn.Module):
    """DeepLab-v3+ for Semantic Image Segmentation.

    ASPP with decoder. Allows to have multiple skip-connections.
    More information about the model: https://arxiv.org/abs/1802.02611

    Args:
      input_sizes (int, or list): number of channels for each input.
                                  Last value represents the input to ASPP,
                                  other values are for skip-connections.
      num_classes (int): number of output channels.
      skip_size (int): common filter size for skip-connections.
      agg_size (int): common filter size.
      rates (list of ints): dilation rates in the ASPP module.

    """

    def __init__(
        self,
        input_sizes,
        num_classes,
        skip_size=48,
        agg_size=256,
        rates=(6, 12, 18),
        **kwargs,
    ):
        super(DLv3plus, self).__init__()

        skip_convs = nn.ModuleList()
        aspp = nn.ModuleList()

        input_sizes = make_list(input_sizes)

        for size in input_sizes[:-1]:
            skip_convs.append(
                nn.Sequential(
                    conv1x1(size, skip_size, bias=False),
                    batchnorm(skip_size),
                    nn.ReLU(inplace=False),
                )
            )
        # ASPP
        aspp.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                conv1x1(input_sizes[-1], agg_size, bias=False),
                batchnorm(agg_size),
                nn.ReLU(inplace=False),
            )
        )
        aspp.append(
            nn.Sequential(
                conv1x1(input_sizes[-1], agg_size, bias=False),
                batchnorm(agg_size),
                nn.ReLU(inplace=False),
            )
        )
        for rate in rates:
            aspp.append(
                sepconv_bn(input_sizes[-1], agg_size, rate=rate, depth_activation=True)
            )
        aspp.append(
            nn.Sequential(
                conv1x1(agg_size * 5, agg_size, bias=False),
                batchnorm(agg_size),
                nn.ReLU(inplace=False),
                nn.Dropout(p=0.1),
            )
        )

        self.skip_convs = skip_convs
        self.aspp = aspp
        self.dec = nn.Sequential(
            sepconv_bn(
                agg_size + len(skip_convs) * skip_size, agg_size, depth_activation=True
            ),
            sepconv_bn(agg_size, agg_size, depth_activation=True),
        )
        self.clf = conv1x1(agg_size, num_classes, bias=True)

    def forward(self, xs):
        xs = make_list(xs)
        skips = [conv(x) for conv, x in zip(self.skip_convs, xs[:-1])]
        aspp = [branch(xs[-1]) for branch in self.aspp[:-1]]
        # Upsample GAP
        aspp[0] = F.interpolate(
            aspp[0], size=xs[-1].size()[2:], mode="bilinear", align_corners=True
        )
        aspp = torch.cat(aspp, dim=1)
        # Apply last conv in ASPP
        aspp = self.aspp[-1](aspp)
        # Connect with skip-connections
        if skips:
            dec = [skips[0]]
            for x in skips[1:] + [aspp]:
                dec.append(
                    F.interpolate(
                        x, size=dec[0].size()[2:], mode="bilinear", align_corners=True
                    )
                )
        else:
            dec = [aspp]
        dec = torch.cat(dec, dim=1)
        dec = self.dec(dec)
        out = self.clf(dec)
        return out


class LWRefineNet(nn.Module):
    """Light-Weight RefineNet for Semantic Image Segmentation.

    More information about the model: https://arxiv.org/abs/1810.03272

    Args:
      input_sizes (int, or list): number of channels for each input.
      collapse_ind (list): which input layers should be united together
                           (via element-wise summation) before CRP.
      num_classes (int): number of output channels.
      agg_size (int): common filter size.
      n_crp (int): number of CRP layers in a single CRP block.

    """

    def __init__(self, input_sizes, collapse_ind, num_classes, agg_size=256, n_crp=4):
        super(LWRefineNet, self).__init__()

        stem_convs = nn.ModuleList()
        crp_blocks = nn.ModuleList()
        adapt_convs = nn.ModuleList()

        input_sizes = make_list(input_sizes)
        # Reverse since we recover information from the end
        input_sizes = list(reversed(input_sizes))
        # No reverse for collapse indices
        self.collapse_ind = make_list(collapse_ind)

        for size in input_sizes:
            stem_convs.append(conv1x1(size, agg_size, bias=False))

        for _ in range(len(self.collapse_ind)):
            crp_blocks.append(self._make_crp(agg_size, agg_size, n_crp))
            adapt_convs.append(conv1x1(agg_size, agg_size, bias=False))

        self.stem_convs = stem_convs
        self.crp_blocks = crp_blocks
        self.adapt_convs = adapt_convs[:-1]

        self.segm = conv3x3(agg_size, num_classes, bias=True)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, xs):
        xs = make_list(xs)
        xs = list(reversed(xs))
        for idx, (conv, x) in enumerate(zip(self.stem_convs, xs)):
            xs[idx] = conv(x)
        # Collapse layers
        c_xs = [
            sum([xs[idx] for idx in make_list(c_idx)]) for c_idx in self.collapse_ind
        ]

        for idx, (crp, x) in enumerate(zip(self.crp_blocks, c_xs)):
            if idx == 0:
                y = self.relu(x)
            else:
                y = self.relu(x + y)
            y = crp(y)
            if idx < (len(c_xs) - 1):
                y = self.adapt_convs[idx](y)
                y = F.interpolate(
                    y,
                    size=c_xs[idx + 1].size()[2:],
                    mode="bilinear",
                    align_corners=True,
                )
        out_segm = self.segm(y)
        return out_segm

    @staticmethod
    def _make_crp(in_planes, out_planes, stages):
        """Creating Light-Weight Chained Residual Pooling (CRP) block.

        Args:
          in_planes (int): number of input channels.
          out_planes (int): number of output channels.
          stages (int): number of times the design is repeated
                        (with new weights)

        Returns:
          `nn.Sequential` of CRP layers.

        """
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)


class MTLWRefineNet(nn.Module):
    """Multi-Task Light-Weight RefineNet for Dense per-pixel tasks.

    More information about the model: https://arxiv.org/abs/1809.04766

    Args:
      input_sizes (int, or list): number of channels for each input.
      collapse_ind (list): which input layers should be united together
                           (via element-wise summation) before CRP.
      num_classes (int or list): number of output channels per each head.
      agg_size (int): common filter size.
      n_crp (int): number of CRP layers in a single CRP block.

    """

    def __init__(self, input_sizes, collapse_ind, num_classes, agg_size=256, n_crp=4):
        super(MTLWRefineNet, self).__init__()

        stem_convs = nn.ModuleList()
        crp_blocks = nn.ModuleList()
        adapt_convs = nn.ModuleList()
        heads = nn.ModuleList()

        input_sizes = make_list(input_sizes)
        # Reverse since we recover information from the end
        input_sizes = list(reversed(input_sizes))
        # No reverse for collapse indices is needed
        self.collapse_ind = make_list(collapse_ind)
        groups = [False] * len(self.collapse_ind)
        groups[-1] = True

        for size in input_sizes:
            stem_convs.append(conv1x1(size, agg_size, bias=False))

        for group in groups:
            crp_blocks.append(self._make_crp(agg_size, agg_size, n_crp, group))
            adapt_convs.append(conv1x1(agg_size, agg_size, bias=False))

        self.stem_convs = stem_convs
        self.crp_blocks = crp_blocks
        self.adapt_convs = adapt_convs[:-1]

        num_classes = make_list(num_classes)
        for n_out in num_classes:
            heads.append(
                nn.Sequential(
                    conv1x1(agg_size, agg_size, groups=agg_size, bias=False),
                    nn.ReLU6(inplace=False),
                    conv3x3(agg_size, n_out, bias=True),
                )
            )

        self.heads = heads
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, xs):
        xs = make_list(xs)
        xs = list(reversed(xs))
        for idx, (conv, x) in enumerate(zip(self.stem_convs, xs)):
            xs[idx] = conv(x)
        # Collapse layers
        c_xs = [
            sum([xs[idx] for idx in make_list(c_idx)]) for c_idx in self.collapse_ind
        ]

        for idx, (crp, x) in enumerate(zip(self.crp_blocks, c_xs)):
            if idx == 0:
                y = self.relu(x)
            else:
                y = self.relu(x + y)
            y = crp(y)
            if idx < (len(c_xs) - 1):
                y = self.adapt_convs[idx](y)
                y = F.interpolate(
                    y,
                    size=c_xs[idx + 1].size()[2:],
                    mode="bilinear",
                    align_corners=True,
                )

        outs = []
        for head in self.heads:
            outs.append(head(y))
        return outs

    @staticmethod
    def _make_crp(in_planes, out_planes, stages, groups):
        """Creating Light-Weight Chained Residual Pooling (CRP) block.

        Args:
          in_planes (int): number of input channels.
          out_planes (int): number of output channels.
          stages (int): number of times the design is repeated
                        (with new weights)
          groups (bool): whether to do groupwise convolution inside CRP.

        Returns:
          `nn.Sequential` of CRP layers.

        """
        layers = [CRPBlock(in_planes, out_planes, stages, groups)]
        return nn.Sequential(*layers)

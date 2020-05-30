from collections import namedtuple

model_urls = {
    # MBv2-encoder pre-trained on ImageNet
    "mobilenetv2": [
        "mobilenetv2-e6e8dd43.pth",
        "https://cloudstor.aarnet.edu.au/plus/s/uRgFbkaRjD3qOg5/download",
    ],
    # Xception-65 encoder from DeepLab-v3+ pre-trained on COCO/VOC
    "xception65": [
        "xception65-81e46d91.pth",
        "https://cloudstor.aarnet.edu.au/plus/s/gvEmFP3ngaJhvje/download",
    ],
    # Residual networks pre-trained on ImageNet
    "resnet18": [
        "resnet18-5c106cde.pth",
        "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    ],
    "resnet34": [
        "resnet34-333f7ec4.pth",
        "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    ],
    "resnet50": [
        "resnet50-19c8e357.pth",
        "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    ],
    "resnet101": [
        "resnet101-5d3b4d8f.pth",
        "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    ],
    "resnet152": [
        "resnet152-b121ed2d.pth",
        "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    ],
}

# Xception with different output strides
# In contrast to DeepLab-v3+, all skip-returns are set to False
blk = namedtuple(
    "Block",
    (
        "stride",
        "in_planes",
        "filters",
        "rate",
        "depth_activation",
        "skip_return",
        "agg",
    ),
)
# Entry
Config8 = {
    0: blk(
        stride=2,
        in_planes=64,
        filters=[128, 128, 128],
        rate=1,
        depth_activation=False,
        skip_return=False,
        agg="conv",
    ),
    1: blk(
        stride=2,
        in_planes=128,
        filters=[256, 256, 256],
        rate=1,
        depth_activation=False,
        skip_return=False,
        agg="conv",
    ),
    2: blk(
        stride=1,
        in_planes=256,
        filters=[728, 728, 728],
        rate=1,
        depth_activation=False,
        skip_return=False,
        agg="conv",
    ),
}
# Middle
for i in range(16):
    Config8[i + 3] = blk(
        stride=1,
        in_planes=728,
        filters=[728, 728, 728],
        rate=2,
        depth_activation=False,
        skip_return=False,
        agg="sum",
    )
# Exit
Config8[19] = blk(
    stride=1,
    in_planes=728,
    filters=[728, 1024, 1024],
    rate=2,
    depth_activation=False,
    skip_return=False,
    agg="conv",
)
Config8[20] = blk(
    stride=1,
    in_planes=1024,
    filters=[1536, 1536, 2048],
    rate=4,
    depth_activation=True,
    skip_return=False,
    agg="none",
)
Config8["rates"] = [12, 24, 36]

## OS16
# Entry
Config16 = {
    0: blk(
        stride=2,
        in_planes=64,
        filters=[128, 128, 128],
        rate=1,
        depth_activation=False,
        skip_return=False,
        agg="conv",
    ),
    1: blk(
        stride=2,
        in_planes=128,
        filters=[256, 256, 256],
        rate=1,
        depth_activation=False,
        skip_return=False,
        agg="conv",
    ),
    2: blk(
        stride=2,
        in_planes=256,
        filters=[728, 728, 728],
        rate=1,
        depth_activation=False,
        skip_return=False,
        agg="conv",
    ),
}
# Middle
for i in range(16):
    Config16[i + 3] = blk(
        stride=1,
        in_planes=728,
        filters=[728, 728, 728],
        rate=1,
        depth_activation=False,
        skip_return=False,
        agg="sum",
    )
# Exit
Config16[19] = blk(
    stride=1,
    in_planes=728,
    filters=[728, 1024, 1024],
    rate=1,
    depth_activation=False,
    skip_return=False,
    agg="conv",
)
Config16[20] = blk(
    stride=1,
    in_planes=1024,
    filters=[1536, 1536, 2048],
    rate=2,
    depth_activation=True,
    skip_return=False,
    agg="none",
)
Config16["rates"] = [6, 12, 18]

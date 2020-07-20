import numpy as np
import pytest
import random
import torch

import densetorch as dt

from src.network import get_network


def get_dummy_input_tensor(height, width, channels=3, batch=4):
    input_tensor = torch.FloatTensor(batch, channels, height, width).float()
    return input_tensor


def get_network_output_shape(h, w, output_stride=4):
    return np.ceil(h / output_stride), np.ceil(w / output_stride)


@pytest.fixture()
def num_classes():
    return [random.randint(1, 40) for _ in range(random.randint(1, 3))]


@pytest.fixture()
def num_channels():
    return random.randint(1, 40)


@pytest.fixture()
def input_height():
    return random.randint(33, 320)


@pytest.fixture()
def input_width():
    return random.randint(33, 320)


@pytest.mark.parametrize(
    "enc_backbone",
    [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "xception65",
        "mobilenetv2",
    ],
)
@pytest.mark.parametrize("enc_pretrained", [False, True])
@pytest.mark.parametrize("dec_backbone", ["lwrefinenet", "dlv3plus"])
def test_networks(
    enc_backbone, enc_pretrained, dec_backbone, input_height, input_width, num_classes
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = get_network(
        enc_backbone=enc_backbone,
        enc_pretrained=enc_pretrained,
        enc_return_layers=[0],
        dec_backbone=dec_backbone,
        dec_combine_layers=[0],
        num_classes=num_classes,
        device=device,
    )
    with torch.no_grad():
        input_tensor = get_dummy_input_tensor(
            height=input_height, width=input_width
        ).to(device)
        outputs = network(input_tensor)
        assert len(outputs) == len(
            num_classes
        ), f"Expected {len(num_classes):d} outputs, got {len(outputs):d}"
        for output, output_classes in zip(outputs, num_classes):
            assert output.size(0) == input_tensor.size(
                0
            ), f"Batch size mismatch, got {output.size(0):d}, expected {input_tensor.size(0):d}"
            assert (
                output.size(1) == output_classes
            ), f"Class dimension mismatch, got {output.size(1):d}, but expected {output_classes:d}"
            assert isinstance(output, torch.Tensor), "Expected a torch.Tensor as output"

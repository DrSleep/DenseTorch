import numpy as np
import pytest
import random
import torch

import densetorch as dt


NUMBER_OF_PARAMETERS_WITH_21_CLASSES = {
    "152": 61993301,
    "101": 46349653,
    "50": 27357525,
    "mbv2": 3284565,
}

NUMBER_OF_ENCODER_DECODER_LAYERS = {
    "152": (465, 28),
    "101": (312, 28),
    "50": (159, 28),
    "mbv2": (156, 27),
}


def get_dummy_input_tensor(height, width, channels=3, batch=4):
    input_tensor = torch.FloatTensor(batch, channels, height, width).float()
    return input_tensor


def get_network_output_shape(h, w, output_stride=4):
    return np.ceil(h / output_stride), np.ceil(w / output_stride)


@pytest.fixture()
def num_classes():
    return random.randint(1, 40)


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
    "enc_fn", [dt.nn.xception65, dt.nn.mobilenetv2, dt.nn.resnet18]
)
def test_encoders(enc_fn, input_height, input_width):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = enc_fn(pretrained=False, return_idx=0).to(device)
    with torch.no_grad():
        input_tensor = get_dummy_input_tensor(
            height=input_height, width=input_width
        ).to(device)
        output = encoder(input_tensor)
        assert len(output) == 1, f"Expected a single output, got {len(output):d}"
        assert output[0].size(0) == input_tensor.size(
            0
        ), f"Batch size mismatch, got {output[0].size(0):d}, expected {input_tensor.size(0):d}"
        assert isinstance(output[0], torch.Tensor), "Expected a torch.Tensor as output"


@pytest.mark.parametrize(
    "dec_fn", [dt.nn.DLv3plus, dt.nn.MTLWRefineNet, dt.nn.LWRefineNet]
)
def test_decoders(dec_fn, input_height, input_width, num_classes, num_channels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = dec_fn(
        input_sizes=num_channels, num_classes=num_classes, collapse_ind=0
    ).to(device)
    with torch.no_grad():
        input_tensor = get_dummy_input_tensor(
            height=input_height, width=input_width, channels=num_channels,
        ).to(device)
        output = decoder(input_tensor)
        if isinstance(output, list):
            assert len(output) == 1, f"Expected a single output, got {len(output):d}"
            output = output[0]
        assert isinstance(output, torch.Tensor), "Expected a torch.Tensor as output"
        assert output.size(0) == input_tensor.size(
            0
        ), f"Batch size mismatch, got {output[0].size(0):d}, expected {input_tensor.size(0):d}"
        assert (
            output.size(1) == num_classes
        ), f"Channel size mismatch, got {output.size(1):d}, expected {num_classes:d}"
        assert output.size(2) == input_tensor.size(
            2
        ), f"Height size mismatch, got {output.size(2):d}, expected {input_tensor.size(2):d}"
        assert output.size(3) == input_tensor.size(
            3
        ), f"Width size mismatch, got {output.size(3):d}, expected {input_tensor.size(3):d}"

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.hub import load_state_dict_from_url
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)

from .CBAM_resnet import ResNet


def _load_state_dict_with_match_stats(module, state_dict):
    model_dict = module.state_dict()
    matched = {
        k: v
        for k, v in state_dict.items()
        if k in model_dict and hasattr(v, "shape") and model_dict[k].shape == v.shape
    }
    model_dict.update(matched)
    module.load_state_dict(model_dict, strict=False)
    return len(matched), len(model_dict)


class ResNetMultiImageInput(ResNet):
    """ResNet encoder supporting stacked multi-image RGB input."""

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=True, num_input_images=1):
    """Construct a multi-image Pose encoder backbone (18/50 only)."""
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        weights = ResNet18_Weights.DEFAULT if num_layers == 18 else ResNet50_Weights.DEFAULT
        loaded = load_state_dict_from_url(weights.url, progress=True)
        loaded["conv1.weight"] = (
            torch.cat([loaded["conv1.weight"]] * num_input_images, 1) / num_input_images
        )
        matched_num, total_num = _load_state_dict_with_match_stats(model, loaded)
        model._pretrained_num_loaded = matched_num
        model._pretrained_num_total = total_num
    return model


class PoseEncoder(nn.Module):
    """Pose encoder that matches original MRFEDepth implementation semantics."""

    def __init__(self, num_layers, pretrained=True, num_input_images=1):
        super().__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.pretrained_num_loaded = None
        self.pretrained_num_total = None

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
            if pretrained:
                self.pretrained_num_loaded = getattr(self.encoder, "_pretrained_num_loaded", None)
                self.pretrained_num_total = getattr(self.encoder, "_pretrained_num_total", None)
        else:
            weights_map = {
                18: ResNet18_Weights.DEFAULT,
                34: ResNet34_Weights.DEFAULT,
                50: ResNet50_Weights.DEFAULT,
                101: ResNet101_Weights.DEFAULT,
                152: ResNet152_Weights.DEFAULT,
            }
            self.encoder = resnets[num_layers](weights=None)
            if pretrained:
                state_dict = load_state_dict_from_url(weights_map[num_layers].url, progress=True)
                matched_num, total_num = _load_state_dict_with_match_stats(self.encoder, state_dict)
                self.pretrained_num_loaded = matched_num
                self.pretrained_num_total = total_num

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        features = []

        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))

        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))
        return features

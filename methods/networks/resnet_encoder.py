# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
from torchvision.models import (
    resnet18,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
)
from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo


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


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a modified ResNet model for multi-image input.

    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Only ResNet-18 or ResNet-50 are supported"

    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]

    # 构建自定义输入通道的 ResNet（定义您自己的 ResNetMultiImageInput）
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        if num_layers == 18:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = ResNet50_Weights.DEFAULT

        # 下载权重
        state_dict = load_state_dict_from_url(weights.url, progress=True)

        # 修改第一层卷积核以适配 num_input_images 输入通道数
        conv1_weight = state_dict['conv1.weight']
        if num_input_images > 1:
            conv1_weight = torch.cat([conv1_weight] * num_input_images, dim=1) / num_input_images
        state_dict['conv1.weight'] = conv1_weight

        matched_num, total_num = _load_state_dict_with_match_stats(model, state_dict)
        model._pretrained_num_loaded = matched_num
        model._pretrained_num_total = total_num

    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.pretrained_num_loaded = None
        self.pretrained_num_total = None

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

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
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

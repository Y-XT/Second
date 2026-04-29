from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from .resnet_encoder import resnet_multiimage_input


class PoseFlowResnetEncoder(nn.Module):
    """Pose encoder with an RGB pretrained stem and a lightweight flow stem."""

    def __init__(
        self,
        num_layers,
        pretrained,
        num_input_images=2,
        num_flow_channels=2,
        flow_init_scale=1.0,
    ):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.encoder = resnet_multiimage_input(
            num_layers,
            pretrained=pretrained,
            num_input_images=num_input_images,
        )
        self.pretrained_num_loaded = getattr(self.encoder, "_pretrained_num_loaded", None)
        self.pretrained_num_total = getattr(self.encoder, "_pretrained_num_total", None)

        self.flow_conv1 = nn.Conv2d(
            int(num_flow_channels), 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.flow_bn1 = nn.BatchNorm2d(64)
        self.fusion_conv = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.kaiming_normal_(self.flow_conv1.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.flow_bn1.weight, 1)
        nn.init.constant_(self.flow_bn1.bias, 0)
        nn.init.zeros_(self.fusion_conv.weight)
        with torch.no_grad():
            for ch in range(64):
                self.fusion_conv.weight[ch, ch, 0, 0] = 1.0
                self.fusion_conv.weight[ch, 64 + ch, 0, 0] = float(flow_init_scale)

    def forward(self, input_image, rotation_flow):
        if input_image.ndim != 4:
            raise ValueError("PoseFlowResnetEncoder expects image input with shape [B, C, H, W].")
        if rotation_flow.ndim != 4:
            raise ValueError("PoseFlowResnetEncoder expects flow input with shape [B, 2, H, W].")
        if input_image.shape[0] != rotation_flow.shape[0]:
            raise ValueError("PoseFlowResnetEncoder image/flow batch sizes must match.")
        if input_image.shape[-2:] != rotation_flow.shape[-2:]:
            raise ValueError("PoseFlowResnetEncoder image/flow spatial sizes must match.")

        self.features = []

        x_rgb = (input_image - 0.45) / 0.225
        x_rgb = self.encoder.conv1(x_rgb)
        x_rgb = self.encoder.bn1(x_rgb)

        x_flow = self.flow_conv1(rotation_flow)
        x_flow = self.flow_bn1(x_flow)

        x = self.fusion_conv(torch.cat([x_rgb, x_flow], dim=1))
        x = self.encoder.relu(x)

        self.features.append(x)
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

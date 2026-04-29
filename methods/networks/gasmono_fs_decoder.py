from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from layers import ConvBlock, Conv3x3, upsample


def upsample_disp(disp, mask, scale):
    """Upsample disparity by convex combination, matching official GasMono fs decoder."""
    n, _, h, w = disp.shape
    mask = mask.view(n, 1, 9, 2 ** scale, 2 ** scale, h, w)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * disp, [3, 3], padding=1)
    up_flow = up_flow.view(n, 1, 9, 1, 1, h, w)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(n, 1, h * 2 ** scale, w * 2 ** scale)


class AttModule(nn.Module):
    """Channel-attention fusion between disparity branch and edge branch."""

    def __init__(self, high_feature_channel, low_feature_channels, output_channel=None):
        super(AttModule, self).__init__()
        in_channel = high_feature_channel + low_feature_channels
        out_channel = high_feature_channel if output_channel is None else output_channel
        reduction = 16

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_se = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, high_features, low_features):
        features = torch.cat([high_features, low_features], dim=1)
        b, c, _, _ = features.size()
        y = self.avg_pool(features).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        features = features * y.expand_as(features)
        return self.relu(self.conv_se(features))


class GasMonoFSDepthDecoder(nn.Module):
    """
    Official GasMono-style fs decoder.
    """

    def __init__(self, num_ch_enc=None, scales=range(4), num_output_channels=1, use_skips=True):
        super(GasMonoFSDepthDecoder, self).__init__()

        if num_ch_enc is None:
            num_ch_enc = [64, 128, 216, 288, 288]

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.use_fs = True

        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs[("upconv_edge", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs[("atten", i, 0)] = AttModule(num_ch_in, num_ch_in)

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)
            self.convs[("upconv_edge", i, 1)] = ConvBlock(self.num_ch_dec[i], num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            if self.use_fs:
                self.convs[("mask", s)] = nn.Sequential(
                    nn.Conv2d(self.num_ch_dec[s], self.num_ch_dec[s], 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.num_ch_dec[s], ((2 ** s) * (2 ** s)) * 9, 1, padding=0),
                )

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}

        x = input_features[-1]
        y = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("atten", i, 0)](x, y)

            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, dim=1)
            x = self.convs[("upconv", i, 1)](x)

            y = self.convs[("upconv_edge", i, 0)](y)
            y = upsample(y)
            y = self.convs[("upconv_edge", i, 1)](y)

            if i in self.scales:
                outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
                if self.use_fs and i > 0:
                    mask = 0.25 * self.convs[("mask", i)](y)
                    outputs[("disp", i)] = upsample_disp(disp=outputs[("disp", i)], mask=mask, scale=i)

        return outputs


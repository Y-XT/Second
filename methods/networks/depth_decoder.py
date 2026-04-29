# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

if __name__ == "__main__":
    import torch
    torch.set_printoptions(sci_mode=False, precision=4)

    # ----------------------------
    # 1) 假定输入图像尺寸（需被 32 整除）
    # ----------------------------
    B, H, W = 2, 480, 640

    # ----------------------------
    # 2) 假定编码器的通道配置（举例：ResNet18/ConvNeXt-S 类似）
    #    对应 5 个尺度特征：1/2, 1/4, 1/8, 1/16, 1/32
    # ----------------------------
    num_ch_enc = [64, 64, 128, 256, 512]

    # ----------------------------
    # 3) 伪造 5 层特征图（模拟 encoder 输出）
    #    注意：DepthDecoder 期望 input_features 的顺序与上面一致
    # ----------------------------
    feats = [
        torch.randn(B, num_ch_enc[0], H // 2,  W // 2),   # C1, stride 2
        torch.randn(B, num_ch_enc[1], H // 4,  W // 4),   # C2, stride 4
        torch.randn(B, num_ch_enc[2], H // 8,  W // 8),   # C3, stride 8
        torch.randn(B, num_ch_enc[3], H // 16, W // 16),  # C4, stride 16
        torch.randn(B, num_ch_enc[4], H // 32, W // 32),  # C5, stride 32
    ]

    # ----------------------------
    # 4) 实例化 DepthDecoder
    #    scales=range(4): 产生 ("disp", 0..3)
    #    其中 ("disp", 0) 为原图分辨率，
    #         ("disp", 1) 为 1/2，
    #         ("disp", 2) 为 1/4，
    #         ("disp", 3) 为 1/8
    # ----------------------------
    decoder = DepthDecoder(num_ch_enc=num_ch_enc,
                           scales=range(4),
                           num_output_channels=1,
                           use_skips=True).eval()

    with torch.no_grad():
        outputs = decoder(feats)

    # ----------------------------
    # 5) 打印各尺度输出的形状与数值范围
    # ----------------------------
    for s in range(4):
        key = ("disp", s)
        disp = outputs[key]
        print(f"{key}: shape={tuple(disp.shape)}, "
              f"min={disp.min().item():.4f}, max={disp.max().item():.4f}")

    # 你也可以把 ("disp", 0) 上采到原图（若它不是原图分辨率）：
    # import torch.nn.functional as F
    # disp_full = F.interpolate(outputs[('disp', 0)], size=(H, W),
    #                           mode='bilinear', align_corners=False)
    # print("disp_full:", disp_full.shape)

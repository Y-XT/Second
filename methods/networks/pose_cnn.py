from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import timm


class PoseCNN(nn.Module):
    """Official SPIdepth-style PoseCNN: timm encoder + lightweight regression head."""

    def __init__(self, num_input_frames, enc_name="resnet18", pretrained=True):
        super(PoseCNN, self).__init__()
        self.n_imgs = int(num_input_frames)
        self.pretrained_num_loaded = None
        self.pretrained_num_total = None
        self.encoder = timm.create_model(
            enc_name,
            in_chans=3 * self.n_imgs,
            features_only=True,
            pretrained=pretrained,
        )
        if pretrained:
            # timm 在 create_model 内部完成加载，这里记录总参数键数作为统一日志统计。
            total_num = len(self.encoder.state_dict())
            self.pretrained_num_loaded = total_num
            self.pretrained_num_total = total_num
        n_chenc = self.encoder.feature_info.channels()
        self.squeeze = self._block(n_chenc[-1], 256, kernel_size=1)
        self.decoder = nn.Sequential(
            self._block(256, 256, kernel_size=3, stride=1, padding=1),
            self._block(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 6 * (self.n_imgs - 1), kernel_size=1),
        )

    @staticmethod
    def _block(in_ch, out_ch, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feat = self.encoder(x)
        out = self.decoder(self.squeeze(feat[-1]))
        out = 0.01 * out.mean(dim=(2, 3)).view(-1, self.n_imgs - 1, 1, 6)
        return out[..., :3], out[..., 3:]

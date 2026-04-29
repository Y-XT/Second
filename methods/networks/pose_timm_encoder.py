from __future__ import absolute_import, division, print_function

import numpy as np
import torch.nn as nn
import timm


class PoseTimmEncoder(nn.Module):
    """Pose encoder backed by a timm features-only backbone."""

    def __init__(self, enc_name="convnext_tiny", pretrained=True, num_input_images=1):
        super().__init__()
        in_chans = int(num_input_images) * 3
        self.encoder = timm.create_model(
            enc_name,
            in_chans=in_chans,
            features_only=True,
            pretrained=bool(pretrained),
        )
        self.num_ch_enc = np.array(self.encoder.feature_info.channels())
        self.pretrained_num_loaded = None
        self.pretrained_num_total = None
        if pretrained:
            total_num = len(self.encoder.state_dict())
            self.pretrained_num_loaded = total_num
            self.pretrained_num_total = total_num

    def forward(self, input_image):
        x = (input_image - 0.45) / 0.225
        return self.encoder(x)

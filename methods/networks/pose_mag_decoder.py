# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseMagDecoder(nn.Module):
    """Pose decoder that predicts only translation magnitude."""

    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1,
                 scale_init=1.0, learnable_scale=False):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.net = nn.ModuleList(list(self.convs.values()))

        scale_init = max(float(scale_init), 1e-6)
        if learnable_scale:
            self.log_scale = nn.Parameter(torch.log(torch.tensor(scale_init, dtype=torch.float32)))
            self.register_buffer("fixed_scale", torch.tensor(1.0, dtype=torch.float32))
        else:
            self.log_scale = None
            self.register_buffer("fixed_scale", torch.tensor(scale_init, dtype=torch.float32))

    def _get_scale(self, device, dtype):
        if self.log_scale is not None:
            scale = torch.exp(self.log_scale)
        else:
            scale = self.fixed_scale
        return scale.to(device=device, dtype=dtype)

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        out = out.view(-1, self.num_frames_to_predict_for, 1, 1)
        out = self.softplus(out)
        out = 0.01 * out

        scale = self._get_scale(out.device, out.dtype)
        return out * scale

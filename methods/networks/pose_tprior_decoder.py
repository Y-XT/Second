# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
import torch.nn as nn


class PoseTPriorDecoder(nn.Module):
    """Single-head pose decoder with translation-prior feature injection."""

    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("fuse")] = nn.Conv2d(num_input_features * (256 + 128), num_input_features * 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 7 * num_frames_to_predict_for, 1)

        self.t_prior_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))

        nn.init.zeros_(self.convs[("pose", 2)].weight)
        nn.init.zeros_(self.convs[("pose", 2)].bias)
        nn.init.zeros_(self.convs[("fuse")].weight)
        if self.convs[("fuse")].bias is not None:
            nn.init.zeros_(self.convs[("fuse")].bias)
        with torch.no_grad():
            out_ch = self.convs[("fuse")].weight.shape[0]
            for ch in range(out_ch):
                self.convs[("fuse")].weight[ch, ch, 0, 0] = 1.0

    def _encode_t_prior(self, t_prior, height, width):
        if not torch.is_tensor(t_prior):
            raise TypeError("PoseTPriorDecoder expects t_prior to be a tensor.")
        if t_prior.ndim == 3 and t_prior.shape[-1] == 1:
            t_prior = t_prior[..., 0]
        if t_prior.ndim != 2 or t_prior.shape[1] != 3:
            raise ValueError("PoseTPriorDecoder expects t_prior with shape [B, 3].")

        mag = torch.linalg.norm(t_prior, dim=1, keepdim=True).clamp_min(1e-6)
        direction = t_prior / mag
        prior_vec = torch.cat([direction, mag], dim=1)
        prior_feat = self.t_prior_mlp(prior_vec)
        return prior_feat[:, :, None, None].expand(-1, -1, height, width)

    def forward(self, input_features, t_prior):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        prior_feat = self._encode_t_prior(
            t_prior.to(device=cat_features.device, dtype=cat_features.dtype),
            height=cat_features.shape[-2],
            width=cat_features.shape[-1],
        )
        out = torch.cat([cat_features, prior_feat], dim=1)
        out = self.relu(self.convs["fuse"](out))

        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 7)

        axisangle = out[..., :3]
        trans_scale = out[..., 3:4]
        trans_res = out[..., 4:]
        return axisangle, trans_scale, trans_res

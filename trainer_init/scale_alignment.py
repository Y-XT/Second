# -*- coding: utf-8 -*-
"""Utilities for per-batch depth scale alignment."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class DepthScaleAligner:
    """Computes a scalar scale to align predicted depth with an anchor depth map."""

    def __init__(self, opt):
        self.mode = str(getattr(opt, "scale_align_mode", "off"))
        self.anchor_key = str(getattr(opt, "scale_align_anchor_key", "depth_gt"))
        self.conf_key = str(getattr(opt, "scale_align_conf_key", "depth_conf"))
        self.min_valid_ratio = float(getattr(opt, "scale_align_min_valid_ratio", 0.01))
        self.min_valid_pixels = int(max(1, getattr(opt, "scale_align_min_valid_pixels", 2048)))
        self.conf_floor = float(getattr(opt, "scale_align_conf_floor", 1e-3))
        self.eps = float(getattr(opt, "scale_align_eps", 1e-6))
        self.scale_min = float(getattr(opt, "scale_align_scale_min", 0.05))
        self.scale_max = float(getattr(opt, "scale_align_scale_max", 40.0))
        self.reference_scale = int(getattr(opt, "scale_align_reference_scale", 0))

    def enabled(self) -> bool:
        return self.mode != "off"

    def apply(self,
              pred_depth: torch.Tensor,
              inputs: dict,
              cached_factor: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mode != "depth":
            return pred_depth, None

        factor = cached_factor
        if factor is None:
            factor = self._compute_depth_scale(pred_depth, inputs)
        if factor is None:
            return pred_depth, None
        return pred_depth * factor, factor

    def _compute_depth_scale(self, pred_depth: torch.Tensor, inputs: dict) -> Optional[torch.Tensor]:
        anchor = inputs.get(self.anchor_key)
        if anchor is None or not torch.is_tensor(anchor):
            return None

        device = pred_depth.device
        dtype = pred_depth.dtype
        anchor = anchor.to(device=device, dtype=dtype)
        if anchor.dim() == 3:
            anchor = anchor.unsqueeze(1)

        target_hw = pred_depth.shape[-2:]
        if anchor.shape[-2:] != target_hw:
            anchor = F.interpolate(anchor, size=target_hw, mode="bilinear", align_corners=False)

        conf = inputs.get(self.conf_key)
        conf_mask = None
        if conf is not None and torch.is_tensor(conf):
            conf = conf.to(device=device, dtype=dtype)
            if conf.dim() == 3:
                conf = conf.unsqueeze(1)
            if conf.shape[-2:] != target_hw:
                conf = F.interpolate(conf, size=target_hw, mode="bilinear", align_corners=False)
            conf_mask = conf > self.conf_floor
            conf = torch.clamp(conf, min=self.conf_floor)
        else:
            conf = torch.ones_like(anchor)

        valid_mask = anchor > 0.0
        if conf_mask is not None:
            valid_mask = valid_mask & conf_mask
        valid_mask = valid_mask.to(dtype)

        total_px = float(target_hw[0] * target_hw[1])
        valid_counts = valid_mask.view(valid_mask.shape[0], -1).sum(dim=1)
        valid_ratio = valid_counts / max(total_px, 1.0)

        has_valid_tensor = inputs.get("depth_has_valid")
        if has_valid_tensor is not None and torch.is_tensor(has_valid_tensor):
            has_valid = has_valid_tensor.to(device=device).view(-1).to(torch.bool)
        else:
            has_valid = torch.ones(valid_mask.shape[0], dtype=torch.bool, device=device)

        enough_pixels = valid_counts >= self.min_valid_pixels
        enough_ratio = valid_ratio >= self.min_valid_ratio
        usable = has_valid & enough_pixels & enough_ratio
        if not bool(usable.any()):
            return None

        weights = conf * valid_mask
        depth_detached = pred_depth.detach()
        numerator = (weights * depth_detached * anchor).view(anchor.shape[0], -1).sum(dim=1)
        denom = (weights * anchor * anchor).view(anchor.shape[0], -1).sum(dim=1) + self.eps

        scale = torch.ones(anchor.shape[0], dtype=dtype, device=device)
        solved = numerator[usable] / denom[usable]
        solved = torch.clamp(solved, self.scale_min, self.scale_max)
        scale[usable] = solved

        return scale.view(-1, 1, 1, 1).detach()

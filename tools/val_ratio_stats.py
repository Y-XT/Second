#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute validation ratios without training.

Metrics (scale=0):
- inbound ratio: warp grid in-bound pixel ratio
- automask pass ratio: ratio of pixels passing auto-mask (if enabled)
- identity ratio: 1 - automask pass ratio
- final supervised ratio: pixels entering photometric loss after min/gate
- gate ratios: VGGT vs PoseNet branch usage (MD2_VGGT_Gated only)
"""

import argparse
import json
import os
import glob
import sys
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from trainer_init.model_init import init_models, get_forward_handler
from trainer_init.loss_init import init_losses
from trainer_init.data_init import init_dataloaders
from trainer_init.geometry_init import init_geometry
from trainer_init.scale_alignment import DepthScaleAligner
from layers import disp_to_depth, transformation_from_parameters, rot_from_axisangle


def _select_weights_dir(exp_dir: str, weights_sel: str) -> Tuple[str, str]:
    exp_dir = os.path.abspath(exp_dir)
    models_dir = os.path.join(exp_dir, "models")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Missing models directory: {models_dir}")

    if weights_sel == "latest":
        cands = [d for d in glob.glob(os.path.join(models_dir, "weights_*")) if os.path.isdir(d)]
        if not cands:
            raise FileNotFoundError(f"No weights_* dirs under: {models_dir}")
        def _suffix(p):
            try:
                return int(os.path.basename(p).split("_")[-1])
            except Exception:
                return -1
        weights_dir = max(cands, key=_suffix)
    else:
        if isinstance(weights_sel, str) and weights_sel.startswith("weights_"):
            weights_dir = os.path.join(models_dir, weights_sel)
        else:
            idx = int(weights_sel)
            weights_dir = os.path.join(models_dir, f"weights_{idx}")
        if not os.path.isdir(weights_dir):
            raise FileNotFoundError(f"Missing weights dir: {weights_dir}")

    return weights_dir, os.path.basename(weights_dir)


def _load_opt(exp_dir: str, opt_json: Optional[str]) -> Dict:
    if opt_json:
        path = opt_json
    else:
        path = os.path.join(exp_dir, "models", "opt.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing opt.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_name)


class RatioMeter:
    def __init__(self):
        self.numer = 0.0
        self.denom = 0.0

    def update(self, values: Optional[torch.Tensor]):
        if values is None or not torch.is_tensor(values):
            return
        values = values.detach()
        self.numer += float(values.sum().item())
        self.denom += float(values.numel())

    def value(self) -> float:
        if self.denom <= 0:
            return float("nan")
        return self.numer / self.denom


class Runner:
    def __init__(self, opt, device: torch.device):
        self.opt = opt
        self.device = device
        self.models = {}
        self.parameters_to_train = []
        self.training = False

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        self.use_pose_net = True
        self.pose_residual_scale = 1.0

        init_models(self)
        init_losses(self)
        init_dataloaders(self)
        init_geometry(self)

        align_mode = str(getattr(self.opt, "scale_align_mode", "off"))
        self.scale_aligner = DepthScaleAligner(self.opt) if align_mode != "off" else None
        self._scale_align_monitor = {
            "train": {"attempts": 0, "success": 0},
            "val": {"attempts": 0, "success": 0},
        }
        self._scale_align_warned = {"train": False, "val": False}

    def set_eval(self):
        for m in self.models.values():
            m.eval()
        self.training = False

    def get_valid_warp_mask(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        x = pixel_coords[..., 0]
        y = pixel_coords[..., 1]
        valid = (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)
        return valid.unsqueeze(1).float()

    def predict_poses(self, inputs: Dict, features: torch.Tensor) -> Dict:
        outputs = {}
        use_vggt_tdir_posenet_mag = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_TDir_PoseMag"
        use_vggt_tprior_alpha = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_TPrior_Alpha"
        use_vggt_tprior_align = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_TPrior_AlignRes"
        alpha_mode = str(getattr(self.opt, "pose_alpha_mode", "tanh")).lower()
        alpha_tanh_scale = float(getattr(self.opt, "pose_alpha_tanh_scale", 0.1))
        alpha_exp_scale = float(getattr(self.opt, "pose_alpha_exp_scale", 0.01))
        align_scale_mode = str(getattr(self.opt, "pose_align_scale_mode", "tanh")).lower()
        align_scale_tanh_scale = float(getattr(self.opt, "pose_align_scale_tanh_scale", 0.1))
        align_scale_exp_scale = float(getattr(self.opt, "pose_align_scale_exp_scale", 0.01))
        align_res_tanh_scale = float(getattr(self.opt, "pose_align_res_tanh_scale", 0.01))
        align_res_scale_by_prior = bool(getattr(self.opt, "pose_align_res_scale_by_prior_norm", False))
        if not self.use_pose_net:
            missing = []
            pose_scale = self.models.get("pose_scale", None)
            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue
                key = ("external_cam_T_cam", 0, f_i)
                T = inputs.get(key, None)
                if T is None:
                    missing.append(f_i)
                    continue
                if pose_scale is not None:
                    scale = pose_scale(T.shape[0], device=T.device, dtype=T.dtype)
                    t_scaled = T[:, :3, 3:4] * scale
                    T = torch.cat([torch.cat([T[:, :3, :3], t_scaled], dim=2), T[:, 3:4, :]], dim=1)
                outputs[("cam_T_cam", 0, f_i)] = T
            if missing:
                raise KeyError(f"Missing external poses for frame ids: {missing}")
            return outputs

        residual_mode = getattr(self, "pose_residual_mode", None)
        pose_reg_terms = []

        if self.num_pose_frames == 2:
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    pose_out = self.models["pose"](pose_inputs)

                    if use_vggt_tdir_posenet_mag:
                        if isinstance(pose_out, (tuple, list)) and len(pose_out) == 2:
                            axisangle, translation = pose_out
                            t_mag = torch.linalg.norm(translation[:, 0], dim=2, keepdim=True)
                        else:
                            axisangle, translation = None, None
                            t_mag = pose_out
                            if t_mag.ndim == 4:
                                t_mag = t_mag[:, 0]
                            elif t_mag.ndim == 3:
                                t_mag = t_mag[:, 0:1]
                            elif t_mag.ndim == 2:
                                t_mag = t_mag[:, None, None]

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for t-dir method, frame id={f_i}")

                        T_prior = base_T.to(t_mag.device)
                        t_prior = T_prior[:, :3, 3]
                        t_norm = torch.linalg.norm(t_prior, dim=1, keepdim=True).clamp_min(1e-6)
                        t_dir = t_prior / t_norm
                        t_mag = t_mag.to(device=t_dir.device, dtype=t_dir.dtype)
                        t_combined = t_dir.unsqueeze(1) * t_mag

                        T = T_prior.clone()
                        T[:, :3, 3] = t_combined[:, 0]
                        outputs[("cam_T_cam", 0, f_i)] = T
                        continue

                    if use_vggt_tprior_alpha:
                        if isinstance(pose_out, (tuple, list)):
                            raise RuntimeError(
                                "MD2_VGGT_TPrior_Alpha expects PoseAlphaDecoder output (tensor), got tuple."
                            )
                        alpha_raw = pose_out
                        if not torch.is_tensor(alpha_raw):
                            raise TypeError("PoseAlphaDecoder output must be a tensor.")
                        if alpha_raw.ndim == 4:
                            alpha_raw = alpha_raw[:, 0]
                        elif alpha_raw.ndim == 3:
                            alpha_raw = alpha_raw[:, 0:1]
                        elif alpha_raw.ndim == 2:
                            alpha_raw = alpha_raw[:, None, None]

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for t-prior method, frame id={f_i}")

                        T_prior = base_T.to(alpha_raw.device)
                        t_prior = T_prior[:, :3, 3]
                        alpha_raw = alpha_raw.to(device=t_prior.device, dtype=t_prior.dtype)
                        if alpha_mode == "exp":
                            alpha = torch.exp(alpha_raw * alpha_exp_scale)
                        else:
                            alpha = 1.0 + alpha_tanh_scale * torch.tanh(alpha_raw)

                        alpha_scalar = alpha.view(alpha.shape[0], -1)
                        if alpha_scalar.shape[1] != 1:
                            alpha_scalar = alpha_scalar[:, 0:1]
                        t_scaled = t_prior * alpha_scalar
                        
                        T = T_prior.clone()
                        T[:, :3, 3] = t_scaled
                        outputs[("cam_T_cam", 0, f_i)] = T
                        continue
                    if use_vggt_tprior_align:
                        align_out = self.models.get("pose_align", None)
                        if align_out is None:
                            raise RuntimeError("MD2_VGGT_TPrior_AlignRes expects pose_align decoder.")
                        align_raw = align_out(pose_inputs)
                        if not torch.is_tensor(align_raw):
                            raise TypeError("PoseAlignDecoder output must be a tensor.")
                        if align_raw.ndim == 4:
                            align_raw = align_raw[:, 0]
                        elif align_raw.ndim == 3:
                            align_raw = align_raw[:, 0:1]
                        elif align_raw.ndim == 2:
                            align_raw = align_raw[:, None, :]
                        if align_raw.shape[-1] != 4:
                            raise RuntimeError("PoseAlignDecoder output last dim must be 4.")

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for align method, frame id={f_i}")

                        T_prior = base_T.to(align_raw.device)
                        t_prior = T_prior[:, :3, 3]
                        t_norm = torch.linalg.norm(t_prior, dim=1, keepdim=True).clamp_min(1e-6)

                        align_raw = align_raw.to(device=t_prior.device, dtype=t_prior.dtype)
                        scale_raw = align_raw[..., :1]
                        res_raw = align_raw[..., 1:]

                        if align_scale_mode == "exp":
                            scale = torch.exp(scale_raw * align_scale_exp_scale)
                        else:
                            scale = 1.0 + align_scale_tanh_scale * torch.tanh(scale_raw)

                        scale_scalar = scale.view(scale.shape[0], -1)
                        if scale_scalar.shape[1] != 1:
                            scale_scalar = scale_scalar[:, 0:1]
                        t_scaled = t_prior * scale_scalar

                        res = torch.tanh(res_raw) * align_res_tanh_scale
                        if align_res_scale_by_prior:
                            res = res * t_norm.view(-1, 1, 1)
                        if res.ndim == 3:
                            t_res = res[:, 0]
                        else:
                            t_res = res.view(res.shape[0], 3)

                        t_corr = t_scaled + t_res

                        if not isinstance(pose_out, (tuple, list)):
                            raise RuntimeError("MD2_VGGT_TPrior_AlignRes expects PoseDecoder output tuple.")
                        axisangle = pose_out[0]
                        if axisangle.ndim == 4:
                            axisangle = axisangle[:, 0]
                        elif axisangle.ndim == 3:
                            axisangle = axisangle[:, 0:1]
                        elif axisangle.ndim == 2:
                            axisangle = axisangle[:, None, :]

                        R_delta = rot_from_axisangle(axisangle)
                        if f_i < 0:
                            R_delta = R_delta.transpose(1, 2)
                        R_prior = T_prior[:, :3, :3]
                        R_final = torch.matmul(R_delta[:, :3, :3], R_prior)

                        T = T_prior.clone()
                        T[:, :3, :3] = R_final
                        T[:, :3, 3] = t_corr
                        outputs[("cam_T_cam", 0, f_i)] = T
                        continue

                    axisangle, translation = pose_out

                    if residual_mode is not None:
                        if residual_mode == "t":
                            axisangle = torch.zeros_like(axisangle)
                        scale = getattr(self, "pose_residual_scale", 1.0)
                        if scale != 1.0:
                            axisangle = axisangle * scale
                            translation = translation * scale

                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation

                        delta_T = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for residual method, frame id={f_i}")
                        T_prior = base_T.to(delta_T.device)
                        order = str(getattr(self, "pose_residual_order", "left")).lower()
                        if order == "right":
                            outputs[("cam_T_cam", 0, f_i)] = torch.matmul(T_prior, delta_T)
                        else:
                            outputs[("cam_T_cam", 0, f_i)] = torch.matmul(delta_T, T_prior)

                        if getattr(self, "pose_residual_reg_weight", 0.0) > 0:
                            reg = (translation[:, 0] ** 2).mean()
                            if residual_mode != "t":
                                reg = reg + (axisangle[:, 0] ** 2).mean()
                            pose_reg_terms.append(reg)
                    else:
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        if pose_reg_terms and getattr(self, "pose_residual_reg_weight", 0.0) > 0:
            outputs["pose_residual_reg"] = torch.stack(pose_reg_terms).sum() * self.pose_residual_reg_weight

        return outputs

    def generate_images_pred(self, inputs: Dict, outputs: Dict) -> None:
        scale_align_factor = None
        scale_align_ref = getattr(self.scale_aligner, "reference_scale", None) if self.scale_aligner else None
        align_mode = "train" if self.training else "val"
        align_attempted = False
        enable_pose_gating = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_Gated"

        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            depth_for_warp = depth
            if self.scale_aligner is not None:
                align_attempted = True
                allow_compute = (scale_align_factor is not None) or (scale_align_ref is None) or (scale == scale_align_ref)
                if allow_compute:
                    depth_for_warp, candidate_factor = self.scale_aligner.apply(
                        depth, inputs, cached_factor=scale_align_factor)
                    if candidate_factor is not None:
                        scale_align_factor = candidate_factor
                        outputs["scale_align_factor"] = scale_align_factor.view(scale_align_factor.shape[0])

            for frame_id in self.opt.frame_ids[1:]:
                T = outputs[("cam_T_cam", 0, frame_id)]
                cam_points = self.backproject_depth[source_scale](
                    depth_for_warp, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    pix_coords,
                    padding_mode="border",
                    align_corners=True,
                )

                if getattr(self.opt, "distorted_mask", False):
                    outputs[("distorted_mask", frame_id, scale)] = self.get_valid_warp_mask(pix_coords)

                if not getattr(self.opt, "disable_automasking", False):
                    outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

                if enable_pose_gating:
                    key = ("external_cam_T_cam", 0, frame_id)
                    T_prior = inputs.get(key, None)
                    if torch.is_tensor(T_prior):
                        T_prior = T_prior.to(cam_points.device)
                        pix_coords_prior = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T_prior)
                        outputs[("color_vggt", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords_prior,
                            padding_mode="border",
                            align_corners=True,
                        )
                        if getattr(self.opt, "distorted_mask", False):
                            outputs[("distorted_mask_vggt", frame_id, scale)] = self.get_valid_warp_mask(
                                pix_coords_prior)

        if align_attempted:
            monitor = self._scale_align_monitor.get(align_mode)
            if monitor is not None:
                monitor["attempts"] += 1
                if scale_align_factor is not None:
                    monitor["success"] += 1


def _move_inputs_to_device(inputs: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, val in inputs.items():
        if isinstance(val, torch.Tensor):
            moved[key] = val.to(device, non_blocking=True)
        elif isinstance(val, (list, tuple)) and val and torch.is_tensor(val[0]):
            moved[key] = [v.to(device, non_blocking=True) for v in val]
        elif isinstance(val, (int, float, str)) or val is None:
            moved[key] = val
        elif hasattr(val, "dtype"):
            moved[key] = torch.as_tensor(val).to(device, non_blocking=True)
        else:
            moved[key] = val
    return moved


def _valid_mask_from_grid(grid: torch.Tensor) -> torch.Tensor:
    x = grid[..., 0]
    y = grid[..., 1]
    return (x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)


class _SimpleProgress:
    def __init__(self, iterable, total: Optional[int] = None, desc: str = "val"):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.count = 0

    def __iter__(self):
        for item in self.iterable:
            self.count += 1
            if self.total:
                msg = f"\r{self.desc}: {self.count}/{self.total}"
            else:
                msg = f"\r{self.desc}: {self.count}"
            print(msg, end="", flush=True)
            yield item
        print("")

    def close(self):
        print("")


def _compute_supervised_mask(runner: Runner, inputs: Dict, outputs: Dict, scale: int) -> Optional[torch.Tensor]:
    opt = runner.opt
    frame_ids = opt.frame_ids[1:]
    if not frame_ids:
        return None

    source_scale = 0
    target = inputs[("color", 0, source_scale)]

    reprojection_losses = []
    use_pose_gating = str(getattr(opt, "methods", "")) == "MD2_VGGT_Gated"
    gate_mode = str(getattr(opt, "pose_gating_mode", "min")).lower()
    gate_tau = float(getattr(opt, "pose_gating_tau", 0.1))

    for frame_id in frame_ids:
        pred = outputs.get(("color", frame_id, scale), None)
        if pred is None:
            continue
        reproj_loss = runner.loss.compute_reprojection_loss(pred, target)
        if getattr(opt, "distorted_mask", False):
            valid_mask = outputs.get(("distorted_mask", frame_id, scale), None)
            if valid_mask is not None:
                reproj_loss = reproj_loss * valid_mask + (1.0 - valid_mask) * 1e5

        if use_pose_gating:
            pred_prior = outputs.get(("color_vggt", frame_id, scale), None)
            if pred_prior is not None:
                reproj_prior = runner.loss.compute_reprojection_loss(pred_prior, target)
                if getattr(opt, "distorted_mask", False):
                    valid_mask_prior = outputs.get(("distorted_mask_vggt", frame_id, scale), None)
                    if valid_mask_prior is not None:
                        reproj_prior = reproj_prior * valid_mask_prior + (1.0 - valid_mask_prior) * 1e5

                if gate_mode == "softmin":
                    tau = max(gate_tau, 1e-6)
                    stacked = torch.stack((-reproj_loss, -reproj_prior), dim=1) / tau
                    weights = torch.softmax(stacked, dim=1)
                    reproj_loss = weights[:, 0] * reproj_loss + weights[:, 1] * reproj_prior
                else:
                    reproj_loss = torch.minimum(reproj_loss, reproj_prior)

        reprojection_losses.append(reproj_loss)

    if not reprojection_losses:
        return None

    reprojection_losses = torch.cat(reprojection_losses, 1)

    identity_loss_tensor = None
    if not getattr(opt, "disable_automasking", False):
        identity_losses = []
        for frame_id in frame_ids:
            pred = inputs[("color", frame_id, source_scale)]
            identity_losses.append(runner.loss.compute_reprojection_loss(pred, target))
        identity_losses = torch.cat(identity_losses, 1)
        if getattr(opt, "avg_reprojection", False):
            identity_loss_tensor = identity_losses.mean(1, keepdim=True)
        else:
            identity_loss_tensor = identity_losses

    if getattr(opt, "avg_reprojection", False):
        reprojection_loss = reprojection_losses.mean(1, keepdim=True)
    else:
        reprojection_loss = reprojection_losses

    if not getattr(opt, "disable_automasking", False) and identity_loss_tensor is not None:
        identity_loss_tensor = identity_loss_tensor + torch.randn(
            identity_loss_tensor.shape, device=identity_loss_tensor.device
        ) * 1e-5
        combined = torch.cat((identity_loss_tensor, reprojection_loss), dim=1)
    else:
        combined = reprojection_loss

    min_vals, min_indices = torch.min(combined, dim=1)
    if identity_loss_tensor is not None:
        offset = identity_loss_tensor.shape[1]
        min_is_reproj = min_indices >= offset
    else:
        min_is_reproj = torch.ones_like(min_indices, dtype=torch.bool)

    invalid_penalty = getattr(opt, "invalid_photometric_penalty", 1e4)
    supervised_mask = min_is_reproj & (min_vals < invalid_penalty)
    return supervised_mask


def _update_gate_meters(
    runner: Runner,
    inputs: Dict,
    outputs: Dict,
    scale: int,
    prior_meter: RatioMeter,
    pose_meter: RatioMeter,
) -> None:
    opt = runner.opt
    if str(getattr(opt, "methods", "")) != "MD2_VGGT_Gated":
        return

    frame_ids = opt.frame_ids[1:]
    if not frame_ids:
        return

    source_scale = 0
    target = inputs[("color", 0, source_scale)]
    gate_mode = str(getattr(opt, "pose_gating_mode", "min")).lower()
    gate_tau = float(getattr(opt, "pose_gating_tau", 0.1))

    for frame_id in frame_ids:
        pred = outputs.get(("color", frame_id, scale), None)
        pred_prior = outputs.get(("color_vggt", frame_id, scale), None)
        if pred is None or pred_prior is None:
            continue

        reproj_pose = runner.loss.compute_reprojection_loss(pred, target)
        reproj_prior = runner.loss.compute_reprojection_loss(pred_prior, target)

        if getattr(opt, "distorted_mask", False):
            valid_mask = outputs.get(("distorted_mask", frame_id, scale), None)
            if valid_mask is not None:
                reproj_pose = reproj_pose * valid_mask + (1.0 - valid_mask) * 1e5
            valid_mask_prior = outputs.get(("distorted_mask_vggt", frame_id, scale), None)
            if valid_mask_prior is not None:
                reproj_prior = reproj_prior * valid_mask_prior + (1.0 - valid_mask_prior) * 1e5

        if gate_mode == "softmin":
            tau = max(gate_tau, 1e-6)
            stacked = torch.stack((-reproj_pose, -reproj_prior), dim=1) / tau
            weights = torch.softmax(stacked, dim=1)
            prior_meter.update(weights[:, 1])
            pose_meter.update(weights[:, 0])
        else:
            prior_mask = (reproj_prior < reproj_pose).float()
            prior_meter.update(prior_mask)
            pose_meter.update(1.0 - prior_mask)


def _load_model_weights(runner: Runner, weights_dir: str, models_to_load: Optional[list]) -> None:
    model_keys = [m for m in (models_to_load or []) if m in runner.models]
    if not model_keys:
        model_keys = list(runner.models.keys())

    for name in model_keys:
        path = os.path.join(weights_dir, f"{name}.pth")
        if not os.path.isfile(path):
            print(f"[WARN] Missing weights for {name}: {path}")
            continue
        state = torch.load(path, map_location=runner.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model_dict = runner.models[name].state_dict()
        filtered = {k: v for k, v in state.items() if k in model_dict}
        model_dict.update(filtered)
        runner.models[name].load_state_dict(model_dict)
        print(f"[Load OK] {name} <- {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation ratio stats (scale=0)")
    parser.add_argument(
        "--exp_dir",
        default=None,
        help="Experiment dir with models/weights_*",
    )
    parser.add_argument("--weights", default=None, help="'latest' | int | weights_x")
    parser.add_argument(
        "--weights_dir",
        default="/home/yxt/文档/mono_result/weights/UAVid_China/monodepth2_uavid2020_512x288_kdef_noapp_bs8_lr1e-04_e40_step20_d1ly/models/weights_35",
        help="Full path to weights dir (e.g. .../models/weights_14). Overrides --exp_dir/--weights",
    )
    parser.add_argument("--opt_json", default=None, help="Override opt.json path")
    parser.add_argument("--dataset", default=None, help="Override dataset name")
    parser.add_argument(
        "--data_path",
        default=None,
        help="Override dataset root",
    )
    parser.add_argument("--split", default=None, help="Override split name")
    parser.add_argument("--triplet_root", default=None, help="Override triplet root")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override num_workers")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"], help="Device")
    parser.add_argument("--max_batches", type=int, default=-1, help="Limit number of val batches")
    parser.add_argument(
        "--output_base",
        default="/home/yxt/文档/mono_result/eval",
        help="Base directory to save txt result (auto subdir by dataset/split)",
    )
    parser.add_argument("--output_dir", default=None, help="Override output directory")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--no_save_txt", action="store_true", help="Disable saving txt output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.weights_dir:
        weights_dir = os.path.abspath(args.weights_dir)
        weights_name = os.path.basename(weights_dir)
        exp_dir = os.path.abspath(os.path.join(weights_dir, "..", ".."))
        args.exp_dir = exp_dir
    else:
        weights_dir = None
        weights_name = None
        exp_dir = args.exp_dir
        if not exp_dir:
            raise ValueError("Please set --weights_dir or --exp_dir")
        if not args.weights:
            raise ValueError("Please set --weights when using --exp_dir")

    opt_dict = _load_opt(exp_dir, args.opt_json)
    opt = SimpleNamespace(**opt_dict)

    if args.dataset is not None:
        opt.dataset = args.dataset
    if args.data_path is not None:
        opt.data_path = args.data_path
    if args.split is not None:
        opt.split = args.split
    if args.triplet_root is not None:
        opt.triplet_root = args.triplet_root
    if args.batch_size is not None:
        opt.batch_size = args.batch_size
    if args.num_workers is not None:
        opt.num_workers = args.num_workers

    device = _resolve_device(args.device)
    opt.no_cuda = (device.type == "cpu")

    if weights_dir is None or weights_name is None:
        weights_dir, weights_name = _select_weights_dir(exp_dir, args.weights)
    print(f"[weights] {weights_name} -> {weights_dir}")

    runner = Runner(opt, device)
    _load_model_weights(runner, weights_dir, getattr(opt, "models_to_load", None))
    runner.set_eval()

    handler = get_forward_handler(getattr(opt, "methods", "Monodepth2"))

    inbound_meter = RatioMeter()
    automask_pass_meter = RatioMeter()
    supervised_meter = RatioMeter()
    gate_prior_meter = RatioMeter()
    gate_pose_meter = RatioMeter()

    total_batches = 0
    total_samples = 0

    try:
        total_len = len(runner.val_loader)
    except TypeError:
        total_len = None

    if tqdm is not None:
        loader_iter = tqdm(runner.val_loader, total=total_len, desc="val", ncols=80)
    else:
        loader_iter = _SimpleProgress(runner.val_loader, total=total_len, desc="val")

    with torch.no_grad():
        for idx, inputs in enumerate(loader_iter):
            if args.max_batches > 0 and idx >= args.max_batches:
                break
            inputs = _move_inputs_to_device(inputs, device)
            outputs, _ = handler(runner, inputs)

            scale = 0
            frame_ids = opt.frame_ids[1:]

            inbound_masks = []
            for frame_id in frame_ids:
                grid = outputs.get(("sample", frame_id, scale), None)
                if grid is None:
                    continue
                inbound_masks.append(_valid_mask_from_grid(grid))
            if inbound_masks:
                inbound_mask = torch.cat([m.reshape(-1) for m in inbound_masks], dim=0).float()
                inbound_meter.update(inbound_mask)

            if not getattr(opt, "disable_automasking", False):
                identity_sel = outputs.get(f"identity_selection/{scale}", None)
                if identity_sel is not None:
                    automask_pass_meter.update(identity_sel)

            supervised_mask = _compute_supervised_mask(runner, inputs, outputs, scale)
            if supervised_mask is not None:
                supervised_meter.update(supervised_mask.float())

            _update_gate_meters(runner, inputs, outputs, scale, gate_prior_meter, gate_pose_meter)

            batch_size = inputs[("color", 0, 0)].shape[0]
            total_samples += int(batch_size)
            total_batches += 1
    if hasattr(loader_iter, "close"):
        loader_iter.close()

    automask_pass_ratio = automask_pass_meter.value()
    identity_ratio = float("nan") if automask_pass_ratio != automask_pass_ratio else 1.0 - automask_pass_ratio

    exp_name = os.path.basename(os.path.abspath(args.exp_dir.rstrip("/")))
    results = {
        "exp_dir": os.path.abspath(args.exp_dir),
        "weights": weights_name,
        "dataset": getattr(opt, "dataset", None),
        "split": getattr(opt, "split", None),
        "num_batches": total_batches,
        "num_samples": total_samples,
        "metrics": {
            "inbound_ratio": inbound_meter.value(),
            "automask_pass_ratio": automask_pass_ratio,
            "identity_ratio": identity_ratio,
            "final_supervised_ratio": supervised_meter.value(),
            "gate_prior_ratio": gate_prior_meter.value(),
            "gate_posenet_ratio": gate_pose_meter.value(),
        },
    }

    print(json.dumps(results, indent=2))

    output_txt = None
    if not args.no_save_txt:
        dataset_lower = str(getattr(opt, "dataset", "")).lower()
        split_lower = str(getattr(opt, "split", "")).lower()
        if args.output_dir:
            output_dir = os.path.abspath(args.output_dir)
        else:
            if "uavula" in dataset_lower or "uavula" in split_lower:
                subdir = "UAVula-R1"
            elif "uavid" in dataset_lower or "uavid" in split_lower:
                if "germany" in split_lower:
                    subdir = "UAVid_germany"
                elif "china" in split_lower:
                    subdir = "UAVid_china"
                else:
                    subdir = "UAVid"
            else:
                subdir = "outputs"
            run_dir_name = f"{exp_name}_{weights_name}"
            output_dir = os.path.join(args.output_base, subdir, run_dir_name)

        os.makedirs(output_dir, exist_ok=True)
        output_txt = os.path.join(output_dir, "ratio_stats.txt")
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(f"exp_dir: {results['exp_dir']}\n")
            f.write(f"weights: {results['weights']}\n")
            f.write(f"dataset: {results['dataset']}\n")
            f.write(f"split: {results['split']}\n")
            f.write(f"num_batches: {results['num_batches']}\n")
            f.write(f"num_samples: {results['num_samples']}\n")
            metrics = results["metrics"]
            for key in [
                "inbound_ratio",
                "automask_pass_ratio",
                "identity_ratio",
                "final_supervised_ratio",
                "gate_prior_ratio",
                "gate_posenet_ratio",
            ]:
                f.write(f"{key}: {metrics.get(key)}\n")
        print(f"[saved txt] {output_txt}")

    if args.output:
        out_path = os.path.abspath(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()

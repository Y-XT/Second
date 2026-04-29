import json
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from image_logger import (
    _disp_to_colormap,
    _log_row_images_to_wandb,
    _log_scalar_row_with_colorbar_to_wandb,
    _scalar_to_colormap_masked,
    _source_fid_to_rgb,
)
from layers import compute_depth_errors
from utils import sec_to_hm_str


def reset_metrics(self, mode):
    if not self.collect_debug_metrics:
        return
    if mode in self._metric_buffers:
        self._metric_buffers[mode] = defaultdict(list)


def accumulate_metrics(self, mode, losses):
    if not self.collect_debug_metrics:
        return
    buffer = self._metric_buffers.get(mode, None)
    if buffer is None:
        return
    for key, value in losses.items():
        if not key.startswith("metrics/"):
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                continue
            val = float(value.detach().mean().cpu().item())
        else:
            val = float(value)
        if math.isnan(val) or math.isinf(val):
            continue
        buffer[key].append(val)


def update_pose_metrics(self, outputs):
    if not self.collect_debug_metrics:
        return {}
    metrics = {}
    frame_ids = [fid for fid in self.opt.frame_ids[1:] if fid != "s"]
    if not frame_ids:
        return metrics

    t_norms = []
    t_mags = []
    prior_norms = []
    t_mags_teacher = []
    prior_norms_teacher = []
    alpha_vals = []
    align_scale_vals = []
    align_res_norms = []
    t_over_z = []
    for fid in frame_ids:
        key = ("cam_T_cam", 0, fid)
        T = outputs.get(key, None)
        if T is None or not torch.is_tensor(T):
            continue
        if T.ndim != 3 or T.shape[-2:] != (4, 4):
            continue
        if self.use_posegt:
            posegt_key = ("posegt_cam_T_cam", 0, fid)
            T_posegt = outputs.get(posegt_key, None)
            if torch.is_tensor(T_posegt) and T_posegt.shape[-2:] == (4, 4):
                T = torch.matmul(T, T_posegt.to(device=T.device, dtype=T.dtype))
        elif getattr(self, "use_vggt_prewarp", False):
            prewarp_key = ("pre_warp_cam_T_cam", 0, fid)
            T_prewarp = outputs.get(prewarp_key, None)
            if torch.is_tensor(T_prewarp) and T_prewarp.shape[-2:] == (4, 4):
                T = torch.matmul(T, T_prewarp.to(device=T.device, dtype=T.dtype))
        t = T[:, :3, 3]
        t_norms.append(torch.linalg.norm(t, dim=1))

        mag_key = ("pose_mag", 0, fid)
        prior_key = ("pose_prior_norm", 0, fid)
        pose_mag = outputs.get(mag_key, None)
        if pose_mag is not None and torch.is_tensor(pose_mag):
            t_mags.append(pose_mag.reshape(-1))
        prior_norm = outputs.get(prior_key, None)
        if prior_norm is not None and torch.is_tensor(prior_norm):
            prior_norms.append(prior_norm.reshape(-1))
        mag_teacher_key = ("pose_mag_teacher", 0, fid)
        prior_teacher_key = ("pose_prior_norm_teacher", 0, fid)
        pose_mag_teacher = outputs.get(mag_teacher_key, None)
        if pose_mag_teacher is not None and torch.is_tensor(pose_mag_teacher):
            t_mags_teacher.append(pose_mag_teacher.reshape(-1))
        prior_norm_teacher = outputs.get(prior_teacher_key, None)
        if prior_norm_teacher is not None and torch.is_tensor(prior_norm_teacher):
            prior_norms_teacher.append(prior_norm_teacher.reshape(-1))
        alpha_key = ("pose_alpha", 0, fid)
        alpha_val = outputs.get(alpha_key, None)
        if alpha_val is not None and torch.is_tensor(alpha_val):
            alpha_vals.append(alpha_val.reshape(-1))
        align_scale_key = ("pose_align_scale", 0, fid)
        align_scale = outputs.get(align_scale_key, None)
        if align_scale is not None and torch.is_tensor(align_scale):
            align_scale_vals.append(align_scale.reshape(-1))
        align_res_key = ("pose_align_res", 0, fid)
        align_res = outputs.get(align_res_key, None)
        if align_res is not None and torch.is_tensor(align_res):
            if align_res.ndim == 3:
                align_res = align_res[:, 0]
            align_res_norms.append(torch.linalg.norm(align_res, dim=1))

    if not t_norms:
        return metrics

    norms = torch.cat(t_norms, dim=0)
    if norms.numel() == 0:
        return metrics

    norms = norms.detach()
    metrics["metrics/pose/t_norm_mean"] = float(norms.mean().item())
    metrics["metrics/pose/t_norm_std"] = float(norms.std(unbiased=False).item())
    metrics["metrics/pose/t_norm_p50"] = float(torch.quantile(norms, 0.5).item())
    metrics["metrics/pose/t_norm_p90"] = float(torch.quantile(norms, 0.9).item())

    self.pose_t_history.append(metrics["metrics/pose/t_norm_mean"])
    if len(self.pose_t_history) > 1:
        hist = torch.tensor(list(self.pose_t_history), dtype=norms.dtype, device=norms.device)
        metrics["metrics/pose/t_norm_roll_var"] = float(hist.var(unbiased=False).item())
    else:
        metrics["metrics/pose/t_norm_roll_var"] = 0.0

    depth = outputs.get(("depth", 0, 0), None)
    if torch.is_tensor(depth):
        depth_map = depth.detach()
        if depth_map.ndim == 4:
            depth_map = depth_map[:, 0]
        if depth_map.ndim == 3:
            flat = depth_map.reshape(depth_map.shape[0], -1).to(torch.float32)
            if flat.numel() > 0:
                q = torch.quantile(
                    flat,
                    torch.tensor([0.05, 0.95], device=flat.device),
                    dim=1,
                )
                q05 = q[0]
                q95 = q[1]
                mask = (flat >= q05[:, None]) & (flat <= q95[:, None])
                mask_f = mask.to(flat.dtype)
                counts = mask_f.sum(dim=1).clamp_min(1.0)
                z_mean = (flat * mask_f).sum(dim=1) / counts
                z_mean = z_mean.clamp_min(1e-6)
                for t_norm in t_norms:
                    t_over_z.append(t_norm.detach() / z_mean)
    if t_over_z:
        t_over_z_all = torch.cat(t_over_z, dim=0)
        if t_over_z_all.numel() > 0:
            metrics["metrics/pose/t_over_z_mean"] = float(t_over_z_all.mean().item())
            metrics["metrics/pose/t_over_z_std"] = float(t_over_z_all.std(unbiased=False).item())
            metrics["metrics/pose/t_over_z_p50"] = float(torch.quantile(t_over_z_all, 0.5).item())
            metrics["metrics/pose/t_over_z_p90"] = float(torch.quantile(t_over_z_all, 0.9).item())

    mags = None
    priors = None
    if t_mags:
        mags = torch.cat(t_mags, dim=0).detach()
        if mags.numel() > 0:
            metrics["metrics/pose/t_mag_mean"] = float(mags.mean().item())
            metrics["metrics/pose/t_mag_std"] = float(mags.std(unbiased=False).item())

    if prior_norms:
        priors = torch.cat(prior_norms, dim=0).detach()
        if priors.numel() > 0:
            metrics["metrics/pose/prior_t_norm_mean"] = float(priors.mean().item())
            metrics["metrics/pose/prior_t_norm_std"] = float(priors.std(unbiased=False).item())

    if mags is not None and priors is not None:
        prior_mean = priors.mean()
        if prior_mean.item() > 0:
            metrics["metrics/pose/t_mag_ratio_mean"] = float((mags.mean() / (prior_mean + 1e-6)).item())

    mags_teacher = None
    priors_teacher = None
    if t_mags_teacher:
        mags_teacher = torch.cat(t_mags_teacher, dim=0).detach()
        if mags_teacher.numel() > 0:
            metrics["metrics/pose_teacher/t_mag_mean"] = float(mags_teacher.mean().item())
            metrics["metrics/pose_teacher/t_mag_std"] = float(mags_teacher.std(unbiased=False).item())

    if prior_norms_teacher:
        priors_teacher = torch.cat(prior_norms_teacher, dim=0).detach()
        if priors_teacher.numel() > 0:
            metrics["metrics/pose_teacher/prior_t_norm_mean"] = float(priors_teacher.mean().item())
            metrics["metrics/pose_teacher/prior_t_norm_std"] = float(priors_teacher.std(unbiased=False).item())

    if mags_teacher is not None and priors_teacher is not None:
        prior_mean = priors_teacher.mean()
        if prior_mean.item() > 0:
            metrics["metrics/pose_teacher/t_mag_ratio_mean"] = float(
                (mags_teacher.mean() / (prior_mean + 1e-6)).item()
            )

    if alpha_vals:
        alphas = torch.cat(alpha_vals, dim=0).detach()
        if alphas.numel() > 0:
            metrics["metrics/pose/alpha_mean"] = float(alphas.mean().item())
            metrics["metrics/pose/alpha_std"] = float(alphas.std(unbiased=False).item())
    if align_scale_vals:
        scales = torch.cat(align_scale_vals, dim=0).detach()
        if scales.numel() > 0:
            metrics["metrics/pose/align_scale_mean"] = float(scales.mean().item())
            metrics["metrics/pose/align_scale_std"] = float(scales.std(unbiased=False).item())
    if align_res_norms:
        res_norms = torch.cat(align_res_norms, dim=0).detach()
        if res_norms.numel() > 0:
            metrics["metrics/pose/align_res_norm_mean"] = float(res_norms.mean().item())
            metrics["metrics/pose/align_res_norm_std"] = float(res_norms.std(unbiased=False).item())
            if prior_norms:
                priors = torch.cat(prior_norms, dim=0).detach()
                if priors.numel() > 0:
                    ratio = res_norms.mean() / (priors.mean() + 1e-6)
                    metrics["metrics/pose/align_res_ratio_mean"] = float(ratio.item())

    return metrics


def compute_depth_losses(self, inputs, outputs, losses):
    """Compute depth metrics, to allow monitoring during training."""
    dataset_name = getattr(self.opt, "dataset", "").lower()

    if dataset_name in {
        "uavid2020",
        "uavid_tridataset",
        "uavid2020_tridataset",
        "uavula",
        "uavula_dataset",
        "uavula_tridataset",
    }:
        compute_uavid_depth_metrics(self, inputs, outputs, losses)
    else:
        compute_kitti_depth_metrics(self, inputs, outputs, losses)


def compute_kitti_depth_metrics(self, inputs, outputs, losses):
    """保留原有 KITTI 评估流程，便于其他数据集复用。"""
    if "depth_gt" not in inputs:
        return

    depth_has_valid = inputs.get("depth_has_valid", None)
    if depth_has_valid is not None:
        flat_valid = depth_has_valid.view(-1).to(torch.bool)
        if not bool(flat_valid.any()):
            return

    depth_pred = outputs[("depth", 0, 0)]
    depth_pred = torch.clamp(
        F.interpolate(depth_pred, [375, 1242], mode="bilinear", align_corners=False),
        1e-3,
        80,
    ).detach()

    depth_gt = inputs["depth_gt"]
    mask = depth_gt > 0

    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask & crop_mask

    depth_gt = depth_gt[mask]
    depth_pred = depth_pred[mask]

    if depth_pred.numel() == 0 or depth_gt.numel() == 0:
        return

    depth_pred *= torch.median(depth_gt) / (torch.median(depth_pred) + 1e-12)
    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

    depth_errors = compute_depth_errors(depth_gt, depth_pred)
    for i, metric in enumerate(self.depth_metric_names):
        losses[metric] = np.array(depth_errors[i].cpu())


def compute_uavid_depth_metrics(self, inputs, outputs, losses):
    """UAVid2020 深度评估：中值缩放 + 5/95 百分位裁剪。"""
    depth_gt = inputs.get("depth_gt", None)
    if depth_gt is None:
        return

    depth_has_valid = inputs.get("depth_has_valid", None)
    if depth_has_valid is not None:
        depth_has_valid = depth_has_valid.view(-1).to(torch.bool)
        if not bool(depth_has_valid.any()):
            return

    depth_pred = outputs.get(("depth", 0, 0), None)
    if depth_pred is None:
        return

    if depth_gt.ndim == 3:
        depth_gt = depth_gt.unsqueeze(1)
    if depth_pred.ndim == 3:
        depth_pred = depth_pred.unsqueeze(1)

    metrics_buffer = []
    ratios = []

    for b in range(depth_pred.shape[0]):
        if depth_has_valid is not None and not bool(depth_has_valid[b]):
            continue

        pred_map = depth_pred[b:b + 1]
        gt_map = depth_gt[b:b + 1]

        if gt_map.numel() == 0:
            continue

        gt_map = gt_map.to(pred_map.device, dtype=pred_map.dtype)
        pred_resized = F.interpolate(
            pred_map, gt_map.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        gt_resized = gt_map[0, 0]

        valid_pred_mask = pred_resized > 0
        valid_gt_mask = torch.isfinite(gt_resized)
        base_mask = valid_pred_mask & valid_gt_mask
        if base_mask.sum() == 0:
            continue

        gt_valid = gt_resized[base_mask]
        q_low = torch.quantile(gt_valid, 0.05)
        q_high = torch.quantile(gt_valid, 0.95)

        if torch.isnan(q_low) or torch.isnan(q_high) or q_high <= q_low:
            continue

        q_low_val = float(q_low.item())
        q_high_val = float(q_high.item())

        gt_clipped = torch.clamp(gt_resized, min=q_low_val, max=q_high_val)
        eval_mask = (gt_clipped > 0) & base_mask
        if eval_mask.sum() == 0:
            continue

        gt_eval = gt_clipped[eval_mask]
        pred_eval = pred_resized[eval_mask]

        median_pred = torch.median(pred_eval)
        if median_pred.abs() < 1e-12:
            continue

        scale_ratio = torch.median(gt_eval) / (median_pred + 1e-12)
        pred_scaled = torch.clamp(pred_eval * scale_ratio, min=q_low_val, max=q_high_val)

        depth_errors = compute_depth_errors(gt_eval, pred_scaled)
        metrics_buffer.append(torch.stack(depth_errors))
        ratios.append(scale_ratio.detach().cpu())

    if not metrics_buffer:
        return

    metrics_tensor = torch.stack(metrics_buffer, dim=0)
    mean_metrics = metrics_tensor.mean(dim=0)
    for i, metric in enumerate(self.depth_metric_names):
        losses[metric] = float(mean_metrics[i].cpu().item())

    if ratios:
        ratios_tensor = torch.stack([r.to(torch.float32) for r in ratios])
        losses["metrics/uavid2020/scale_ratio_median"] = float(ratios_tensor.median().item())


def log_time(self, batch_idx, duration, loss):
    """Print a logging statement to the terminal."""
    samples_per_sec = self.opt.batch_size / duration
    time_sofar = time.time() - self.start_time
    training_time_left = (
        (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
    )
    print_string = (
        "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}"
        + " | loss: {:.5f} | time elapsed: {} | time left: {}"
    )
    print(
        print_string.format(
            self.epoch,
            batch_idx,
            samples_per_sec,
            loss,
            sec_to_hm_str(time_sofar),
            sec_to_hm_str(training_time_left),
        )
    )


def log_epoch_metrics(self, mode):
    if not self.collect_debug_metrics:
        return
    buffer = self._metric_buffers.get(mode, None)
    if not buffer:
        return

    aggregated = {}
    for key, values in buffer.items():
        if not values:
            continue
        metric_mean = sum(values) / len(values)
        metric_name = key[len("metrics/"):]
        aggregated[f"{mode}/{metric_name}"] = metric_mean

    if not aggregated:
        self._reset_metrics(mode)
        return

    header = f"[Metrics][{mode}] step={self.step}"
    lines = [header]
    for k in sorted(aggregated.keys()):
        v = aggregated[k]
        if isinstance(v, (int, float)):
            lines.append(f"  {k} = {v:.6f}")
        else:
            lines.append(f"  {k} = {v}")
    print("\n".join(lines))

    if getattr(self, "using_wandb", False):
        wandb.log(aggregated, step=self.step)

    self._reset_metrics(mode)


def log(self, mode, inputs, outputs, losses):
    s_main = 0 if (hasattr(self.opt, "scales") and 0 in self.opt.scales) else max(self.opt.scales)
    is_spidepth = str(getattr(self.opt, "methods", "")) == "SPIDepth"
    enable_automask_margin_viz = bool(getattr(self.opt, "enable_automask_margin_viz", False))
    save_automask_margin_viz_local = bool(getattr(self.opt, "save_automask_margin_viz_local", False))
    automask_margin_viz_samples = max(1, int(getattr(self.opt, "automask_margin_viz_samples", 1)))

    def _maybe_log_automask_margin_panels(log_to_wandb: bool):
        if self.opt.disable_automasking:
            return
        if not (enable_automask_margin_viz or save_automask_margin_viz_local):
            return

        identity_comp = outputs.get(f"automask_identity_comp/{s_main}", None)
        reproj_comp = outputs.get(f"automask_reproj_comp/{s_main}", None)
        margin_map = outputs.get(f"automask_margin/{s_main}", None)
        keep_map = outputs.get(f"identity_selection/{s_main}", None)
        if not (
            torch.is_tensor(identity_comp)
            and torch.is_tensor(reproj_comp)
            and torch.is_tensor(margin_map)
            and torch.is_tensor(keep_map)
        ):
            return

        save_dir = None
        if save_automask_margin_viz_local:
            save_dir = os.path.join(self.log_path, "automask_margin_viz", mode)
            os.makedirs(save_dir, exist_ok=True)

        batch_n = identity_comp.shape[0]
        num_vis = min(automask_margin_viz_samples, batch_n)
        for j in range(num_vis):
            id_2d = identity_comp[j].detach().squeeze()
            reproj_2d = reproj_comp[j].detach().squeeze()
            margin_2d = margin_map[j].detach().squeeze()
            keep_2d = keep_map[j].detach().squeeze() > 0.5
            finite_2d = torch.isfinite(id_2d) & torch.isfinite(reproj_2d) & torch.isfinite(margin_2d)
            if not bool(finite_2d.any()):
                continue

            pair_vals = torch.cat([id_2d[finite_2d], reproj_2d[finite_2d]], dim=0)
            loss_lo = float(torch.quantile(pair_vals, 0.02).item())
            loss_hi = float(torch.quantile(pair_vals, 0.98).item())
            if (not math.isfinite(loss_lo)) or (not math.isfinite(loss_hi)) or loss_hi <= loss_lo:
                loss_lo = float(pair_vals.min().item())
                loss_hi = float(pair_vals.max().item())
            if loss_hi <= loss_lo:
                loss_hi = loss_lo + 1e-6

            margin_vals = margin_2d[finite_2d]
            margin_abs_hi = float(torch.quantile(margin_vals.abs(), 0.98).item())
            if (not math.isfinite(margin_abs_hi)) or margin_abs_hi <= 0.0:
                margin_abs_hi = float(margin_vals.abs().max().item()) if margin_vals.numel() > 0 else 1.0
            margin_abs_hi = max(margin_abs_hi, 1e-6)

            keep_valid = keep_2d & finite_2d
            masked_valid = (~keep_2d) & finite_2d
            keep_ratio = float(
                keep_valid.float().sum().item() / max(1.0, float(finite_2d.float().sum().item()))
            )
            margin_mean = float(margin_vals.mean().item())
            keep_mean = float(margin_2d[keep_valid].mean().item()) if bool(keep_valid.any()) else float("nan")
            masked_mean = float(margin_2d[masked_valid].mean().item()) if bool(masked_valid.any()) else float("nan")

            save_path = None
            if save_dir is not None:
                save_path = os.path.join(
                    save_dir,
                    f"ep{int(self.epoch):03d}_step{int(self.step):08d}_s{s_main}_b{j}.png",
                )

            _log_scalar_row_with_colorbar_to_wandb(
                scalar_images=[
                    id_2d,
                    reproj_2d,
                    margin_2d,
                ],
                scalar_titles=[
                    f"id_min(mean={float(id_2d[finite_2d].mean().item()):.4f})",
                    f"reproj_min(mean={float(reproj_2d[finite_2d].mean().item()):.4f})",
                    f"margin(mean={margin_mean:.4f})",
                ],
                vmin=loss_lo,
                vmax=loss_hi,
                cmap="magma",
                extra_images=[
                    keep_2d.float().unsqueeze(0),
                    inputs[("color", 0, s_main)][j].detach().clamp(0.0, 1.0),
                ],
                extra_titles=[
                    f"keep(r={keep_ratio:.3f}, keep={keep_mean:.4f}, mask={masked_mean:.4f})",
                    "target",
                ],
                scalar_vmins=[loss_lo, loss_lo, -margin_abs_hi],
                scalar_vmaxs=[loss_hi, loss_hi, margin_abs_hi],
                colorbar_groups=[0, 0, 1],
                scalar_cmaps=["magma", "magma", "coolwarm"],
                scalar_valid_masks=[finite_2d, finite_2d, finite_2d],
                masked_color=(0.0, 0.0, 0.0, 1.0),
                key=f"{mode}/automask_compare_{s_main}_{j}",
                step=self.step,
                save_path=save_path,
                log_to_wandb=log_to_wandb and enable_automask_margin_viz,
            )

    if self.using_wandb:
        val = losses["loss"]
        vloss = float(val.detach().cpu().item() if isinstance(val, torch.Tensor) else val)
        log_dict = {f"{mode}/loss": vloss}
        depth_pairs = []

        scale_align = outputs.get("scale_align_factor", None)
        if isinstance(scale_align, torch.Tensor) and scale_align.numel() > 0:
            s_mean = float(scale_align.detach().mean().cpu().item())
            log_dict[f"{mode}/scale_align_factor_mean"] = s_mean

        pose_scale = self.models.get("pose_scale", None)
        if pose_scale is not None and hasattr(pose_scale, "log_scale"):
            try:
                scale_val = float(torch.exp(pose_scale.log_scale.detach()).item())
                log_dict[f"{mode}/pose_scale"] = scale_val
            except Exception:
                pass

        monitor = self._scale_align_monitor.get(mode)
        if monitor and monitor["attempts"] > 0:
            success_ratio = monitor["success"] / float(max(1, monitor["attempts"]))
            log_dict[f"{mode}/scale_align_success_ratio"] = success_ratio

        # 附加深度相关标量（de/* 与 metrics/uavid2020/*）
        for key, value in losses.items():
            if key == "loss":
                continue
            if not (key.startswith("de/") or key.startswith("metrics/uavid2020/")):
                continue
            if isinstance(value, torch.Tensor):
                val_float = float(value.detach().cpu().item())
            else:
                val_float = float(value)
            log_dict[f"{mode}/{key}"] = val_float
            depth_pairs.append((key, val_float))

        # 蒸馏相关指标可视化
        distill_keys = (
            "loss/pose_teacher",
            "loss/pose_teacher_rot",
            "loss/pose_teacher_trans",
            "loss/pose_teacher_weight",
            "loss/teacher_photo",
            "loss/teacher_photo_raw",
            "loss/teacher_photo_weight",
        )
        for key in distill_keys:
            if key not in losses:
                continue
            value = losses[key]
            if isinstance(value, torch.Tensor):
                val_float = float(value.detach().cpu().item())
            else:
                val_float = float(value)
            log_dict[f"{mode}/{key}"] = val_float

        depth_sens_weight_map = outputs.get(f"depth_sens_weight_selected/{s_main}", None)
        if torch.is_tensor(depth_sens_weight_map):
            w_valid = outputs.get(f"depth_sens_weight_valid_selected/{s_main}", None)
            if torch.is_tensor(w_valid):
                valid_mask = w_valid > 0.5
                w_flat = depth_sens_weight_map.detach()[valid_mask]
            else:
                w_flat = depth_sens_weight_map.detach().reshape(-1)
            w_flat = w_flat[torch.isfinite(w_flat)]
            if w_flat.numel() > 0:
                log_dict[f"{mode}/depth_sens_weight/mean"] = float(w_flat.mean().item())
                log_dict[f"{mode}/depth_sens_weight/p90"] = float(torch.quantile(w_flat, 0.9).item())

        sens_delta_means = []
        sens_rel_delta_means = []
        sens_rel_delta_means_after_auto = []
        pix_delta_means = []
        pix_rel_delta_means = []
        pix_rel_delta_means_after_auto = []
        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                continue
            valid_map = outputs.get(("depth_sens_valid", frame_id, s_main), None)
            if not torch.is_tensor(valid_map):
                continue
            valid_bool = valid_map > 0.5
            auto_keep = outputs.get(("automask_keep_fid", frame_id, s_main), None)
            if not torch.is_tensor(auto_keep):
                auto_keep = outputs.get(f"identity_selection/{s_main}", None)
            if torch.is_tensor(auto_keep):
                auto_keep_bool = auto_keep > 0.5
                if auto_keep_bool.shape != valid_bool.shape:
                    auto_keep_bool = auto_keep_bool.expand_as(valid_bool)
            else:
                auto_keep_bool = torch.ones_like(valid_bool, dtype=torch.bool)
            keep_bool = valid_bool & auto_keep_bool
            delta_map = outputs.get(("depth_sens_loss_delta", frame_id, s_main), None)
            rel_delta_map = outputs.get(("depth_sens_loss_rel_delta", frame_id, s_main), None)
            if torch.is_tensor(delta_map) and torch.is_tensor(rel_delta_map):
                valid = valid_bool.to(dtype=delta_map.dtype)
                denom = valid.sum().clamp_min(1.0)
                delta_mean = float(((delta_map * valid).sum() / denom).detach().cpu().item())
                rel_delta_mean = float(((rel_delta_map * valid).sum() / denom).detach().cpu().item())
                if bool(keep_bool.any()):
                    keep = keep_bool.to(dtype=rel_delta_map.dtype)
                    denom_keep = keep.sum().clamp_min(1.0)
                    rel_delta_mean_after_auto = float(
                        ((rel_delta_map * keep).sum() / denom_keep).detach().cpu().item()
                    )
                else:
                    rel_delta_mean_after_auto = float("nan")

                log_dict[f"{mode}/depth_sensitivity/delta_mean_f{frame_id}"] = delta_mean
                log_dict[f"{mode}/depth_sensitivity/rel_delta_mean_f{frame_id}"] = rel_delta_mean
                log_dict[f"{mode}/depth_sensitivity/rel_delta_mean_after_automask_f{frame_id}"] = (
                    rel_delta_mean_after_auto
                )
                sens_delta_means.append(delta_mean)
                sens_rel_delta_means.append(rel_delta_mean)
                if math.isfinite(rel_delta_mean_after_auto):
                    sens_rel_delta_means_after_auto.append(rel_delta_mean_after_auto)

            pix_delta_map = outputs.get(("depth_sens_pixmag_delta", frame_id, s_main), None)
            pix_rel_delta_map = outputs.get(("depth_sens_pixmag_rel_delta", frame_id, s_main), None)
            if torch.is_tensor(pix_delta_map) and torch.is_tensor(pix_rel_delta_map):
                valid = valid_bool.to(dtype=pix_delta_map.dtype)
                denom = valid.sum().clamp_min(1.0)
                pix_delta_mean = float(((pix_delta_map * valid).sum() / denom).detach().cpu().item())
                pix_rel_delta_mean = float(((pix_rel_delta_map * valid).sum() / denom).detach().cpu().item())
                if bool(keep_bool.any()):
                    keep = keep_bool.to(dtype=pix_rel_delta_map.dtype)
                    denom_keep = keep.sum().clamp_min(1.0)
                    pix_rel_delta_mean_after_auto = float(
                        ((pix_rel_delta_map * keep).sum() / denom_keep).detach().cpu().item()
                    )
                else:
                    pix_rel_delta_mean_after_auto = float("nan")

                log_dict[f"{mode}/depth_pixshift/delta_mean_f{frame_id}"] = pix_delta_mean
                log_dict[f"{mode}/depth_pixshift/rel_delta_mean_f{frame_id}"] = pix_rel_delta_mean
                log_dict[f"{mode}/depth_pixshift/rel_delta_mean_after_automask_f{frame_id}"] = (
                    pix_rel_delta_mean_after_auto
                )
                pix_delta_means.append(pix_delta_mean)
                pix_rel_delta_means.append(pix_rel_delta_mean)
                if math.isfinite(pix_rel_delta_mean_after_auto):
                    pix_rel_delta_means_after_auto.append(pix_rel_delta_mean_after_auto)
        if sens_delta_means:
            log_dict[f"{mode}/depth_sensitivity/delta_mean"] = float(
                sum(sens_delta_means) / len(sens_delta_means)
            )
        if sens_rel_delta_means:
            log_dict[f"{mode}/depth_sensitivity/rel_delta_mean"] = float(
                sum(sens_rel_delta_means) / len(sens_rel_delta_means)
            )
        if sens_rel_delta_means_after_auto:
            log_dict[f"{mode}/depth_sensitivity/rel_delta_mean_after_automask"] = float(
                sum(sens_rel_delta_means_after_auto) / len(sens_rel_delta_means_after_auto)
            )
        if pix_delta_means:
            log_dict[f"{mode}/depth_pixshift/delta_mean"] = float(
                sum(pix_delta_means) / len(pix_delta_means)
            )
        if pix_rel_delta_means:
            log_dict[f"{mode}/depth_pixshift/rel_delta_mean"] = float(
                sum(pix_rel_delta_means) / len(pix_rel_delta_means)
            )
        if pix_rel_delta_means_after_auto:
            log_dict[f"{mode}/depth_pixshift/rel_delta_mean_after_automask"] = float(
                sum(pix_rel_delta_means_after_auto) / len(pix_rel_delta_means_after_auto)
            )

        if not self.collect_debug_metrics:
            # 兼容未开启 debug 聚合时的旧行为：按 step 直接记录。
            for key, value in losses.items():
                if not (
                    key.startswith("metrics/hrmask/")
                    or key.startswith("metrics/derot/")
                    or key.startswith("metrics/automask/")
                    or key.startswith("metrics/final_loss_mask/")
                ):
                    continue
                if isinstance(value, torch.Tensor):
                    val_float = float(value.detach().cpu().item())
                else:
                    val_float = float(value)
                log_dict[f"{mode}/{key[len('metrics/') :]}"] = val_float
        # Debug metrics/* (包括 hrmask / derot / automask / final_loss_mask)
        # 在开启 debug 聚合时，统一由 _log_epoch_metrics 按 epoch 记录，避免 step 级抖动。

        wandb.log(log_dict, step=self.step)
        _maybe_log_automask_margin_panels(log_to_wandb=True)

        if depth_pairs:
            summary = ", ".join(f"{name}={val:.4f}" for name, val in depth_pairs)
            print(f"[DepthMetrics][{mode}] step={self.step} | {summary}")

        # 在 log() 里，记录完 vloss 之后追加：
        if mode == "val":
            print(f"[Validation] step={self.step} | loss={vloss:.6f}")
            best = wandb.run.summary.get("val/loss.min", float("inf"))
            if vloss < best:
                wandb.run.summary["val/loss.min"] = vloss

        int_frame_ids = [fid for fid in self.opt.frame_ids[1:] if isinstance(fid, int)]
        prev_fid = max([fid for fid in int_frame_ids if fid < 0], default=None)
        next_fid = min([fid for fid in int_frame_ids if fid > 0], default=None)
        has_bidirectional_pair = prev_fid is not None and next_fid is not None
        is_derot_sigmoid = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_PoseGT_DeRotSigmoidWeight"

        def _shared_range(t0, t1, q_low=0.02, q_high=0.98, clip_max=None):
            vals = torch.cat([t0.reshape(-1), t1.reshape(-1)], dim=0).detach().float()
            vals = vals[torch.isfinite(vals)]
            if clip_max is not None:
                vals = vals[vals < float(clip_max)]
            if vals.numel() == 0:
                return 0.0, 1.0
            lo = float(torch.quantile(vals, q_low).item())
            hi = float(torch.quantile(vals, q_high).item())
            if (not math.isfinite(lo)) or (not math.isfinite(hi)) or hi <= lo:
                lo = float(vals.min().item())
                hi = float(vals.max().item())
            if hi <= lo:
                hi = lo + 1e-6
            return lo, hi

        def _single_range(t0, q_low=0.02, q_high=0.98, clip_max=None):
            vals = t0.reshape(-1).detach().float()
            vals = vals[torch.isfinite(vals)]
            if clip_max is not None:
                vals = vals[vals < float(clip_max)]
            if vals.numel() == 0:
                return 0.0, 1.0
            lo = float(torch.quantile(vals, q_low).item())
            hi = float(torch.quantile(vals, q_high).item())
            if (not math.isfinite(lo)) or (not math.isfinite(hi)) or hi <= lo:
                lo = float(vals.min().item())
                hi = float(vals.max().item())
            if hi <= lo:
                hi = lo + 1e-6
            return lo, hi

        def _to_2d(x, index, dtype=None):
            y = x[index]
            if y.ndim == 3:
                y = y[0]
            if dtype is not None:
                y = y.to(dtype=dtype)
            return y

        def _tensor_scalar(x, index):
            if not torch.is_tensor(x):
                return None
            vals = x[index].detach().reshape(-1)
            if vals.numel() < 1:
                return None
            return float(vals[0].cpu().item())

        def _get_logged_warp(frame_id, scale):
            pred = outputs.get(("color", frame_id, scale), None)
            if torch.is_tensor(pred):
                return pred
            pred = outputs.get(("color_MiS", frame_id, scale), None)
            if torch.is_tensor(pred):
                return pred
            return None

        for j in range(min(4, self.opt.batch_size)):
            # 只用 s_main
            for frame_id in self.opt.frame_ids:
                # 收集需要拼的图
                imgs, titles = [], []

                if frame_id == 0:
                    imgs.append(inputs[("color", frame_id, s_main)][j])
                    titles.append("target")
                else:
                    aux_img = outputs.get(("pre_warp_img", frame_id, 0), None)
                    aux_title = "pre_warp(R)"
                    if not torch.is_tensor(aux_img):
                        aux_img = outputs.get(("irw_img", frame_id, 0), None)
                        aux_title = "irw_img"
                    pred_rdistill = outputs.get(("color_rdistill", frame_id, s_main), None)
                    rdistill_gate_pair = _tensor_scalar(outputs.get(("r_distill_gate_pair", frame_id, s_main), None), j)
                    rdistill_active_pair = _tensor_scalar(outputs.get(("r_distill_active_pair", frame_id, s_main), None), j)
                    rdistill_valid_pair = _tensor_scalar(outputs.get(("r_distill_valid_pair", frame_id, s_main), None), j)
                    rdistill_delta_pair = _tensor_scalar(outputs.get(("r_distill_delta_rel_pair", frame_id, s_main), None), j)
                    rdistill_rot_pair = _tensor_scalar(outputs.get(("r_distill_rot_pair", frame_id, s_main), None), j)
                    rdistill_margin_mean = _tensor_scalar(outputs.get("r_distill_margin_mean/0", None), j)
                    rdistill_margin_gate = _tensor_scalar(outputs.get("r_distill_margin_gate/0", None), j)
                    if torch.is_tensor(pred_rdistill):
                        imgs.append(inputs[("color", 0, s_main)][j])
                        titles.append("target")

                    imgs.append(inputs[("color", frame_id, s_main)][j])
                    titles.append(f"source[{frame_id}]")

                    if torch.is_tensor(aux_img):
                        imgs.append(aux_img[j])
                        titles.append(aux_title)

                # 2) 预测 warp（通常 frame_id != 0）
                if frame_id != 0:
                    pred_warp = _get_logged_warp(frame_id, s_main)
                    if not torch.is_tensor(pred_warp):
                        continue
                    imgs.append(pred_warp[j])
                    pose_title = "warp_final"
                    if rdistill_delta_pair is not None or rdistill_gate_pair is not None:
                        pose_bits = []
                        if rdistill_delta_pair is not None:
                            pose_bits.append(f"dr={rdistill_delta_pair:.3f}")
                        if rdistill_gate_pair is not None:
                            pose_bits.append(f"g={rdistill_gate_pair:.2f}")
                        if rdistill_active_pair is not None:
                            pose_bits.append(f"a={int(rdistill_active_pair > 0.5)}")
                        if rdistill_valid_pair is not None:
                            pose_bits.append(f"v={int(rdistill_valid_pair > 0.5)}")
                        pose_title = f"warp_final({' '.join(pose_bits)})"
                    titles.append(pose_title)

                    if torch.is_tensor(pred_rdistill):
                        imgs.append(pred_rdistill[j])
                        ext_bits = []
                        if rdistill_rot_pair is not None:
                            ext_bits.append(f"rot={rdistill_rot_pair:.3f}")
                        if rdistill_margin_mean is not None:
                            ext_bits.append(f"mm={rdistill_margin_mean:.3f}")
                        if rdistill_margin_gate is not None:
                            ext_bits.append(f"m_ok={int(rdistill_margin_gate > 0.5)}")
                        ext_title = "warp_extR" if not ext_bits else f"warp_extR({' '.join(ext_bits)})"
                        titles.append(ext_title)

                # 根据收集到的张数（2 或 3）生成一行子图，并一次性记录到一个格子
                _log_row_images_to_wandb(
                    images=imgs,
                    titles=titles,
                    key=f"{mode}/color_panel_{frame_id}_{s_main}_{j}",
                    step=self.step,
                )
                if frame_id != 0 and torch.is_tensor(pred_rdistill):
                    target_rgb = inputs[("color", 0, s_main)][j].detach().clamp(0.0, 1.0)
                    pred_warp = _get_logged_warp(frame_id, s_main)
                    if not torch.is_tensor(pred_warp):
                        continue
                    pose_rgb = pred_warp[j].detach().clamp(0.0, 1.0)
                    rdistill_rgb = pred_rdistill[j].detach().clamp(0.0, 1.0)

                    pose_err_2d = (pose_rgb - target_rgb).abs().mean(dim=0)
                    rdistill_err_2d = (rdistill_rgb - target_rgb).abs().mean(dim=0)
                    improve_2d = pose_err_2d - rdistill_err_2d

                    pose_valid_2d = None
                    pose_valid = outputs.get(("distorted_mask", frame_id, s_main), None)
                    if torch.is_tensor(pose_valid):
                        pose_valid_2d = _to_2d(pose_valid, j, dtype=torch.float32) > 0.5

                    rdistill_valid_2d = None
                    rdistill_valid = outputs.get(("distorted_mask_rdistill", frame_id, s_main), None)
                    if torch.is_tensor(rdistill_valid):
                        rdistill_valid_2d = _to_2d(rdistill_valid, j, dtype=torch.float32) > 0.5

                    improve_valid_2d = None
                    if pose_valid_2d is not None and rdistill_valid_2d is not None:
                        improve_valid_2d = pose_valid_2d & rdistill_valid_2d
                    elif pose_valid_2d is not None:
                        improve_valid_2d = pose_valid_2d
                    elif rdistill_valid_2d is not None:
                        improve_valid_2d = rdistill_valid_2d

                    err_vmin, err_vmax = _shared_range(
                        pose_err_2d,
                        rdistill_err_2d,
                        q_low=0.0,
                        q_high=0.98,
                        clip_max=1.0,
                    )
                    improve_vals = improve_2d.detach()
                    if improve_valid_2d is not None and bool(improve_valid_2d.any()):
                        improve_vals = improve_vals[improve_valid_2d]
                    else:
                        improve_vals = improve_vals.reshape(-1)
                    improve_vals = improve_vals[torch.isfinite(improve_vals)]
                    if improve_vals.numel() > 0:
                        improve_abs = float(torch.quantile(improve_vals.abs(), 0.98).item())
                    else:
                        improve_abs = 1.0
                    improve_abs = max(improve_abs, 1e-6)

                    if pose_valid_2d is not None and bool(pose_valid_2d.any()):
                        pose_err_mean = float(pose_err_2d[pose_valid_2d].mean().item())
                    else:
                        pose_err_mean = float(pose_err_2d.mean().item())
                    if rdistill_valid_2d is not None and bool(rdistill_valid_2d.any()):
                        rdistill_err_mean = float(rdistill_err_2d[rdistill_valid_2d].mean().item())
                    else:
                        rdistill_err_mean = float(rdistill_err_2d.mean().item())
                    if improve_valid_2d is not None and bool(improve_valid_2d.any()):
                        improve_mean = float(improve_2d[improve_valid_2d].mean().item())
                    else:
                        improve_mean = float(improve_2d.mean().item())

                    _log_scalar_row_with_colorbar_to_wandb(
                        scalar_images=[
                            pose_err_2d,
                            rdistill_err_2d,
                            improve_2d,
                        ],
                        scalar_titles=[
                            f"pose_err(mean={pose_err_mean:.4f})",
                            f"extR_err(mean={rdistill_err_mean:.4f})",
                            (
                                f"improve(mean={improve_mean:.4f}, "
                                f"dr={0.0 if rdistill_delta_pair is None else rdistill_delta_pair:.4f}, "
                                f"g={0.0 if rdistill_gate_pair is None else rdistill_gate_pair:.2f}, "
                                f"a={0 if rdistill_active_pair is None else int(rdistill_active_pair > 0.5)}, "
                                f"m_ok={0 if rdistill_margin_gate is None else int(rdistill_margin_gate > 0.5)})"
                            ),
                        ],
                        vmin=err_vmin,
                        vmax=err_vmax,
                        scalar_vmins=[err_vmin, err_vmin, -improve_abs],
                        scalar_vmaxs=[err_vmax, err_vmax, improve_abs],
                        colorbar_groups=[0, 0, 1],
                        scalar_cmaps=["magma", "magma", "coolwarm"],
                        scalar_valid_masks=[
                            pose_valid_2d,
                            rdistill_valid_2d,
                            improve_valid_2d,
                        ],
                        masked_color=(0.0, 0.0, 0.0, 1.0),
                        key=f"{mode}/warp_error_panel_{frame_id}_{s_main}_{j}",
                        step=self.step,
                    )
                if j == 0 and frame_id != 0:
                    sens_base = outputs.get(("depth_sens_color_base", frame_id, s_main), None)
                    sens_perturbed = outputs.get(("depth_sens_color_perturbed", frame_id, s_main), None)
                    sens_loss_base = outputs.get(("depth_sens_loss_base", frame_id, s_main), None)
                    sens_loss_perturbed = outputs.get(("depth_sens_loss_perturbed", frame_id, s_main), None)
                    sens_loss_delta = outputs.get(("depth_sens_loss_delta", frame_id, s_main), None)
                    sens_loss_rel_delta = outputs.get(("depth_sens_loss_rel_delta", frame_id, s_main), None)
                    sens_pix_base = outputs.get(("depth_sens_pixmag_base", frame_id, s_main), None)
                    sens_pix_perturbed = outputs.get(("depth_sens_pixmag_perturbed", frame_id, s_main), None)
                    sens_pix_delta = outputs.get(("depth_sens_pixmag_delta", frame_id, s_main), None)
                    sens_pix_rel_delta = outputs.get(("depth_sens_pixmag_rel_delta", frame_id, s_main), None)
                    sens_valid = outputs.get(("depth_sens_valid", frame_id, s_main), None)
                    if (
                        torch.is_tensor(sens_base)
                        and torch.is_tensor(sens_perturbed)
                        and torch.is_tensor(sens_loss_base)
                        and torch.is_tensor(sens_loss_perturbed)
                        and torch.is_tensor(sens_loss_delta)
                        and torch.is_tensor(sens_loss_rel_delta)
                        and torch.is_tensor(sens_valid)
                    ):
                        valid_j = sens_valid[j] > 0.5
                        if valid_j.ndim == 3:
                            valid_2d = valid_j[0]
                        else:
                            valid_2d = valid_j

                        auto_keep_j = outputs.get(("automask_keep_fid", frame_id, s_main), None)
                        auto_title = f"keep(fid={frame_id})"
                        if not torch.is_tensor(auto_keep_j):
                            auto_keep_j = outputs.get(f"identity_selection/{s_main}", None)
                            auto_title = "keep(union)"
                        if torch.is_tensor(auto_keep_j):
                            auto_keep_j = auto_keep_j[j]
                            if auto_keep_j.ndim == 3:
                                auto_keep_2d = auto_keep_j[0] > 0.5
                            elif auto_keep_j.ndim == 2:
                                auto_keep_2d = auto_keep_j > 0.5
                            else:
                                auto_keep_2d = torch.ones_like(valid_2d, dtype=torch.bool)
                        else:
                            auto_keep_2d = torch.ones_like(valid_2d, dtype=torch.bool)
                            auto_title = "keep(?)"

                        keep_2d = valid_2d & auto_keep_2d
                        keep_2d_f = keep_2d.float()

                        base_loss_2d = sens_loss_base[j].squeeze(0)
                        pert_loss_2d = sens_loss_perturbed[j].squeeze(0)
                        delta_2d = sens_loss_delta[j].squeeze(0)
                        rel_delta_2d = sens_loss_rel_delta[j].squeeze(0)
                        base_loss_masked_2d = base_loss_2d * keep_2d_f
                        pert_loss_masked_2d = pert_loss_2d * keep_2d_f
                        delta_masked_2d = delta_2d * keep_2d_f
                        rel_delta_masked_2d = rel_delta_2d * keep_2d_f

                        if bool(keep_2d.any()):
                            base_pert_vals = torch.cat(
                                [
                                    base_loss_2d[keep_2d],
                                    pert_loss_2d[keep_2d],
                                ],
                                dim=0,
                            )
                            delta_vals = delta_2d[keep_2d]
                            rel_vals = rel_delta_2d[keep_2d]
                            loss_vmax = float(torch.quantile(base_pert_vals, 0.98).item())
                            delta_vmax = float(torch.quantile(delta_vals, 0.98).item())
                            rel_vmax = float(torch.quantile(rel_vals, 0.98).item())
                        else:
                            loss_vmax = 1.0
                            delta_vmax = 1.0
                            rel_vmax = 1.0
                        loss_vmax = max(loss_vmax, 1e-6)
                        delta_vmax = max(delta_vmax, 1e-6)
                        rel_vmax = max(rel_vmax, 1e-6)

                        # 第一行：target / warp / depth*1.1 的 warp
                        _log_row_images_to_wandb(
                            images=[
                                inputs[("color", 0, s_main)][j].detach().clamp(0.0, 1.0),
                                sens_base[j].detach().clamp(0.0, 1.0),
                                sens_perturbed[j].detach().clamp(0.0, 1.0),
                            ],
                            titles=[
                                "target",
                                "warp",
                                f"warp(x{self.depth_sensitivity_factor:.2f})",
                            ],
                            key=f"{mode}/depth_sens_rgb_{frame_id}_{s_main}_{j}",
                            step=self.step,
                        )
                        # 第二行：loss split 风格，loss 图叠加 automask（keep 之外为 0）
                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=[
                                base_loss_masked_2d,
                                pert_loss_masked_2d,
                            ],
                            scalar_titles=[
                                "loss(base)",
                                f"loss(x{self.depth_sensitivity_factor:.2f})",
                            ],
                            vmin=0.0,
                            vmax=loss_vmax,
                            cmap="magma",
                            extra_images=[
                                auto_keep_2d.float().unsqueeze(0),
                                keep_2d.float().unsqueeze(0),
                            ],
                            extra_titles=[
                                auto_title,
                                "keep&valid",
                            ],
                            key=f"{mode}/depth_sens_loss_split_{frame_id}_{s_main}_{j}",
                            step=self.step,
                        )
                        # 差值单独归一化，并单独提供 colorbar
                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=[
                                delta_masked_2d,
                            ],
                            scalar_titles=[
                                "d_loss",
                            ],
                            vmin=0.0,
                            vmax=delta_vmax,
                            cmap="magma",
                            extra_images=[
                                auto_keep_2d.float().unsqueeze(0),
                                keep_2d.float().unsqueeze(0),
                            ],
                            extra_titles=[
                                auto_title,
                                "keep&valid",
                            ],
                            key=f"{mode}/depth_sens_delta_split_{frame_id}_{s_main}_{j}",
                            step=self.step,
                        )
                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=[
                                rel_delta_masked_2d,
                            ],
                            scalar_titles=[
                                "r_loss",
                            ],
                            vmin=0.0,
                            vmax=rel_vmax,
                            cmap="magma",
                            extra_images=[
                                auto_keep_2d.float().unsqueeze(0),
                                keep_2d.float().unsqueeze(0),
                            ],
                            extra_titles=[
                                auto_title,
                                "keep&valid",
                            ],
                            key=f"{mode}/depth_sens_rel_delta_split_{frame_id}_{s_main}_{j}",
                            step=self.step,
                        )
                        if (
                            torch.is_tensor(sens_pix_base)
                            and torch.is_tensor(sens_pix_perturbed)
                            and torch.is_tensor(sens_pix_delta)
                            and torch.is_tensor(sens_pix_rel_delta)
                        ):
                            pix_base_2d = sens_pix_base[j].squeeze(0)
                            pix_perturbed_2d = sens_pix_perturbed[j].squeeze(0)
                            pix_delta_2d = sens_pix_delta[j].squeeze(0)
                            pix_rel_delta_2d = sens_pix_rel_delta[j].squeeze(0)
                            pix_base_masked_2d = pix_base_2d * keep_2d_f
                            pix_perturbed_masked_2d = pix_perturbed_2d * keep_2d_f
                            pix_delta_masked_2d = pix_delta_2d * keep_2d_f
                            pix_rel_delta_masked_2d = pix_rel_delta_2d * keep_2d_f

                            if bool(keep_2d.any()):
                                pix_base_vals = pix_base_2d[keep_2d]
                                pix_perturbed_vals = pix_perturbed_2d[keep_2d]
                                pix_delta_vals = pix_delta_2d[keep_2d]
                                pix_rel_vals = pix_rel_delta_2d[keep_2d]
                                pix_base_vmax = float(torch.quantile(pix_base_vals, 0.98).item())
                                pix_perturbed_vmax = float(torch.quantile(pix_perturbed_vals, 0.98).item())
                                pix_delta_vmax = float(torch.quantile(pix_delta_vals, 0.98).item())
                                pix_rel_vmax = float(torch.quantile(pix_rel_vals, 0.98).item())
                            else:
                                pix_base_vmax = 1.0
                                pix_perturbed_vmax = 1.0
                                pix_delta_vmax = 1.0
                                pix_rel_vmax = 1.0
                            pix_base_vmax = max(pix_base_vmax, 1e-6)
                            pix_perturbed_vmax = max(pix_perturbed_vmax, 1e-6)
                            pix_delta_vmax = max(pix_delta_vmax, 1e-6)
                            pix_rel_vmax = max(pix_rel_vmax, 1e-6)

                            pix_mag_vmax = max(pix_base_vmax, pix_perturbed_vmax)
                            _log_scalar_row_with_colorbar_to_wandb(
                                scalar_images=[
                                    pix_base_masked_2d,
                                    pix_perturbed_masked_2d,
                                ],
                                scalar_titles=[
                                    "pix_mag(base)",
                                    f"pix_mag(x{self.depth_sensitivity_factor:.2f})",
                                ],
                                vmin=0.0,
                                vmax=pix_mag_vmax,
                                cmap="magma",
                                extra_images=[
                                    auto_keep_2d.float().unsqueeze(0),
                                    keep_2d.float().unsqueeze(0),
                                ],
                                extra_titles=[
                                    auto_title,
                                    "keep&valid",
                                ],
                                key=f"{mode}/depth_pixshift_mag_split_{frame_id}_{s_main}_{j}",
                                step=self.step,
                            )
                            _log_scalar_row_with_colorbar_to_wandb(
                                scalar_images=[
                                    pix_delta_masked_2d,
                                ],
                                scalar_titles=[
                                    "pix_delta",
                                ],
                                vmin=0.0,
                                vmax=pix_delta_vmax,
                                cmap="magma",
                                extra_images=[
                                    auto_keep_2d.float().unsqueeze(0),
                                    keep_2d.float().unsqueeze(0),
                                ],
                                extra_titles=[
                                    auto_title,
                                    "keep&valid",
                                ],
                                key=f"{mode}/depth_pixshift_delta_split_{frame_id}_{s_main}_{j}",
                                step=self.step,
                            )
                            _log_scalar_row_with_colorbar_to_wandb(
                                scalar_images=[
                                    pix_rel_delta_masked_2d,
                                ],
                                scalar_titles=[
                                    "pix_rel",
                                ],
                                vmin=0.0,
                                vmax=pix_rel_vmax,
                                cmap="magma",
                                extra_images=[
                                    auto_keep_2d.float().unsqueeze(0),
                                    keep_2d.float().unsqueeze(0),
                                ],
                                extra_titles=[
                                    auto_title,
                                    "keep&valid",
                                ],
                                key=f"{mode}/depth_pixshift_rel_delta_split_{frame_id}_{s_main}_{j}",
                                step=self.step,
                            )
                if (
                    j == 0
                    and frame_id != 0
                    and str(getattr(self.opt, "methods", "")) in {
                        "MD2_VGGT_PoseGT_DeRotHardMask",
                        "MD2_VGGT_PoseGT_DeRotSigmoidWeight",
                    }
                ):
                    is_derot_weight = (
                        str(getattr(self.opt, "methods", "")) == "MD2_VGGT_PoseGT_DeRotSigmoidWeight"
                    )
                    derot_map = outputs.get(("derot_parallax", frame_id, s_main), None)
                    derot_mask = outputs.get(("derot_mask", frame_id, s_main), None)
                    derot_mask_raw = outputs.get(("derot_mask_raw", frame_id, s_main), None)
                    derot_weight = outputs.get(("derot_weight", frame_id, s_main), None)
                    derot_weight_raw = outputs.get(("derot_weight_raw", frame_id, s_main), None)
                    derot_vis_tensor = derot_weight if is_derot_weight else derot_mask
                    if torch.is_tensor(derot_map) and torch.is_tensor(derot_vis_tensor):
                        if is_derot_weight:
                            derot_vis = derot_weight_raw[j] if torch.is_tensor(derot_weight_raw) else derot_weight[j]
                            final_weight_fid = outputs.get(("final_loss_weight_fid", frame_id, s_main), None)
                            if torch.is_tensor(final_weight_fid):
                                keep_vis = final_weight_fid[j]
                            else:
                                final_weight = outputs.get(f"final_loss_weight/{s_main}", None)
                                if torch.is_tensor(final_weight):
                                    keep_vis = final_weight[j]
                                else:
                                    keep_final = outputs.get(f"final_loss_mask/{s_main}", None)
                                    if torch.is_tensor(keep_final):
                                        keep_vis = keep_final[j].float()
                                    else:
                                        keep_vis = derot_vis.float()
                        else:
                            derot_vis = derot_mask_raw[j] if torch.is_tensor(derot_mask_raw) else derot_mask[j]
                            keep_final = outputs.get(f"final_loss_mask/{s_main}", None)
                            if torch.is_tensor(keep_final):
                                keep_mask = keep_final[j] > 0.5
                            else:
                                # Fallback for methods without stored final keep mask.
                                keep_mask = derot_mask[j] > 0.5
                                auto_sel = outputs.get(f"identity_selection/{s_main}", None)
                                if torch.is_tensor(auto_sel):
                                    keep_mask = keep_mask & (auto_sel[j].unsqueeze(0) > 0.5)
                                ext_mask = inputs.get(("mask", 0, s_main), None)
                                if torch.is_tensor(ext_mask):
                                    keep_mask = keep_mask & (ext_mask[j] > 0.5)
                                if self.opt.distorted_mask:
                                    warp_valid = outputs.get(("distorted_mask", frame_id, s_main), None)
                                    if torch.is_tensor(warp_valid):
                                        keep_mask = keep_mask & (warp_valid[j] > 0.5)
                            keep_vis = keep_mask.float()

                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=[derot_map[j]],
                            scalar_titles=["P_derot"],
                            vmin=0.0,
                            vmax=25.0,
                            cmap="magma",
                            extra_images=[
                                derot_vis.float(),
                                keep_vis.float(),
                            ],
                            extra_titles=[
                                "W_derot_sigmoid" if is_derot_weight else "W_derot",
                                "final_loss_weight(fid)" if is_derot_weight else "final_loss_mask",
                            ],
                            key=f"{mode}/derot_panel_{frame_id}_{s_main}_{j}",
                            step=self.step,
                        )
                if j == 0 and frame_id != 0:
                    depth_t = outputs.get(("depth_cycle_target_depth", frame_id, s_main), None)
                    depth_t_from_s = outputs.get(("depth_cycle_target_from_source", frame_id, s_main), None)
                    depth_logdiff = outputs.get(("depth_cycle_logdiff_masked", frame_id, s_main), None)
                    depth_valid = outputs.get(("depth_cycle_valid", frame_id, s_main), None)
                    if (
                        torch.is_tensor(depth_t)
                        and torch.is_tensor(depth_t_from_s)
                        and torch.is_tensor(depth_logdiff)
                        and torch.is_tensor(depth_valid)
                    ):
                        eps = 1e-6
                        depth_t_j = depth_t[j].clamp_min(eps)
                        depth_t_from_s_j = depth_t_from_s[j].clamp_min(eps)
                        valid_j = depth_valid[j] > 0.5
                        if valid_j.ndim == 3:
                            valid_2d = valid_j[0]
                        else:
                            valid_2d = valid_j

                        abs_rel_j = torch.abs(depth_t_from_s_j - depth_t_j) / depth_t_j
                        logdiff_j = torch.abs(torch.log(depth_t_j) - torch.log(depth_t_from_s_j))
                        abs_rel_2d = abs_rel_j.squeeze(0)
                        logdiff_2d = logdiff_j.squeeze(0)

                        # Fixed ranges make this panel easier to compare across batches/epochs.
                        rel_vmax = 0.30
                        logdiff_vmax = 0.20
                        bad_rel_thresh = 0.15

                        target_rgb = inputs[("color", 0, s_main)][j].detach().clamp(0.0, 1.0)

                        bad_rel = (abs_rel_2d > bad_rel_thresh) & valid_2d
                        valid_count = float(valid_2d.float().sum().item())
                        bad_count = float(bad_rel.float().sum().item())

                        # Current-frame contribution that actually survives into final loss.
                        keep_fid_2d = None
                        keep_fid_raw = outputs.get(("automask_keep_fid", frame_id, s_main), None)
                        if torch.is_tensor(keep_fid_raw):
                            keep_fid_2d = keep_fid_raw[j].squeeze(0) > 0.5

                        final_weight_fid = outputs.get(("final_loss_weight_fid", frame_id, s_main), None)
                        if torch.is_tensor(final_weight_fid):
                            keep_fid_2d = final_weight_fid[j].squeeze(0) > 1e-6
                        else:
                            final_keep = outputs.get(f"final_loss_mask/{s_main}", None)
                            if torch.is_tensor(final_keep):
                                final_keep_2d = final_keep[j]
                                if final_keep_2d.ndim == 3:
                                    final_keep_2d = final_keep_2d[0]
                                final_keep_2d = final_keep_2d > 0.5
                                if keep_fid_2d is None:
                                    keep_fid_2d = final_keep_2d
                                else:
                                    keep_fid_2d = keep_fid_2d & final_keep_2d

                        if keep_fid_2d is None:
                            keep_fid_2d = torch.zeros_like(valid_2d, dtype=torch.bool)
                        keep_fid_2d = keep_fid_2d & valid_2d

                        keep_count = float(keep_fid_2d.float().sum().item())
                        bad_kept = bad_rel & keep_fid_2d
                        bad_kept_count = float(bad_kept.float().sum().item())
                        # Visualize only pixels that survive final masking.
                        abs_rel_heat = _scalar_to_colormap_masked(
                            abs_rel_2d,
                            keep_fid_2d.float(),
                            to_tensor=True,
                            cmap="magma",
                            vmin=0.0,
                            vmax=rel_vmax,
                        ).to(device=target_rgb.device, dtype=target_rgb.dtype)
                        overlay_alpha = 0.65
                        rel_overlay = target_rgb * (1.0 - overlay_alpha) + abs_rel_heat * overlay_alpha
                        keep_3c = keep_fid_2d.unsqueeze(0).expand_as(rel_overlay)
                        rel_overlay = torch.where(keep_3c, rel_overlay, target_rgb * 0.20)

                        # red = bad_rel that finally enters loss
                        bad_keep_overlay = target_rgb * 0.20
                        bad_keep_overlay[0] = torch.where(
                            bad_kept, torch.ones_like(bad_keep_overlay[0]), bad_keep_overlay[0]
                        )
                        bad_keep_overlay[1] = torch.where(
                            bad_kept, torch.zeros_like(bad_keep_overlay[1]), bad_keep_overlay[1]
                        )
                        bad_keep_overlay[2] = torch.where(
                            bad_kept, torch.zeros_like(bad_keep_overlay[2]), bad_keep_overlay[2]
                        )

                        keep_vis = keep_fid_2d.float().unsqueeze(0).repeat(3, 1, 1)

                        _log_row_images_to_wandb(
                            images=[
                                target_rgb,
                                rel_overlay,
                                _scalar_to_colormap_masked(
                                    abs_rel_2d,
                                    keep_fid_2d.float(),
                                    to_tensor=False,
                                    cmap="magma",
                                    vmin=0.0,
                                    vmax=rel_vmax,
                                ),
                                _scalar_to_colormap_masked(
                                    logdiff_2d,
                                    keep_fid_2d.float(),
                                    to_tensor=False,
                                    cmap="magma",
                                    vmin=0.0,
                                    vmax=logdiff_vmax,
                                ),
                                bad_keep_overlay,
                                keep_vis,
                            ],
                            titles=[
                                "target",
                                "rel_overlay",
                                "rel_err",
                                "log_err",
                                "bad_kept",
                                f"keep(fid={frame_id})",
                            ],
                            key=f"{mode}/depth_cycle_panel_{frame_id}_{s_main}_{j}",
                            step=self.step,
                        )
            rmask_pose_auto = outputs.get("r_mask_switch_identity_selection_pose/0", None)
            rmask_ext_auto = outputs.get("r_mask_switch_identity_selection_external/0", None)
            rmask_pose_keep = outputs.get("r_mask_switch_final_keep_pose/0", None)
            rmask_ext_keep = outputs.get("r_mask_switch_final_keep_external/0", None)
            rmask_sel_keep = outputs.get("r_mask_switch_final_keep_selected/0", None)
            if (
                torch.is_tensor(rmask_pose_keep)
                and torch.is_tensor(rmask_ext_keep)
                and torch.is_tensor(rmask_sel_keep)
            ):
                target_rgb = inputs[("color", 0, s_main)][j].detach().clamp(0.0, 1.0)
                pose_keep_ratio = _tensor_scalar(outputs.get("r_mask_switch_pose_keep/0", None), j)
                ext_keep_ratio = _tensor_scalar(outputs.get("r_mask_switch_external_keep/0", None), j)
                delta_keep = _tensor_scalar(outputs.get("r_mask_switch_delta_keep/0", None), j)
                route_gate = _tensor_scalar(outputs.get("r_mask_switch_gate/0", None), j)
                route_valid = _tensor_scalar(outputs.get("r_mask_switch_valid/0", None), j)

                pose_auto_img = None
                ext_auto_img = None
                if torch.is_tensor(rmask_pose_auto):
                    pose_auto_img = _to_2d(rmask_pose_auto, j, dtype=torch.float32)
                if torch.is_tensor(rmask_ext_auto):
                    ext_auto_img = _to_2d(rmask_ext_auto, j, dtype=torch.float32)
                pose_keep_img = _to_2d(rmask_pose_keep, j, dtype=torch.float32)
                ext_keep_img = _to_2d(rmask_ext_keep, j, dtype=torch.float32)
                sel_keep_img = _to_2d(rmask_sel_keep, j, dtype=torch.float32)

                route_vis = torch.full_like(target_rgb, 0.10)
                if route_valid is None or route_valid > 0.5:
                    if route_gate is not None and route_gate > 0.5:
                        route_vis[1] = 0.90
                        route_vis[0] = 0.10
                        route_vis[2] = 0.10
                        route_title = "route=external-R"
                    else:
                        route_vis[2] = 0.90
                        route_vis[0] = 0.10
                        route_vis[1] = 0.10
                        route_title = "route=pose"
                else:
                    route_vis[:] = 0.40
                    route_title = "route=invalid"
                if delta_keep is not None:
                    route_title += f"(dk={delta_keep:.3f})"

                images = [target_rgb]
                titles = ["target"]
                if pose_auto_img is not None:
                    pose_title = "pose_auto_keep"
                    if pose_keep_ratio is not None:
                        pose_title += f"(k={pose_keep_ratio:.3f})"
                    images.append(pose_auto_img)
                    titles.append(pose_title)
                if ext_auto_img is not None:
                    ext_title = "ext_auto_keep"
                    if ext_keep_ratio is not None:
                        ext_title += f"(k={ext_keep_ratio:.3f})"
                    images.append(ext_auto_img)
                    titles.append(ext_title)
                images.extend([pose_keep_img, ext_keep_img, sel_keep_img, route_vis])
                titles.extend(["pose_final_keep", "ext_final_keep", "selected_keep", route_title])
                _log_row_images_to_wandb(
                    images=images,
                    titles=titles,
                    key=f"{mode}/r_mask_switch_panel_{s_main}_{j}",
                    step=self.step,
                )
            if j == 0 and has_bidirectional_pair:
                auto_source = outputs.get(f"automask_source_fid/{s_main}", None)
                fid_images = []
                fid_titles = []
                if torch.is_tensor(auto_source):
                    fid_images.append(
                        _source_fid_to_rgb(
                            auto_source[j],
                            prev_fid=prev_fid,
                            next_fid=next_fid,
                            to_tensor=False,
                        )
                    )
                    fid_titles.append("auto_src_fid")

                fw_source = None
                if is_derot_sigmoid:
                    fw_source = outputs.get(f"final_loss_weight_source_fid/{s_main}", None)

                loss_prev = outputs.get(("reproj_loss_fid", prev_fid, s_main), None)
                loss_next = outputs.get(("reproj_loss_fid", next_fid, s_main), None)
                loss_source = outputs.get(f"reproj_loss_source_fid/{s_main}", None)
                auto_union = outputs.get(f"identity_selection/{s_main}", None)
                final_weight = outputs.get(f"final_loss_weight/{s_main}", None)
                if torch.is_tensor(loss_prev) and torch.is_tensor(loss_next):
                    lvmin, lvmax = _shared_range(loss_prev[j], loss_next[j], q_low=0.02, q_high=0.98, clip_max=1e4)
                    loss_min = torch.minimum(loss_prev[j], loss_next[j])

                    loss_scalar_images = [loss_prev[j], loss_next[j], loss_min]
                    loss_titles = [
                        f"loss(fid{prev_fid})",
                        f"loss(fid{next_fid})",
                        "loss(min)",
                    ]
                    if torch.is_tensor(auto_union):
                        auto_mask = auto_union[j]
                        if auto_mask.dim() == 2:
                            auto_mask = auto_mask.unsqueeze(0)
                        loss_after_auto = loss_min * auto_mask
                        loss_scalar_images.append(loss_after_auto)
                        loss_titles.append("loss(auto)")
                    if torch.is_tensor(final_weight):
                        fw = final_weight[j]
                        if fw.dim() == 2:
                            fw = fw.unsqueeze(0)
                        loss_after_final_weight = loss_min * fw
                        loss_scalar_images.append(loss_after_final_weight)
                        badscorelocal_weight_mean = outputs.get(f"badscorelocal_weight_mean/{s_main}", None)
                        badscorelocal_w_scalar = _tensor_scalar(badscorelocal_weight_mean, j)
                        if badscorelocal_w_scalar is not None:
                            loss_titles.append(f"loss(weight, W_mean={badscorelocal_w_scalar:.3f})")
                        else:
                            badscore_w_img = outputs.get(f"badscore_w_img/{s_main}", None)
                            badscore_w_scalar = _tensor_scalar(badscore_w_img, j)
                            if badscore_w_scalar is not None:
                                loss_titles.append(f"loss(weight, w_img={badscore_w_scalar:.3f})")
                            else:
                                loss_titles.append("loss(weight)")
                    _log_scalar_row_with_colorbar_to_wandb(
                        scalar_images=loss_scalar_images,
                        scalar_titles=loss_titles,
                        vmin=lvmin,
                        vmax=lvmax,
                        cmap="magma",
                        key=f"{mode}/loss_split_{s_main}_{j}",
                        step=self.step,
                    )

                if torch.is_tensor(loss_source):
                    fid_images.append(
                        _source_fid_to_rgb(
                            loss_source[j],
                            prev_fid=prev_fid,
                            next_fid=next_fid,
                            to_tensor=False,
                        )
                    )
                    fid_titles.append("loss_src_fid")
                if torch.is_tensor(fw_source):
                    fid_images.append(
                        _source_fid_to_rgb(
                            fw_source[j],
                            prev_fid=prev_fid,
                            next_fid=next_fid,
                            to_tensor=False,
                        )
                    )
                    fid_titles.append("weight_src_fid")
                if fid_images:
                    _log_row_images_to_wandb(
                        images=fid_images,
                        titles=fid_titles,
                        key=f"{mode}/fid_split_{s_main}_{j}",
                        step=self.step,
                    )
                badscore_R = outputs.get(f"badscore_R_map/{s_main}", None)
                badscore_O = outputs.get(f"badscore_O_map/{s_main}", None)
                badscore_K = outputs.get(f"badscore_K_map/{s_main}", None)
                badscore_r = outputs.get(f"badscore_r_map/{s_main}", None)
                badscore_o = outputs.get(f"badscore_o_map/{s_main}", None)
                badscore_b = outputs.get(f"badscore_b_map/{s_main}", None)
                badscore_valid = outputs.get(f"badscore_valid/{s_main}", None)
                badscore_w_img = outputs.get(f"badscore_w_img/{s_main}", None)
                badscore_w_img_map = outputs.get(f"badscore_weight_map/{s_main}", None)
                badscorelocal_fragile = outputs.get(f"badscorelocal_fragile_map/{s_main}", None)
                if badscorelocal_fragile is None:
                    badscorelocal_fragile = outputs.get(f"badscorelocal_s_map/{s_main}", None)
                badscorelocal_margin_tilde = outputs.get(f"badscorelocal_margin_tilde_map/{s_main}", None)
                badscorelocal_margin_gate = outputs.get(f"badscorelocal_margin_gate_map/{s_main}", None)
                badscorelocal_w_pix = outputs.get(f"badscorelocal_w_pix_map/{s_main}", None)
                if badscorelocal_w_pix is None:
                    badscorelocal_w_pix = outputs.get(f"badscorelocal_w_loc_map/{s_main}", None)
                badscorelocal_w = outputs.get(f"badscorelocal_weight_map/{s_main}", None)
                badscorelocal_loss_img_pix = outputs.get(
                    f"badscorelocal_loss_min_weighted_after_mask/{s_main}", None
                )
                badscorelocal_valid = outputs.get(f"badscorelocal_weight_valid/{s_main}", None)
                badscore_q_title = "q_map" if torch.is_tensor(badscore_b) else "b_map"
                if (
                    torch.is_tensor(badscore_R)
                    and torch.is_tensor(badscore_O)
                    and torch.is_tensor(badscore_K)
                    and torch.is_tensor(badscore_r)
                    and torch.is_tensor(badscore_o)
                ):
                    R_2d = _to_2d(badscore_R, j, dtype=torch.float32)
                    O_2d = _to_2d(badscore_O, j, dtype=torch.float32)
                    K_2d = _to_2d(badscore_K, j, dtype=torch.float32)
                    r_2d = _to_2d(badscore_r, j, dtype=torch.float32)
                    o_2d = _to_2d(badscore_o, j, dtype=torch.float32)
                    if torch.is_tensor(badscore_b):
                        b_2d = _to_2d(badscore_b, j, dtype=torch.float32)
                    else:
                        b_2d = torch.zeros_like(K_2d, dtype=torch.float32)
                    if torch.is_tensor(badscore_valid):
                        valid_2d = _to_2d(badscore_valid, j, dtype=torch.float32) > 0.5
                    else:
                        valid_2d = K_2d > 0.5

                    R_vals = R_2d[valid_2d & torch.isfinite(R_2d)]
                    O_vals = O_2d[valid_2d & torch.isfinite(O_2d)]
                    if R_vals.numel() > 0:
                        R_vmax = float(torch.quantile(R_vals, 0.98).item())
                        if (not math.isfinite(R_vmax)) or R_vmax <= 0.0:
                            R_vmax = float(R_vals.max().item())
                    else:
                        R_vmax = 1.0
                    if O_vals.numel() > 0:
                        O_vmax = float(torch.quantile(O_vals, 0.98).item())
                        if (not math.isfinite(O_vmax)) or O_vmax <= 0.0:
                            O_vmax = float(O_vals.max().item())
                    else:
                        O_vmax = 1.0
                    R_vmax = max(R_vmax, 1e-6)
                    O_vmax = max(O_vmax, 1e-6)
                    b_vmin, b_vmax = _single_range(b_2d, q_low=0.02, q_high=0.98, clip_max=1e4)
                    w_img_scalar = _tensor_scalar(badscore_w_img, j)

                    _log_scalar_row_with_colorbar_to_wandb(
                        scalar_images=[R_2d, b_2d],
                        scalar_titles=["R_map", badscore_q_title],
                        vmin=0.0,
                        vmax=R_vmax,
                        cmap="magma",
                        scalar_valid_masks=[valid_2d, valid_2d],
                        masked_color=(0.0, 0.0, 0.0, 1.0),
                        extra_images=[
                            K_2d.unsqueeze(0),
                        ],
                        extra_titles=[
                            "K_map",
                        ],
                        scalar_vmins=[
                            0.0,
                            b_vmin,
                        ],
                        scalar_vmaxs=[
                            R_vmax,
                            b_vmax,
                        ],
                        colorbar_groups=[0, 1],
                        scalar_cmaps=["magma", "magma"],
                        key=f"{mode}/badscore_R_map_{s_main}_{j}",
                        step=self.step,
                    )
                    _log_scalar_row_with_colorbar_to_wandb(
                        scalar_images=[O_2d, b_2d],
                        scalar_titles=["O_map", badscore_q_title],
                        vmin=0.0,
                        vmax=O_vmax,
                        cmap="magma",
                        scalar_valid_masks=[valid_2d, valid_2d],
                        masked_color=(0.0, 0.0, 0.0, 1.0),
                        extra_images=[
                            K_2d.unsqueeze(0),
                        ],
                        extra_titles=[
                            "K_map",
                        ],
                        scalar_vmins=[
                            0.0,
                            b_vmin,
                        ],
                        scalar_vmaxs=[
                            O_vmax,
                            b_vmax,
                        ],
                        colorbar_groups=[0, 1],
                        scalar_cmaps=["magma", "magma"],
                        key=f"{mode}/badscore_O_map_{s_main}_{j}",
                        step=self.step,
                    )
                    ro_vmin, ro_vmax = _shared_range(r_2d, o_2d, q_low=0.02, q_high=0.98, clip_max=1e4)
                    _log_scalar_row_with_colorbar_to_wandb(
                        scalar_images=[r_2d, o_2d, b_2d],
                        scalar_titles=["r_map", "o_map(low_obs)", badscore_q_title],
                        vmin=0.0,
                        vmax=1.0,
                        cmap="magma",
                        scalar_valid_masks=[valid_2d, valid_2d, valid_2d],
                        masked_color=(0.0, 0.0, 0.0, 1.0),
                        extra_images=[
                            K_2d.unsqueeze(0),
                        ],
                        extra_titles=[
                            "K_map",
                        ],
                        scalar_vmins=[
                            ro_vmin,
                            ro_vmin,
                            b_vmin,
                        ],
                        scalar_vmaxs=[
                            ro_vmax,
                            ro_vmax,
                            b_vmax,
                        ],
                        colorbar_groups=[0, 0, 1],
                        scalar_cmaps=["magma", "magma", "magma"],
                        key=f"{mode}/badscore_norm_maps_{s_main}_{j}",
                        step=self.step,
                    )
                if (
                    torch.is_tensor(badscorelocal_fragile)
                    and torch.is_tensor(badscorelocal_margin_tilde)
                    and torch.is_tensor(badscorelocal_margin_gate)
                    and torch.is_tensor(badscorelocal_w_pix)
                    and torch.is_tensor(badscorelocal_w)
                ):
                    fragile_2d = _to_2d(badscorelocal_fragile, j, dtype=torch.float32)
                    margin_tilde_2d = _to_2d(badscorelocal_margin_tilde, j, dtype=torch.float32)
                    margin_gate_2d = _to_2d(badscorelocal_margin_gate, j, dtype=torch.float32)
                    w_pix_2d = _to_2d(badscorelocal_w_pix, j, dtype=torch.float32)
                    w_2d = _to_2d(badscorelocal_w, j, dtype=torch.float32)
                    if torch.is_tensor(badscorelocal_valid):
                        pix_valid_2d = _to_2d(badscorelocal_valid, j, dtype=torch.float32) > 0.5
                    elif torch.is_tensor(badscore_valid):
                        pix_valid_2d = _to_2d(badscore_valid, j, dtype=torch.float32) > 0.5
                    else:
                        pix_valid_2d = torch.ones_like(fragile_2d, dtype=torch.bool)

                    w_img_scalar = _tensor_scalar(badscore_w_img, j)
                    if torch.is_tensor(badscore_w_img_map) and torch.is_tensor(badscorelocal_loss_img_pix):
                        w_img_2d = _to_2d(badscore_w_img_map, j, dtype=torch.float32)
                        loss_img_pix_2d = _to_2d(badscorelocal_loss_img_pix, j, dtype=torch.float32)
                        loss_vals = loss_img_pix_2d[pix_valid_2d & torch.isfinite(loss_img_pix_2d)]
                        if loss_vals.numel() > 0:
                            loss_vmin, loss_vmax = _single_range(loss_vals)
                        else:
                            loss_vmin, loss_vmax = _single_range(loss_img_pix_2d)
                        w_img_title = "w_img" if w_img_scalar is None else f"w_img={w_img_scalar:.3f}"
                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=[w_img_2d, w_pix_2d, w_2d, loss_img_pix_2d],
                            scalar_titles=[w_img_title, "w_pix", "w_img*w_pix", "loss*img*pix"],
                            vmin=0.0,
                            vmax=1.0,
                            cmap="magma",
                            scalar_valid_masks=[
                                pix_valid_2d, pix_valid_2d, pix_valid_2d, pix_valid_2d,
                            ],
                            masked_color=(0.0, 0.0, 0.0, 1.0),
                            scalar_vmins=[
                                0.0,
                                0.0,
                                0.0,
                                max(0.0, loss_vmin),
                            ],
                            scalar_vmaxs=[
                                1.0,
                                1.0,
                                1.0,
                                max(loss_vmax, 1e-6),
                            ],
                            colorbar_groups=[0, 0, 0, 1],
                            scalar_cmaps=["magma", "magma", "magma", "magma"],
                            key=f"{mode}/badscorelocal_maps_{s_main}_{j}",
                            step=self.step,
                        )
                    total_weight_title = "W"
                    if w_img_scalar is not None:
                        total_weight_title = f"W(w_img={w_img_scalar:.3f})"
                    margin_tilde_vmin, margin_tilde_vmax = _single_range(
                        margin_tilde_2d[pix_valid_2d] if bool(pix_valid_2d.any()) else margin_tilde_2d
                    )
                    _log_scalar_row_with_colorbar_to_wandb(
                        scalar_images=[margin_tilde_2d, margin_gate_2d, fragile_2d, w_pix_2d, w_2d],
                        scalar_titles=["M_tilde", "margin_gate", "fragile_score", "W_pix", total_weight_title],
                        vmin=0.0,
                        vmax=1.0,
                        cmap="magma",
                        scalar_valid_masks=[
                            pix_valid_2d, pix_valid_2d, pix_valid_2d, pix_valid_2d, pix_valid_2d,
                        ],
                        masked_color=(0.0, 0.0, 0.0, 1.0),
                        extra_images=[
                            K_2d.unsqueeze(0),
                        ] if torch.is_tensor(badscore_K) else [],
                        extra_titles=[
                            "K_map",
                        ] if torch.is_tensor(badscore_K) else [],
                        scalar_vmins=[
                            margin_tilde_vmin,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],
                        scalar_vmaxs=[
                            max(margin_tilde_vmax, margin_tilde_vmin + 1e-6),
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        colorbar_groups=[0, 1, 2, 2, 2],
                        scalar_cmaps=["magma", "magma", "magma", "magma", "magma"],
                        key=f"{mode}/badscorelocal_score_maps_{s_main}_{j}",
                        step=self.step,
                    )
                # Target-fused sense maps: stitch prev/next sensitivity onto target grid
                # according to per-pixel supervising source assignment.
                sens_loss_prev = outputs.get(("depth_sens_loss_delta", prev_fid, s_main), None)
                sens_loss_next = outputs.get(("depth_sens_loss_delta", next_fid, s_main), None)
                sens_pix_prev = outputs.get(("depth_sens_pixmag_delta", prev_fid, s_main), None)
                sens_pix_next = outputs.get(("depth_sens_pixmag_delta", next_fid, s_main), None)
                sens_valid_prev = outputs.get(("depth_sens_valid", prev_fid, s_main), None)
                sens_valid_next = outputs.get(("depth_sens_valid", next_fid, s_main), None)
                if (
                    torch.is_tensor(sens_loss_prev)
                    and torch.is_tensor(sens_loss_next)
                    and torch.is_tensor(sens_valid_prev)
                    and torch.is_tensor(sens_valid_next)
                ):

                    def _to_2d_float(x):
                        y = x[j]
                        if y.ndim == 3:
                            y = y[0]
                        return y.float()

                    def _to_2d_bool(x, thresh=0.5):
                        y = x[j]
                        if y.ndim == 3:
                            y = y[0]
                        return y > thresh

                    valid_prev_2d = _to_2d_bool(sens_valid_prev, thresh=0.5)
                    valid_next_2d = _to_2d_bool(sens_valid_next, thresh=0.5)

                    def _build_keep_for_fid(fid, valid_2d):
                        keep_fid_weight = outputs.get(("final_loss_weight_fid", fid, s_main), None)
                        if torch.is_tensor(keep_fid_weight):
                            keep_2d = _to_2d_bool(keep_fid_weight, thresh=1e-6)
                            return keep_2d & valid_2d

                        keep_fid_auto = outputs.get(("automask_keep_fid", fid, s_main), None)
                        if torch.is_tensor(keep_fid_auto):
                            keep_2d = _to_2d_bool(keep_fid_auto, thresh=0.5)
                            return keep_2d & valid_2d

                        src_map = outputs.get(f"final_loss_weight_source_fid/{s_main}", None)
                        if not torch.is_tensor(src_map):
                            src_map = outputs.get(f"automask_source_fid/{s_main}", None)
                        if not torch.is_tensor(src_map):
                            src_map = outputs.get(f"reproj_loss_source_fid/{s_main}", None)
                        if torch.is_tensor(src_map):
                            src_2d = _to_2d_float(src_map)
                            keep_2d = torch.isclose(
                                src_2d,
                                torch.full_like(src_2d, float(fid)),
                                atol=0.1,
                            )
                            if torch.is_tensor(auto_union):
                                keep_2d = keep_2d & _to_2d_bool(auto_union, thresh=0.5)
                            return keep_2d & valid_2d

                        return valid_2d.clone()

                    keep_prev_2d = _build_keep_for_fid(prev_fid, valid_prev_2d)
                    keep_next_2d = _build_keep_for_fid(next_fid, valid_next_2d)

                    overlap = keep_prev_2d & keep_next_2d
                    if bool(overlap.any()):
                        if torch.is_tensor(loss_prev) and torch.is_tensor(loss_next):
                            loss_prev_2d = _to_2d_float(loss_prev)
                            loss_next_2d = _to_2d_float(loss_next)
                            prev_wins = loss_prev_2d <= loss_next_2d
                            keep_prev_2d = (keep_prev_2d & ~overlap) | (overlap & prev_wins)
                            keep_next_2d = (keep_next_2d & ~overlap) | (overlap & ~prev_wins)
                        else:
                            keep_next_2d = keep_next_2d & ~overlap

                    fused_keep_2d = keep_prev_2d | keep_next_2d
                    fused_src_2d = torch.zeros_like(_to_2d_float(sens_loss_prev))
                    fused_src_2d[keep_prev_2d] = float(prev_fid)
                    fused_src_2d[keep_next_2d] = float(next_fid)
                    fused_src_rgb = _source_fid_to_rgb(
                        fused_src_2d,
                        prev_fid=prev_fid,
                        next_fid=next_fid,
                        to_tensor=False,
                    )

                    def _log_fused_scalar(prev_map, next_map, title, key_suffix):
                        prev_2d = _to_2d_float(prev_map)
                        next_2d = _to_2d_float(next_map)
                        fused_2d = torch.zeros_like(prev_2d)
                        fused_2d[keep_prev_2d] = prev_2d[keep_prev_2d]
                        fused_2d[keep_next_2d] = next_2d[keep_next_2d]
                        fused_2d = torch.where(
                            fused_keep_2d & torch.isfinite(fused_2d),
                            fused_2d,
                            torch.zeros_like(fused_2d),
                        )

                        vals = fused_2d[fused_keep_2d]
                        vals = vals[torch.isfinite(vals)]
                        if vals.numel() > 0:
                            vmax = float(torch.quantile(vals, 0.98).item())
                            if (not math.isfinite(vmax)) or vmax <= 0.0:
                                vmax = float(vals.max().item())
                        else:
                            vmax = 1.0
                        vmax = max(vmax, 1e-6)

                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=[fused_2d],
                            scalar_titles=[title],
                            vmin=0.0,
                            vmax=vmax,
                            cmap="magma",
                            extra_images=[
                                fused_keep_2d.float().unsqueeze(0),
                                fused_src_rgb,
                            ],
                            extra_titles=[
                                "fused_keep",
                                "fused_src_fid",
                            ],
                            key=f"{mode}/{key_suffix}_{s_main}_{j}",
                            step=self.step,
                        )

                    _log_fused_scalar(
                        sens_loss_prev,
                        sens_loss_next,
                        f"d_loss(fused:{prev_fid}/{next_fid})",
                        "depth_sens_target_fused_delta",
                    )
                    if torch.is_tensor(sens_pix_prev) and torch.is_tensor(sens_pix_next):
                        _log_fused_scalar(
                            sens_pix_prev,
                            sens_pix_next,
                            f"pix_delta(fused:{prev_fid}/{next_fid})",
                            "depth_pixshift_target_fused_delta",
                        )
            if j == 0:
                ds_loss_after_mask = outputs.get(f"depth_sens_loss_min_after_mask/{s_main}", None)
                ds_loss_weighted_after_mask = outputs.get(
                    f"depth_sens_loss_min_weighted_after_mask/{s_main}", None
                )
                ds_weight_map = outputs.get(f"depth_sens_weight_selected/{s_main}", None)
                ds_weight_after_mask = outputs.get(f"final_loss_weight/{s_main}", None)
                badscore_loss_after_mask = outputs.get(f"badscore_loss_min_after_mask/{s_main}", None)
                badscore_loss_weighted_after_mask = outputs.get(
                    f"badscore_loss_min_weighted_after_mask/{s_main}", None
                )
                badscore_weight_map = outputs.get(f"badscore_weight_map/{s_main}", None)
                badscore_w_img = outputs.get(f"badscore_w_img/{s_main}", None)
                badscorelocal_loss_after_mask = outputs.get(f"badscorelocal_loss_min_after_mask/{s_main}", None)
                badscorelocal_loss_weighted_after_mask = outputs.get(
                    f"badscorelocal_loss_min_weighted_after_mask/{s_main}", None
                )
                badscorelocal_weight_map = outputs.get(f"badscorelocal_weight_map/{s_main}", None)
                badscorelocal_weight_mean = outputs.get(f"badscorelocal_weight_mean/{s_main}", None)
                badscorelocal_weight_valid = outputs.get(f"badscorelocal_weight_valid/{s_main}", None)
                if torch.is_tensor(ds_loss_after_mask) and torch.is_tensor(ds_loss_weighted_after_mask):
                    ds_vmin, ds_vmax = _shared_range(
                        ds_loss_after_mask[j],
                        ds_loss_weighted_after_mask[j],
                        q_low=0.02,
                        q_high=0.98,
                        clip_max=1e4,
                    )
                    loss_scalar_images = [ds_loss_after_mask[j], ds_loss_weighted_after_mask[j]]
                    loss_scalar_titles = ["loss(mask)", "loss(mask*w)"]
                    _log_scalar_row_with_colorbar_to_wandb(
                        scalar_images=loss_scalar_images,
                        scalar_titles=loss_scalar_titles,
                        vmin=ds_vmin,
                        vmax=ds_vmax,
                        cmap="magma",
                        key=f"{mode}/depth_sens_weighted_loss_{s_main}_{j}",
                        step=self.step,
                    )
                    if torch.is_tensor(ds_weight_after_mask):
                        weight_map_j = ds_weight_after_mask[j]
                        if weight_map_j.dim() == 2:
                            weight_map_j = weight_map_j.unsqueeze(0)
                        denom_w = weight_map_j.sum().clamp_min(1.0)
                        weighted_mean = (ds_loss_weighted_after_mask[j].sum() / denom_w).detach()
                        contrib_map = (ds_loss_weighted_after_mask[j] / denom_w).detach()
                        contrib_valid = (weight_map_j > 0.0) & torch.isfinite(contrib_map)
                        contrib_map = torch.where(contrib_valid, contrib_map, torch.zeros_like(contrib_map))

                        valid_flat = contrib_map[contrib_valid]
                        if valid_flat.numel() > 1:
                            c_vmin = float(torch.quantile(valid_flat, 0.02).item())
                            c_vmax = float(torch.quantile(valid_flat, 0.98).item())
                            if (not math.isfinite(c_vmin)) or (not math.isfinite(c_vmax)) or c_vmax <= c_vmin:
                                c_vmin = float(valid_flat.min().item())
                                c_vmax = float(valid_flat.max().item())
                        elif valid_flat.numel() == 1:
                            c_vmin = 0.0
                            c_vmax = max(float(valid_flat.item()), 1e-6)
                        else:
                            c_vmin = 0.0
                            c_vmax = 1.0
                        if c_vmax <= c_vmin:
                            c_vmax = c_vmin + 1e-6

                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=[contrib_map],
                            scalar_titles=[
                                f"contrib((loss*w)/sum(w), mean={float(weighted_mean.item()):.6f})"
                            ],
                            vmin=c_vmin,
                            vmax=c_vmax,
                            cmap="magma",
                            key=f"{mode}/depth_sens_contrib_{s_main}_{j}",
                            step=self.step,
                        )
                if torch.is_tensor(ds_weight_map) or torch.is_tensor(ds_weight_after_mask):
                    derot_prev = outputs.get(("derot_parallax", prev_fid, s_main), None) if has_bidirectional_pair else None
                    derot_next = outputs.get(("derot_parallax", next_fid, s_main), None) if has_bidirectional_pair else None
                    fw_source = outputs.get(f"final_loss_weight_source_fid/{s_main}", None)
                    if (
                        is_derot_sigmoid
                        and has_bidirectional_pair
                        and torch.is_tensor(ds_weight_after_mask)
                        and torch.is_tensor(derot_prev)
                        and torch.is_tensor(derot_next)
                    ):
                        derot_prev_2d = derot_prev[j]
                        derot_next_2d = derot_next[j]
                        if derot_prev_2d.ndim == 3:
                            derot_prev_2d = derot_prev_2d[0]
                        if derot_next_2d.ndim == 3:
                            derot_next_2d = derot_next_2d[0]

                        fused_derot = torch.zeros_like(derot_prev_2d)
                        fused_assigned = torch.zeros_like(derot_prev_2d, dtype=torch.bool)

                        src_2d = None
                        if torch.is_tensor(fw_source):
                            src_2d = fw_source[j]
                            if src_2d.ndim == 3:
                                src_2d = src_2d[0]
                            src_2d = src_2d.float()

                        if src_2d is not None:
                            prev_mask = torch.isclose(
                                src_2d,
                                torch.full_like(src_2d, float(prev_fid)),
                                atol=0.1,
                            )
                            next_mask = torch.isclose(
                                src_2d,
                                torch.full_like(src_2d, float(next_fid)),
                                atol=0.1,
                            )
                            fused_derot[prev_mask] = derot_prev_2d[prev_mask]
                            fused_derot[next_mask] = derot_next_2d[next_mask]
                            fused_assigned = prev_mask | next_mask
                        else:
                            prev_fw = outputs.get(("final_loss_weight_fid", prev_fid, s_main), None)
                            next_fw = outputs.get(("final_loss_weight_fid", next_fid, s_main), None)
                            if torch.is_tensor(prev_fw):
                                prev_fw_2d = prev_fw[j]
                                if prev_fw_2d.ndim == 3:
                                    prev_fw_2d = prev_fw_2d[0]
                                prev_mask = prev_fw_2d > 1e-6
                                fused_derot[prev_mask] = derot_prev_2d[prev_mask]
                                fused_assigned = fused_assigned | prev_mask
                            if torch.is_tensor(next_fw):
                                next_fw_2d = next_fw[j]
                                if next_fw_2d.ndim == 3:
                                    next_fw_2d = next_fw_2d[0]
                                next_mask = next_fw_2d > 1e-6
                                fused_derot[next_mask] = derot_next_2d[next_mask]
                                fused_assigned = fused_assigned | next_mask

                        if not bool(fused_assigned.any()):
                            weight_keep_2d = ds_weight_after_mask[j]
                            if weight_keep_2d.ndim == 3:
                                weight_keep_2d = weight_keep_2d[0]
                            fused_derot = torch.where(
                                weight_keep_2d > 1e-6,
                                torch.maximum(derot_prev_2d, derot_next_2d),
                                torch.zeros_like(derot_prev_2d),
                            )

                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=[
                                derot_prev_2d,
                                derot_next_2d,
                                fused_derot,
                            ],
                            scalar_titles=[
                                f"P_derot(fid={prev_fid})",
                                f"P_derot(fid={next_fid})",
                                "P_derot(fused/pre-sigmoid)",
                            ],
                            vmin=0.0,
                            vmax=25.0,
                            cmap="magma",
                            extra_images=[ds_weight_after_mask[j]],
                            extra_titles=["sens_w(mask)"],
                            key=f"{mode}/depth_sens_weight_map_{s_main}_{j}",
                            step=self.step,
                        )
                    else:
                        weight_scalar_images = []
                        weight_titles = []
                        if torch.is_tensor(ds_weight_map):
                            weight_scalar_images.append(ds_weight_map[j])
                            weight_titles.append("sens_w(raw)")
                        if torch.is_tensor(ds_weight_after_mask):
                            weight_scalar_images.append(ds_weight_after_mask[j])
                            weight_titles.append("sens_w(mask)")
                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=weight_scalar_images,
                            scalar_titles=weight_titles,
                            vmin=0.0,
                            vmax=1.0,
                            cmap="magma",
                            key=f"{mode}/depth_sens_weight_map_{s_main}_{j}",
                            step=self.step,
                        )
                if torch.is_tensor(badscore_loss_after_mask) and torch.is_tensor(badscore_loss_weighted_after_mask):
                    bs_vmin, bs_vmax = _shared_range(
                        badscore_loss_after_mask[j],
                        badscore_loss_weighted_after_mask[j],
                        q_low=0.02,
                        q_high=0.98,
                        clip_max=1e4,
                    )
                    badscore_w_scalar = _tensor_scalar(badscore_w_img, j)
                    weight_title = "loss(mask*w_img)"
                    weight_map_title = "w_img"
                    if badscore_w_scalar is not None:
                        weight_title = f"loss(mask*w_img={badscore_w_scalar:.3f})"
                        weight_map_title = f"w_img={badscore_w_scalar:.3f}"
                    _log_scalar_row_with_colorbar_to_wandb(
                        scalar_images=[
                            badscore_loss_after_mask[j],
                            badscore_loss_weighted_after_mask[j],
                        ],
                        scalar_titles=[
                            "loss(mask)",
                            weight_title,
                        ],
                        vmin=bs_vmin,
                        vmax=bs_vmax,
                        cmap="magma",
                        key=f"{mode}/badscore_weighted_loss_{s_main}_{j}",
                        step=self.step,
                    )
                    if torch.is_tensor(badscore_weight_map):
                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=[badscore_weight_map[j]],
                            scalar_titles=[weight_map_title],
                            vmin=0.0,
                            vmax=1.0,
                            cmap="magma",
                            key=f"{mode}/badscore_weight_map_{s_main}_{j}",
                            step=self.step,
                        )
                if torch.is_tensor(badscorelocal_loss_after_mask) and torch.is_tensor(badscorelocal_loss_weighted_after_mask):
                    bsp_vmin, bsp_vmax = _shared_range(
                        badscorelocal_loss_after_mask[j],
                        badscorelocal_loss_weighted_after_mask[j],
                        q_low=0.02,
                        q_high=0.98,
                        clip_max=1e4,
                    )
                    badscorelocal_w_scalar = _tensor_scalar(badscorelocal_weight_mean, j)
                    pix_weight_title = "loss(mask*w_img*w_pix)"
                    pix_weight_map_title = "w_img*w_pix"
                    if badscorelocal_w_scalar is not None:
                        pix_weight_title = f"loss(mask*w_img*w_pix, mean={badscorelocal_w_scalar:.3f})"
                        pix_weight_map_title = f"w_img*w_pix(mean={badscorelocal_w_scalar:.3f})"
                    _log_scalar_row_with_colorbar_to_wandb(
                        scalar_images=[
                            badscorelocal_loss_after_mask[j],
                            badscorelocal_loss_weighted_after_mask[j],
                        ],
                        scalar_titles=[
                            "loss(mask)",
                            pix_weight_title,
                        ],
                        vmin=bsp_vmin,
                        vmax=bsp_vmax,
                        cmap="magma",
                        key=f"{mode}/badscorelocal_weighted_loss_{s_main}_{j}",
                        step=self.step,
                    )
                    if torch.is_tensor(badscorelocal_weight_map):
                        pix_weight_valid_2d = None
                        if torch.is_tensor(badscorelocal_weight_valid):
                            pix_weight_valid_2d = _to_2d(badscorelocal_weight_valid, j, dtype=torch.float32) > 0.5
                        _log_scalar_row_with_colorbar_to_wandb(
                            scalar_images=[badscorelocal_weight_map[j]],
                            scalar_titles=[pix_weight_map_title],
                            vmin=0.0,
                            vmax=1.0,
                            cmap="magma",
                            scalar_valid_masks=[pix_weight_valid_2d] if pix_weight_valid_2d is not None else None,
                            masked_color=(0.0, 0.0, 0.0, 1.0),
                            key=f"{mode}/badscorelocal_weight_map_{s_main}_{j}",
                            step=self.step,
                        )
            # 其余单图统一记录为 disp_*，便于跨方法对齐看板。
            # SPIDepth: 先将 depth 显式转换为 disp(=1/depth) 后再可视化。
            if is_spidepth:
                depth_vis = outputs.get(("depth", 0, s_main), outputs.get(("disp", s_main), None))
                if torch.is_tensor(depth_vis):
                    min_depth = float(getattr(self.opt, "min_depth", 0.1))
                    max_depth = float(getattr(self.opt, "max_depth", 100.0))
                    min_depth = max(min_depth, 1e-6)
                    max_depth = max(max_depth, min_depth + 1e-6)
                    min_disp = 1.0 / max_depth
                    max_disp = 1.0 / min_depth
                    disp_vis = torch.reciprocal(depth_vis.clamp_min(min_depth))
                    disp_vis = disp_vis.clamp(min=min_disp, max=max_disp)
                    disp_color = _disp_to_colormap(disp_vis[j], to_tensor=False, cmap="magma")
                    wandb.log({f"{mode}/disp_{s_main}_{j}": wandb.Image(disp_color)}, step=self.step)
            else:
                disp_color = _disp_to_colormap(outputs[("disp", s_main)][j], to_tensor=False, cmap="magma")
                wandb.log({f"{mode}/disp_{s_main}_{j}": wandb.Image(disp_color)}, step=self.step)

            if not self.opt.disable_automasking:
                wandb.log({
                    f"{mode}/automask_{s_main}_{j}": wandb.Image(
                        outputs[f"identity_selection/{s_main}"][j][None, ...]
                    )
                }, step=self.step)

    else:
        val = losses["loss"]
        vloss = float(val.detach().cpu().item() if isinstance(val, torch.Tensor) else val)
        summary = [f"[Log][{mode}] step={self.step} | loss={vloss:.6f}"]
        depth_pairs = []
        for key, value in losses.items():
            if key == "loss":
                continue
            if isinstance(value, torch.Tensor):
                val_float = float(value.detach().cpu().item())
            else:
                val_float = float(value)
            if key.startswith("de/") or key.startswith("metrics/uavid2020/"):
                depth_pairs.append(f"{key}={val_float:.4f}")
            elif not self.collect_debug_metrics and (
                key.startswith("metrics/hrmask/")
                or key.startswith("metrics/derot/")
                or key.startswith("metrics/automask/")
                or key.startswith("metrics/final_loss_mask/")
            ):
                summary.append(f"{key[len('metrics/'):]}={val_float:.4f}")
            # 开启 debug 聚合时，metrics/* 在 _log_epoch_metrics 中统一输出 epoch 聚合值。
        if depth_pairs:
            summary.append(" | ".join(depth_pairs))
        monitor = self._scale_align_monitor.get(mode)
        if monitor and monitor["attempts"] > 0:
            ratio = monitor["success"] / float(max(1, monitor["attempts"]))
            summary.append(f"scale_align_success_ratio={ratio:.3f}")
        print(" ".join(summary))
        _maybe_log_automask_margin_panels(log_to_wandb=False)


def save_opts(self):
    models_dir = os.path.join(self.log_path, "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    to_save = self.opt.__dict__.copy()

    with open(os.path.join(models_dir, "opt.json"), "w") as f:
        json.dump(to_save, f, indent=2)

    if self.using_wandb:
        EXCLUDE = {
            "model_name",
            "log_dir",
            "log_path",
        }
        safe_cfg = {k: v for k, v in to_save.items() if k not in EXCLUDE}
        wandb.config.update(safe_cfg, allow_val_change=True)

        try:
            wandb.run.name = self.opt.model_name
            wandb.run.save()
        except Exception:
            pass


def save_model(self):
    """Save model weights to disk."""
    save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for model_name, model in self.models.items():
        save_path = os.path.join(save_folder, "{}.pth".format(model_name))
        to_save = model.state_dict()

        if self.opt.methods == "MRFEDepth":
            if model_name == "DepthEncoder":
                to_save["height"] = self.opt.height
                to_save["width"] = self.opt.width
        else:
            if model_name == "encoder":
                to_save["height"] = self.opt.height
                to_save["width"] = self.opt.width

        torch.save(to_save, save_path)

    save_path = os.path.join(save_folder, "{}.pth".format("adam"))
    torch.save(self.model_optimizer.state_dict(), save_path)


def load_model(self):
    """Load model(s) from disk."""
    self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

    assert os.path.isdir(self.opt.load_weights_folder), (
        "Cannot find folder {}".format(self.opt.load_weights_folder)
    )
    print("loading model from folder {}".format(self.opt.load_weights_folder))

    for n in self.opt.models_to_load:
        if n not in self.models:
            print(f"[load_model] Skip loading '{n}' (model not initialized for {self.opt.methods})")
            continue
        path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
        if not os.path.isfile(path):
            print(f"[load_model] Skip loading '{n}' (missing file: {path})")
            continue
        print("Loading {} weights...".format(n))
        model_dict = self.models[n].state_dict()
        pretrained_dict = torch.load(path, map_location=self.device)
        matched = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(matched)
        self.models[n].load_state_dict(model_dict)
        # 统一 checkpoint 加载成功提示，便于快速确认每个模块都已正确读取权重。
        print(f"[load_model] Loaded '{n}' ({len(matched)} tensors) from {path}")

    optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
    if os.path.isfile(optimizer_load_path):
        print("Loading Adam weights")
        optimizer_dict = torch.load(optimizer_load_path)
        self.model_optimizer.load_state_dict(optimizer_dict)
        print(f"[load_model] Loaded Adam optimizer state from {optimizer_load_path}")
    else:
        print("Cannot find Adam weights so Adam is randomly initialized")

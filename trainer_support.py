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
    t_over_z = []
    for fid in frame_ids:
        key = ("cam_T_cam", 0, fid)
        T = outputs.get(key, None)
        if T is None or not torch.is_tensor(T):
            continue
        if T.ndim != 3 or T.shape[-2:] != (4, 4):
            continue
        t = T[:, :3, 3]
        t_norms.append(torch.linalg.norm(t, dim=1))

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
    enable_automask_margin_viz = bool(getattr(self.opt, "enable_automask_margin_viz", False))
    save_automask_margin_viz_local = bool(getattr(self.opt, "save_automask_margin_viz_local", False))
    automask_margin_viz_samples = max(1, int(getattr(self.opt, "automask_margin_viz_samples", 1)))

    def _to_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().mean().cpu().item())
        return float(value)

    def _get_color(frame_id, scale):
        image = inputs.get(("color", frame_id, scale), None)
        if torch.is_tensor(image):
            return image
        return inputs.get(("color", frame_id, 0), None)

    def _maybe_log_automask_margin_panels(log_to_wandb: bool):
        if self.opt.disable_automasking:
            return
        if not (enable_automask_margin_viz or save_automask_margin_viz_local):
            return

        identity_comp = outputs.get(f"automask_identity_comp/{s_main}", None)
        reproj_comp = outputs.get(f"automask_reproj_comp/{s_main}", None)
        margin_map = outputs.get(f"automask_margin/{s_main}", None)
        keep_map = outputs.get(f"identity_selection/{s_main}", None)
        target = _get_color(0, s_main)
        if not (
            torch.is_tensor(identity_comp)
            and torch.is_tensor(reproj_comp)
            and torch.is_tensor(margin_map)
            and torch.is_tensor(keep_map)
            and torch.is_tensor(target)
        ):
            return

        save_dir = None
        if save_automask_margin_viz_local:
            save_dir = os.path.join(self.log_path, "automask_margin_viz", mode)
            os.makedirs(save_dir, exist_ok=True)

        num_vis = min(automask_margin_viz_samples, identity_comp.shape[0])
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
            keep_ratio = float(keep_valid.float().sum().item() / max(1.0, float(finite_2d.float().sum().item())))
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
                scalar_images=[id_2d, reproj_2d, margin_2d],
                scalar_titles=[
                    f"id_min(mean={float(id_2d[finite_2d].mean().item()):.4f})",
                    f"reproj_min(mean={float(reproj_2d[finite_2d].mean().item()):.4f})",
                    f"margin(mean={margin_mean:.4f})",
                ],
                vmin=loss_lo,
                vmax=loss_hi,
                cmap="magma",
                extra_images=[keep_2d.float().unsqueeze(0), target[j].detach().clamp(0.0, 1.0)],
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

    loss_value = _to_float(losses["loss"])
    depth_pairs = []
    log_dict = {f"{mode}/loss": loss_value}

    for key, value in losses.items():
        if key == "loss":
            continue
        if key.startswith("de/") or key.startswith("metrics/uavid2020/"):
            val_float = _to_float(value)
            log_dict[f"{mode}/{key}"] = val_float
            depth_pairs.append((key, val_float))
        elif not self.collect_debug_metrics and key.startswith("metrics/automask/"):
            log_dict[f"{mode}/{key[len('metrics/'):]}"] = _to_float(value)

    if getattr(self, "using_wandb", False):
        wandb.log(log_dict, step=self.step)
        _maybe_log_automask_margin_panels(log_to_wandb=True)

        if mode == "val":
            print(f"[Validation] step={self.step} | loss={loss_value:.6f}")
            try:
                best = wandb.run.summary.get("val/loss.min", float("inf"))
                if loss_value < best:
                    wandb.run.summary["val/loss.min"] = loss_value
            except Exception:
                pass

        target = _get_color(0, s_main)
        if torch.is_tensor(target):
            batch_n = min(4, target.shape[0])
        else:
            batch_n = 0

        for j in range(batch_n):
            rows = [target[j].detach().clamp(0.0, 1.0)]
            titles = ["target"]
            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                source = _get_color(frame_id, s_main)
                pred = outputs.get(("color", frame_id, s_main), None)
                if torch.is_tensor(source):
                    rows.append(source[j].detach().clamp(0.0, 1.0))
                    titles.append(f"source[{frame_id}]")
                if torch.is_tensor(pred):
                    rows.append(pred[j].detach().clamp(0.0, 1.0))
                    titles.append(f"warp[{frame_id}]")
                if not self.opt.disable_automasking:
                    identity = outputs.get(("color_identity", frame_id, s_main), None)
                    if torch.is_tensor(identity):
                        rows.append(identity[j].detach().clamp(0.0, 1.0))
                        titles.append(f"identity[{frame_id}]")

            if rows:
                _log_row_images_to_wandb(
                    images=rows,
                    titles=titles,
                    key=f"{mode}/reprojection_{s_main}_{j}",
                    step=self.step,
                )

            disp = outputs.get(("disp", s_main), None)
            if torch.is_tensor(disp):
                disp_color = _disp_to_colormap(disp[j], to_tensor=False, cmap="magma")
                wandb.log({f"{mode}/disp_{s_main}_{j}": wandb.Image(disp_color)}, step=self.step)

            automask = outputs.get(f"identity_selection/{s_main}", None)
            if not self.opt.disable_automasking and torch.is_tensor(automask):
                wandb.log({f"{mode}/automask_{s_main}_{j}": wandb.Image(automask[j].detach().float().cpu())}, step=self.step)
    else:
        summary = [f"[Log][{mode}] step={self.step} | loss={loss_value:.6f}"]
        if depth_pairs:
            summary.append(" | ".join(f"{name}={val:.4f}" for name, val in depth_pairs))
        if not self.collect_debug_metrics:
            automask_pairs = [
                (key, _to_float(value))
                for key, value in losses.items()
                if key.startswith("metrics/automask/")
            ]
            if automask_pairs:
                summary.append(" | ".join(f"{key[len('metrics/'):]}={val:.4f}" for key, val in automask_pairs))
        print(" ".join(summary))
        _maybe_log_automask_margin_panels(log_to_wandb=False)

    if depth_pairs:
        summary = ", ".join(f"{name}={val:.4f}" for name, val in depth_pairs)
        print(f"[DepthMetrics][{mode}] step={self.step} | {summary}")


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
        optimizer_dict = torch.load(optimizer_load_path, map_location=self.device)
        self.model_optimizer.load_state_dict(optimizer_dict)
        print(f"[load_model] Loaded Adam optimizer state from {optimizer_load_path}")
    else:
        print("Cannot find Adam weights so Adam is randomly initialized")

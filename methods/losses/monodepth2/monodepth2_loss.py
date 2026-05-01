# monodepth2_loss.py

import numbers

import torch
import torch.nn.functional as F

from layers import SSIM, get_smooth_loss


class Monodepth2Loss:
    """Photometric self-supervised loss used by Monodepth2, MonoViT and RFlow-TInj."""

    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.ssim = SSIM().to(self.device)
        self.global_step = 0
        self.global_epoch = 0
        self.total_steps = None
        self.steps_per_epoch = None

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            return l1_loss

        ssim_loss = self.ssim(pred, target).mean(1, True)
        return 0.85 * ssim_loss + 0.15 * l1_loss

    @staticmethod
    def _masked_mean(loss_map, keep_mask):
        if keep_mask is None:
            return loss_map.mean()
        if loss_map.dim() == 3:
            loss_map = loss_map.unsqueeze(1)
        if keep_mask.dim() == 3:
            keep_mask = keep_mask.unsqueeze(1)
        keep_f = keep_mask.to(dtype=loss_map.dtype)
        denom = keep_f.sum().clamp_min(1.0)
        return (loss_map * keep_f).sum() / denom

    @staticmethod
    def _combine_keep_mask(identity_selection):
        if identity_selection is None:
            return None
        keep = identity_selection > 0.5
        if keep.dim() == 3:
            keep = keep.unsqueeze(1)
        return keep

    @staticmethod
    def _compute_min_reproj(reprojection_losses, avg_reprojection):
        if avg_reprojection:
            return reprojection_losses.mean(1, keepdim=True)
        if reprojection_losses.shape[1] == 1:
            return reprojection_losses
        return torch.min(reprojection_losses, dim=1, keepdim=True)[0]

    @staticmethod
    def _reduce_automask_compare_loss(loss_tensor, avg_reprojection):
        if loss_tensor is None:
            return None
        if loss_tensor.dim() == 3:
            loss_tensor = loss_tensor.unsqueeze(1)
        if avg_reprojection:
            return loss_tensor.mean(1, keepdim=True)
        if loss_tensor.shape[1] == 1:
            return loss_tensor
        return torch.min(loss_tensor, dim=1, keepdim=True)[0]

    def _collect_debug_metrics(
        self,
        scale,
        reprojection_losses,
        identity_loss,
        combined,
        to_optimise,
        frame_offsets,
    ):
        if not getattr(self.opt, "enable_debug_metrics", False):
            return {}
        if getattr(self.opt, "avg_reprojection", False):
            return {}

        frame_ids = self.opt.frame_ids[1:]
        if not frame_ids:
            return {}

        metrics = {}
        metric_root = "metrics/photometric"
        frame_to_idx = {fid: idx for idx, fid in enumerate(frame_ids)}

        def _select_frame(predicate, fallback=None):
            candidates = [
                fid for fid in frame_ids
                if isinstance(fid, numbers.Integral) and predicate(fid)
            ]
            if not candidates:
                return fallback
            selected = min(candidates, key=lambda v: abs(int(v)))
            return frame_to_idx.get(selected, fallback)

        prev_idx = _select_frame(lambda fid: fid < 0)
        next_idx = _select_frame(lambda fid: fid > 0)
        if prev_idx is None and next_idx is None:
            return metrics

        prev_fid = frame_ids[prev_idx] if prev_idx is not None else None
        next_fid = frame_ids[next_idx] if next_idx is not None else None
        prev_offset = None if prev_fid is None else frame_offsets.get(prev_fid, prev_fid)
        next_offset = None if next_fid is None else frame_offsets.get(next_fid, next_fid)

        invalid_penalty = getattr(self.opt, "invalid_photometric_penalty", 1e4)
        offset = identity_loss.shape[1] if identity_loss is not None else 0

        with torch.no_grad():
            combined_det = combined.detach()
            reproj_det = reprojection_losses.detach()
            to_opt_det = to_optimise.detach()

            min_vals, min_indices = torch.min(combined_det, dim=1)
            if identity_loss is not None:
                min_is_reproj = min_indices >= offset
                choice = min_indices - offset
            else:
                min_is_reproj = torch.ones_like(min_indices, dtype=torch.bool)
                choice = min_indices

            supervised_mask = min_is_reproj & (min_vals < invalid_penalty)
            total_pixels = float(min_vals.numel())
            supervised_pixels = supervised_mask.float().sum().item()
            total_den = total_pixels if total_pixels > 0 else 1.0
            supervised_den = supervised_pixels if supervised_pixels > 0 else 1.0

            def _ratio(mask, denom):
                if denom <= 0:
                    return float("nan")
                return mask.float().sum().item() / denom

            def _quantiles(values, mask, prefix):
                if mask is None:
                    metrics[prefix + "p50"] = float("nan")
                    metrics[prefix + "p90"] = float("nan")
                    return
                mask_flat = mask.reshape(-1)
                if mask_flat.float().sum().item() < 1:
                    metrics[prefix + "p50"] = float("nan")
                    metrics[prefix + "p90"] = float("nan")
                    return
                vals = values.reshape(-1)[mask_flat]
                metrics[prefix + "p50"] = float(torch.quantile(vals, 0.5).item())
                metrics[prefix + "p90"] = float(torch.quantile(vals, 0.9).item())

            if prev_idx is not None:
                valid_prev = supervised_mask & (choice == prev_idx)
                metrics[f"{metric_root}/scale{scale}_prev_frame_id"] = float(prev_offset)
                metrics[f"{metric_root}/scale{scale}_valid_ratio_prev"] = _ratio(valid_prev, total_den)
                metrics[f"{metric_root}/scale{scale}_min_from_prev_ratio"] = _ratio(valid_prev, supervised_den)
                _quantiles(reproj_det[:, prev_idx], valid_prev, f"{metric_root}/scale{scale}_prev_")
            else:
                metrics[f"{metric_root}/scale{scale}_prev_frame_id"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_valid_ratio_prev"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_min_from_prev_ratio"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_prev_p50"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_prev_p90"] = float("nan")

            if next_idx is not None:
                valid_next = supervised_mask & (choice == next_idx)
                metrics[f"{metric_root}/scale{scale}_next_frame_id"] = float(next_offset)
                metrics[f"{metric_root}/scale{scale}_valid_ratio_next"] = _ratio(valid_next, total_den)
                metrics[f"{metric_root}/scale{scale}_min_from_next_ratio"] = _ratio(valid_next, supervised_den)
                _quantiles(reproj_det[:, next_idx], valid_next, f"{metric_root}/scale{scale}_next_")
            else:
                metrics[f"{metric_root}/scale{scale}_next_frame_id"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_valid_ratio_next"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_min_from_next_ratio"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_next_p50"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_next_p90"] = float("nan")

            if prev_idx is not None and next_idx is not None:
                prev_valid_raw = reproj_det[:, prev_idx] < invalid_penalty
                next_valid_raw = reproj_det[:, next_idx] < invalid_penalty
                valid_both = supervised_mask & prev_valid_raw & next_valid_raw
                metrics[f"{metric_root}/scale{scale}_valid_ratio_both"] = _ratio(valid_both, total_den)
            else:
                metrics[f"{metric_root}/scale{scale}_valid_ratio_both"] = float("nan")

            _quantiles(to_opt_det, supervised_mask, f"{metric_root}/scale{scale}_min_")

        return metrics

    def _collect_automask_metrics(self, scale, reprojection_losses, identity_selection):
        if not getattr(self.opt, "enable_debug_metrics", False):
            return {}
        if self.opt.disable_automasking or identity_selection is None:
            return {}

        hr_percentile = float(getattr(self.opt, "automask_hr_percentile", 90.0))
        hr_scope = str(getattr(self.opt, "automask_hr_scope", "all")).lower()
        if hr_scope not in ("all", "mask"):
            hr_scope = "all"

        with torch.no_grad():
            min_reproj = self._compute_min_reproj(reprojection_losses, self.opt.avg_reprojection).squeeze(1)
            mask_keep = identity_selection.squeeze(1) > 0.5
            finite = torch.isfinite(min_reproj)

            total_pixels = 0
            keep_pixels = 0
            masked_pixels = 0
            high_all = 0
            high_keep = 0
            high_masked = 0
            q = max(0.0, min(hr_percentile / 100.0, 1.0))

            for b in range(min_reproj.shape[0]):
                mr = min_reproj[b]
                fin = finite[b]
                total = int(fin.sum().item())
                if total <= 0:
                    continue

                keep = int((mask_keep[b] & fin).sum().item())
                masked = total - keep
                total_pixels += total
                keep_pixels += keep
                masked_pixels += masked

                if hr_scope == "mask" and mask_keep[b].any():
                    values = mr[mask_keep[b] & fin]
                else:
                    values = mr[fin]
                if values.numel() == 0:
                    continue

                threshold = float(torch.quantile(values, q).item())
                high = (mr > threshold) & fin
                high_all += int(high.sum().item())
                high_keep += int((high & mask_keep[b]).sum().item())
                high_masked += int((high & (~mask_keep[b]) & fin).sum().item())

        return {
            f"metrics/automask/scale{scale}_keep_ratio": (
                keep_pixels / total_pixels if total_pixels > 0 else float("nan")
            ),
            f"metrics/automask/scale{scale}_high_res_ratio_in_mask": (
                high_masked / masked_pixels if masked_pixels > 0 else float("nan")
            ),
            f"metrics/automask/scale{scale}_bad_keep_ratio": (
                high_keep / high_all if high_all > 0 else float("nan")
            ),
        }

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0

        frame_offsets = {}
        if getattr(self.opt, "enable_debug_metrics", False):
            for frame_id in self.opt.frame_ids[1:]:
                val = inputs.get(("frame_offset", frame_id), None)
                if isinstance(val, torch.Tensor):
                    if val.numel() == 0:
                        val = float(frame_id)
                    else:
                        val = float(val.detach().float().mean().cpu().item())
                elif val is not None:
                    val = float(val)
                else:
                    val = float(frame_id)
                frame_offsets[frame_id] = val

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            if color.shape[-2:] != disp.shape[-2:]:
                color = F.interpolate(
                    color,
                    size=disp.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            target = inputs[("color", 0, 0)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_loss_tensor = None
            identity_losses = None
            if not self.opt.disable_automasking:
                identity_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, 0)]
                    identity_losses.append(self.compute_reprojection_loss(pred, target))
                identity_losses = torch.cat(identity_losses, 1)

                if self.opt.avg_reprojection:
                    identity_loss_tensor = identity_losses.mean(1, keepdim=True)
                else:
                    identity_loss_tensor = identity_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            automask_identity_comp = None
            automask_reproj_comp = None
            automask_margin = None
            if identity_losses is not None:
                automask_identity_comp = self._reduce_automask_compare_loss(
                    identity_losses,
                    avg_reprojection=self.opt.avg_reprojection,
                )
                automask_reproj_comp = self._reduce_automask_compare_loss(
                    reprojection_losses,
                    avg_reprojection=self.opt.avg_reprojection,
                )
                automask_margin = automask_identity_comp - automask_reproj_comp

            if not self.opt.disable_automasking and identity_loss_tensor is not None:
                identity_loss_tensor = identity_loss_tensor + torch.randn(
                    identity_loss_tensor.shape, device=self.device
                ) * 1e-5
                combined = torch.cat((identity_loss_tensor, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
                combined_argmin = None
            else:
                to_optimise, combined_argmin = torch.min(combined, dim=1)

            identity_selection = None
            if not self.opt.disable_automasking and identity_loss_tensor is not None:
                if combined_argmin is None:
                    combined_argmin = torch.min(combined, dim=1)[1]
                identity_selection = (
                    combined_argmin > identity_loss_tensor.shape[1] - 1
                ).float()
                outputs[f"identity_selection/{scale}"] = identity_selection

            keep_mask = self._combine_keep_mask(identity_selection)
            loss += self._masked_mean(to_optimise, keep_mask)

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses[f"loss/{scale}"] = loss

            if scale == 0:
                if automask_identity_comp is not None and automask_reproj_comp is not None:
                    outputs["automask_identity_comp/0"] = automask_identity_comp.detach()
                    outputs["automask_reproj_comp/0"] = automask_reproj_comp.detach()
                    outputs["automask_margin/0"] = automask_margin.detach()

                    finite_margin = torch.isfinite(automask_margin)
                    margin_flat = automask_margin[finite_margin]
                    if margin_flat.numel() > 0:
                        losses["metrics/automask/scale0_margin_mean"] = float(margin_flat.mean().item())
                        losses["metrics/automask/scale0_margin_abs_mean"] = float(
                            margin_flat.abs().mean().item()
                        )
                    if identity_selection is not None:
                        keep_bool = identity_selection > 0.5
                        if keep_bool.dim() == 3:
                            keep_bool = keep_bool.unsqueeze(1)
                        keep_margin = automask_margin[keep_bool & finite_margin]
                        masked_margin = automask_margin[(~keep_bool) & finite_margin]
                        if keep_margin.numel() > 0:
                            losses["metrics/automask/scale0_margin_keep_mean"] = float(
                                keep_margin.mean().item()
                            )
                        if masked_margin.numel() > 0:
                            losses["metrics/automask/scale0_margin_masked_mean"] = float(
                                masked_margin.mean().item()
                            )

                if (
                    identity_selection is not None
                    and combined_argmin is not None
                    and (not self.opt.avg_reprojection)
                ):
                    id_ch = identity_loss_tensor.shape[1]
                    reproj_selected = identity_selection > 0.5
                    source_choice = (combined_argmin - id_ch).clamp_min(0)
                    source_map = torch.full_like(source_choice, -1, dtype=torch.float32)
                    for idx, fid in enumerate(self.opt.frame_ids[1:]):
                        fid_keep = reproj_selected.bool() & (source_choice == idx)
                        if isinstance(fid, numbers.Integral):
                            source_map[fid_keep] = float(int(fid))
                        outputs[("automask_keep_fid", fid, 0)] = fid_keep.unsqueeze(1).float().detach()
                    outputs["automask_source_fid/0"] = source_map.unsqueeze(1).detach()

                if identity_selection is not None:
                    losses.update(
                        self._collect_automask_metrics(scale, reprojection_losses, identity_selection)
                    )
                losses.update(
                    self._collect_debug_metrics(
                        scale,
                        reprojection_losses,
                        identity_loss_tensor,
                        combined,
                        to_optimise,
                        frame_offsets,
                    )
                )

        total_loss /= len(self.opt.scales)
        losses["loss"] = total_loss
        return losses

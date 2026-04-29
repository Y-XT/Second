# monodepth2_loss.py

import numbers
import math
import torch
import torch.nn.functional as F
from layers import SSIM, get_smooth_loss

class Monodepth2Loss:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.ssim = SSIM().to(self.device)
        self.global_step = 0
        self.global_epoch = 0
        self.total_steps = None
        self.steps_per_epoch = None
        self.bestpp = {}

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            return l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            return 0.85 * ssim_loss + 0.15 * l1_loss

    def compute_posegt_reprojection_loss(self, pred, target):
        mode = str(getattr(self.opt, "posegt_reprojection_mode", "gasmono")).lower()
        if mode == "gasmono":
            abs_diff = torch.abs(target - pred) / (target + 0.01)
        else:
            abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if self.opt.no_ssim:
            return l1_loss
        ssim_loss = self.ssim(pred, target).mean(1, True)
        return 0.85 * ssim_loss + 0.15 * l1_loss

    def _compute_gasmono_selfpp_loss(self, disp_best, outputs):
        if not torch.is_tensor(disp_best):
            return None
        disp_best = disp_best.detach()
        pp_loss = None
        for scale in self.opt.scales:
            disp_pred = outputs.get(("disp", scale), None)
            if not torch.is_tensor(disp_pred):
                continue
            if disp_pred.shape[-2:] != disp_best.shape[-2:]:
                disp_pred = F.interpolate(
                    disp_pred,
                    size=disp_best.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            term = torch.log(torch.abs(disp_best - disp_pred) + 1.0).mean()
            pp_loss = term if pp_loss is None else (pp_loss + term)
        return pp_loss

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
    def _masked_sample_reduce(loss_map, keep_mask=None):
        if loss_map is None or (not torch.is_tensor(loss_map)):
            return None, None, None
        if loss_map.dim() == 4 and loss_map.shape[1] == 1:
            loss_map = loss_map[:, 0]
        elif loss_map.dim() != 3:
            return None, None, None

        valid = torch.isfinite(loss_map)
        if keep_mask is not None and torch.is_tensor(keep_mask):
            if keep_mask.dim() == 4 and keep_mask.shape[1] == 1:
                keep_mask = keep_mask[:, 0]
            elif keep_mask.dim() != 3:
                return None, None, None
            valid = valid & (keep_mask > 0.5)

        zeros = torch.zeros_like(loss_map)
        masked = torch.where(valid, loss_map, zeros)
        numer = masked.reshape(masked.shape[0], -1).sum(dim=1)
        denom = valid.reshape(valid.shape[0], -1).sum(dim=1)
        has_valid = denom > 0
        denom = denom.to(dtype=loss_map.dtype)
        numer = torch.where(torch.isfinite(numer), numer, torch.zeros_like(numer))
        return numer, denom, has_valid

    def _get_external_keep_mask(self, inputs, scale, loss_map, device, dtype):
        # Prefer scale-0 mask since reprojection loss is computed at full resolution.
        key = ("mask", 0, 0)
        mask = inputs.get(key, None)
        if mask is None or (not torch.is_tensor(mask)):
            key = ("mask", 0, scale)
            mask = inputs.get(key, None)
        if mask is None or (not torch.is_tensor(mask)):
            return None
        mask = mask.to(device=device, dtype=dtype)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        if loss_map.dim() == 3:
            target_h, target_w = loss_map.shape[-2], loss_map.shape[-1]
        else:
            target_h, target_w = loss_map.shape[-2], loss_map.shape[-1]
        if mask.shape[-2] != target_h or mask.shape[-1] != target_w:
            mask = F.interpolate(mask, size=(target_h, target_w), mode="nearest")
        return mask > 0.5

    @staticmethod
    def _combine_keep_mask(external_keep, identity_selection):
        keep = external_keep
        if identity_selection is not None:
            id_keep = identity_selection > 0.5
            if id_keep.dim() == 3:
                id_keep = id_keep.unsqueeze(1)
            keep = id_keep if keep is None else (keep & id_keep)
        return keep

    @staticmethod
    def _compute_sample_keep_ratio(reprojection_losses, identity_selection, avg_reprojection):
        if identity_selection is None or (not torch.is_tensor(identity_selection)):
            return None, None

        min_reproj = Monodepth2Loss._compute_min_reproj(reprojection_losses, avg_reprojection)
        if min_reproj.dim() == 4 and min_reproj.shape[1] == 1:
            min_reproj = min_reproj[:, 0]
        elif min_reproj.dim() != 3:
            return None, None

        if identity_selection.dim() == 4 and identity_selection.shape[1] == 1:
            identity_selection = identity_selection[:, 0]
        elif identity_selection.dim() != 3:
            return None, None

        finite = torch.isfinite(min_reproj)
        keep = (identity_selection > 0.5) & finite
        keep_count = keep.reshape(keep.shape[0], -1).sum(dim=1).to(dtype=min_reproj.dtype)
        total = finite.reshape(finite.shape[0], -1).sum(dim=1)
        has_valid = total > 0
        total_f = total.to(dtype=min_reproj.dtype).clamp_min(1.0)
        ratio = keep_count / total_f
        ratio = torch.where(has_valid, ratio, torch.zeros_like(ratio))
        ratio = torch.where(torch.isfinite(ratio), ratio, torch.zeros_like(ratio))
        return ratio, has_valid

    def _build_reprojection_branch_pack(
        self,
        reprojection_losses,
        identity_compare_tensor,
        external_keep=None,
    ):
        if (not torch.is_tensor(reprojection_losses)) or reprojection_losses.dim() != 4:
            return None

        if self.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if identity_compare_tensor is not None:
            combined = torch.cat((identity_compare_tensor, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        combined_argmin = None
        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, combined_argmin = torch.min(combined, dim=1)

        identity_selection = None
        if identity_compare_tensor is not None:
            if combined_argmin is None:
                combined_argmin = torch.min(combined, dim=1)[1]
            identity_selection = (
                combined_argmin > identity_compare_tensor.shape[1] - 1
            ).float()

        keep_mask = self._combine_keep_mask(external_keep, identity_selection)
        numer, denom, valid = self._masked_sample_reduce(to_optimise, keep_mask)
        if numer is None or denom is None or valid is None:
            return None
        sample_loss = numer / denom.clamp_min(1.0)
        sample_loss = torch.where(valid, sample_loss, torch.zeros_like(sample_loss))
        sample_loss = torch.where(torch.isfinite(sample_loss), sample_loss, torch.zeros_like(sample_loss))

        keep_ratio, keep_valid = self._compute_sample_keep_ratio(
            reprojection_losses,
            identity_selection,
            avg_reprojection=self.opt.avg_reprojection,
        )
        if keep_ratio is None or keep_valid is None:
            keep_ratio = torch.zeros(reprojection_losses.shape[0], device=reprojection_losses.device, dtype=reprojection_losses.dtype)
            keep_valid = torch.zeros(reprojection_losses.shape[0], device=reprojection_losses.device, dtype=torch.bool)

        to_optimise_map = to_optimise if to_optimise.dim() == 4 else to_optimise.unsqueeze(1)
        return {
            "reprojection_loss": reprojection_loss,
            "combined": combined,
            "to_optimise": to_optimise,
            "to_optimise_map": to_optimise_map,
            "identity_selection": identity_selection,
            "keep_mask": keep_mask,
            "numer": numer,
            "denom": denom,
            "valid": valid,
            "sample_loss": sample_loss,
            "keep_ratio": keep_ratio,
            "keep_valid": keep_valid,
        }

    @staticmethod
    def _reduce_automask_compare_loss(loss_tensor, avg_reprojection):
        """Aggregate competing automask losses into a single comparison map."""
        if loss_tensor is None:
            return None
        if loss_tensor.dim() == 3:
            loss_tensor = loss_tensor.unsqueeze(1)
        if avg_reprojection:
            return loss_tensor.mean(1, keepdim=True)
        if loss_tensor.shape[1] == 1:
            return loss_tensor
        return torch.min(loss_tensor, dim=1, keepdim=True)[0]

    @staticmethod
    def _build_fid_aligned_derot_weight(
        derot_weights,
        derot_weight_fids,
        frame_ids,
        reproj_argmin,
        derot_weight_union,
    ):
        """Select per-pixel de-rotation weight following reprojection winner fid."""
        if reproj_argmin is None or (not derot_weights):
            return derot_weight_union

        # stacked_derot: [B, N, H, W], N follows derot_weight_fids order
        stacked_derot = torch.stack([w.squeeze(1) for w in derot_weights], dim=1)
        fid_to_derot_idx = {fid: idx for idx, fid in enumerate(derot_weight_fids)}

        gather_idx = torch.zeros_like(reproj_argmin, dtype=torch.long)
        has_mapping = torch.zeros_like(reproj_argmin, dtype=torch.bool)
        for reproj_idx, fid in enumerate(frame_ids):
            derot_idx = fid_to_derot_idx.get(fid, None)
            if derot_idx is None:
                continue
            mask = reproj_argmin == reproj_idx
            if mask.any():
                gather_idx[mask] = int(derot_idx)
                has_mapping = has_mapping | mask

        selected = torch.gather(stacked_derot, 1, gather_idx.unsqueeze(1))
        if derot_weight_union is not None:
            selected = torch.where(has_mapping.unsqueeze(1), selected, derot_weight_union)
        return selected

    @staticmethod
    def _percentile_normalize_map(value_map, valid_mask, q_low=5.0, q_high=95.0, eps=1e-6):
        """Per-image percentile normalization on valid pixels, output clipped to [0, 1]."""
        if value_map.dim() == 3:
            value_map = value_map.unsqueeze(1)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)

        out = torch.zeros_like(value_map)
        ql = max(0.0, min(float(q_low) / 100.0, 1.0))
        qh = max(0.0, min(float(q_high) / 100.0, 1.0))
        if qh <= ql:
            qh = min(1.0, ql + 1e-3)

        for b in range(value_map.shape[0]):
            valid_flat = (valid_mask[b] > 0.5).reshape(-1)
            if not bool(valid_flat.any()):
                continue
            vals = value_map[b].reshape(-1)[valid_flat]
            vals = vals[torch.isfinite(vals)]
            if vals.numel() < 1:
                continue
            lo = torch.quantile(vals, ql)
            hi = torch.quantile(vals, qh)
            if (not torch.isfinite(lo)) or (not torch.isfinite(hi)) or float(hi.item()) <= float(lo.item()):
                lo = vals.min()
                hi = vals.max()
            denom = (hi - lo).clamp_min(eps)
            norm_b = ((value_map[b] - lo) / denom).clamp(0.0, 1.0)
            norm_b = torch.where(torch.isfinite(norm_b), norm_b, torch.zeros_like(norm_b))
            out[b] = norm_b
        return out

    @staticmethod
    def _normalize_map_with_bounds(value_map, lo, hi, eps=1e-6):
        """Normalize map with shared scalar bounds, output clipped to [0, 1]."""
        if value_map.dim() == 3:
            value_map = value_map.unsqueeze(1)
        lo_t = torch.as_tensor(lo, device=value_map.device, dtype=value_map.dtype)
        hi_t = torch.as_tensor(hi, device=value_map.device, dtype=value_map.dtype)
        denom = (hi_t - lo_t).clamp_min(eps)
        norm = ((value_map - lo_t) / denom).clamp(0.0, 1.0)
        norm = torch.where(torch.isfinite(norm), norm, torch.zeros_like(norm))
        return norm

    def _compute_depth_sens_batch_norm_stats(
        self,
        outputs,
        frame_ids,
        scale,
        device,
        dtype,
        q_low,
        q_high,
    ):
        """Shared percentile bounds over current mini-batch and all fids."""
        ql = max(0.0, min(float(q_low) / 100.0, 1.0))
        qh = max(0.0, min(float(q_high) / 100.0, 1.0))
        if qh <= ql:
            qh = min(1.0, ql + 1e-3)

        sens_g_vals = []
        sens_p_vals = []

        for fid in frame_ids:
            sens_g = outputs.get(("depth_sens_loss_delta", fid, scale), None)
            sens_p = outputs.get(("depth_sens_pixmag_delta", fid, scale), None)
            sens_valid = outputs.get(("depth_sens_valid", fid, scale), None)
            if (not torch.is_tensor(sens_g)) or (not torch.is_tensor(sens_p)) or (not torch.is_tensor(sens_valid)):
                continue

            warp_valid = outputs.get(("warp_valid", fid, scale), None)
            if not torch.is_tensor(warp_valid):
                warp_valid = outputs.get(("distorted_mask", fid, scale), None)

            sens_g = sens_g.to(device=device, dtype=dtype)
            sens_p = sens_p.to(device=device, dtype=dtype)
            sens_valid = sens_valid.to(device=device)
            if torch.is_tensor(warp_valid):
                warp_valid = warp_valid.to(device=device)

            if sens_g.dim() == 3:
                sens_g = sens_g.unsqueeze(1)
            if sens_p.dim() == 3:
                sens_p = sens_p.unsqueeze(1)
            if sens_valid.dim() == 3:
                sens_valid = sens_valid.unsqueeze(1)
            if torch.is_tensor(warp_valid) and warp_valid.dim() == 3:
                warp_valid = warp_valid.unsqueeze(1)

            valid = sens_valid > 0.5
            if torch.is_tensor(warp_valid):
                valid = valid & (warp_valid > 0.5)
            if not bool(valid.any()):
                continue

            g_vals = sens_g[valid]
            p_vals = sens_p[valid]
            g_vals = g_vals[torch.isfinite(g_vals)]
            p_vals = p_vals[torch.isfinite(p_vals)]
            if g_vals.numel() > 0:
                sens_g_vals.append(g_vals.reshape(-1))
            if p_vals.numel() > 0:
                sens_p_vals.append(p_vals.reshape(-1))

        def _compute_bounds(vals_list):
            if not vals_list:
                return None
            vals = torch.cat(vals_list, dim=0)
            if vals.numel() < 1:
                return None
            lo = torch.quantile(vals, ql)
            hi = torch.quantile(vals, qh)
            if (not torch.isfinite(lo)) or (not torch.isfinite(hi)) or float(hi.item()) <= float(lo.item()):
                lo = vals.min()
                hi = vals.max()
            if float(hi.item()) <= float(lo.item()):
                hi = lo + 1e-6
            return lo, hi

        g_bounds = _compute_bounds(sens_g_vals)
        p_bounds = _compute_bounds(sens_p_vals)
        if g_bounds is None or p_bounds is None:
            return None

        g_lo, g_hi = g_bounds
        p_lo, p_hi = p_bounds
        return {
            "g_lo": g_lo.to(device=device, dtype=dtype),
            "g_hi": g_hi.to(device=device, dtype=dtype),
            "p_lo": p_lo.to(device=device, dtype=dtype),
            "p_hi": p_hi.to(device=device, dtype=dtype),
        }

    def _build_depth_sens_fid_weight(
        self,
        outputs,
        frame_id,
        scale,
        device,
        dtype,
        q_low,
        q_high,
        wpix_scale,
        w_min,
        shared_norm_stats=None,
    ):
        sens_g = outputs.get(("depth_sens_loss_delta", frame_id, scale), None)
        sens_p = outputs.get(("depth_sens_pixmag_delta", frame_id, scale), None)
        sens_valid = outputs.get(("depth_sens_valid", frame_id, scale), None)
        if (not torch.is_tensor(sens_g)) or (not torch.is_tensor(sens_p)) or (not torch.is_tensor(sens_valid)):
            return None
        warp_valid = outputs.get(("warp_valid", frame_id, scale), None)
        if not torch.is_tensor(warp_valid):
            warp_valid = outputs.get(("distorted_mask", frame_id, scale), None)

        sens_g = sens_g.to(device=device, dtype=dtype)
        sens_p = sens_p.to(device=device, dtype=dtype)
        sens_valid = sens_valid.to(device=device)
        if torch.is_tensor(warp_valid):
            warp_valid = warp_valid.to(device=device)
        if sens_g.dim() == 3:
            sens_g = sens_g.unsqueeze(1)
        if sens_p.dim() == 3:
            sens_p = sens_p.unsqueeze(1)
        if sens_valid.dim() == 3:
            sens_valid = sens_valid.unsqueeze(1)
        if torch.is_tensor(warp_valid) and warp_valid.dim() == 3:
            warp_valid = warp_valid.unsqueeze(1)

        valid = sens_valid > 0.5
        if torch.is_tensor(warp_valid):
            valid = valid & (warp_valid > 0.5)
        if shared_norm_stats is not None:
            s_g = self._normalize_map_with_bounds(
                sens_g,
                shared_norm_stats["g_lo"],
                shared_norm_stats["g_hi"],
            )
            s_p = self._normalize_map_with_bounds(
                sens_p,
                shared_norm_stats["p_lo"],
                shared_norm_stats["p_hi"],
            )
        else:
            s_g = self._percentile_normalize_map(sens_g, valid, q_low=q_low, q_high=q_high)
            s_p = self._percentile_normalize_map(sens_p, valid, q_low=q_low, q_high=q_high)
        joint = (s_g * s_p).clamp(0.0, 1.0)
        mismatch = (s_g * (1.0 - s_p)).clamp(0.0, 1.0)

        suppress = max(0.0, min(float(wpix_scale), 1.0))
        w_floor = max(0.0, min(float(w_min), 1.0))
        w_pix = 1.0 - suppress * mismatch
        w_pix = w_pix.clamp(min=w_floor, max=1.0)
        w_pix = torch.where(valid, w_pix, torch.zeros_like(w_pix))
        w = w_pix
        w = torch.where(valid, w, torch.zeros_like(w))
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))

        valid_f = valid.to(dtype=dtype)
        mismatch_sum = (mismatch * valid_f).reshape(mismatch.shape[0], -1).sum(dim=1)
        valid_count = valid_f.reshape(valid_f.shape[0], -1).sum(dim=1)
        mismatch_mean = mismatch_sum / valid_count.clamp_min(1.0)
        has_valid = valid_count > 0.5

        return {
            "weight": w.to(dtype=dtype),
            "joint": joint.to(dtype=dtype),
            "mismatch": mismatch.to(dtype=dtype),
            "wpix": w_pix.to(dtype=dtype),
            "valid": valid,
            "mismatch_mean": mismatch_mean.to(dtype=dtype),
            "has_valid": has_valid,
        }

    @staticmethod
    def _build_depth_sens_frame_weights(mismatch_mean, has_valid, wimg_scale, w_min, eps=1e-6):
        """
        Build per-frame scalar suppression weights from frame-wise mismatch means.
        Lower mismatch frame -> higher weight (up to 1), higher mismatch frame -> lower weight.
        """
        if mismatch_mean is None or has_valid is None:
            return None
        if mismatch_mean.dim() != 2 or has_valid.dim() != 2:
            return None
        out = torch.ones_like(mismatch_mean)
        scale = max(0.0, min(float(wimg_scale), 1.0))
        floor = max(0.0, min(float(w_min), 1.0))
        if scale <= 0.0:
            return out

        bsz, nframe = mismatch_mean.shape
        for b in range(bsz):
            valid_idx = has_valid[b] > 0.5
            n_valid = int(valid_idx.sum().item())
            if n_valid <= 0:
                continue
            vals = mismatch_mean[b, valid_idx]
            finite = torch.isfinite(vals)
            if not bool(finite.any()):
                continue
            vals = vals.clone()
            vals[~finite] = 0.0
            vmin = vals.min()
            vmax = vals.max()
            if float((vmax - vmin).abs().item()) <= eps:
                rel = torch.zeros_like(vals)
            else:
                rel = (vals - vmin) / (vmax - vmin + eps)
            w = 1.0 - scale * rel
            w = w.clamp(min=floor, max=1.0)

            out[b, valid_idx] = w

        if nframe > 0:
            out = torch.where(torch.isfinite(out), out, torch.ones_like(out))
        return out

    @staticmethod
    def _compute_valid_batch_bounds(value_map, valid_mask, q_low=5.0, q_high=95.0, eps=1e-6):
        """Compute shared [low, high] bounds from valid pixels over the whole mini-batch."""
        if value_map.dim() == 3:
            value_map = value_map.unsqueeze(1)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)

        valid = (valid_mask > 0.5) & torch.isfinite(value_map)
        vals = value_map[valid]
        if vals.numel() < 1:
            return None

        ql = max(0.0, min(float(q_low) / 100.0, 1.0))
        qh = max(0.0, min(float(q_high) / 100.0, 1.0))
        if qh <= ql:
            qh = min(1.0, ql + 1e-3)

        lo = torch.quantile(vals, ql)
        hi = torch.quantile(vals, qh)
        if (not torch.isfinite(lo)) or (not torch.isfinite(hi)) or float(hi.item()) <= float(lo.item()):
            lo = vals.min()
            hi = vals.max()
        if float(hi.item()) <= float(lo.item()):
            hi = lo + eps
        return lo, hi

    def _build_depth_sens_target_weight(
        self,
        outputs,
        frame_ids,
        scale,
        winner_index,
        keep_mask,
        device,
        dtype,
        q_low,
        q_high,
        wpix_scale,
        wimg_scale,
        w_min,
    ):
        """
        Build target-space depth sensitivity weights.
        - Intra-image (pixel) suppression: per-target-image percentile normalization.
        - Inter-image (sample) suppression: batch-level normalization on raw target mismatch.
        """
        if not frame_ids:
            return None
        if winner_index is None or (not torch.is_tensor(winner_index)):
            return None

        if winner_index.dim() == 4 and winner_index.shape[1] == 1:
            winner_index = winner_index[:, 0]
        if winner_index.dim() != 3:
            return None

        bsz, h_t, w_t = winner_index.shape
        nframe = len(frame_ids)
        winner = winner_index.to(device=device, dtype=torch.long).clamp(0, nframe - 1)

        keep_bool = None
        if keep_mask is not None and torch.is_tensor(keep_mask):
            keep_bool = keep_mask.to(device=device)
            if keep_bool.dim() == 3:
                keep_bool = keep_bool.unsqueeze(1)
            keep_bool = keep_bool > 0.5

        sens_g_list = []
        sens_p_list = []
        sens_valid_list = []
        has_any = False

        for fid in frame_ids:
            sens_g = outputs.get(("depth_sens_loss_delta", fid, scale), None)
            sens_p = outputs.get(("depth_sens_pixmag_delta", fid, scale), None)
            sens_valid = outputs.get(("depth_sens_valid", fid, scale), None)
            warp_valid = outputs.get(("warp_valid", fid, scale), None)
            if not torch.is_tensor(warp_valid):
                warp_valid = outputs.get(("distorted_mask", fid, scale), None)

            if (not torch.is_tensor(sens_g)) or (not torch.is_tensor(sens_p)) or (not torch.is_tensor(sens_valid)):
                sens_g = torch.zeros((bsz, 1, h_t, w_t), device=device, dtype=dtype)
                sens_p = torch.zeros((bsz, 1, h_t, w_t), device=device, dtype=dtype)
                valid = torch.zeros((bsz, 1, h_t, w_t), device=device, dtype=torch.bool)
                sens_g_list.append(sens_g)
                sens_p_list.append(sens_p)
                sens_valid_list.append(valid)
                continue

            sens_g = sens_g.to(device=device, dtype=dtype)
            sens_p = sens_p.to(device=device, dtype=dtype)
            sens_valid = sens_valid.to(device=device)
            if torch.is_tensor(warp_valid):
                warp_valid = warp_valid.to(device=device)

            if sens_g.dim() == 3:
                sens_g = sens_g.unsqueeze(1)
            if sens_p.dim() == 3:
                sens_p = sens_p.unsqueeze(1)
            if sens_valid.dim() == 3:
                sens_valid = sens_valid.unsqueeze(1)
            if torch.is_tensor(warp_valid) and warp_valid.dim() == 3:
                warp_valid = warp_valid.unsqueeze(1)

            if sens_g.shape[-2:] != (h_t, w_t):
                sens_g = F.interpolate(sens_g, size=(h_t, w_t), mode="bilinear", align_corners=False)
            if sens_p.shape[-2:] != (h_t, w_t):
                sens_p = F.interpolate(sens_p, size=(h_t, w_t), mode="bilinear", align_corners=False)
            if sens_valid.shape[-2:] != (h_t, w_t):
                sens_valid = F.interpolate(sens_valid.float(), size=(h_t, w_t), mode="nearest") > 0.5
            if torch.is_tensor(warp_valid) and warp_valid.shape[-2:] != (h_t, w_t):
                warp_valid = F.interpolate(warp_valid.float(), size=(h_t, w_t), mode="nearest") > 0.5

            valid = sens_valid > 0.5
            if torch.is_tensor(warp_valid):
                valid = valid & (warp_valid > 0.5)
            if keep_bool is not None:
                valid = valid & keep_bool

            sens_g = torch.where(torch.isfinite(sens_g), sens_g, torch.zeros_like(sens_g))
            sens_p = torch.where(torch.isfinite(sens_p), sens_p, torch.zeros_like(sens_p))
            valid = valid & torch.isfinite(sens_g) & torch.isfinite(sens_p)

            if bool(valid.any()):
                has_any = True
            sens_g_list.append(sens_g)
            sens_p_list.append(sens_p)
            sens_valid_list.append(valid)

        if not has_any:
            return None

        g_stack = torch.cat(sens_g_list, dim=1)         # [B, N, H, W]
        p_stack = torch.cat(sens_p_list, dim=1)         # [B, N, H, W]
        valid_stack = torch.cat(
            [v.to(dtype=dtype) for v in sens_valid_list],
            dim=1,
        ) > 0.5                                          # [B, N, H, W]

        gather_idx = winner.unsqueeze(1)
        g_tgt = torch.gather(g_stack, 1, gather_idx)
        p_tgt = torch.gather(p_stack, 1, gather_idx)
        valid_tgt = torch.gather(valid_stack.to(dtype=dtype), 1, gather_idx) > 0.5
        if keep_bool is not None:
            valid_tgt = valid_tgt & keep_bool

        # Intra-image pixel suppression: per-target-image normalization.
        s_g_intra = self._percentile_normalize_map(g_tgt, valid_tgt, q_low=q_low, q_high=q_high)
        s_p_intra = self._percentile_normalize_map(p_tgt, valid_tgt, q_low=q_low, q_high=q_high)
        mismatch_intra = (s_g_intra * (1.0 - s_p_intra)).clamp(0.0, 1.0)

        suppress_pix = max(0.0, min(float(wpix_scale), 1.0))
        w_floor = max(0.0, min(float(w_min), 1.0))
        w_intra = (1.0 - suppress_pix * mismatch_intra).clamp(min=w_floor, max=1.0)
        w_intra = torch.where(valid_tgt, w_intra, torch.zeros_like(w_intra))

        # Inter-image sample suppression: batch-normalized target raw mismatch.
        g_bounds = self._compute_valid_batch_bounds(g_tgt, valid_tgt, q_low=q_low, q_high=q_high)
        p_bounds = self._compute_valid_batch_bounds(p_tgt, valid_tgt, q_low=q_low, q_high=q_high)
        if g_bounds is not None and p_bounds is not None:
            s_g_batch = self._normalize_map_with_bounds(g_tgt, g_bounds[0], g_bounds[1])
            s_p_batch = self._normalize_map_with_bounds(p_tgt, p_bounds[0], p_bounds[1])
        else:
            s_g_batch = torch.zeros_like(g_tgt)
            s_p_batch = torch.zeros_like(p_tgt)
        mismatch_batch = (s_g_batch * (1.0 - s_p_batch)).clamp(0.0, 1.0)

        valid_f = valid_tgt.to(dtype=dtype)
        score_num = (mismatch_batch * valid_f).reshape(bsz, -1).sum(dim=1)
        score_den = valid_f.reshape(bsz, -1).sum(dim=1)
        has_valid = score_den > 0.5
        score_raw = score_num / score_den.clamp_min(1.0)
        score_raw = torch.where(torch.isfinite(score_raw), score_raw, torch.zeros_like(score_raw))

        score_norm = torch.zeros_like(score_raw)
        if bool(has_valid.any()):
            vals = score_raw[has_valid]
            vals = vals[torch.isfinite(vals)]
            if vals.numel() > 0:
                ql = max(0.0, min(float(q_low) / 100.0, 1.0))
                qh = max(0.0, min(float(q_high) / 100.0, 1.0))
                if qh <= ql:
                    qh = min(1.0, ql + 1e-3)
                lo = torch.quantile(vals, ql)
                hi = torch.quantile(vals, qh)
                if (not torch.isfinite(lo)) or (not torch.isfinite(hi)) or float(hi.item()) <= float(lo.item()):
                    lo = vals.min()
                    hi = vals.max()
                denom = (hi - lo).clamp_min(1e-6)
                score_norm[has_valid] = ((score_raw[has_valid] - lo) / denom).clamp(0.0, 1.0)

        suppress_img = max(0.0, min(float(wimg_scale), 1.0))
        w_inter_scalar = torch.ones_like(score_raw)
        if suppress_img > 0.0:
            w_inter_scalar[has_valid] = (
                1.0 - suppress_img * score_norm[has_valid]
            ).clamp(min=w_floor, max=1.0)
        w_inter_scalar = torch.where(
            torch.isfinite(w_inter_scalar),
            w_inter_scalar,
            torch.ones_like(w_inter_scalar),
        )
        w_inter_map = w_inter_scalar.view(bsz, 1, 1, 1).expand_as(w_intra)
        w_inter_map = torch.where(valid_tgt, w_inter_map, torch.zeros_like(w_inter_map))

        weight = w_intra * w_inter_map
        weight = torch.where(valid_tgt, weight, torch.zeros_like(weight))
        weight = torch.where(torch.isfinite(weight), weight, torch.zeros_like(weight))

        winner_fid = torch.zeros_like(weight)
        for idx, fid in enumerate(frame_ids):
            if not isinstance(fid, numbers.Integral):
                continue
            mask = (winner == idx).unsqueeze(1) & valid_tgt
            if bool(mask.any()):
                winner_fid[mask] = float(int(fid))

        return {
            "weight": weight.to(dtype=dtype),
            "valid": valid_tgt,
            "w_intra": w_intra.to(dtype=dtype),
            "w_inter_scalar": w_inter_scalar.to(dtype=dtype),
            "w_inter_map": w_inter_map.to(dtype=dtype),
            "mismatch_intra": mismatch_intra.to(dtype=dtype),
            "mismatch_batch": mismatch_batch.to(dtype=dtype),
            "score_raw": score_raw.to(dtype=dtype),
            "score_norm": score_norm.to(dtype=dtype),
            "winner_fid": winner_fid.to(dtype=dtype),
        }

    @staticmethod
    def _compute_log_standardized_map(value_map, valid_mask, clip_value=3.0, eps=1e-6):
        """Batch-shared log standardization over valid pixels."""
        if value_map.dim() == 3:
            value_map = value_map.unsqueeze(1)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)

        log_map = torch.log(value_map.clamp_min(eps))
        valid = (valid_mask > 0.5) & torch.isfinite(log_map)
        score_map = torch.zeros_like(log_map)
        mean = torch.zeros((), device=value_map.device, dtype=value_map.dtype)
        std = torch.zeros((), device=value_map.device, dtype=value_map.dtype)

        if bool(valid.any()):
            vals = log_map[valid]
            mean = vals.mean()
            std = vals.std(unbiased=False)
            score_map = (log_map - mean) / (std + eps)
            if clip_value > 0.0:
                score_map = score_map.clamp(min=-float(clip_value), max=float(clip_value))
            score_map = torch.where(valid, score_map, torch.zeros_like(score_map))

        score_map = torch.where(torch.isfinite(score_map), score_map, torch.zeros_like(score_map))
        return {
            "score": score_map,
            "log_map": log_map,
            "valid": valid,
            "mean": mean,
            "std": std,
        }

    @staticmethod
    def _compute_standardized_map(value_map, valid_mask, clip_value=3.0, eps=1e-6):
        """Batch-shared standardization over valid pixels without log transform."""
        if value_map.dim() == 3:
            value_map = value_map.unsqueeze(1)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)

        valid = (valid_mask > 0.5) & torch.isfinite(value_map)
        score_map = torch.zeros_like(value_map)
        mean = torch.zeros((), device=value_map.device, dtype=value_map.dtype)
        std = torch.zeros((), device=value_map.device, dtype=value_map.dtype)

        if bool(valid.any()):
            vals = value_map[valid]
            mean = vals.mean()
            std = vals.std(unbiased=False)
            score_map = (value_map - mean) / (std + eps)
            if clip_value > 0.0:
                score_map = score_map.clamp(min=-float(clip_value), max=float(clip_value))
            score_map = torch.where(valid, score_map, torch.zeros_like(score_map))

        score_map = torch.where(torch.isfinite(score_map), score_map, torch.zeros_like(score_map))
        return {
            "score": score_map,
            "valid": valid,
            "mean": mean,
            "std": std,
        }

    def _build_badscore_target_weight(
        self,
        outputs,
        frame_ids,
        scale,
        reprojection_losses,
        winner_index,
        identity_selection,
        device,
        dtype,
        alpha_r,
        alpha_o,
        wimg_scale,
        w_min,
        norm_clip,
    ):
        """Build target-space residual/observability bad-score image weights."""
        if (not frame_ids) or (winner_index is None) or (not torch.is_tensor(winner_index)):
            return None
        if (not torch.is_tensor(reprojection_losses)) or reprojection_losses.dim() != 4:
            return None

        if winner_index.dim() == 4 and winner_index.shape[1] == 1:
            winner_index = winner_index[:, 0]
        if winner_index.dim() != 3:
            return None

        bsz, nframe, h_t, w_t = reprojection_losses.shape
        if len(frame_ids) != nframe:
            return None

        winner = winner_index.to(device=device, dtype=torch.long).clamp(0, nframe - 1)
        gather_idx = winner.unsqueeze(1)

        reproj_stack = reprojection_losses.to(device=device, dtype=dtype)
        R_map = torch.gather(reproj_stack, 1, gather_idx)

        obs_list = []
        obs_valid_list = []
        has_obs = False
        for fid in frame_ids:
            obs = outputs.get(("badscore_obs", fid, scale), None)
            obs_valid = outputs.get(("badscore_obs_valid", fid, scale), None)
            if (not torch.is_tensor(obs)) or (not torch.is_tensor(obs_valid)):
                obs = torch.zeros((bsz, 1, h_t, w_t), device=device, dtype=dtype)
                obs_valid = torch.zeros((bsz, 1, h_t, w_t), device=device, dtype=torch.bool)
            else:
                obs = obs.to(device=device, dtype=dtype)
                obs_valid = obs_valid.to(device=device)
                if obs.dim() == 3:
                    obs = obs.unsqueeze(1)
                if obs_valid.dim() == 3:
                    obs_valid = obs_valid.unsqueeze(1)
                if obs.shape[-2:] != (h_t, w_t):
                    obs = F.interpolate(obs, size=(h_t, w_t), mode="bilinear", align_corners=False)
                if obs_valid.shape[-2:] != (h_t, w_t):
                    obs_valid = F.interpolate(obs_valid.float(), size=(h_t, w_t), mode="nearest") > 0.5
                obs = torch.where(torch.isfinite(obs), obs, torch.zeros_like(obs))
                obs_valid = (obs_valid > 0.5) & torch.isfinite(obs)
                has_obs = has_obs or bool(obs_valid.any())
            obs_list.append(obs)
            obs_valid_list.append(obs_valid)

        if not has_obs:
            return None

        O_stack = torch.cat(obs_list, dim=1)
        O_valid_stack = torch.cat([v.to(dtype=dtype) for v in obs_valid_list], dim=1) > 0.5
        O_map = torch.gather(O_stack, 1, gather_idx)
        O_valid = torch.gather(O_valid_stack.to(dtype=dtype), 1, gather_idx) > 0.5

        if identity_selection is not None:
            K_map = identity_selection.to(device=device)
            if K_map.dim() == 3:
                K_map = K_map.unsqueeze(1)
            K_map = K_map > 0.5
        else:
            K_map = torch.ones_like(R_map, dtype=torch.bool)

        invalid_penalty = float(getattr(self.opt, "invalid_photometric_penalty", 1e4))
        residual_valid = torch.isfinite(R_map) & (R_map < invalid_penalty)
        obs_valid = torch.isfinite(O_map) & O_valid
        score_valid = K_map & residual_valid & obs_valid
        if not bool(score_valid.any()):
            return None

        r_pack = self._compute_log_standardized_map(R_map, score_valid, clip_value=norm_clip)
        obs_pack = self._compute_log_standardized_map(O_map, score_valid, clip_value=norm_clip)
        r_map = r_pack["score"]
        # Raw O_map means observability and is "larger = better".
        # We flip the standardized score so o_map means low-observability severity:
        # larger o_map => less observable => more suspicious.
        o_map = -obs_pack["score"]

        alpha_r = float(alpha_r)
        alpha_o = float(alpha_o)
        sig_r = torch.sigmoid(alpha_r * r_map)
        sig_o = torch.sigmoid(alpha_o * o_map)
        b_map = sig_r * sig_o
        b_map = torch.where(score_valid, b_map, torch.zeros_like(b_map))
        b_map = torch.where(torch.isfinite(b_map), b_map, torch.zeros_like(b_map))

        score_valid_f = score_valid.to(dtype=dtype)
        keep_count = score_valid_f.reshape(bsz, -1).sum(dim=1)
        has_valid = keep_count > 0.5
        B_img = (b_map.reshape(bsz, -1).sum(dim=1) / keep_count.clamp_min(1.0)).to(dtype=dtype)
        B_img = torch.where(torch.isfinite(B_img), B_img, torch.zeros_like(B_img))

        B_hat = torch.zeros_like(B_img)
        if bool(has_valid.any()):
            vals = B_img[has_valid]
            vals = vals[torch.isfinite(vals)]
            if vals.numel() > 0:
                mean_b = vals.mean()
                std_b = vals.std(unbiased=False)
                B_hat[has_valid] = (B_img[has_valid] - mean_b) / (std_b + 1e-6)
                if norm_clip > 0.0:
                    B_hat[has_valid] = B_hat[has_valid].clamp(
                        min=-float(norm_clip),
                        max=float(norm_clip),
                    )
        B_hat = torch.where(torch.isfinite(B_hat), B_hat, torch.zeros_like(B_hat))

        suppress_img = max(0.0, min(float(wimg_scale), 1.0))
        w_floor = max(0.0, min(float(w_min), 1.0))
        W_img = torch.ones_like(B_img)
        if suppress_img > 0.0:
            W_img[has_valid] = (
                1.0 - suppress_img * torch.sigmoid(B_hat[has_valid])
            ).clamp(min=w_floor, max=1.0)
        W_img = torch.where(torch.isfinite(W_img), W_img, torch.ones_like(W_img))
        W_img_map = W_img.view(bsz, 1, 1, 1).expand_as(R_map)

        winner_fid = torch.zeros_like(R_map)
        for idx, fid in enumerate(frame_ids):
            if not isinstance(fid, numbers.Integral):
                continue
            mask = (winner == idx).unsqueeze(1)
            if bool(mask.any()):
                winner_fid[mask] = float(int(fid))

        return {
            "R_map": R_map.to(dtype=dtype),
            "O_map": O_map.to(dtype=dtype),
            "K_map": K_map,
            "valid": score_valid,
            "r_map": r_map.to(dtype=dtype),
            "o_map": o_map.to(dtype=dtype),
            "b_map": b_map.to(dtype=dtype),
            "B_img": B_img.to(dtype=dtype),
            "B_hat": B_hat.to(dtype=dtype),
            "w_img": W_img.to(dtype=dtype),
            "w_img_map": W_img_map.to(dtype=dtype),
            "winner_fid": winner_fid.to(dtype=dtype),
            "r_mean": r_pack["mean"].to(dtype=dtype),
            "r_std": r_pack["std"].to(dtype=dtype),
            "o_mean": obs_pack["mean"].to(dtype=dtype),
            "o_std": obs_pack["std"].to(dtype=dtype),
        }

    def _build_badscore_local_target_weight(
        self,
        base_pack,
        margin_map,
        dtype,
        beta_m,
        local_scale,
        w_min,
        norm_clip,
    ):
        """Build no-threshold pixel weights from fragile score and combine them with image-level weight."""
        if base_pack is None:
            return None

        fragile_base = base_pack["b_map"].to(dtype=dtype)
        w_img_map = base_pack["w_img_map"].to(dtype=dtype)
        valid = base_pack["valid"]
        if valid.dim() == 3:
            valid = valid.unsqueeze(1)
        valid = valid > 0.5
        valid_f = valid.to(dtype=dtype)

        if torch.is_tensor(margin_map):
            margin_map = margin_map.to(device=fragile_base.device, dtype=dtype)
            if margin_map.dim() == 3:
                margin_map = margin_map.unsqueeze(1)
            if margin_map.shape[-2:] != fragile_base.shape[-2:]:
                margin_map = F.interpolate(
                    margin_map,
                    size=fragile_base.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            margin_pack = self._compute_standardized_map(
                margin_map,
                valid,
                clip_value=norm_clip,
            )
            margin_tilde_map = margin_pack["score"]
            beta_m = float(beta_m)
            margin_gate_map = torch.sigmoid(-beta_m * margin_tilde_map)
            margin_gate_map = torch.where(valid, margin_gate_map, torch.zeros_like(margin_gate_map))
            margin_gate_map = torch.where(
                torch.isfinite(margin_gate_map),
                margin_gate_map,
                torch.zeros_like(margin_gate_map),
            )
            margin_mean = margin_pack["mean"].to(dtype=dtype)
            margin_std = margin_pack["std"].to(dtype=dtype)
        else:
            margin_tilde_map = torch.zeros_like(fragile_base)
            margin_gate_map = torch.ones_like(fragile_base)
            margin_gate_map = torch.where(valid, margin_gate_map, torch.zeros_like(margin_gate_map))
            margin_mean = torch.zeros((), device=fragile_base.device, dtype=dtype)
            margin_std = torch.zeros((), device=fragile_base.device, dtype=dtype)

        fragile_map = fragile_base * margin_gate_map
        fragile_map = torch.where(valid, fragile_map, torch.zeros_like(fragile_map))
        fragile_map = torch.where(torch.isfinite(fragile_map), fragile_map, torch.zeros_like(fragile_map))

        bsz = fragile_map.shape[0]
        flat_valid = valid_f.reshape(bsz, -1)
        count = flat_valid.sum(dim=1)
        has_valid = count > 0.5

        local_scale = max(0.0, float(local_scale))
        w_floor = max(0.0, min(float(w_min), 1.0))
        w_pix = torch.ones_like(fragile_map)
        if local_scale > 0.0:
            w_pix_valid = (1.0 - local_scale * fragile_map[valid]).clamp(min=w_floor, max=1.0)
            w_pix[valid] = w_pix_valid
        w_pix = torch.where(torch.isfinite(w_pix), w_pix, torch.ones_like(w_pix))
        weight = w_img_map * w_pix
        weight = torch.where(torch.isfinite(weight), weight, torch.ones_like(weight))
        weight_map = weight.to(dtype=dtype)

        weight_mean = torch.ones_like(count)
        if bool(has_valid.any()):
            mean_vals = (weight_map.reshape(bsz, -1) * flat_valid).sum(dim=1) / count.clamp_min(1.0)
            weight_mean[has_valid] = mean_vals[has_valid]
        weight_mean = torch.where(torch.isfinite(weight_mean), weight_mean, torch.ones_like(weight_mean))

        return {
            "margin_tilde_map": margin_tilde_map.to(dtype=dtype),
            "margin_gate_map": margin_gate_map.to(dtype=dtype),
            "fragile_map": fragile_map.to(dtype=dtype),
            "w_pix": w_pix.to(dtype=dtype),
            "weight": weight_map,
            "weight_mean": weight_mean.to(dtype=dtype),
            "margin_mean": margin_mean,
            "margin_std": margin_std,
            "valid": valid,
        }

    def _collect_debug_metrics(
        self,
        scale,
        reprojection_losses,
        identity_loss,
        combined,
        to_optimise,
        frame_offsets,
        prefix="",
    ):
        if not getattr(self.opt, "enable_debug_metrics", False):
            return {}
        metrics = {}
        metric_root = "metrics/photometric" if not prefix else f"metrics/{prefix}photometric"
        if getattr(self.opt, "avg_reprojection", False):
            return metrics

        frame_ids = self.opt.frame_ids[1:]
        if not frame_ids:
            return metrics

        frame_to_idx = {fid: idx for idx, fid in enumerate(frame_ids)}

        def _is_integer_frame(fid):
            return isinstance(fid, numbers.Integral)

        def _select_frame(predicate, fallback=None):
            candidates = [fid for fid in frame_ids if _is_integer_frame(fid) and predicate(fid)]
            if not candidates:
                return fallback
            # 选取离 0 最近的帧（时间上最邻近的 prev/next）
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

            valid_prev = None
            valid_next = None

            if prev_idx is not None:
                valid_prev = supervised_mask & (choice == prev_idx)
                metrics[f"{metric_root}/scale{scale}_prev_frame_id"] = (
                    float(prev_offset) if prev_offset is not None else float(prev_fid)
                )
                metrics[f"{metric_root}/scale{scale}_valid_ratio_prev"] = _ratio(valid_prev, total_den)
                metrics[f"{metric_root}/scale{scale}_min_from_prev_ratio"] = _ratio(valid_prev, supervised_den)
                _quantiles(
                    reproj_det[:, prev_idx],
                    valid_prev,
                    f"{metric_root}/scale{scale}_prev_",
                )
            else:
                metrics[f"{metric_root}/scale{scale}_prev_frame_id"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_valid_ratio_prev"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_min_from_prev_ratio"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_prev_p50"] = float("nan")
                metrics[f"{metric_root}/scale{scale}_prev_p90"] = float("nan")

            if next_idx is not None:
                valid_next = supervised_mask & (choice == next_idx)
                metrics[f"{metric_root}/scale{scale}_next_frame_id"] = (
                    float(next_offset) if next_offset is not None else float(next_fid)
                )
                metrics[f"{metric_root}/scale{scale}_valid_ratio_next"] = _ratio(valid_next, total_den)
                metrics[f"{metric_root}/scale{scale}_min_from_next_ratio"] = _ratio(valid_next, supervised_den)
                _quantiles(
                    reproj_det[:, next_idx],
                    valid_next,
                    f"{metric_root}/scale{scale}_next_",
                )
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

            _quantiles(
                to_opt_det,
                supervised_mask,
                f"{metric_root}/scale{scale}_min_",
            )

        return metrics

    @staticmethod
    def _compute_min_reproj(reprojection_losses, avg_reprojection):
        """Return [B,1,H,W] min reprojection loss (avg or min, matching training logic)."""
        if avg_reprojection:
            return reprojection_losses.mean(1, keepdim=True)
        if reprojection_losses.shape[1] == 1:
            return reprojection_losses
        return torch.min(reprojection_losses, dim=1, keepdim=True)[0]

    def _collect_automask_metrics(
        self,
        scale,
        reprojection_losses,
        identity_selection,
        external_keep=None,
    ):
        method_name = str(getattr(self.opt, "methods", ""))
        collect_debug = bool(getattr(self.opt, "enable_debug_metrics", False))
        if (not collect_debug) and method_name not in (
            "MD2_VGGT_PoseGT_DeRotHardMask",
            "MD2_VGGT_PoseGT_DeRotSigmoidWeight",
        ):
            return {}
        if self.opt.disable_automasking or identity_selection is None:
            return {}

        hr_percentile = float(getattr(self.opt, "automask_hr_percentile", 90.0))
        hr_scope = str(getattr(self.opt, "automask_hr_scope", "all")).lower()
        if hr_scope not in ("all", "mask"):
            hr_scope = "all"

        with torch.no_grad():
            min_reproj = self._compute_min_reproj(reprojection_losses, self.opt.avg_reprojection)
            min_reproj = min_reproj.squeeze(1)
            mask_keep = identity_selection.squeeze(1) > 0.5
            ext_keep = None
            if external_keep is not None:
                ext_keep = external_keep.squeeze(1) > 0.5
            finite = torch.isfinite(min_reproj)

            total_pixels = 0
            keep_pixels = 0
            masked_pixels = 0
            high_all = 0
            high_keep = 0
            high_masked = 0

            final_total = 0
            final_keep_pixels = 0
            final_masked_pixels = 0
            final_high_all = 0
            final_high_keep = 0
            final_high_masked = 0

            batch_size = min_reproj.shape[0]
            q = max(0.0, min(hr_percentile / 100.0, 1.0))
            for b in range(batch_size):
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

                if ext_keep is not None:
                    final_keep = (mask_keep[b] & ext_keep[b] & fin)
                    f_keep = int(final_keep.sum().item())
                    f_masked = total - f_keep
                    final_total += total
                    final_keep_pixels += f_keep
                    final_masked_pixels += f_masked

                    if hr_scope == "mask" and final_keep.any():
                        f_values = mr[final_keep]
                    else:
                        f_values = mr[fin]
                    if f_values.numel() == 0:
                        continue
                    f_threshold = float(torch.quantile(f_values, q).item())
                    f_high = (mr > f_threshold) & fin
                    final_high_all += int(f_high.sum().item())
                    final_high_keep += int((f_high & final_keep).sum().item())
                    final_high_masked += int((f_high & (~final_keep) & fin).sum().item())

        metrics = {}
        metrics[f"metrics/automask/scale{scale}_keep_ratio"] = (
            keep_pixels / total_pixels if total_pixels > 0 else float("nan")
        )
        metrics[f"metrics/automask/scale{scale}_high_res_ratio_in_mask"] = (
            high_masked / masked_pixels if masked_pixels > 0 else float("nan")
        )
        metrics[f"metrics/automask/scale{scale}_bad_keep_ratio"] = (
            high_keep / high_all if high_all > 0 else float("nan")
        )
        if ext_keep is not None:
            metrics[f"metrics/final_loss_mask/scale{scale}_keep_ratio"] = (
                final_keep_pixels / final_total if final_total > 0 else float("nan")
            )
            metrics[f"metrics/final_loss_mask/scale{scale}_high_res_ratio_in_mask"] = (
                final_high_masked / final_masked_pixels if final_masked_pixels > 0 else float("nan")
            )
            metrics[f"metrics/final_loss_mask/scale{scale}_bad_keep_ratio"] = (
                final_high_keep / final_high_all if final_high_all > 0 else float("nan")
            )
        return metrics

    def _apply_high_res_mask(
        self,
        loss_map,
        reprojection_losses,
        identity_selection,
        hr_percentile,
        hr_scope,
        extra_keep=None,
    ):
        """Mask out top high-residual pixels (per-image), return masked mean."""
        if loss_map.dim() == 3:
            loss_map = loss_map.unsqueeze(1)
        masks = self._compute_hrmask_masks(
            reprojection_losses,
            identity_selection,
            hr_percentile,
            hr_scope,
            extra_keep=extra_keep,
        )
        keep_mask = masks["final_keep"].unsqueeze(1)
        denom = keep_mask.float().sum().clamp_min(1.0)
        return (loss_map * keep_mask.float()).sum() / denom

    def _compute_hrmask_masks(
        self,
        reprojection_losses,
        identity_selection,
        hr_percentile,
        hr_scope,
        extra_keep=None,
    ):
        """Return masks for HRMask: base_keep, high_res, hrmask_bad, final_keep, finite."""
        hr_scope = str(hr_scope).lower()
        if hr_scope not in ("all", "mask"):
            hr_scope = "all"
        q = max(0.0, min(float(hr_percentile) / 100.0, 1.0))

        with torch.no_grad():
            min_reproj = self._compute_min_reproj(reprojection_losses, self.opt.avg_reprojection)
            min_reproj = min_reproj.squeeze(1)
            finite = torch.isfinite(min_reproj)

            if identity_selection is not None:
                base_keep = (identity_selection.squeeze(1) > 0.5) & finite
            else:
                base_keep = finite
            if extra_keep is not None:
                if extra_keep.dim() == 4:
                    extra_keep = extra_keep.squeeze(1)
                base_keep = base_keep & (extra_keep > 0.5)

            high_masks = []
            for b in range(min_reproj.shape[0]):
                mr = min_reproj[b]
                fin = finite[b]
                base = base_keep[b]

                if hr_scope == "mask" and base.any():
                    values = mr[base]
                else:
                    values = mr[fin]
                if values.numel() == 0:
                    high = torch.zeros_like(fin)
                else:
                    threshold = float(torch.quantile(values, q).item())
                    high = (mr > threshold) & fin
                    if identity_selection is not None:
                        high = high & base
                high_masks.append(high)

            high_res = torch.stack(high_masks, dim=0)
            hrmask_bad = high_res
            final_keep = base_keep & (~hrmask_bad)

        return {
            "base_keep": base_keep,
            "high_res": high_res,
            "hrmask_bad": hrmask_bad,
            "final_keep": final_keep,
            "finite": finite,
        }

    def _get_pose_teacher_progress(self):
        schedule_by = str(getattr(self.opt, "pose_teacher_schedule_by", "epoch")).lower()
        if schedule_by == "step":
            return float(getattr(self, "global_step", 0))

        total_steps = getattr(self, "total_steps", None)
        steps_per_epoch = getattr(self, "steps_per_epoch", None)
        if steps_per_epoch is None and total_steps:
            num_epochs = float(getattr(self.opt, "num_epochs", 0) or 0)
            if num_epochs > 0:
                steps_per_epoch = float(total_steps) / max(1.0, num_epochs)
        if steps_per_epoch:
            return float(getattr(self, "global_step", 0)) / max(1.0, float(steps_per_epoch))
        return float(getattr(self, "global_epoch", 0))

    def _get_pose_teacher_schedule_weight(self):
        schedule_mode = str(getattr(self.opt, "pose_teacher_schedule", "off")).lower()
        method_name = str(getattr(self.opt, "methods", ""))
        if schedule_mode in ("off", "none", ""):
            if method_name == "MD2_VGGT_Teacher_Distill":
                schedule_mode = "linear"
            else:
                return 1.0

        phase0_end = float(getattr(self.opt, "pose_teacher_phase0_end", -1))
        phase1_end = float(getattr(self.opt, "pose_teacher_phase1_end", -1))
        if phase1_end <= 0:
            schedule_by = str(getattr(self.opt, "pose_teacher_schedule_by", "epoch")).lower()
            if schedule_by == "step":
                total_steps = getattr(self, "total_steps", None)
                if total_steps is None:
                    steps_per_epoch = getattr(self, "steps_per_epoch", None)
                    num_epochs = float(getattr(self.opt, "num_epochs", 0) or 0)
                    if steps_per_epoch and num_epochs > 0:
                        total_steps = float(steps_per_epoch) * num_epochs
                if total_steps and float(total_steps) > 0:
                    total = float(total_steps)
                    phase0_end = max(1.0, round(total * 0.2))
                    phase1_end = max(phase0_end + 1.0, round(total * 0.8))
            else:
                num_epochs = float(getattr(self.opt, "num_epochs", 0) or 0)
                if num_epochs > 0:
                    phase0_end = max(1.0, round(num_epochs * 0.2))
                    phase1_end = max(phase0_end + 1.0, round(num_epochs * 0.8))
            if phase1_end <= 0:
                return 1.0
        if phase0_end < 0:
            phase0_end = 0.0
        if phase1_end < phase0_end:
            phase1_end = phase0_end

        w0 = float(getattr(self.opt, "pose_teacher_w0", 1.0))
        w1 = float(getattr(self.opt, "pose_teacher_w1", w0))
        w2 = float(getattr(self.opt, "pose_teacher_w2", 0.0))

        cur = self._get_pose_teacher_progress()
        if cur < phase0_end:
            return w0
        if cur < phase1_end:
            denom = max(1.0, phase1_end - phase0_end)
            alpha = (cur - phase0_end) / denom
            if schedule_mode == "cosine":
                alpha = 0.5 * (1.0 - math.cos(math.pi * float(alpha)))
            return w1 + (w2 - w1) * float(alpha)
        return w2

    def _get_pose_teacher_conf_weight(self, inputs, batch_size, device, dtype):
        key = str(getattr(self.opt, "pose_teacher_conf_key", "vggt_conf"))
        if not key:
            return torch.ones(batch_size, device=device, dtype=dtype)
        conf = inputs.get(key, None)
        if conf is None or not torch.is_tensor(conf):
            return torch.ones(batch_size, device=device, dtype=dtype)
        conf = conf.to(device=device, dtype=dtype)
        if conf.ndim > 1:
            conf_flat = conf.reshape(conf.shape[0], -1)
            conf_mean = conf_flat.mean(dim=1)
        else:
            conf_mean = conf.reshape(-1)

        conf_mean = torch.where(torch.isfinite(conf_mean), conf_mean, torch.zeros_like(conf_mean))
        conf_weight = conf_mean.clamp(0.0, 1.0)

        conf_floor = float(getattr(self.opt, "pose_teacher_conf_floor", 0.0))
        if conf_floor > 0.0:
            denom = max(1e-6, 1.0 - conf_floor)
            conf_weight = (conf_weight - conf_floor) / denom
            conf_weight = conf_weight.clamp(0.0, 1.0)

        conf_thresh = float(getattr(self.opt, "pose_teacher_conf_thresh", 0.0))
        if conf_thresh > 0.0:
            conf_weight = torch.where(conf_mean >= conf_thresh, conf_weight, torch.zeros_like(conf_weight))

        if conf_weight.shape[0] != batch_size:
            return torch.ones(batch_size, device=device, dtype=dtype)
        return conf_weight

    def _compute_pose_teacher_loss_legacy(self, inputs, outputs):
        rot_weight = float(getattr(self.opt, "pose_teacher_rot_weight", 0.0))
        trans_weight = float(getattr(self.opt, "pose_teacher_trans_weight", 0.0))
        if rot_weight <= 0.0 and trans_weight <= 0.0:
            return None, None, None, 1.0

        rot_terms = []
        trans_terms = []
        eps = 1e-7

        for frame_id in self.opt.frame_ids[1:]:
            T_pose = outputs.get(("cam_T_cam", 0, frame_id), None)
            T_prior = inputs.get(("external_cam_T_cam", 0, frame_id), None)
            if T_pose is None or T_prior is None:
                continue
            if not (torch.is_tensor(T_pose) and torch.is_tensor(T_prior)):
                continue
            if T_pose.shape[-2:] != (4, 4) or T_prior.shape[-2:] != (4, 4):
                continue

            T_prior = T_prior.to(T_pose.device)
            R_pose = T_pose[:, :3, :3]
            R_prior = T_prior[:, :3, :3]
            R_delta = torch.matmul(R_prior.transpose(1, 2), R_pose)
            trace = R_delta[..., 0, 0] + R_delta[..., 1, 1] + R_delta[..., 2, 2]
            cos_angle = (trace - 1.0) * 0.5
            cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
            rot_terms.append(torch.acos(cos_angle))

            t_pose = T_pose[:, :3, 3]
            t_prior = T_prior[:, :3, 3]
            t_pose_norm = t_pose / (t_pose.norm(dim=1, keepdim=True) + eps)
            t_prior_norm = t_prior / (t_prior.norm(dim=1, keepdim=True) + eps)
            cos_dir = (t_pose_norm * t_prior_norm).sum(-1)
            cos_dir = torch.clamp(cos_dir, -1.0 + eps, 1.0 - eps)
            trans_terms.append(1.0 - cos_dir)

        if not rot_terms and not trans_terms:
            return None, None, None, 1.0

        rot_loss = torch.cat(rot_terms).mean() if rot_terms else None
        trans_loss = torch.cat(trans_terms).mean() if trans_terms else None

        total = 0.0
        if rot_loss is not None:
            total = total + rot_weight * rot_loss
        if trans_loss is not None:
            total = total + trans_weight * trans_loss
        return total, rot_loss, trans_loss, 1.0

    def _compute_pose_teacher_loss(self, inputs, outputs):
        method_name = str(getattr(self.opt, "methods", ""))
        if method_name == "MD2_VGGT_Teacher":
            return self._compute_pose_teacher_loss_legacy(inputs, outputs)

        rot_weight = float(getattr(self.opt, "pose_teacher_rot_weight", 0.0))
        trans_weight = float(getattr(self.opt, "pose_teacher_trans_weight", 0.0))
        distill_weight = self._get_pose_teacher_schedule_weight()
        if distill_weight <= 0.0 or (rot_weight <= 0.0 and trans_weight <= 0.0):
            return None, None, None, distill_weight

        rot_num = None
        rot_den = None
        trans_num = None
        trans_den = None

        conf_weight = None
        eps = 1e-7
        min_prior_norm = float(getattr(self.opt, "pose_teacher_min_prior_t_norm", 0.0))
        min_pred_norm = float(getattr(self.opt, "pose_teacher_min_pred_t_norm", 0.0))
        max_rot_deg = float(getattr(self.opt, "pose_teacher_max_rot_deg", 0.0))
        max_rot_rad = math.radians(max_rot_deg) if max_rot_deg > 0.0 else None

        for frame_id in self.opt.frame_ids[1:]:
            T_pose = outputs.get(("cam_T_cam", 0, frame_id), None)
            T_prior = inputs.get(("external_cam_T_cam", 0, frame_id), None)
            if T_pose is None or T_prior is None:
                continue
            if not (torch.is_tensor(T_pose) and torch.is_tensor(T_prior)):
                continue
            if T_pose.shape[-2:] != (4, 4) or T_prior.shape[-2:] != (4, 4):
                continue

            T_prior = T_prior.to(T_pose.device)
            if conf_weight is None:
                conf_weight = self._get_pose_teacher_conf_weight(
                    inputs, T_pose.shape[0], T_pose.device, T_pose.dtype
                )
            conf_weight = conf_weight.to(device=T_pose.device, dtype=T_pose.dtype)

            finite_pose = torch.isfinite(T_pose).all(dim=(1, 2))
            finite_prior = torch.isfinite(T_prior).all(dim=(1, 2))
            valid = finite_pose & finite_prior

            R_pose = T_pose[:, :3, :3]
            R_prior = T_prior[:, :3, :3]
            t_pose = T_pose[:, :3, 3]
            t_prior = T_prior[:, :3, 3]

            eye = torch.eye(3, device=R_pose.device, dtype=R_pose.dtype)
            R_pose_safe = torch.where(valid[:, None, None], R_pose, eye)
            R_prior_safe = torch.where(valid[:, None, None], R_prior, eye)
            t_pose_safe = torch.where(valid[:, None], t_pose, torch.zeros_like(t_pose))
            t_prior_safe = torch.where(valid[:, None], t_prior, torch.zeros_like(t_prior))

            R_delta = torch.matmul(R_prior_safe.transpose(1, 2), R_pose_safe)
            trace = R_delta[..., 0, 0] + R_delta[..., 1, 1] + R_delta[..., 2, 2]
            cos_angle = (trace - 1.0) * 0.5
            cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
            rot_angle = torch.acos(cos_angle)

            prior_norm = torch.linalg.norm(t_prior_safe, dim=1)
            pred_norm = torch.linalg.norm(t_pose_safe, dim=1)
            prior_ok = prior_norm >= min_prior_norm if min_prior_norm > 0.0 else torch.ones_like(valid)
            conf_ok = conf_weight > 0.0

            base_mask = valid & prior_ok & conf_ok
            if max_rot_rad is not None:
                base_mask = base_mask & (rot_angle <= max_rot_rad)

            if rot_weight > 0.0:
                rot_w = torch.where(base_mask, conf_weight, torch.zeros_like(conf_weight))
                if rot_w.sum().item() > 0.0:
                    if rot_num is None:
                        rot_num = torch.zeros((), device=rot_w.device, dtype=rot_w.dtype)
                        rot_den = torch.zeros((), device=rot_w.device, dtype=rot_w.dtype)
                    rot_num = rot_num + (rot_angle * rot_w).sum()
                    rot_den = rot_den + rot_w.sum()

            if trans_weight > 0.0:
                pred_ok = pred_norm >= min_pred_norm if min_pred_norm > 0.0 else torch.ones_like(valid)
                trans_mask = base_mask & pred_ok
                t_pose_norm = t_pose_safe / (pred_norm.unsqueeze(1).clamp_min(eps))
                t_prior_norm = t_prior_safe / (prior_norm.unsqueeze(1).clamp_min(eps))
                cos_dir = (t_pose_norm * t_prior_norm).sum(-1)
                cos_dir = torch.clamp(cos_dir, -1.0 + eps, 1.0 - eps)
                trans_loss = 1.0 - cos_dir
                trans_w = torch.where(trans_mask, conf_weight, torch.zeros_like(conf_weight))
                if trans_w.sum().item() > 0.0:
                    if trans_num is None:
                        trans_num = torch.zeros((), device=trans_w.device, dtype=trans_w.dtype)
                        trans_den = torch.zeros((), device=trans_w.device, dtype=trans_w.dtype)
                    trans_num = trans_num + (trans_loss * trans_w).sum()
                    trans_den = trans_den + trans_w.sum()

        rot_loss = rot_num / rot_den if rot_den is not None and rot_den.item() > 0 else None
        trans_loss = trans_num / trans_den if trans_den is not None and trans_den.item() > 0 else None

        if rot_loss is None and trans_loss is None:
            return None, None, None, distill_weight

        total = 0.0
        if rot_loss is not None:
            total = total + rot_weight * rot_loss
        if trans_loss is not None:
            total = total + trans_weight * trans_loss
        total = total * float(distill_weight)
        return total, rot_loss, trans_loss, distill_weight

    @staticmethod
    def _masked_sample_mean(loss_map, keep_mask=None):
        numer, denom, has_valid = Monodepth2Loss._masked_sample_reduce(loss_map, keep_mask=keep_mask)
        if numer is None or denom is None or has_valid is None:
            return None, None
        mean = numer / denom.clamp_min(1.0)
        mean = torch.where(has_valid, mean, torch.zeros_like(mean))
        mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        return mean, has_valid

    def _compute_rotation_geodesic_per_frame(self, inputs, outputs, frame_ids, batch_size, device, dtype):
        rot_terms = []
        valid_terms = []
        eps = 1e-7
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)

        for frame_id in frame_ids:
            T_pose = outputs.get(("cam_T_cam", 0, frame_id), None)
            T_prior = inputs.get(("external_cam_T_cam", 0, frame_id), None)
            if (
                T_pose is None
                or T_prior is None
                or (not torch.is_tensor(T_pose))
                or (not torch.is_tensor(T_prior))
                or T_pose.shape[-2:] != (4, 4)
                or T_prior.shape[-2:] != (4, 4)
            ):
                rot_terms.append(torch.zeros(batch_size, device=device, dtype=dtype))
                valid_terms.append(torch.zeros(batch_size, device=device, dtype=torch.bool))
                continue

            T_pose = T_pose.to(device=device, dtype=dtype)
            T_prior = T_prior.to(device=device, dtype=dtype)
            valid = torch.isfinite(T_pose).all(dim=(1, 2)) & torch.isfinite(T_prior).all(dim=(1, 2))

            R_pose = T_pose[:, :3, :3]
            R_prior = T_prior[:, :3, :3]
            R_pose_safe = torch.where(valid[:, None, None], R_pose, eye)
            R_prior_safe = torch.where(valid[:, None, None], R_prior, eye)

            R_delta = torch.matmul(R_prior_safe.transpose(1, 2), R_pose_safe)
            trace = R_delta[..., 0, 0] + R_delta[..., 1, 1] + R_delta[..., 2, 2]
            cos_angle = (trace - 1.0) * 0.5
            cos_angle = torch.clamp(cos_angle, -1.0 + eps, 1.0 - eps)
            rot_terms.append(torch.acos(cos_angle))
            valid_terms.append(valid)

        if not rot_terms:
            return None, None

        rot_stack = torch.stack(rot_terms, dim=1)
        valid_stack = torch.stack(valid_terms, dim=1)
        rot_stack = torch.where(torch.isfinite(rot_stack), rot_stack, torch.zeros_like(rot_stack))
        return rot_stack, valid_stack

    def _build_rdistill_pack(
        self,
        inputs,
        outputs,
        frame_ids,
        pose_reprojection_losses,
        rdistill_frame_ids,
        rdistill_reprojection_losses,
        automask_margin,
        keep_mask,
    ):
        method_name = str(getattr(self.opt, "methods", ""))
        if method_name != "MonoViT_VGGT_RDistill":
            return None

        loss_weight = float(getattr(self.opt, "r_distill_loss_weight", 0.0))
        warmup_epochs = int(getattr(self.opt, "r_distill_warmup_epochs", 5))
        margin_thresh = float(getattr(self.opt, "r_distill_margin_thresh", 0.10))
        delta_rel_min = float(getattr(self.opt, "r_distill_delta_rel_min", 0.02))
        delta_rel_max = float(getattr(self.opt, "r_distill_delta_rel_max", 0.10))
        delta_rel_max = max(delta_rel_max, delta_rel_min + 1e-6)

        if (
            loss_weight <= 0.0
            or int(getattr(self, "global_epoch", 0)) < warmup_epochs
            or rdistill_reprojection_losses is None
        ):
            return None

        if (
            (not torch.is_tensor(pose_reprojection_losses))
            or pose_reprojection_losses.dim() != 4
            or len(frame_ids) != pose_reprojection_losses.shape[1]
            or (not torch.is_tensor(rdistill_reprojection_losses))
            or rdistill_reprojection_losses.dim() != 4
            or len(rdistill_frame_ids) != rdistill_reprojection_losses.shape[1]
        ):
            return None

        batch_size = pose_reprojection_losses.shape[0]
        device = pose_reprojection_losses.device
        dtype = pose_reprojection_losses.dtype
        rdistill_fid_to_idx = {fid: idx for idx, fid in enumerate(rdistill_frame_ids)}

        if torch.is_tensor(automask_margin):
            margin_mean, margin_valid = self._masked_sample_mean(automask_margin.detach(), keep_mask=None)
        else:
            margin_mean = torch.zeros(batch_size, device=device, dtype=dtype)
            margin_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)

        rot_stack, rot_valid = self._compute_rotation_geodesic_per_frame(
            inputs=inputs,
            outputs=outputs,
            frame_ids=frame_ids,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        if rot_stack is None or rot_valid is None:
            return None

        pose_mean_terms = []
        rfix_mean_terms = []
        valid_terms = []
        delta_terms = []

        for pose_idx, frame_id in enumerate(frame_ids):
            pose_mean_f, pose_valid_f = self._masked_sample_mean(
                pose_reprojection_losses[:, pose_idx:pose_idx + 1].detach(),
                keep_mask=None,
            )
            if pose_mean_f is None or pose_valid_f is None:
                return None

            rfix_idx = rdistill_fid_to_idx.get(frame_id, None)
            if rfix_idx is None:
                rfix_mean_f = torch.zeros_like(pose_mean_f)
                rfix_valid_f = torch.zeros_like(pose_valid_f)
            else:
                rfix_mean_f, rfix_valid_f = self._masked_sample_mean(
                    rdistill_reprojection_losses[:, rfix_idx:rfix_idx + 1].detach(),
                    keep_mask=None,
                )
                if rfix_mean_f is None or rfix_valid_f is None:
                    rfix_mean_f = torch.zeros_like(pose_mean_f)
                    rfix_valid_f = torch.zeros_like(pose_valid_f)

            valid_f = pose_valid_f & rfix_valid_f & rot_valid[:, pose_idx]
            delta_rel_f = torch.zeros_like(pose_mean_f)
            if bool(valid_f.any()):
                rel_f = (pose_mean_f[valid_f] - rfix_mean_f[valid_f]) / pose_mean_f[valid_f].clamp_min(1e-6)
                rel_f = torch.where(torch.isfinite(rel_f), rel_f, torch.zeros_like(rel_f))
                delta_rel_f[valid_f] = rel_f

            pose_mean_terms.append(pose_mean_f)
            rfix_mean_terms.append(rfix_mean_f)
            valid_terms.append(valid_f)
            delta_terms.append(delta_rel_f)

        pose_mean = torch.stack(pose_mean_terms, dim=1)
        rdistill_mean = torch.stack(rfix_mean_terms, dim=1)
        valid = torch.stack(valid_terms, dim=1)
        delta_rel = torch.stack(delta_terms, dim=1)

        margin_gate_sample = margin_valid & (margin_mean <= margin_thresh)
        gate = torch.zeros_like(delta_rel)
        if bool(valid.any()):
            gate[valid] = (
                (delta_rel[valid] - delta_rel_min) / (delta_rel_max - delta_rel_min)
            ).clamp(0.0, 1.0)
        gate = torch.where(torch.isfinite(gate), gate, torch.zeros_like(gate))
        gate_detached = gate.detach()
        active = valid & (gate_detached > 0.0)

        loss = None
        if bool(active.any()):
            numer = (gate_detached[active] * rot_stack[active]).sum()
            denom = gate_detached[active].sum().clamp_min(1e-6)
            loss = numer / denom * float(loss_weight)

        valid_sample = valid.any(dim=1)
        active_sample = active.any(dim=1)

        pose_mean_valid = pose_mean[valid]
        rdistill_mean_valid = rdistill_mean[valid]
        delta_rel_valid = delta_rel[valid]
        rot_valid_flat = rot_stack[valid]
        active_gate_flat = gate_detached[active]
        active_delta_flat = delta_rel[active]
        active_rot_flat = rot_stack[active]

        pose_mean_sample = torch.zeros(batch_size, device=device, dtype=dtype)
        rdistill_mean_sample = torch.zeros(batch_size, device=device, dtype=dtype)
        delta_rel_sample = torch.zeros(batch_size, device=device, dtype=dtype)
        rot_mean_sample = torch.zeros(batch_size, device=device, dtype=dtype)
        if bool(valid_sample.any()):
            valid_f = valid.to(dtype=dtype)
            denom = valid_f.sum(dim=1).clamp_min(1.0)
            pose_mean_sample[valid_sample] = (pose_mean * valid_f).sum(dim=1)[valid_sample] / denom[valid_sample]
            rdistill_mean_sample[valid_sample] = (
                (rdistill_mean * valid_f).sum(dim=1)[valid_sample] / denom[valid_sample]
            )
            delta_rel_sample[valid_sample] = (delta_rel * valid_f).sum(dim=1)[valid_sample] / denom[valid_sample]
            rot_mean_sample[valid_sample] = (rot_stack * valid_f).sum(dim=1)[valid_sample] / denom[valid_sample]
        gate_sample = gate_detached.max(dim=1)[0]

        metrics = {
            "loss/r_distill_weight": float(loss_weight),
            "metrics/r_distill/valid_ratio": float(valid_sample.float().mean().item()),
            "metrics/r_distill/valid_pair_ratio": float(valid.float().mean().item()),
            "metrics/r_distill/margin_gate_ratio": float(margin_gate_sample.float().mean().item()),
            "metrics/r_distill/active_ratio": float(active_sample.float().mean().item()),
            "metrics/r_distill/active_pair_ratio": float(active.float().mean().item()),
        }
        if bool(valid_sample.any()):
            metrics["metrics/r_distill/margin_mean"] = float(margin_mean[valid_sample].mean().item())
        if pose_mean_valid.numel() > 0:
            metrics["metrics/r_distill/pose_photo_mean"] = float(pose_mean_valid.mean().item())
        if rdistill_mean_valid.numel() > 0:
            metrics["metrics/r_distill/rfix_photo_mean"] = float(rdistill_mean_valid.mean().item())
        if delta_rel_valid.numel() > 0:
            metrics["metrics/r_distill/delta_rel_mean"] = float(delta_rel_valid.mean().item())
        if rot_valid_flat.numel() > 0:
            metrics["metrics/r_distill/rot_mean"] = float(rot_valid_flat.detach().mean().item())
        if active_gate_flat.numel() > 0:
            metrics["metrics/r_distill/gate_mean"] = float(active_gate_flat.mean().item())
        else:
            metrics["metrics/r_distill/gate_mean"] = 0.0
        if active_delta_flat.numel() > 0:
            metrics["metrics/r_distill/delta_rel_active_mean"] = float(active_delta_flat.mean().item())
        else:
            metrics["metrics/r_distill/delta_rel_active_mean"] = 0.0
        if active_rot_flat.numel() > 0:
            metrics["metrics/r_distill/rot_active_mean"] = float(active_rot_flat.detach().mean().item())
        else:
            metrics["metrics/r_distill/rot_active_mean"] = 0.0

        return {
            "loss": loss,
            "metrics": metrics,
            "margin_mean": margin_mean.detach(),
            "margin_gate_sample": margin_gate_sample.detach(),
            "pose_mean": pose_mean_sample.detach(),
            "rfix_mean": rdistill_mean_sample.detach(),
            "delta_rel": delta_rel_sample.detach(),
            "gate": gate_sample.detach(),
            "active": active_sample.detach(),
            "valid": valid_sample.detach(),
            "rot_mean": rot_mean_sample.detach(),
            "pair_delta_rel": delta_rel.detach(),
            "pair_gate": gate_detached.detach(),
            "pair_active": active.detach(),
            "pair_valid": valid.detach(),
            "pair_rot": rot_stack.detach(),
        }

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0
        method_name = str(getattr(self.opt, "methods", ""))
        use_gasmono_selfpp = method_name == "GasMono"
        use_pose_gating = method_name == "MD2_VGGT_Gated"
        use_pose_teacher = method_name in ("MD2_VGGT_Teacher", "MD2_VGGT_Teacher_Distill")
        use_rdistill = method_name == "MonoViT_VGGT_RDistill"
        use_rmask_switch = method_name == "MonoViT_VGGT_RMaskSwitch"
        use_teacher_photo = method_name == "MD2_VGGT_Teacher_Photo"
        use_derot_hardmask = method_name == "MD2_VGGT_PoseGT_DeRotHardMask"
        use_derot_sigmoid_weight = method_name == "MD2_VGGT_PoseGT_DeRotSigmoidWeight"
        use_depth_sens_weight = method_name == "MD2_VGGT_PoseGT_DepthSensWeight"
        use_badscore_weight = method_name in (
            "MD2_VGGT_PoseGT_BadScoreWeight",
            "MonoViT_VGGT_PoseGT_BadScoreWeight",
        )
        use_badscore_local_weight = method_name == "MD2_VGGT_PoseGT_BadScoreLocalWeight"
        use_posegt = method_name in (
            "MD2_VGGT_PoseGT", "MD2_VGGT_PoseGT_DepthCycleViz",
            "MD2_VGGT_PoseGT_DepthSensitivityViz",
            "MD2_VGGT_PoseGT_DepthSensViz",
            "MD2_VGGT_PoseGT_DepthSensWeight",
            "MD2_VGGT_PoseGT_BadScoreWeight",
            "MD2_VGGT_PoseGT_BadScoreLocalWeight",
            "MD2_VGGT_PoseGT_HRMask", "MD2_VGGT_PoseGT_Mask",
            "MD2_VGGT_PoseGT_DeRotHardMask",
            "MD2_VGGT_PoseGT_DeRotSigmoidWeight",
            "GasMono",
            "MonoViT_PoseGT",
            "MonoViT_PoseGT_Mask",
            "MonoViT_PoseGT_HRMask",
            "MonoViT_VGGT_PoseGT_BadScoreWeight",
        )
        use_hr_mask = method_name in ("MD2_VGGT_PoseGT_HRMask", "MonoViT_PoseGT_HRMask")
        gate_mode = str(getattr(self.opt, "pose_gating_mode", "min")).lower()
        gate_tau = float(getattr(self.opt, "pose_gating_tau", 0.1))
        teacher_photo_weight = float(getattr(self.opt, "teacher_photo_weight", 0.0))
        if teacher_photo_weight <= 0.0:
            use_teacher_photo = False
        posegt_weight = float(getattr(self.opt, "www", 0.0))
        if posegt_weight <= 0.0:
            use_posegt = False
        depth_sens_weight_start_epoch = int(getattr(self.opt, "depth_sens_weight_start_epoch", 10))
        depth_sens_weight_q_low = float(getattr(self.opt, "depth_sens_weight_q_low", 5.0))
        depth_sens_weight_q_high = float(getattr(self.opt, "depth_sens_weight_q_high", 95.0))
        depth_sens_wpix_scale = float(getattr(self.opt, "depth_sens_wpix_scale", 0.8))
        depth_sens_wimg_scale = float(getattr(self.opt, "depth_sens_wimg_scale", 0.6))
        depth_sens_weight_min = float(getattr(self.opt, "depth_sens_weight_min", 0.2))
        depth_sens_weight_source_scale = int(getattr(self.opt, "depth_sensitivity_viz_scale", 0))
        depth_sens_weight_enabled = bool(
            use_depth_sens_weight and int(getattr(self, "global_epoch", 0)) >= depth_sens_weight_start_epoch
        )
        badscore_start_epoch = int(getattr(self.opt, "badscore_start_epoch", 10))
        badscore_alpha_r = float(getattr(self.opt, "badscore_alpha_r", 1.0))
        badscore_alpha_o = float(getattr(self.opt, "badscore_alpha_o", 1.0))
        badscore_beta_m = float(getattr(self.opt, "badscore_beta_m", 1.0))
        badscore_wimg_scale = float(getattr(self.opt, "badscore_wimg_scale", 0.6))
        badscore_weight_min = float(getattr(self.opt, "badscore_weight_min", 0.2))
        badscore_norm_clip = float(getattr(self.opt, "badscore_norm_clip", 3.0))
        badscore_local_scale = float(getattr(self.opt, "badscore_local_scale", 0.2))
        badscore_local_weight_min = float(getattr(self.opt, "badscore_local_weight_min", 0.2))
        badscore_local_gamma = float(getattr(self.opt, "badscore_local_gamma", 1.0))
        badscore_local_z_thresh = float(getattr(self.opt, "badscore_local_z_thresh", 1.0))
        badscore_weight_enabled = bool(
            (use_badscore_weight or use_badscore_local_weight)
            and int(getattr(self, "global_epoch", 0)) >= badscore_start_epoch
        )

        teacher_photo_loss = 0.0
        teacher_photo_scales = 0
        posegt_total = 0.0
        rdistill_total = None
        rswitch_route_sample = None
        disp_best = None
        reproj_loss_min = None
        if use_gasmono_selfpp and isinstance(self.bestpp, dict):
            cached_disp = self.bestpp.get("disp", None)
            cached_err = self.bestpp.get("error", None)
            if torch.is_tensor(cached_disp) and torch.is_tensor(cached_err):
                disp_best = cached_disp
                reproj_loss_min = cached_err

        frame_offsets = {}
        if getattr(self.opt, "enable_debug_metrics", False):
            for frame_id in self.opt.frame_ids[1:]:
                key = ("frame_offset", frame_id)
                val = inputs.get(key, None)
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
        is_spidepth = method_name == "SPIDepth"

        def _smoothness_source_map(scale_idx, disp_map):
            if not is_spidepth:
                return disp_map
            depth_map = outputs.get(("depth", 0, scale_idx), None)
            if not torch.is_tensor(depth_map):
                depth_map = disp_map
            return torch.reciprocal(depth_map.clamp_min(1e-3))

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            rdistill_reprojection_losses = [] if (use_rdistill and scale == 0) else None
            rdistill_frame_ids = [] if (use_rdistill and scale == 0) else None
            rswitch_reprojection_losses = [] if use_rmask_switch else None
            derot_masks = []
            derot_weights = []
            derot_weight_fids = []
            derot_masks_raw = []
            derot_parallax_vals = []
            frame_ids = list(self.opt.frame_ids[1:])

            source_scale = 0
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            if color.shape[-2:] != disp.shape[-2:]:
                color = F.interpolate(
                    color,
                    size=disp.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reproj_loss = self.compute_reprojection_loss(pred, target)

                if self.opt.distorted_mask:
                    valid_mask = outputs.get(("distorted_mask", frame_id, scale), None)
                    if valid_mask is not None:
                        reproj_loss = reproj_loss * valid_mask + (1.0 - valid_mask) * 1e5

                if use_derot_hardmask:
                    derot_mask = outputs.get(("derot_mask", frame_id, scale), None)
                    derot_mask_raw = outputs.get(("derot_mask_raw", frame_id, scale), None)
                    derot_parallax = outputs.get(("derot_parallax", frame_id, scale), None)
                    if torch.is_tensor(derot_mask):
                        # Keep automask semantics unchanged:
                        # hard mask is merged only at final keep-mask stage.
                        derot_mask = derot_mask.to(device=reproj_loss.device)
                        derot_masks.append(derot_mask > 0.5)
                    if torch.is_tensor(derot_mask_raw):
                        derot_masks_raw.append(derot_mask_raw.to(device=reproj_loss.device) > 0.5)
                    if torch.is_tensor(derot_parallax):
                        derot_parallax_vals.append(derot_parallax.detach())
                elif use_derot_sigmoid_weight:
                    derot_weight = outputs.get(("derot_weight", frame_id, scale), None)
                    derot_mask_raw = outputs.get(("derot_mask_raw", frame_id, scale), None)
                    derot_parallax = outputs.get(("derot_parallax", frame_id, scale), None)
                    if torch.is_tensor(derot_weight):
                        derot_weight = derot_weight.to(device=reproj_loss.device, dtype=reproj_loss.dtype)
                        if derot_weight.dim() == 3:
                            derot_weight = derot_weight.unsqueeze(1)
                        derot_weights.append(derot_weight)
                        derot_weight_fids.append(frame_id)
                    if torch.is_tensor(derot_mask_raw):
                        derot_masks_raw.append(derot_mask_raw.to(device=reproj_loss.device) > 0.5)
                    if torch.is_tensor(derot_parallax):
                        derot_parallax_vals.append(derot_parallax.detach())

                if use_pose_gating:
                    pred_prior = outputs.get(("color_vggt", frame_id, scale), None)
                    if pred_prior is not None:
                        reproj_prior = self.compute_reprojection_loss(pred_prior, target)
                        if self.opt.distorted_mask:
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
                if rdistill_reprojection_losses is not None:
                    pred_rdistill = outputs.get(("color_rdistill", frame_id, scale), None)
                    if pred_rdistill is not None:
                        reproj_rdistill = self.compute_reprojection_loss(pred_rdistill, target)
                        if self.opt.distorted_mask:
                            valid_mask_rdistill = outputs.get(("distorted_mask_rdistill", frame_id, scale), None)
                            if valid_mask_rdistill is not None:
                                reproj_rdistill = (
                                    reproj_rdistill * valid_mask_rdistill
                                    + (1.0 - valid_mask_rdistill) * 1e5
                                )
                        rdistill_reprojection_losses.append(reproj_rdistill)
                        rdistill_frame_ids.append(frame_id)
                if rswitch_reprojection_losses is not None:
                    pred_rswitch = outputs.get(("color_rswitch", frame_id, scale), None)
                    if pred_rswitch is not None:
                        reproj_rswitch = self.compute_reprojection_loss(pred_rswitch, target)
                        if self.opt.distorted_mask:
                            valid_mask_rswitch = outputs.get(("distorted_mask_rswitch", frame_id, scale), None)
                            if valid_mask_rswitch is not None:
                                reproj_rswitch = (
                                    reproj_rswitch * valid_mask_rswitch
                                    + (1.0 - valid_mask_rswitch) * 1e5
                                )
                    else:
                        reproj_rswitch = reproj_loss.detach()
                    rswitch_reprojection_losses.append(reproj_rswitch)

            reprojection_losses = torch.cat(reprojection_losses, 1)
            if rdistill_reprojection_losses:
                rdistill_reprojection_losses = torch.cat(rdistill_reprojection_losses, 1)
            else:
                rdistill_reprojection_losses = None
            if rswitch_reprojection_losses:
                rswitch_reprojection_losses = torch.cat(rswitch_reprojection_losses, 1)
            else:
                rswitch_reprojection_losses = None
            derot_keep = None
            derot_weight_union = None
            if derot_masks:
                derot_keep = torch.stack(derot_masks, dim=0).any(dim=0)
            if derot_weights:
                derot_weight_union = torch.stack(derot_weights, dim=0).max(dim=0)[0]

            identity_loss_tensor = None
            identity_losses = None
            if not self.opt.disable_automasking:
                identity_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
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

            depth_sens_select_idx = None
            if (
                reprojection_losses.dim() == 4
                and reprojection_losses.shape[1] == len(frame_ids)
            ):
                depth_sens_select_idx = torch.argmin(reprojection_losses, dim=1)

            reproj_argmin = None
            if (
                (not self.opt.avg_reprojection)
                and reprojection_loss.dim() == 4
                and reprojection_loss.shape[1] == len(frame_ids)
            ):
                reproj_argmin = torch.argmin(reprojection_loss, dim=1)

            if not self.opt.disable_automasking and identity_loss_tensor is not None:
                identity_loss_tensor = identity_loss_tensor + torch.randn(
                    identity_loss_tensor.shape, device=self.device
                ) * 1e-5
                combined = torch.cat((identity_loss_tensor, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            combined_argmin = None
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, combined_argmin = torch.min(combined, dim=1)
            if use_gasmono_selfpp:
                disp_full = disp
                target_hw = to_optimise.shape[-2:]
                if disp_full.shape[-2:] != target_hw:
                    disp_full = F.interpolate(
                        disp_full,
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    )
                error_map = to_optimise
                if error_map.dim() == 3:
                    error_map = error_map.unsqueeze(1)
                if (
                    disp_best is None
                    or reproj_loss_min is None
                    or (not torch.is_tensor(disp_best))
                    or (not torch.is_tensor(reproj_loss_min))
                    or disp_best.shape != disp_full.shape
                    or reproj_loss_min.shape != error_map.shape
                ):
                    disp_best = disp_full.detach().clone()
                    reproj_loss_min = torch.full_like(error_map, 10.0)
                better = error_map.detach() < reproj_loss_min
                disp_best = torch.where(better, disp_full.detach(), disp_best)
                reproj_loss_min = torch.minimum(error_map.detach(), reproj_loss_min)

            identity_selection = None
            if not self.opt.disable_automasking and identity_loss_tensor is not None:
                if combined_argmin is None:
                    combined_argmin = torch.min(combined, dim=1)[1]
                identity_selection = (
                    combined_argmin > identity_loss_tensor.shape[1] - 1
                ).float()
                outputs["identity_selection/{}".format(scale)] = identity_selection
            external_keep = self._get_external_keep_mask(
                inputs, scale, to_optimise, to_optimise.device, to_optimise.dtype
            )
            final_keep_seed = external_keep
            if derot_keep is not None:
                final_keep_seed = derot_keep if final_keep_seed is None else (final_keep_seed & derot_keep)
            keep_mask = self._combine_keep_mask(final_keep_seed, identity_selection)

            if use_rmask_switch:
                pose_numer, pose_denom, pose_valid = self._masked_sample_reduce(to_optimise, keep_mask)
                if pose_numer is None or pose_denom is None or pose_valid is None:
                    pose_numer = torch.zeros(reprojection_losses.shape[0], device=reprojection_losses.device, dtype=reprojection_losses.dtype)
                    pose_denom = torch.zeros_like(pose_numer)
                    pose_valid = torch.zeros(reprojection_losses.shape[0], device=reprojection_losses.device, dtype=torch.bool)
                pose_sample_loss = pose_numer / pose_denom.clamp_min(1.0)
                pose_sample_loss = torch.where(pose_valid, pose_sample_loss, torch.zeros_like(pose_sample_loss))
                pose_keep_ratio, pose_keep_valid = self._compute_sample_keep_ratio(
                    reprojection_losses,
                    identity_selection,
                    avg_reprojection=self.opt.avg_reprojection,
                )
                if pose_keep_ratio is None or pose_keep_valid is None:
                    pose_keep_ratio = torch.zeros(reprojection_losses.shape[0], device=reprojection_losses.device, dtype=reprojection_losses.dtype)
                    pose_keep_valid = torch.zeros(reprojection_losses.shape[0], device=reprojection_losses.device, dtype=torch.bool)

                if rswitch_reprojection_losses is None:
                    rswitch_reprojection_losses = reprojection_losses.detach()
                rswitch_pack = self._build_reprojection_branch_pack(
                    reprojection_losses=rswitch_reprojection_losses,
                    identity_compare_tensor=identity_loss_tensor,
                    external_keep=external_keep,
                )

                route_sample = rswitch_route_sample
                if route_sample is None:
                    route_sample = torch.zeros(
                        reprojection_losses.shape[0],
                        device=reprojection_losses.device,
                        dtype=torch.bool,
                    )

                selected_reprojection_losses = reprojection_losses
                selected_combined = combined
                pose_to_optimise_map = to_optimise if to_optimise.dim() == 4 else to_optimise.unsqueeze(1)
                selected_to_optimise_map = pose_to_optimise_map
                selected_identity_selection = identity_selection
                selected_numer = pose_numer
                selected_denom = pose_denom
                pose_identity_map = None
                if identity_selection is not None:
                    pose_identity_map = identity_selection
                    if pose_identity_map.dim() == 3:
                        pose_identity_map = pose_identity_map.unsqueeze(1)
                pose_keep_map = torch.ones_like(pose_to_optimise_map, dtype=pose_to_optimise_map.dtype)
                if keep_mask is not None:
                    pose_keep_map = keep_mask.to(dtype=pose_to_optimise_map.dtype)
                    if pose_keep_map.dim() == 3:
                        pose_keep_map = pose_keep_map.unsqueeze(1)
                ext_identity_map = pose_identity_map
                ext_keep_map = pose_keep_map
                selected_keep_map = pose_keep_map

                if rswitch_pack is not None:
                    if scale == 0 or rswitch_route_sample is None:
                        keep_thresh = float(getattr(self.opt, "r_mask_switch_keep_thresh", 0.03))
                        delta_keep = (rswitch_pack["keep_ratio"] - pose_keep_ratio).detach()
                        route_valid = (pose_keep_valid & rswitch_pack["keep_valid"]).detach()
                        rswitch_route_sample = route_valid & (delta_keep > keep_thresh)
                        route_sample = rswitch_route_sample
                        outputs["r_mask_switch_pose_keep/0"] = pose_keep_ratio.detach().view(-1, 1, 1, 1)
                        outputs["r_mask_switch_external_keep/0"] = rswitch_pack["keep_ratio"].detach().view(-1, 1, 1, 1)
                        outputs["r_mask_switch_delta_keep/0"] = delta_keep.view(-1, 1, 1, 1)
                        outputs["r_mask_switch_gate/0"] = route_sample.float().view(-1, 1, 1, 1)
                        outputs["r_mask_switch_valid/0"] = route_valid.float().view(-1, 1, 1, 1)
                        outputs["r_mask_switch_thresh/0"] = torch.full(
                            (route_sample.shape[0], 1, 1, 1),
                            fill_value=keep_thresh,
                            device=route_sample.device,
                            dtype=pose_keep_ratio.dtype,
                        )
                    else:
                        route_sample = rswitch_route_sample

                    route_mask_4d = route_sample.view(-1, 1, 1, 1)
                    route_mask_3d = route_sample.view(-1, 1, 1)
                    ext_to_optimise_map = rswitch_pack["to_optimise_map"]
                    ext_identity_map = rswitch_pack["identity_selection"]
                    if ext_identity_map is not None and ext_identity_map.dim() == 3:
                        ext_identity_map = ext_identity_map.unsqueeze(1)
                    ext_keep_map = torch.ones_like(ext_to_optimise_map, dtype=ext_to_optimise_map.dtype)
                    if rswitch_pack["keep_mask"] is not None:
                        ext_keep_map = rswitch_pack["keep_mask"].to(dtype=ext_to_optimise_map.dtype)
                        if ext_keep_map.dim() == 3:
                            ext_keep_map = ext_keep_map.unsqueeze(1)
                    selected_reprojection_losses = torch.where(
                        route_mask_4d,
                        rswitch_reprojection_losses,
                        reprojection_losses,
                    )
                    selected_combined = torch.where(
                        route_mask_4d,
                        rswitch_pack["combined"],
                        combined,
                    )
                    selected_to_optimise_map = torch.where(
                        route_mask_4d,
                        rswitch_pack["to_optimise_map"],
                        selected_to_optimise_map,
                    )
                    if identity_selection is None:
                        selected_identity_selection = rswitch_pack["identity_selection"]
                    elif rswitch_pack["identity_selection"] is not None:
                        selected_identity_selection = torch.where(
                            route_mask_3d,
                            rswitch_pack["identity_selection"],
                            identity_selection,
                        )
                    selected_numer = torch.where(route_sample, rswitch_pack["numer"], pose_numer)
                    selected_denom = torch.where(route_sample, rswitch_pack["denom"], pose_denom)
                    selected_keep_map = torch.where(route_mask_4d, ext_keep_map, pose_keep_map)

                photo_loss = selected_numer.sum() / selected_denom.sum().clamp_min(1.0)
                loss = loss + photo_loss

                if selected_identity_selection is not None:
                    outputs["identity_selection/{}".format(scale)] = selected_identity_selection

                smooth_source = _smoothness_source_map(scale, disp)
                mean_disp = smooth_source.mean(2, True).mean(3, True)
                norm_disp = smooth_source / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)
                loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

                total_loss += loss
                losses["loss/{}".format(scale)] = loss

                if scale == 0:
                    def _valid_mean(values, valid_mask):
                        if values is None or valid_mask is None or (not bool(valid_mask.any())):
                            return float("nan")
                        vals = values[valid_mask]
                        vals = vals[torch.isfinite(vals)]
                        if vals.numel() < 1:
                            return float("nan")
                        return float(vals.mean().item())

                    ext_keep_ratio = rswitch_pack["keep_ratio"] if rswitch_pack is not None else None
                    ext_keep_valid = rswitch_pack["keep_valid"] if rswitch_pack is not None else None
                    ext_sample_loss = rswitch_pack["sample_loss"] if rswitch_pack is not None else None
                    delta_keep = None if ext_keep_ratio is None else (ext_keep_ratio - pose_keep_ratio)
                    route_valid = None if ext_keep_valid is None else (pose_keep_valid & ext_keep_valid)
                    if pose_identity_map is not None:
                        outputs["r_mask_switch_identity_selection_pose/0"] = pose_identity_map.detach()
                    if ext_identity_map is not None:
                        outputs["r_mask_switch_identity_selection_external/0"] = ext_identity_map.detach()
                    outputs["r_mask_switch_final_keep_pose/0"] = pose_keep_map.detach()
                    outputs["r_mask_switch_final_keep_external/0"] = ext_keep_map.detach()
                    outputs["r_mask_switch_final_keep_selected/0"] = selected_keep_map.detach()

                    losses["metrics/r_mask_switch/scale0_route_ratio"] = float(route_sample.float().mean().item())
                    if route_valid is not None:
                        losses["metrics/r_mask_switch/scale0_route_valid_ratio"] = float(route_valid.float().mean().item())
                    losses["metrics/r_mask_switch/scale0_pose_keep_mean"] = _valid_mean(pose_keep_ratio, pose_keep_valid)
                    losses["metrics/r_mask_switch/scale0_external_keep_mean"] = _valid_mean(ext_keep_ratio, ext_keep_valid)
                    losses["metrics/r_mask_switch/scale0_delta_keep_mean"] = _valid_mean(delta_keep, route_valid)
                    losses["metrics/r_mask_switch/scale0_pose_loss_mean"] = _valid_mean(pose_sample_loss, pose_valid)
                    losses["metrics/r_mask_switch/scale0_external_loss_mean"] = _valid_mean(ext_sample_loss, rswitch_pack["valid"] if rswitch_pack is not None else None)

                    if selected_identity_selection is not None:
                        metrics = self._collect_automask_metrics(
                            scale,
                            selected_reprojection_losses,
                            selected_identity_selection,
                            external_keep=external_keep,
                        )
                        losses.update(metrics)
                    metrics = self._collect_debug_metrics(
                        scale,
                        selected_reprojection_losses,
                        identity_loss_tensor,
                        selected_combined,
                        selected_to_optimise_map,
                        frame_offsets,
                    )
                    losses.update(metrics)
                continue

            loss_weight = None
            if use_derot_sigmoid_weight:
                if keep_mask is None:
                    loss_weight = torch.ones_like(to_optimise, dtype=to_optimise.dtype).unsqueeze(1)
                else:
                    loss_weight = keep_mask.to(dtype=to_optimise.dtype)
                    if loss_weight.dim() == 3:
                        loss_weight = loss_weight.unsqueeze(1)
                derot_weight_selected = self._build_fid_aligned_derot_weight(
                    derot_weights=derot_weights,
                    derot_weight_fids=derot_weight_fids,
                    frame_ids=frame_ids,
                    reproj_argmin=reproj_argmin,
                    derot_weight_union=derot_weight_union,
                )
                if derot_weight_selected is not None:
                    loss_weight = loss_weight * derot_weight_selected

            depth_sens_weight_selected = None
            depth_sens_valid_selected = None
            depth_sens_weight_available = False
            depth_sens_img_weight_stack = None
            badscore_pack = None
            badscore_img_weight = None
            badscore_local_pack = None
            badscore_local_weight = None
            if depth_sens_weight_enabled:
                target_pack = self._build_depth_sens_target_weight(
                    outputs=outputs,
                    frame_ids=frame_ids,
                    scale=depth_sens_weight_source_scale,
                    winner_index=depth_sens_select_idx,
                    keep_mask=keep_mask,
                    device=to_optimise.device,
                    dtype=to_optimise.dtype,
                    q_low=depth_sens_weight_q_low,
                    q_high=depth_sens_weight_q_high,
                    wpix_scale=depth_sens_wpix_scale,
                    wimg_scale=depth_sens_wimg_scale,
                    w_min=depth_sens_weight_min,
                )
                if target_pack is not None:
                    depth_sens_weight_selected = target_pack["weight"]
                    depth_sens_valid_selected = target_pack["valid"]
                    depth_sens_weight_available = bool(depth_sens_valid_selected.any().item())
                    depth_sens_img_weight_stack = target_pack["w_inter_scalar"]
                    if scale == 0:
                        outputs["depth_sens_weight_intra/0"] = target_pack["w_intra"].detach()
                        outputs["depth_sens_weight_inter/0"] = target_pack["w_inter_map"].detach()
                        outputs["depth_sens_mismatch_norm/0"] = target_pack["mismatch_intra"].detach()
                        outputs["depth_sens_mismatch_batch_norm/0"] = target_pack["mismatch_batch"].detach()
                        outputs["depth_sens_weight_img_shared/0"] = (
                            target_pack["w_inter_scalar"].view(-1, 1, 1, 1).detach()
                        )
                        outputs["depth_sens_score_target_raw/0"] = target_pack["score_raw"].detach()
                        outputs["depth_sens_score_target_norm/0"] = target_pack["score_norm"].detach()
                        outputs["depth_sens_winner_fid/0"] = target_pack["winner_fid"].detach()
                        for fid in frame_ids:
                            outputs[("depth_sens_weight_img", fid, 0)] = (
                                target_pack["w_inter_scalar"].view(-1, 1, 1, 1).detach()
                            )

            if badscore_weight_enabled:
                badscore_pack = self._build_badscore_target_weight(
                    outputs=outputs,
                    frame_ids=frame_ids,
                    scale=scale,
                    reprojection_losses=reprojection_losses,
                    winner_index=depth_sens_select_idx,
                    identity_selection=identity_selection,
                    device=to_optimise.device,
                    dtype=to_optimise.dtype,
                    alpha_r=badscore_alpha_r,
                    alpha_o=badscore_alpha_o,
                    wimg_scale=badscore_wimg_scale,
                    w_min=badscore_weight_min,
                    norm_clip=badscore_norm_clip,
                )
                if badscore_pack is not None:
                    if use_badscore_weight:
                        badscore_img_weight = badscore_pack["w_img_map"]
                    if use_badscore_local_weight:
                        badscore_local_pack = self._build_badscore_local_target_weight(
                            base_pack=badscore_pack,
                            margin_map=automask_margin,
                            dtype=to_optimise.dtype,
                            beta_m=badscore_beta_m,
                            local_scale=badscore_local_scale,
                            w_min=badscore_local_weight_min,
                            norm_clip=badscore_norm_clip,
                        )
                        if badscore_local_pack is not None:
                            badscore_local_weight = badscore_local_pack["weight"]

            if depth_sens_weight_selected is not None:
                if loss_weight is None:
                    if keep_mask is None:
                        loss_weight = torch.ones_like(to_optimise, dtype=to_optimise.dtype).unsqueeze(1)
                    else:
                        loss_weight = keep_mask.to(dtype=to_optimise.dtype)
                        if loss_weight.dim() == 3:
                            loss_weight = loss_weight.unsqueeze(1)
                loss_weight = loss_weight * depth_sens_weight_selected
            if badscore_img_weight is not None:
                if loss_weight is None:
                    if keep_mask is None:
                        loss_weight = torch.ones_like(to_optimise, dtype=to_optimise.dtype).unsqueeze(1)
                    else:
                        loss_weight = keep_mask.to(dtype=to_optimise.dtype)
                        if loss_weight.dim() == 3:
                            loss_weight = loss_weight.unsqueeze(1)
                loss_weight = loss_weight * badscore_img_weight
            if badscore_local_weight is not None:
                if loss_weight is None:
                    if keep_mask is None:
                        loss_weight = torch.ones_like(to_optimise, dtype=to_optimise.dtype).unsqueeze(1)
                    else:
                        loss_weight = keep_mask.to(dtype=to_optimise.dtype)
                        if loss_weight.dim() == 3:
                            loss_weight = loss_weight.unsqueeze(1)
                loss_weight = loss_weight * badscore_local_weight

            if use_rdistill and scale == 0:
                rdistill_pack = self._build_rdistill_pack(
                    inputs=inputs,
                    outputs=outputs,
                    frame_ids=frame_ids,
                    pose_reprojection_losses=reprojection_losses,
                    rdistill_frame_ids=rdistill_frame_ids,
                    rdistill_reprojection_losses=rdistill_reprojection_losses,
                    automask_margin=automask_margin,
                    keep_mask=keep_mask,
                )
                if rdistill_pack is not None:
                    losses.update(rdistill_pack["metrics"])
                    if rdistill_pack["loss"] is not None:
                        rdistill_total = rdistill_pack["loss"]
                    outputs["r_distill_margin_mean/0"] = rdistill_pack["margin_mean"].view(-1, 1, 1, 1)
                    outputs["r_distill_margin_gate/0"] = (
                        rdistill_pack["margin_gate_sample"].float().view(-1, 1, 1, 1)
                    )
                    outputs["r_distill_pose_photo_mean/0"] = rdistill_pack["pose_mean"].view(-1, 1, 1, 1)
                    outputs["r_distill_rfix_photo_mean/0"] = rdistill_pack["rfix_mean"].view(-1, 1, 1, 1)
                    outputs["r_distill_delta_rel/0"] = rdistill_pack["delta_rel"].view(-1, 1, 1, 1)
                    outputs["r_distill_gate/0"] = rdistill_pack["gate"].view(-1, 1, 1, 1)
                    outputs["r_distill_active/0"] = rdistill_pack["active"].float().view(-1, 1, 1, 1)
                    outputs["r_distill_valid/0"] = rdistill_pack["valid"].float().view(-1, 1, 1, 1)
                    outputs["r_distill_rot_mean/0"] = rdistill_pack["rot_mean"].view(-1, 1, 1, 1)
                    for pair_idx, frame_id in enumerate(frame_ids):
                        outputs[("r_distill_delta_rel_pair", frame_id, 0)] = (
                            rdistill_pack["pair_delta_rel"][:, pair_idx].view(-1, 1, 1, 1)
                        )
                        outputs[("r_distill_gate_pair", frame_id, 0)] = (
                            rdistill_pack["pair_gate"][:, pair_idx].view(-1, 1, 1, 1)
                        )
                        outputs[("r_distill_active_pair", frame_id, 0)] = (
                            rdistill_pack["pair_active"][:, pair_idx].float().view(-1, 1, 1, 1)
                        )
                        outputs[("r_distill_valid_pair", frame_id, 0)] = (
                            rdistill_pack["pair_valid"][:, pair_idx].float().view(-1, 1, 1, 1)
                        )
                        outputs[("r_distill_rot_pair", frame_id, 0)] = (
                            rdistill_pack["pair_rot"][:, pair_idx].view(-1, 1, 1, 1)
                        )

            if scale == 0:
                outputs["depth_sens_weight_enabled"] = float(1.0 if depth_sens_weight_enabled else 0.0)
                outputs["depth_sens_weight_start_epoch"] = float(depth_sens_weight_start_epoch)
                outputs["depth_sens_wpix_scale"] = float(depth_sens_wpix_scale)
                outputs["depth_sens_wimg_scale"] = float(depth_sens_wimg_scale)
                outputs["depth_sens_weight_min"] = float(depth_sens_weight_min)
                outputs["badscore_weight_enabled"] = float(
                    1.0 if (badscore_weight_enabled and use_badscore_weight) else 0.0
                )
                outputs["badscorelocal_weight_enabled"] = float(
                    1.0 if (badscore_weight_enabled and use_badscore_local_weight) else 0.0
                )
                outputs["badscore_start_epoch"] = float(badscore_start_epoch)
                outputs["badscore_alpha_r"] = float(badscore_alpha_r)
                outputs["badscore_alpha_o"] = float(badscore_alpha_o)
                outputs["badscore_beta_m"] = float(badscore_beta_m)
                outputs["badscore_wimg_scale"] = float(badscore_wimg_scale)
                outputs["badscore_weight_min"] = float(badscore_weight_min)
                outputs["badscore_norm_clip"] = float(badscore_norm_clip)
                outputs["badscore_local_scale"] = float(badscore_local_scale)
                outputs["badscore_local_weight_min"] = float(badscore_local_weight_min)
                outputs["badscore_local_gamma"] = float(badscore_local_gamma)
                outputs["badscore_local_z_thresh"] = float(badscore_local_z_thresh)
                if automask_identity_comp is not None and automask_reproj_comp is not None and automask_margin is not None:
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
                if derot_keep is not None:
                    outputs["derot_keep_union/0"] = derot_keep.float().detach()
                if keep_mask is not None:
                    outputs["final_loss_mask/0"] = keep_mask.float().detach()
                if loss_weight is not None:
                    outputs["final_loss_weight/0"] = loss_weight.detach()
                if depth_sens_weight_selected is not None:
                    loss_min_map = to_optimise.unsqueeze(1)
                    if keep_mask is None:
                        keep_map = torch.ones_like(loss_min_map)
                    else:
                        keep_map = keep_mask.to(dtype=loss_min_map.dtype)
                        if keep_map.dim() == 3:
                            keep_map = keep_map.unsqueeze(1)
                    outputs["depth_sens_weight_selected/0"] = depth_sens_weight_selected.detach()
                    if depth_sens_valid_selected is not None:
                        outputs["depth_sens_weight_valid_selected/0"] = (
                            depth_sens_valid_selected.float().detach()
                        )
                    outputs["depth_sens_loss_min/0"] = loss_min_map.detach()
                    outputs["depth_sens_loss_min_weighted/0"] = (
                        loss_min_map * depth_sens_weight_selected
                    ).detach()
                    outputs["depth_sens_loss_min_after_mask/0"] = (loss_min_map * keep_map).detach()
                    outputs["depth_sens_loss_min_weighted_after_mask/0"] = (
                        loss_min_map * depth_sens_weight_selected * keep_map
                    ).detach()
                    if depth_sens_valid_selected is not None:
                        valid_flat = depth_sens_weight_selected.detach()[depth_sens_valid_selected]
                    else:
                        valid_flat = depth_sens_weight_selected.detach().reshape(-1)
                    valid_flat = valid_flat[torch.isfinite(valid_flat)]
                    if valid_flat.numel() > 0:
                        losses["metrics/depth_sens_weight/scale0_weight_mean"] = float(valid_flat.mean().item())
                        if depth_sens_weight_available:
                            losses["metrics/depth_sens_weight/scale0_weight_p90"] = float(
                                torch.quantile(valid_flat, 0.9).item()
                            )
                    if depth_sens_img_weight_stack is not None:
                        img_flat = depth_sens_img_weight_stack.detach().reshape(-1)
                        img_flat = img_flat[torch.isfinite(img_flat)]
                        if img_flat.numel() > 0:
                            losses["metrics/depth_sens_weight/scale0_img_weight_mean"] = float(img_flat.mean().item())
                            losses["metrics/depth_sens_weight/scale0_img_weight_min"] = float(img_flat.min().item())
                if badscore_pack is not None:
                    badscore_valid = badscore_pack["valid"]
                    masked_R = torch.where(badscore_valid, badscore_pack["R_map"], torch.zeros_like(badscore_pack["R_map"]))
                    masked_O = torch.where(badscore_valid, badscore_pack["O_map"], torch.zeros_like(badscore_pack["O_map"]))
                    masked_r = torch.where(badscore_valid, badscore_pack["r_map"], torch.zeros_like(badscore_pack["r_map"]))
                    masked_o = torch.where(badscore_valid, badscore_pack["o_map"], torch.zeros_like(badscore_pack["o_map"]))
                    masked_b = torch.where(badscore_valid, badscore_pack["b_map"], torch.zeros_like(badscore_pack["b_map"]))
                    outputs["badscore_R_map/0"] = masked_R.detach()
                    outputs["badscore_O_map/0"] = masked_O.detach()
                    outputs["badscore_K_map/0"] = badscore_pack["K_map"].float().detach()
                    outputs["badscore_valid/0"] = badscore_valid.float().detach()
                    outputs["badscore_r_map/0"] = masked_r.detach()
                    outputs["badscore_o_map/0"] = masked_o.detach()
                    outputs["badscore_b_map/0"] = masked_b.detach()
                    outputs["badscore_winner_fid/0"] = badscore_pack["winner_fid"].detach()
                    outputs["badscore_B_img/0"] = badscore_pack["B_img"].view(-1, 1, 1, 1).detach()
                    outputs["badscore_B_hat/0"] = badscore_pack["B_hat"].view(-1, 1, 1, 1).detach()
                    loss_min_map = to_optimise.unsqueeze(1)
                    if keep_mask is None:
                        keep_map = torch.ones_like(loss_min_map)
                    else:
                        keep_map = keep_mask.to(dtype=loss_min_map.dtype)
                        if keep_map.dim() == 3:
                            keep_map = keep_map.unsqueeze(1)
                    outputs["badscore_loss_min/0"] = loss_min_map.detach()
                    outputs["badscore_loss_min_after_mask/0"] = (loss_min_map * keep_map).detach()
                    b_img_flat = badscore_pack["B_img"].detach().reshape(-1)
                    b_img_flat = b_img_flat[torch.isfinite(b_img_flat)]
                    if b_img_flat.numel() > 0:
                        losses["metrics/badscore/scale0_B_img_mean"] = float(b_img_flat.mean().item())
                        losses["metrics/badscore/scale0_B_img_max"] = float(b_img_flat.max().item())
                    b_map_flat = masked_b.detach()[badscore_valid]
                    b_map_flat = b_map_flat[torch.isfinite(b_map_flat)]
                    if b_map_flat.numel() > 0:
                        losses["metrics/badscore/scale0_b_map_mean"] = float(b_map_flat.mean().item())
                    outputs["badscore_w_img/0"] = badscore_pack["w_img"].view(-1, 1, 1, 1).detach()
                    outputs["badscore_weight_map/0"] = badscore_pack["w_img_map"].detach()
                    outputs["badscore_loss_min_weighted_after_mask/0"] = (
                        loss_min_map * keep_map * badscore_pack["w_img_map"]
                    ).detach()

                    w_img_flat = badscore_pack["w_img"].detach().reshape(-1)
                    w_img_flat = w_img_flat[torch.isfinite(w_img_flat)]
                    if w_img_flat.numel() > 0:
                        losses["metrics/badscore/scale0_w_img_mean"] = float(w_img_flat.mean().item())
                        losses["metrics/badscore/scale0_w_img_min"] = float(w_img_flat.min().item())
                    if badscore_local_pack is not None:
                        masked_margin_tilde = torch.where(
                            badscore_local_pack["valid"],
                            badscore_local_pack["margin_tilde_map"],
                            torch.zeros_like(badscore_local_pack["margin_tilde_map"]),
                        )
                        masked_margin_gate = torch.where(
                            badscore_local_pack["valid"],
                            badscore_local_pack["margin_gate_map"],
                            torch.zeros_like(badscore_local_pack["margin_gate_map"]),
                        )
                        masked_fragile = torch.where(
                            badscore_local_pack["valid"],
                            badscore_local_pack["fragile_map"],
                            torch.zeros_like(badscore_local_pack["fragile_map"]),
                        )
                        masked_w_pix = torch.where(
                            badscore_local_pack["valid"],
                            badscore_local_pack["w_pix"],
                            torch.ones_like(badscore_local_pack["w_pix"]),
                        )
                        outputs["badscorelocal_margin_tilde_map/0"] = masked_margin_tilde.detach()
                        outputs["badscorelocal_margin_gate_map/0"] = masked_margin_gate.detach()
                        outputs["badscorelocal_fragile_map/0"] = masked_fragile.detach()
                        outputs["badscorelocal_w_pix_map/0"] = masked_w_pix.detach()
                        # Backward-compatible aliases for older visualization keys.
                        outputs["badscorelocal_s_map/0"] = masked_fragile.detach()
                        outputs["badscorelocal_w_loc_map/0"] = masked_w_pix.detach()
                        outputs["badscorelocal_weight_valid/0"] = badscore_local_pack["valid"].float().detach()
                        outputs["badscorelocal_weight_map/0"] = badscore_local_pack["weight"].detach()
                        outputs["badscorelocal_margin_mean/0"] = badscore_local_pack["margin_mean"].view(1, 1, 1, 1).detach()
                        outputs["badscorelocal_margin_std/0"] = badscore_local_pack["margin_std"].view(1, 1, 1, 1).detach()
                        outputs["badscorelocal_weight_mean/0"] = badscore_local_pack["weight_mean"].view(-1, 1, 1, 1).detach()
                        outputs["badscorelocal_loss_min_after_mask/0"] = (loss_min_map * keep_map).detach()
                        outputs["badscorelocal_loss_min_weighted_after_mask/0"] = (
                            loss_min_map * keep_map * badscore_local_pack["weight"]
                        ).detach()

                        weight_flat = badscore_local_pack["weight"].detach()[badscore_local_pack["valid"]]
                        weight_flat = weight_flat[torch.isfinite(weight_flat)]
                        if weight_flat.numel() > 0:
                            losses["metrics/badscorelocal/scale0_weight_mean"] = float(weight_flat.mean().item())
                            losses["metrics/badscorelocal/scale0_weight_min"] = float(weight_flat.min().item())
                        fragile_flat = masked_fragile.detach()[badscore_local_pack["valid"]]
                        fragile_flat = fragile_flat[torch.isfinite(fragile_flat)]
                        if fragile_flat.numel() > 0:
                            losses["metrics/badscorelocal/scale0_fragile_mean"] = float(fragile_flat.mean().item())
                        margin_tilde_flat = masked_margin_tilde.detach()[badscore_local_pack["valid"]]
                        margin_tilde_flat = margin_tilde_flat[torch.isfinite(margin_tilde_flat)]
                        if margin_tilde_flat.numel() > 0:
                            losses["metrics/badscorelocal/scale0_margin_tilde_mean"] = float(
                                margin_tilde_flat.mean().item()
                            )
                        margin_gate_flat = masked_margin_gate.detach()[badscore_local_pack["valid"]]
                        margin_gate_flat = margin_gate_flat[torch.isfinite(margin_gate_flat)]
                        if margin_gate_flat.numel() > 0:
                            losses["metrics/badscorelocal/scale0_margin_gate_mean"] = float(
                                margin_gate_flat.mean().item()
                            )
                        w_pix_flat = badscore_local_pack["w_pix"].detach()[badscore_local_pack["valid"]]
                        w_pix_flat = w_pix_flat[torch.isfinite(w_pix_flat)]
                        if w_pix_flat.numel() > 0:
                            losses["metrics/badscorelocal/scale0_w_pix_mean"] = float(w_pix_flat.mean().item())
                if derot_weight_union is not None:
                    outputs["derot_weight_union/0"] = derot_weight_union.detach()
                if reproj_argmin is not None:
                    for idx, fid in enumerate(frame_ids):
                        outputs[("reproj_loss_fid", fid, 0)] = reprojection_loss[:, idx:idx + 1].detach()

                    if reprojection_loss.shape[1] > 1:
                        source_map = torch.zeros_like(reproj_argmin, dtype=torch.float32)
                        for idx, fid in enumerate(frame_ids):
                            if isinstance(fid, numbers.Integral):
                                source_map[reproj_argmin == idx] = float(int(fid))
                        outputs["reproj_loss_source_fid/0"] = source_map.unsqueeze(1).detach()

                    if identity_selection is not None and combined_argmin is not None:
                        id_ch = identity_loss_tensor.shape[1]
                        reproj_winner = combined_argmin - id_ch
                        reproj_selected = identity_selection > 0.5
                        source_map = torch.zeros_like(combined_argmin, dtype=torch.float32)
                        for idx, fid in enumerate(frame_ids):
                            keep_fid = reproj_selected & (reproj_winner == idx)
                            outputs[("automask_keep_fid", fid, 0)] = keep_fid.unsqueeze(1).float().detach()
                            if isinstance(fid, numbers.Integral):
                                source_map[keep_fid] = float(int(fid))
                        outputs["automask_source_fid/0"] = source_map.unsqueeze(1).detach()

                if use_derot_sigmoid_weight and loss_weight is not None and reproj_argmin is not None:
                    if keep_mask is None:
                        final_keep_bool = torch.ones_like(reproj_argmin, dtype=torch.bool)
                    else:
                        final_keep_bool = (keep_mask > 0.5)
                        if final_keep_bool.dim() == 4:
                            final_keep_bool = final_keep_bool[:, 0]

                    source_map = torch.zeros_like(reproj_argmin, dtype=torch.float32)
                    for idx, fid in enumerate(frame_ids):
                        fid_keep = final_keep_bool & (reproj_argmin == idx)
                        outputs[("final_loss_weight_fid", fid, 0)] = (
                            loss_weight * fid_keep.unsqueeze(1).to(dtype=loss_weight.dtype)
                        ).detach()
                        if isinstance(fid, numbers.Integral):
                            source_map[fid_keep] = float(int(fid))
                    outputs["final_loss_weight_source_fid/0"] = source_map.unsqueeze(1).detach()

            if use_hr_mask:
                loss = loss + self._apply_high_res_mask(
                    to_optimise,
                    reprojection_losses,
                    identity_selection,
                    getattr(self.opt, "posegt_hr_percentile", 90.0),
                    getattr(self.opt, "posegt_hr_scope", "mask"),
                    extra_keep=final_keep_seed,
                )
            else:
                if loss_weight is not None:
                    loss += self._masked_mean(to_optimise, loss_weight)
                else:
                    loss += self._masked_mean(to_optimise, keep_mask)

            if use_posegt:
                posegt_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred_posegt = outputs.get(("irw_img", frame_id, 0), None)
                    if pred_posegt is None:
                        continue
                    posegt_loss = self.compute_posegt_reprojection_loss(pred_posegt, target)
                    posegt_losses.append(posegt_loss)

                if posegt_losses:
                    posegt_losses = torch.cat(posegt_losses, 1)
                    if self.opt.avg_reprojection:
                        posegt_loss = posegt_losses.mean(1, keepdim=True)
                    else:
                        posegt_loss = posegt_losses

                    if not self.opt.disable_automasking and identity_loss_tensor is not None:
                        combined_posegt = torch.cat((identity_loss_tensor, posegt_loss), dim=1)
                    else:
                        combined_posegt = posegt_loss

                    posegt_to_optimise = (
                        combined_posegt if combined_posegt.shape[1] == 1 else torch.min(combined_posegt, dim=1)[0]
                    )
                    posegt_identity_selection = None
                    if not self.opt.disable_automasking and identity_loss_tensor is not None:
                        posegt_identity_selection = (
                            torch.min(combined_posegt, dim=1)[1] > identity_loss_tensor.shape[1] - 1
                        ).float()
                    posegt_keep = self._combine_keep_mask(final_keep_seed, posegt_identity_selection)
                    if use_hr_mask:
                        posegt_weighted = (
                            self._apply_high_res_mask(
                                posegt_to_optimise,
                                posegt_losses,
                                posegt_identity_selection,
                                getattr(self.opt, "posegt_hr_percentile", 90.0),
                                getattr(self.opt, "posegt_hr_scope", "mask"),
                                extra_keep=final_keep_seed,
                            )
                            * float(posegt_weight)
                        )
                    else:
                        if use_derot_sigmoid_weight:
                            if posegt_keep is None:
                                posegt_loss_weight = torch.ones_like(
                                    posegt_to_optimise, dtype=posegt_to_optimise.dtype
                                ).unsqueeze(1)
                            else:
                                posegt_loss_weight = posegt_keep.to(dtype=posegt_to_optimise.dtype)
                                if posegt_loss_weight.dim() == 3:
                                    posegt_loss_weight = posegt_loss_weight.unsqueeze(1)
                            posegt_reproj_argmin = None
                            if (
                                (not self.opt.avg_reprojection)
                                and posegt_loss.dim() == 4
                                and posegt_loss.shape[1] == len(frame_ids)
                            ):
                                posegt_reproj_argmin = torch.argmin(posegt_loss, dim=1)
                            posegt_derot_weight = self._build_fid_aligned_derot_weight(
                                derot_weights=derot_weights,
                                derot_weight_fids=derot_weight_fids,
                                frame_ids=frame_ids,
                                reproj_argmin=posegt_reproj_argmin,
                                derot_weight_union=derot_weight_union,
                            )
                            if posegt_derot_weight is not None:
                                posegt_loss_weight = posegt_loss_weight * posegt_derot_weight
                            posegt_weighted = (
                                self._masked_mean(posegt_to_optimise, posegt_loss_weight) * float(posegt_weight)
                            )
                        else:
                            posegt_weighted = self._masked_mean(posegt_to_optimise, posegt_keep) * float(posegt_weight)
                    loss = loss + posegt_weighted
                    posegt_total = posegt_total + posegt_weighted

            if use_teacher_photo:
                teacher_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred_teacher = outputs.get(("color_teacher", frame_id, scale), None)
                    if pred_teacher is None:
                        continue
                    reproj_teacher = self.compute_reprojection_loss(pred_teacher, target)

                    if self.opt.distorted_mask:
                        valid_mask_teacher = outputs.get(("distorted_mask_teacher", frame_id, scale), None)
                        if valid_mask_teacher is not None:
                            reproj_teacher = reproj_teacher * valid_mask_teacher + (1.0 - valid_mask_teacher) * 1e5

                    teacher_reprojection_losses.append(reproj_teacher)

                if teacher_reprojection_losses:
                    teacher_reprojection_losses = torch.cat(teacher_reprojection_losses, 1)
                    if self.opt.avg_reprojection:
                        teacher_reprojection_loss = teacher_reprojection_losses.mean(1, keepdim=True)
                    else:
                        teacher_reprojection_loss = teacher_reprojection_losses

                    if not self.opt.disable_automasking and identity_loss_tensor is not None:
                        combined_teacher = torch.cat((identity_loss_tensor, teacher_reprojection_loss), dim=1)
                    else:
                        combined_teacher = teacher_reprojection_loss

                    teacher_to_optimise = (
                        combined_teacher if combined_teacher.shape[1] == 1 else torch.min(combined_teacher, dim=1)[0]
                    )
                    teacher_photo_loss = teacher_photo_loss + teacher_to_optimise.mean()
                    teacher_photo_scales += 1

                    if scale == 0:
                        metrics = self._collect_debug_metrics(
                            scale,
                            teacher_reprojection_losses,
                            identity_loss_tensor,
                            combined_teacher,
                            teacher_to_optimise,
                            frame_offsets,
                            prefix="teacher_",
                        )
                        losses.update(metrics)

            smooth_source = _smoothness_source_map(scale, disp)
            mean_disp = smooth_source.mean(2, True).mean(3, True)
            norm_disp = smooth_source / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

            if scale == 0:
                if use_hr_mask:
                    masks = self._compute_hrmask_masks(
                        reprojection_losses,
                        identity_selection,
                        getattr(self.opt, "posegt_hr_percentile", 90.0),
                        getattr(self.opt, "posegt_hr_scope", "mask"),
                        extra_keep=final_keep_seed,
                    )
                    finite = masks["finite"]
                    base_keep = masks["base_keep"]
                    high_res = masks["high_res"]
                    hrmask_bad = masks["hrmask_bad"]
                    final_keep = masks["final_keep"]

                    finite_count = float(finite.sum().item())
                    high_count = float(high_res.sum().item())
                    final_keep_count = float(final_keep.sum().item())
                    bad_keep_num = float((high_res & base_keep & (~hrmask_bad)).sum().item())

                    losses["metrics/hrmask/scale0_final_keep_ratio"] = (
                        final_keep_count / finite_count if finite_count > 0 else float("nan")
                    )
                    losses["metrics/hrmask/scale0_final_bad_keep_ratio"] = (
                        bad_keep_num / high_count if high_count > 0 else float("nan")
                    )
                if identity_selection is not None:
                    metrics = self._collect_automask_metrics(
                        scale,
                        reprojection_losses,
                        identity_selection,
                        external_keep=final_keep_seed,
                    )
                    losses.update(metrics)
                if use_derot_hardmask:
                    if derot_masks_raw:
                        derot_keep_raw = torch.stack(derot_masks_raw, dim=0).any(dim=0)
                        losses["metrics/derot/scale0_keep_ratio"] = float(derot_keep_raw.float().mean().item())
                    elif derot_keep is not None:
                        losses["metrics/derot/scale0_keep_ratio"] = float(derot_keep.float().mean().item())
                    if derot_parallax_vals:
                        derot_concat = torch.cat([v.reshape(-1) for v in derot_parallax_vals], dim=0)
                        finite = torch.isfinite(derot_concat)
                        if finite.any():
                            derot_valid = derot_concat[finite]
                            losses["metrics/derot/scale0_p50"] = float(torch.quantile(derot_valid, 0.50).item())
                            losses["metrics/derot/scale0_p90"] = float(torch.quantile(derot_valid, 0.90).item())
                            losses["metrics/derot/scale0_p95"] = float(torch.quantile(derot_valid, 0.95).item())
                elif use_derot_sigmoid_weight:
                    if derot_masks_raw:
                        derot_keep_raw = torch.stack(derot_masks_raw, dim=0).any(dim=0)
                        losses["metrics/derot/scale0_keep_ratio"] = float(derot_keep_raw.float().mean().item())
                    if derot_weight_union is not None:
                        losses["metrics/derot/scale0_weight_mean"] = float(derot_weight_union.mean().item())
                    if loss_weight is not None:
                        losses["metrics/derot/scale0_final_weight_mean"] = float(loss_weight.mean().item())
                    if derot_parallax_vals:
                        derot_concat = torch.cat([v.reshape(-1) for v in derot_parallax_vals], dim=0)
                        finite = torch.isfinite(derot_concat)
                        if finite.any():
                            derot_valid = derot_concat[finite]
                            losses["metrics/derot/scale0_p50"] = float(torch.quantile(derot_valid, 0.50).item())
                            losses["metrics/derot/scale0_p90"] = float(torch.quantile(derot_valid, 0.90).item())
                            losses["metrics/derot/scale0_p95"] = float(torch.quantile(derot_valid, 0.95).item())
                metrics = self._collect_debug_metrics(
                    scale,
                    reprojection_losses,
                    identity_loss_tensor,
                    combined,
                    to_optimise,
                    frame_offsets,
                )
                losses.update(metrics)

        if use_gasmono_selfpp and torch.is_tensor(disp_best) and torch.is_tensor(reproj_loss_min):
            selfpp_raw = self._compute_gasmono_selfpp_loss(disp_best, outputs)
            if selfpp_raw is not None:
                warmup_epochs = max(1, int(getattr(self.opt, "gasmono_selfpp_warmup_epochs", 20)))
                full_weight = float(
                    getattr(self.opt, "wpp", getattr(self.opt, "gasmono_selfpp_weight", 0.1))
                )
                cur_epoch = int(getattr(self, "global_epoch", 0))
                if cur_epoch < warmup_epochs:
                    selfpp_weight = full_weight * float(cur_epoch) / float(warmup_epochs)
                else:
                    selfpp_weight = full_weight
                selfpp_weighted = selfpp_raw * selfpp_weight
                total_loss = total_loss + selfpp_weighted
                losses["loss/gasmono_selfpp_raw"] = selfpp_raw
                losses["loss/gasmono_selfpp_weight"] = float(selfpp_weight)
                losses["loss/gasmono_selfpp"] = selfpp_weighted
            self.bestpp = {
                "disp": disp_best.detach(),
                "error": reproj_loss_min.detach(),
            }

        total_loss /= len(self.opt.scales)
        if use_teacher_photo:
            losses["loss/teacher_photo_weight"] = float(teacher_photo_weight)
            if teacher_photo_scales > 0:
                teacher_photo_loss = teacher_photo_loss / float(len(self.opt.scales))
                losses["loss/teacher_photo_raw"] = teacher_photo_loss
                teacher_photo_weighted = teacher_photo_loss * float(teacher_photo_weight)
                losses["loss/teacher_photo"] = teacher_photo_weighted
                total_loss = total_loss + teacher_photo_weighted
        if use_posegt:
            losses["loss/posegt_weight"] = float(posegt_weight)
            losses["loss/posegt"] = posegt_total / float(len(self.opt.scales))
        if rdistill_total is not None:
            losses["loss/r_distill"] = rdistill_total
            total_loss = total_loss + rdistill_total
        if use_pose_teacher:
            teacher_loss, rot_loss, trans_loss, distill_weight = self._compute_pose_teacher_loss(inputs, outputs)
            if distill_weight is not None:
                losses["loss/pose_teacher_weight"] = float(distill_weight)
            if teacher_loss is not None:
                losses["loss/pose_teacher"] = teacher_loss
                if rot_loss is not None:
                    losses["loss/pose_teacher_rot"] = rot_loss
                if trans_loss is not None:
                    losses["loss/pose_teacher_trans"] = trans_loss
                total_loss = total_loss + teacher_loss
        losses["loss"] = total_loss
        return losses

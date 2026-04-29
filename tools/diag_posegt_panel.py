#!/usr/bin/env python3
"""
Export PoseGT qualitative panels:
  source -> irw_img -> final_warp -> target + error maps.

This script runs the MD2_VGGT_PoseGT forward path and saves image panels for
quick visual inspection on train/val split.

Example:
  python tools/diag_posegt_panel.py \
    --dataset UAVid_TriDataset \
    --split UAVid2020_Germany \
    --data_path /path/to/UAVid/Germany \
    --triplet_root /path/to/uavid_triplets \
    --weights_folder /path/to/models/weights_XX \
    --eval_split train \
    --max_images 200 \
    --output_dir results/diag_posegt_panel
"""
import argparse
import copy
import json
import numbers
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import matplotlib.cm as mpl_cm
except Exception:
    mpl_cm = None

# Disable W&B side effects in diag mode
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from options import MonodepthOptions
from trainer import Trainer
from trainer_init.data_init import _infer_norm_cfg, parse_split_pairs, _infer_uavid_k_region
from trainer_init.geometry_init import init_geometry
from trainer_init.loss_init import init_losses
from trainer_init.model_init import get_forward_handler, init_models
from methods.datasets import UAVTripletJsonDataset, UAVid2020TripletJsonDataset
from layers import BackprojectDepth, Project3D
from utils import readlines

POSEGT_METHODS = {"MD2_VGGT_PoseGT", "MD2_VGGT_PoseGT_HRMask", "MD2_VGGT_PoseGT_Mask"}


def load_model_weights_only(runner: "EvalRunner", folder: str, model_names=None) -> None:
    """Load model weights only (no optimizer state)."""
    if not folder:
        raise ValueError("weights_folder is required")
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"weights folder not found: {folder}")

    if model_names is None:
        model_names = list(runner.models.keys())

    for n in model_names:
        if n not in runner.models:
            continue
        path = os.path.join(folder, f"{n}.pth")
        if not os.path.isfile(path):
            print(f"[warn] weight file not found: {path}, skip")
            continue
        model_dict = runner.models[n].state_dict()
        pretrained_dict = torch.load(path, map_location="cpu")
        # Drop metadata keys if present
        pretrained_dict.pop("height", None)
        pretrained_dict.pop("width", None)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        runner.models[n].load_state_dict(model_dict)


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def _safe_name(s: str) -> str:
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("-")
    name = "".join(out).strip("-")
    return name or "default"


def _ensure_geometry_for_batch(runner: "EvalRunner", batch_size: int) -> None:
    for scale in runner.opt.scales:
        h = runner.opt.height // (2 ** scale)
        w = runner.opt.width // (2 ** scale)
        module = runner.backproject_depth.get(scale, None)
        if module is None or getattr(module, "batch_size", None) != batch_size:
            runner.backproject_depth[scale] = BackprojectDepth(batch_size, h, w).to(runner.device)
            runner.project_3d[scale] = Project3D(batch_size, h, w).to(runner.device)


def _filter_samples_for_diag(
    dataset,
    data_root: str,
    frame_ids: List[Any],
) -> None:
    if not dataset.samples:
        return
    keep = []
    missing_img = 0
    missing_pose = 0
    for rec in dataset.samples:
        ok = True
        for key in ("center", "prev", "next"):
            path = rec.get(key)
            if path and not os.path.isabs(path):
                path = os.path.join(data_root, path)
            if not path or not os.path.exists(path):
                ok = False
                missing_img += 1
                break
        if ok:
            pose = rec.get("_external_pose")
            if not pose:
                ok = False
                missing_pose += 1
            else:
                for fid in frame_ids[1:]:
                    if fid == "s":
                        continue
                    if isinstance(fid, numbers.Integral) and fid < 0:
                        if pose.get("prev_to_center") is None:
                            ok = False
                            missing_pose += 1
                            break
                    elif isinstance(fid, numbers.Integral) and fid > 0:
                        if pose.get("next_to_center") is None:
                            ok = False
                            missing_pose += 1
                            break
        if ok:
            keep.append(rec)
    if missing_img:
        print(f"[diag] filtered {missing_img} samples without images under {data_root}")
    if missing_pose:
        print(f"[diag] filtered {missing_pose} samples without external pose under {data_root}")
    dataset.samples = keep


def _build_loader(
    opt,
    split: str,
    use_split_files: bool,
    use_train_aug: bool,
) -> DataLoader:
    assert split in {"train", "val"}
    dataset_kwargs = _infer_norm_cfg(opt)
    manifest_glob = "triplets_uniform_t.jsonl" if "UniformT" in str(opt.methods) else "triplets.jsonl"
    root = "Train" if split == "train" else "Validation"

    if str(opt.dataset) == "UAVid_TriDataset":
        k_region = _infer_uavid_k_region(opt)
        dataset = UAVid2020TripletJsonDataset(
            data_path=os.path.join(opt.data_path, root),
            triplet_root=opt.triplet_root,
            height=opt.height,
            width=opt.width,
            frame_idxs=opt.frame_ids,
            num_scales=len(opt.scales),
            is_train=bool(use_train_aug and split == "train"),
            img_ext=".png" if opt.png else ".jpg",
            allow_flip=bool(use_train_aug and getattr(opt, "enable_flip", False)),
            use_triplet_pose=True,
            triplet_manifest_glob=manifest_glob,
            k_region=k_region,
            **dataset_kwargs,
        )
    elif str(opt.dataset) == "UAVula_TriDataset":
        dataset = UAVTripletJsonDataset(
            data_path=os.path.join(opt.data_path, root),
            triplet_root=opt.triplet_root,
            height=opt.height,
            width=opt.width,
            frame_idxs=opt.frame_ids,
            num_scales=len(opt.scales),
            is_train=bool(use_train_aug and split == "train"),
            img_ext=".png" if opt.png else ".jpg",
            allow_flip=bool(use_train_aug and getattr(opt, "enable_flip", False)),
            use_triplet_pose=True,
            triplet_manifest_glob=manifest_glob,
            **dataset_kwargs,
        )
    else:
        raise ValueError("Only triplet datasets are supported: UAVula_TriDataset / UAVid_TriDataset")

    if use_split_files:
        split_file = os.path.join(PROJECT_ROOT, "methods", "splits", opt.split, f"{split}_files.txt")
        if os.path.isfile(split_file):
            lines = readlines(split_file)
            pairs = parse_split_pairs(lines)
            if pairs:
                dataset.samples = [s for s in dataset.samples if (s.get("seq"), s.get("center_idx")) in pairs]
        else:
            print(f"[warn] split file not found: {split_file} (skip filtering)")

    _filter_samples_for_diag(
        dataset,
        data_root=os.path.join(opt.data_path, root),
        frame_ids=opt.frame_ids,
    )

    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(int(getattr(opt, "seed", 42))),
    )
    return loader


class EvalRunner(Trainer):
    """Lightweight Trainer: init models/losses/geometry only."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.models: Dict[str, torch.nn.Module] = {}
        self.parameters_to_train: List[torch.nn.Parameter] = []
        self.training = False

        assert self.opt.height % 32 == 0
        assert self.opt.width % 32 == 0
        assert self.opt.frame_ids[0] == 0

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        self.use_pose_net = True
        self.use_posegt = str(getattr(self.opt, "methods", "")) in POSEGT_METHODS

        init_models(self)
        init_losses(self)
        init_geometry(self)

        self.scale_aligner = None
        self._scale_align_monitor = {"train": {"attempts": 0, "success": 0}, "val": {"attempts": 0, "success": 0}}
        self._scale_align_warned = {"train": False, "val": False}
        self.collect_debug_metrics = False
        self.set_eval()


def move_batch_to_device(batch: Dict[Any, Any], device: torch.device) -> Dict[Any, Any]:
    out: Dict[Any, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _extract_meta(inputs: Dict[Any, Any]) -> Tuple[List[str], List[Optional[int]]]:
    seqs = inputs.get("seq", None)
    if isinstance(seqs, (list, tuple)):
        seq_list = [str(s) for s in seqs]
    else:
        seq_list = [str(seqs)] if seqs is not None else []
    center_idx = inputs.get("center_idx", None)
    if torch.is_tensor(center_idx):
        idx_list = [int(x.item()) for x in center_idx]
    elif isinstance(center_idx, (list, tuple)):
        idx_list = [int(x) if x is not None else None for x in center_idx]
    else:
        idx_list = [int(center_idx)] if center_idx is not None else []
    return seq_list, idx_list


def _resize_tensor_to_hw(t: torch.Tensor, h: int, w: int, mode: str) -> torch.Tensor:
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
        squeeze_mode = "hw"
    elif t.dim() == 3:
        t = t.unsqueeze(0)
        squeeze_mode = "chw"
    else:
        squeeze_mode = None
    if mode in {"bilinear", "bicubic"}:
        out = F.interpolate(t.float(), size=(h, w), mode=mode, align_corners=False)
    else:
        out = F.interpolate(t.float(), size=(h, w), mode=mode)
    if squeeze_mode == "hw":
        return out[0, 0]
    if squeeze_mode == "chw":
        return out[0]
    return out


def _tensor_to_uint8_rgb(img_t: torch.Tensor) -> np.ndarray:
    if img_t.dim() == 4:
        img_t = img_t[0]
    if img_t.shape[0] == 1:
        img_t = img_t.repeat(3, 1, 1)
    arr = img_t.detach().float().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _apply_colormap(norm_arr: np.ndarray, cmap: str) -> np.ndarray:
    if mpl_cm is None:
        gray = (norm_arr * 255.0 + 0.5).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    cm = mpl_cm.get_cmap(cmap)
    colored = cm(norm_arr)[..., :3]
    return (colored * 255.0 + 0.5).astype(np.uint8)


def _normalize_err_pair(
    err_a: torch.Tensor,
    err_b: torch.Tensor,
    qmin: float,
    qmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    vals = torch.cat([err_a.reshape(-1), err_b.reshape(-1)], dim=0)
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        z = np.zeros(err_a.shape, dtype=np.float32)
        return z, z
    qa = max(0.0, min(100.0, float(qmin))) / 100.0
    qb = max(qa + 1e-3, min(100.0, float(qmax))) / 100.0
    vmin = torch.quantile(vals, qa)
    vmax = torch.quantile(vals, qb)
    denom = (vmax - vmin).clamp_min(1e-6)
    n1 = ((err_a - vmin) / denom).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
    n2 = ((err_b - vmin) / denom).clamp(0.0, 1.0).detach().cpu().numpy().astype(np.float32)
    return n1, n2


def _concat_with_gap(images: List[np.ndarray], gap: int = 4) -> np.ndarray:
    if not images:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    h = images[0].shape[0]
    gap_img = np.full((h, gap, 3), 255, dtype=np.uint8)
    cols = []
    for i, im in enumerate(images):
        cols.append(im)
        if i != len(images) - 1:
            cols.append(gap_img)
    return np.concatenate(cols, axis=1)


def _compute_loss_contrib_maps(
    runner: "EvalRunner",
    inputs: Dict[Any, Any],
    outputs: Dict[Any, Any],
    scale: int = 0,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Compute per-frame pixel-wise contribution maps that actually enter the
    photometric loss aggregation for:
      - main branch (final_warp)
      - posegt branch (irw_img)
    Returns:
      final_maps[fid]: [B,H,W]
      posegt_maps[fid]: [B,H,W]
    """
    opt = runner.opt
    frame_ids = [fid for fid in opt.frame_ids[1:] if fid != "s"]
    target = inputs[("color", 0, 0)]
    device = target.device

    main_losses = []
    posegt_losses = []
    identity_losses = []
    for fid in frame_ids:
        pred_main = outputs[("color", fid, scale)]
        main_losses.append(runner.loss.compute_reprojection_loss(pred_main, target))

        pred_posegt = outputs.get(("irw_img", fid, 0), None)
        if pred_posegt is None:
            posegt_losses.append(torch.zeros_like(main_losses[-1]))
        else:
            posegt_losses.append(runner.loss.compute_posegt_reprojection_loss(pred_posegt, target))

        pred_id = inputs[("color", fid, 0)]
        identity_losses.append(runner.loss.compute_reprojection_loss(pred_id, target))

    main_losses = torch.cat(main_losses, dim=1)        # [B,N,H,W]
    posegt_losses = torch.cat(posegt_losses, dim=1)    # [B,N,H,W]
    identity_losses = torch.cat(identity_losses, dim=1)  # [B,N,H,W]

    # Reproduce training logic: one shared identity tensor (with tiny noise)
    # is used by both main and posegt branches when automasking is enabled.
    identity_noise = identity_losses
    if not opt.disable_automasking:
        identity_noise = identity_noise + torch.randn(identity_noise.shape, device=device) * 1e-5

    final_maps: Dict[int, torch.Tensor] = {}
    posegt_maps: Dict[int, torch.Tensor] = {}

    if opt.avg_reprojection:
        # avg_reprojection has no per-frame winner; use shared map for all frame_ids.
        main_avg = main_losses.mean(1, keepdim=True)      # [B,1,H,W]
        posegt_avg = posegt_losses.mean(1, keepdim=True)  # [B,1,H,W]

        if not opt.disable_automasking:
            id_avg = identity_noise.mean(1, keepdim=True)
            comb_main = torch.cat((id_avg, main_avg), dim=1)
            main_to_opt, main_idx = torch.min(comb_main, dim=1)
            main_id_sel = (main_idx > id_avg.shape[1] - 1).float().unsqueeze(1)

            comb_posegt = torch.cat((id_avg, posegt_avg), dim=1)
            posegt_to_opt, posegt_idx = torch.min(comb_posegt, dim=1)
            posegt_id_sel = (posegt_idx > id_avg.shape[1] - 1).float().unsqueeze(1)
        else:
            main_to_opt = main_avg[:, 0]
            posegt_to_opt = posegt_avg[:, 0]
            main_id_sel = None
            posegt_id_sel = None

        external_keep = runner.loss._get_external_keep_mask(
            inputs, scale, main_to_opt, main_to_opt.device, main_to_opt.dtype
        )
        main_keep = runner.loss._combine_keep_mask(external_keep, main_id_sel)
        posegt_keep = runner.loss._combine_keep_mask(external_keep, posegt_id_sel)

        if main_keep is None:
            main_keep_f = torch.ones_like(main_to_opt)
        else:
            main_keep_f = main_keep.squeeze(1).to(dtype=main_to_opt.dtype)
        if posegt_keep is None:
            posegt_keep_f = torch.ones_like(posegt_to_opt)
        else:
            posegt_keep_f = posegt_keep.squeeze(1).to(dtype=posegt_to_opt.dtype)

        main_shared = main_to_opt * main_keep_f
        posegt_shared = posegt_to_opt * posegt_keep_f
        for fid in frame_ids:
            final_maps[fid] = main_shared
            posegt_maps[fid] = posegt_shared
        return final_maps, posegt_maps

    # avg_reprojection=False (normal case): per-frame winner is available.
    if not opt.disable_automasking:
        comb_main = torch.cat((identity_noise, main_losses), dim=1)   # [B,2N,H,W]
        _, main_idx = torch.min(comb_main, dim=1)                     # [B,H,W]
        main_id_sel = (main_idx > identity_noise.shape[1] - 1).float().unsqueeze(1)
        main_offset = identity_noise.shape[1]

        comb_posegt = torch.cat((identity_noise, posegt_losses), dim=1)
        _, posegt_idx = torch.min(comb_posegt, dim=1)
        posegt_id_sel = (posegt_idx > identity_noise.shape[1] - 1).float().unsqueeze(1)
        posegt_offset = identity_noise.shape[1]
    else:
        _, main_idx = torch.min(main_losses, dim=1)
        _, posegt_idx = torch.min(posegt_losses, dim=1)
        main_id_sel = None
        posegt_id_sel = None
        main_offset = 0
        posegt_offset = 0

    external_keep = runner.loss._get_external_keep_mask(
        inputs, scale, main_losses[:, 0], main_losses.device, main_losses.dtype
    )
    main_keep = runner.loss._combine_keep_mask(external_keep, main_id_sel)
    posegt_keep = runner.loss._combine_keep_mask(external_keep, posegt_id_sel)

    if main_keep is None:
        main_keep_f = torch.ones_like(main_idx, dtype=main_losses.dtype)
    else:
        main_keep_f = main_keep.squeeze(1).to(dtype=main_losses.dtype)
    if posegt_keep is None:
        posegt_keep_f = torch.ones_like(posegt_idx, dtype=posegt_losses.dtype)
    else:
        posegt_keep_f = posegt_keep.squeeze(1).to(dtype=posegt_losses.dtype)

    for i, fid in enumerate(frame_ids):
        main_chosen = (main_idx == (i + main_offset)).to(dtype=main_losses.dtype)
        posegt_chosen = (posegt_idx == (i + posegt_offset)).to(dtype=posegt_losses.dtype)
        final_maps[fid] = main_losses[:, i] * main_chosen * main_keep_f
        posegt_maps[fid] = posegt_losses[:, i] * posegt_chosen * posegt_keep_f

    return final_maps, posegt_maps


def run_posegt_panel_diag(
    opt,
    split: str,
    output_dir: Path,
    max_images: int,
    output_height: int,
    output_width: int,
    error_qmin: float,
    error_qmax: float,
    error_cmap: str,
    error_mode: str,
) -> Dict[str, Any]:
    if Image is None:
        raise RuntimeError("PIL not available, please install pillow")
    method_name = str(getattr(opt, "methods", ""))
    is_posegt_method = method_name in POSEGT_METHODS

    loader = _build_loader(
        opt,
        split=split,
        use_split_files=not bool(getattr(opt, "ignore_split", False)),
        use_train_aug=bool(getattr(opt, "use_train_aug", False)),
    )
    runner = EvalRunner(opt)
    load_model_weights_only(runner, opt.weights_folder)
    handler = get_forward_handler(opt.methods)

    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)
    stats_path = split_out / "stats.jsonl"
    stats_desc_path = split_out / "stats_desc.json"

    panel_columns = (
        ["source", "aux_image", "final_warp", "target", "err_aux_target", "err_final_target"]
        if is_posegt_method
        else ["source", "final_warp", "target", "err_final_target"]
    )
    descriptions = {
        "panel_columns": {
            "source": "source frame image I_s (raw source, frame_id in filename)",
            "aux_image": "PoseGT methods only: irw_img (pre-warp image)",
            "final_warp": "final warped image used by photometric supervision after PoseNet residual pose",
            "target": "target frame image I_t (center frame)",
            "err_aux_target": "PoseGT methods only: error map of aux_image branch",
            "err_final_target": "error map of final_warp branch; meaning depends on error_mode",
        },
        "error_mode": {
            "rgb_mae": "heatmap value = mean_c(|pred - target|) per pixel",
            "loss_contrib": "heatmap value = per-pixel loss contribution after min-reproj + automask (+ external mask if enabled)",
        },
        "formula_notes": {
            "mae_final": "mae_final = mean_{all pixels}( mean_c(|final_warp - target|) ); independent of error_mode",
            "mae_irw": "mae_irw = mean_{all pixels}( mean_c(|irw_img - target|) ); PoseGT methods only",
            "metric_final_rgb_mae": "if error_mode=rgb_mae: metric_final == mae_final (numerically equal up to float precision)",
            "metric_final_loss_contrib": "if error_mode=loss_contrib: metric_final = mean_{all pixels}(loss contribution map), not equal to mae_final",
            "metric_vs_loss": "metric_final is not the training total loss scalar; it is a per-image map average for visualization",
        },
        "stats_fields": {
            "split": "evaluated split",
            "seq": "sequence name from dataset",
            "center_idx": "center frame index",
            "frame_id": "source frame id relative to center (e.g. -1 or +1)",
            "panel_path": "saved panel image path",
            "panel_layout": "ordered column names in the saved panel",
            "aux_image_kind": "PoseGT methods only: image type of aux_image (normally irw_img)",
            "error_mode": "error-map mode used for panel",
            "mae_final": "mean absolute reconstruction error of final_warp vs target (formula in formula_notes.mae_final)",
            "metric_final": "mean value of err_final_target map (mode-dependent; see formula_notes)",
            "mae_irw": "PoseGT methods only: mean absolute reconstruction error of irw_img vs target",
            "mae_delta": "PoseGT methods only: mae_irw - mae_final (positive means final_warp is better)",
            "metric_irw": "PoseGT methods only: mean value of err_aux_target map",
            "metric_delta": "PoseGT methods only: metric_irw - metric_final",
        },
        "summary_fields": {
            "mae_final_mean": "dataset mean of mae_final over saved panels",
            "metric_final_mean": "dataset mean of metric_final over saved panels",
            "mae_irw_mean": "PoseGT methods only: dataset mean of mae_irw over saved panels",
            "mae_delta_mean": "PoseGT methods only: dataset mean of mae_delta; >0 means final_warp improves on average",
            "metric_irw_mean": "PoseGT methods only: dataset mean of metric_irw over saved panels",
            "metric_delta_mean": "PoseGT methods only: dataset mean of metric_delta; interpretation depends on error_mode",
            "num_center_images": "number of center images exported",
            "num_panels": "number of saved panel files",
            "error_qmin": "lower percentile for error-map normalization",
            "error_qmax": "upper percentile for error-map normalization",
            "error_cmap": "colormap used for error maps",
        },
    }
    with open(stats_desc_path, "w", encoding="utf-8") as f_desc:
        json.dump(descriptions, f_desc, indent=2, ensure_ascii=False)

    saved_images = 0
    saved_panels = 0
    stats: List[Dict[str, Any]] = []

    with open(stats_path, "w", encoding="utf-8") as f_stats, torch.no_grad():
        for batch in tqdm(loader, desc=f"posegt-panel/{split}"):
            if saved_images >= max_images:
                break

            inputs = move_batch_to_device(batch, runner.device)
            batch_size = inputs[("color", 0, 0)].shape[0]
            _ensure_geometry_for_batch(runner, batch_size)
            outputs, _ = handler(runner, inputs)
            final_loss_maps, posegt_loss_maps = _compute_loss_contrib_maps(
                runner, inputs, outputs, scale=0
            )

            seq_list, idx_list = _extract_meta(inputs)
            if not seq_list:
                seq_list = ["seq"] * batch_size
            if not idx_list:
                idx_list = [None] * batch_size

            for b in range(batch_size):
                if saved_images >= max_images:
                    break

                seq = _safe_name(seq_list[b] if b < len(seq_list) else "seq")
                idx = idx_list[b] if b < len(idx_list) else None
                idx_str = f"{idx:06d}" if isinstance(idx, int) else f"b{saved_images:06d}"
                target = inputs[("color", 0, 0)][b]
                target_hw = (
                    int(output_height) if output_height > 0 else int(target.shape[-2]),
                    int(output_width) if output_width > 0 else int(target.shape[-1]),
                )

                had_any_frame = False
                for fid in opt.frame_ids[1:]:
                    if fid == "s":
                        continue
                    final = outputs.get(("color", fid, 0), None)
                    if final is None:
                        continue

                    source = inputs[("color", fid, 0)][b]
                    irw = outputs.get(("irw_img", fid, 0), None) if is_posegt_method else None
                    aux_image_kind = None
                    irw_b = None
                    if is_posegt_method:
                        aux_image_kind = "irw_img"
                        irw_b = source if irw is None else irw[b]
                    final_b = final[b]
                    tgt_b = target

                    if source.shape[-2:] != target_hw:
                        source = _resize_tensor_to_hw(source, target_hw[0], target_hw[1], "bilinear")
                    if irw_b is not None and irw_b.shape[-2:] != target_hw:
                        irw_b = _resize_tensor_to_hw(irw_b, target_hw[0], target_hw[1], "bilinear")
                    if final_b.shape[-2:] != target_hw:
                        final_b = _resize_tensor_to_hw(final_b, target_hw[0], target_hw[1], "bilinear")
                    if tgt_b.shape[-2:] != target_hw:
                        tgt_b = _resize_tensor_to_hw(tgt_b, target_hw[0], target_hw[1], "bilinear")

                    err_final = (final_b - tgt_b).abs().mean(dim=0)
                    mae_final = float(err_final.mean().item())
                    err_irw = None
                    mae_irw = None
                    if irw_b is not None:
                        err_irw = (irw_b - tgt_b).abs().mean(dim=0)
                        mae_irw = float(err_irw.mean().item())

                    if error_mode == "loss_contrib":
                        loss_final = final_loss_maps[fid][b]
                        if is_posegt_method and aux_image_kind == "irw_img":
                            loss_irw = posegt_loss_maps[fid][b]
                        if loss_final.shape[-2:] != target_hw:
                            loss_final = _resize_tensor_to_hw(loss_final, target_hw[0], target_hw[1], "bilinear")
                        vis_irw = None
                        if is_posegt_method and aux_image_kind == "irw_img":
                            if loss_irw.shape[-2:] != target_hw:
                                loss_irw = _resize_tensor_to_hw(loss_irw, target_hw[0], target_hw[1], "bilinear")
                            vis_irw = loss_irw
                        vis_final = loss_final
                    else:
                        vis_irw = err_irw
                        vis_final = err_final

                    metric_final = float(vis_final.mean().item())
                    metric_irw = float(vis_irw.mean().item()) if vis_irw is not None else None

                    if vis_irw is not None:
                        err_irw_n, err_final_n = _normalize_err_pair(vis_irw, vis_final, error_qmin, error_qmax)
                        err_irw_rgb = _apply_colormap(err_irw_n, error_cmap)
                    else:
                        err_final_n, _tmp = _normalize_err_pair(vis_final, vis_final, error_qmin, error_qmax)
                        err_irw_rgb = None
                    err_final_rgb = _apply_colormap(err_final_n, error_cmap)

                    src_rgb = _tensor_to_uint8_rgb(source)
                    final_rgb = _tensor_to_uint8_rgb(final_b)
                    tgt_rgb = _tensor_to_uint8_rgb(tgt_b)
                    panel_images = [src_rgb]
                    if irw_b is not None:
                        irw_rgb = _tensor_to_uint8_rgb(irw_b)
                        panel_images.append(irw_rgb)
                    panel_images.extend([final_rgb, tgt_rgb])
                    if err_irw_rgb is not None:
                        panel_images.append(err_irw_rgb)
                    panel_images.append(err_final_rgb)
                    panel = _concat_with_gap(panel_images, gap=4)
                    out_dir = split_out / seq
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{idx_str}_f{int(fid):+d}.png"
                    Image.fromarray(panel).save(str(out_path))

                    rec = {
                        "split": split,
                        "seq": seq_list[b] if b < len(seq_list) else "seq",
                        "center_idx": idx,
                        "frame_id": int(fid),
                        "panel_path": str(out_path),
                        "panel_layout": panel_columns,
                        "error_mode": error_mode,
                        "mae_final": mae_final,
                        "metric_final": metric_final,
                    }
                    if aux_image_kind is not None:
                        rec["aux_image_kind"] = aux_image_kind
                    if mae_irw is not None:
                        rec["mae_irw"] = mae_irw
                        rec["mae_delta"] = mae_irw - mae_final
                    if metric_irw is not None:
                        rec["metric_irw"] = metric_irw
                        rec["metric_delta"] = metric_irw - metric_final
                    stats.append(rec)
                    f_stats.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    had_any_frame = True
                    saved_panels += 1

                if had_any_frame:
                    saved_images += 1

    mae_final_all = float(np.mean([x["mae_final"] for x in stats])) if stats else None
    metric_final_all = float(np.mean([x["metric_final"] for x in stats])) if stats else None
    posegt_stats = [x for x in stats if "mae_irw" in x]
    mae_irw_all = float(np.mean([x["mae_irw"] for x in posegt_stats])) if posegt_stats else None
    mae_delta_all = float(np.mean([x["mae_delta"] for x in posegt_stats])) if posegt_stats else None
    metric_irw_all = float(np.mean([x["metric_irw"] for x in posegt_stats if "metric_irw" in x])) if posegt_stats else None
    metric_delta_all = float(np.mean([x["metric_delta"] for x in posegt_stats if "metric_delta" in x])) if posegt_stats else None
    summary = {
        "split": split,
        "method": opt.methods,
        "weights_folder": opt.weights_folder,
        "num_center_images": saved_images,
        "num_panels": saved_panels,
        "is_posegt_method": is_posegt_method,
        "error_mode": error_mode,
        "mae_final_mean": mae_final_all,
        "metric_final_mean": metric_final_all,
        "output_height": output_height,
        "output_width": output_width,
        "error_qmin": error_qmin,
        "error_qmax": error_qmax,
        "error_cmap": error_cmap,
        "descriptions": descriptions,
    }
    if mae_irw_all is not None:
        summary["mae_irw_mean"] = mae_irw_all
        summary["mae_delta_mean"] = mae_delta_all
    if metric_irw_all is not None:
        summary["metric_irw_mean"] = metric_irw_all
        summary["metric_delta_mean"] = metric_delta_all
    with open(split_out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    base = MonodepthOptions()
    parser: argparse.ArgumentParser = base.parser
    parser.add_argument("--weights_folder", type=str, default="/home/yxt/文档/mono_result/weights/UAVid_China/md2_vggt_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_u0ui/models/weights_32", help="Weights folder (required)")
    parser.add_argument("--output_dir", type=str, default="results/diag_posegt_panel")
    parser.add_argument("--eval_split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--max_images", type=int, default=200, help="Number of center images to export")
    parser.add_argument("--use_train_aug", type=_str2bool, default=False, help="Use train aug when eval_split=train")
    parser.add_argument("--ignore_split", type=_str2bool, default=False, help="Ignore split files filtering")
    parser.add_argument("--output_height", type=int, default=576, help="Saved panel height")
    parser.add_argument("--output_width", type=int, default=1024, help="Saved panel width")
    parser.add_argument("--error_qmin", type=float, default=2.0, help="Error-map lower percentile")
    parser.add_argument("--error_qmax", type=float, default=98.0, help="Error-map upper percentile")
    parser.add_argument("--error_cmap", type=str, default="magma", help="Error-map colormap")
    parser.add_argument(
        "--error_mode",
        type=str,
        default="loss_contrib",
        choices=["rgb_mae", "loss_contrib"],
        help="Error map mode: rgb_mae or loss_contrib (actual per-pixel loss contribution)",
    )

    # Keep defaults minimal for manually filled args in local runs.
    parser.set_defaults(
        methods="MD2_VGGT",
        dataset="UAVid_TriDataset",
        split="UAVid2020_China",
        data_path="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China",
        triplet_root="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China_tri/tri_win20_disp0.03_lap0_rot20d_U0.5_S0",
        seed=42,
    )

    opt = base.parse()

    if not opt.weights_folder:
        raise ValueError("Please set --weights_folder")
    if not opt.data_path:
        raise ValueError("Please set --data_path")
    if not opt.triplet_root:
        raise ValueError("Please set --triplet_root")
    if str(opt.dataset) not in {"UAVula_TriDataset", "UAVid_TriDataset"}:
        raise ValueError("This script only supports triplet datasets: UAVula_TriDataset / UAVid_TriDataset")

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    method_out_dir = output_dir / _safe_name(opt.methods)
    method_out_dir.mkdir(parents=True, exist_ok=True)

    with open(method_out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(opt), f, indent=2, ensure_ascii=False)

    summary = run_posegt_panel_diag(
        opt=copy.deepcopy(opt),
        split=opt.eval_split,
        output_dir=method_out_dir,
        max_images=int(opt.max_images),
        output_height=int(opt.output_height),
        output_width=int(opt.output_width),
        error_qmin=float(opt.error_qmin),
        error_qmax=float(opt.error_qmax),
        error_cmap=str(opt.error_cmap),
        error_mode=str(opt.error_mode),
    )
    with open(method_out_dir / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump([summary], f, indent=2, ensure_ascii=False)
    print("[diag] done")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate and save training-time automasks (identity_selection), plus automask ratio
and high-residual ratio statistics.

Example:
  python tools/diag_automask.py \\
    --dataset UAVula_TriDataset --split UAVula \\
    --data_path /path/to/UAVula --triplet_root /path/to/uav_triplets \\
    --posegt_weights /path/to/posegt_ckpt \\
    --md2_weights /path/to/md2_ckpt \\
    --batch_size 4 --height 320 --width 1024 \\
    --eval_splits train --output_dir results/diag_automask
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

# Disable W&B to avoid offline errors
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from options import MonodepthOptions
from trainer import Trainer
from trainer_init.data_init import _infer_norm_cfg, parse_split_pairs, _infer_uavid_k_region
from trainer_init.geometry_init import init_geometry
from trainer_init.loss_init import init_losses
from trainer_init.model_init import get_forward_handler, init_models
from methods import datasets as kitti_datasets
from methods.datasets import (
    UAVTripletJsonDataset,
    UAVid2020TripletJsonDataset,
    UAVid2020_Dataset,
    UAVula_Dataset,
    wilduav_dataset,
)
from layers import SSIM, BackprojectDepth, Project3D
from utils import readlines


POSE_PRIOR_METHODS = {
    "MD2_VGGT_NoPose", "MD2_VGGT_NoPose_UniformT", "MD2_VGGT_NoPose_SAlign",
    "MD2_VGGT_NoPose_TScale",
    "MD2_VGGT_ResPose_RT", "MD2_VGGT_ResPose_RT_Reg", "MD2_VGGT_ResPose_RT_RMul",
    "MD2_VGGT_ResPose_T", "MD2_VGGT_ResPose_T_Reg",
    "MD2_VGGT_Gated", "MD2_VGGT_Teacher", "MD2_VGGT_Teacher_Distill",
    "MD2_VGGT_Teacher_Photo",
    "MD2_VGGT_ResPose_Decay", "MD2_VGGT_PoseToRes",
    "MD2_VGGT_TDir_PoseMag", "MD2_VGGT_TPrior_Alpha",
    "MD2_VGGT_TPrior_AlignRes",
    "MD2_VGGT_RPrior_TPose",
    "MD2_VGGT_PoseGT",
}


def load_model_weights_only(runner: "EvalRunner", folder: str, model_names=None) -> None:
    """Load model weights only (no optimizer)."""
    if not folder:
        raise ValueError("load_model_weights_only: weights folder is required")
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
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        runner.models[n].load_state_dict(model_dict)


def _safe_name(s: str) -> str:
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("-")
    name = "".join(out).strip("-")
    return name or "default"


def _ensure_geometry_for_batch(runner: "EvalRunner", batch_size: int) -> None:
    """Recreate geometry layers if batch size changes."""
    for scale in runner.opt.scales:
        h = runner.opt.height // (2 ** scale)
        w = runner.opt.width // (2 ** scale)
        module = runner.backproject_depth.get(scale, None)
        if module is None or getattr(module, "batch_size", None) != batch_size:
            runner.backproject_depth[scale] = BackprojectDepth(batch_size, h, w).to(runner.device)
            runner.project_3d[scale] = Project3D(batch_size, h, w).to(runner.device)


def _filter_samples_for_diag(
    dataset: UAVTripletJsonDataset,
    data_root: str,
    require_pose: bool,
    frame_ids: List[Any],
) -> None:
    """Filter triplet samples: ensure images exist and (optionally) poses are present."""
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
        if ok and require_pose:
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
    if require_pose and missing_pose:
        print(f"[diag] filtered {missing_pose} samples without external pose under {data_root}")
    dataset.samples = keep


def _build_loader(
    opt,
    split: str,
    use_split_files: bool,
    use_train_aug: bool,
) -> DataLoader:
    """Build DataLoader for the given split (train/val)."""
    assert split in {"train", "val"}
    dataset_kwargs = _infer_norm_cfg(opt)
    manifest_glob = "triplets_uniform_t.jsonl" if "UniformT" in str(opt.methods) else "triplets.jsonl"
    root = "Train" if split == "train" else "Validation"
    use_triplet_pose = str(opt.methods) in POSE_PRIOR_METHODS

    triplet_datasets = {"UAVula_TriDataset", "UAVid_TriDataset"}
    mono_datasets = {"UAVula_Dataset", "UAVid2020", "WildUAV", "kitti", "kitti_odom"}

    if use_triplet_pose and str(opt.dataset) not in triplet_datasets:
        raise ValueError(
            f"methods={opt.methods} requires external pose, "
            f"but dataset={opt.dataset} is not a triplet dataset"
        )

    if str(opt.dataset) in triplet_datasets:
        if str(opt.dataset) == "UAVid_TriDataset":
            k_region = _infer_uavid_k_region(opt)
            dataset = UAVid2020TripletJsonDataset(
                data_path=os.path.join(opt.data_path, root),
                triplet_root=opt.triplet_root,
                height=opt.height,
                width=opt.width,
                frame_idxs=opt.frame_ids,
                num_scales=len(opt.scales),
                is_train=use_train_aug if split == "train" else False,
                img_ext=".png" if opt.png else ".jpg",
                allow_flip=bool(use_train_aug and getattr(opt, "enable_flip", False)),
                vggt_target_width=getattr(opt, "vggt_target_width", 518),
                use_triplet_pose=use_triplet_pose,
                triplet_manifest_glob=manifest_glob,
                k_region=k_region,
                **dataset_kwargs,
            )
        else:
            dataset = UAVTripletJsonDataset(
                data_path=os.path.join(opt.data_path, root),
                triplet_root=opt.triplet_root,
                height=opt.height,
                width=opt.width,
                frame_idxs=opt.frame_ids,
                num_scales=len(opt.scales),
                is_train=use_train_aug if split == "train" else False,
                img_ext=".png" if opt.png else ".jpg",
                allow_flip=bool(use_train_aug and getattr(opt, "enable_flip", False)),
                vggt_target_width=getattr(opt, "vggt_target_width", 518),
                use_triplet_pose=use_triplet_pose,
                triplet_manifest_glob=manifest_glob,
                **dataset_kwargs,
            )

        if use_split_files:
            split_file = os.path.join(PROJECT_ROOT, "methods", "splits", opt.split, f"{split}_files.txt")
            if os.path.isfile(split_file):
                lines = readlines(split_file)
                pairs = parse_split_pairs(lines)
                if pairs:
                    dataset.samples = [
                        s for s in dataset.samples if (s.get("seq"), s.get("center_idx")) in pairs
                    ]
            else:
                print(f"[warn] split file not found: {split_file} (skip filtering)")

        _filter_samples_for_diag(
            dataset,
            data_root=os.path.join(opt.data_path, root),
            require_pose=use_triplet_pose,
            frame_ids=opt.frame_ids,
        )
    elif str(opt.dataset) in mono_datasets:
        if not use_split_files:
            print("[warn] ignore_split=True is not supported for mono datasets; using split file list.")
        split_file = os.path.join(PROJECT_ROOT, "methods", "splits", opt.split, f"{split}_files.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"split file not found: {split_file}")
        filenames = readlines(split_file)

        if str(opt.dataset) == "UAVid2020":
            k_region = _infer_uavid_k_region(opt)
            dataset = UAVid2020_Dataset(
                opt.data_path,
                filenames,
                opt.height,
                opt.width,
                opt.frame_ids,
                len(opt.scales),
                is_train=use_train_aug if split == "train" else False,
                img_ext=".png" if opt.png else ".jpg",
                allow_flip=bool(use_train_aug and getattr(opt, "enable_flip", False)),
                k_region=k_region,
                **dataset_kwargs,
            )
        elif str(opt.dataset) == "UAVula_Dataset":
            dataset = UAVula_Dataset(
                opt.data_path,
                filenames,
                opt.height,
                opt.width,
                opt.frame_ids,
                len(opt.scales),
                is_train=use_train_aug if split == "train" else False,
                img_ext=".png" if opt.png else ".jpg",
                allow_flip=bool(use_train_aug and getattr(opt, "enable_flip", False)),
                **dataset_kwargs,
            )
        elif str(opt.dataset) == "WildUAV":
            dataset = wilduav_dataset.WildUAVDataset(
                opt.data_path,
                filenames,
                opt.height,
                opt.width,
                opt.frame_ids,
                len(opt.scales),
                is_train=use_train_aug if split == "train" else False,
                img_ext=".png" if opt.png else ".jpg",
                allow_flip=bool(use_train_aug and getattr(opt, "enable_flip", False)),
                **dataset_kwargs,
            )
        elif str(opt.dataset) == "kitti":
            dataset = kitti_datasets.KITTIRAWDataset(
                opt.data_path,
                filenames,
                opt.height,
                opt.width,
                opt.frame_ids,
                len(opt.scales),
                is_train=use_train_aug if split == "train" else False,
                img_ext=".png" if opt.png else ".jpg",
                allow_flip=bool(use_train_aug and getattr(opt, "enable_flip", False)),
            )
        elif str(opt.dataset) == "kitti_odom":
            dataset = kitti_datasets.KITTIOdomDataset(
                opt.data_path,
                filenames,
                opt.height,
                opt.width,
                opt.frame_ids,
                len(opt.scales),
                is_train=use_train_aug if split == "train" else False,
                img_ext=".png" if opt.png else ".jpg",
                allow_flip=bool(use_train_aug and getattr(opt, "enable_flip", False)),
            )
        else:
            raise ValueError(f"unsupported dataset: {opt.dataset}")
    else:
        raise ValueError(f"unsupported dataset: {opt.dataset}")

    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
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
        self.use_posegt = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_PoseGT"

        init_models(self)
        init_losses(self)
        init_geometry(self)

        self.scale_aligner = None
        self._scale_align_monitor = {"train": {"attempts": 0, "success": 0}, "val": {"attempts": 0, "success": 0}}
        self._scale_align_warned = {"train": False, "val": False}

        if not self.opt.no_ssim:
            self.ssim = SSIM().to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3",
        ]
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


def _compute_reprojection_losses(
    runner: "EvalRunner",
    inputs: Dict[Any, Any],
    outputs: Dict[Any, Any],
    scale: int,
) -> torch.Tensor:
    """Return reprojection losses with shape [B, N, H, W]."""
    opt = runner.opt
    reprojection_losses = []
    target = inputs[("color", 0, 0)]
    for frame_id in opt.frame_ids[1:]:
        if frame_id == "s":
            continue
        pred = outputs[("color", frame_id, scale)]
        reproj_loss = runner.loss.compute_reprojection_loss(pred, target)
        if opt.distorted_mask:
            valid_mask = outputs.get(("distorted_mask", frame_id, scale), None)
            if valid_mask is not None:
                reproj_loss = reproj_loss * valid_mask + (1.0 - valid_mask) * 1e5
        reprojection_losses.append(reproj_loss)
    return torch.cat(reprojection_losses, 1)


def _compute_identity_selection(
    runner: "EvalRunner",
    inputs: Dict[Any, Any],
    reprojection_losses: torch.Tensor,
    scale: int,
) -> torch.Tensor:
    """Compute training-time identity_selection mask (1=keep, 0=masked)."""
    opt = runner.opt
    if opt.disable_automasking:
        return torch.ones(
            reprojection_losses.shape[0],
            1,
            reprojection_losses.shape[2],
            reprojection_losses.shape[3],
            device=reprojection_losses.device,
            dtype=reprojection_losses.dtype,
        )

    target = inputs[("color", 0, 0)]
    identity_losses = []
    for frame_id in opt.frame_ids[1:]:
        if frame_id == "s":
            continue
        pred = inputs[("color", frame_id, 0)]
        identity_losses.append(runner.loss.compute_reprojection_loss(pred, target))
    identity_losses = torch.cat(identity_losses, 1)

    if opt.avg_reprojection:
        identity_loss_tensor = identity_losses.mean(1, keepdim=True)
        reprojection_loss = reprojection_losses.mean(1, keepdim=True)
    else:
        identity_loss_tensor = identity_losses
        reprojection_loss = reprojection_losses

    identity_loss_tensor = identity_loss_tensor + torch.randn(
        identity_loss_tensor.shape, device=runner.device
    ) * 1e-5
    combined = torch.cat((identity_loss_tensor, reprojection_loss), dim=1)
    selection = torch.min(combined, dim=1)[1]
    return (selection > identity_loss_tensor.shape[1] - 1).float()


def _compute_min_reproj(
    reprojection_losses: torch.Tensor,
    avg_reprojection: bool,
) -> torch.Tensor:
    """Return [B,1,H,W] min reproj (avg or min, matching training logic)."""
    if avg_reprojection:
        return reprojection_losses.mean(1, keepdim=True)
    if reprojection_losses.shape[1] == 1:
        return reprojection_losses
    return torch.min(reprojection_losses, dim=1, keepdim=True)[0]


def _save_mask_image(mask: torch.Tensor, path: Path, invert: bool) -> None:
    if Image is None:
        raise RuntimeError("PIL not available; install pillow or use --save_masks npy/both")
    path.parent.mkdir(parents=True, exist_ok=True)
    m = mask.detach().float().cpu()
    if invert:
        m = 1.0 - m
    m = (m > 0.5).byte().numpy() * 255
    Image.fromarray(m).save(str(path))


def _save_mask_npy(mask: torch.Tensor, path: Path, invert: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    m = mask.detach().float().cpu()
    if invert:
        m = 1.0 - m
    np.save(str(path), m.numpy())


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


def _overlay_mask_rgb(
    base_rgb: np.ndarray,
    mask_bool: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    if mask_bool.dtype != np.bool_:
        mask_bool = mask_bool.astype(bool)
    if not mask_bool.any():
        return base_rgb
    overlay = base_rgb.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask_bool] = overlay[mask_bool] * (1.0 - alpha) + color_arr * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _apply_colormap(norm_arr: np.ndarray, cmap: str) -> np.ndarray:
    if mpl_cm is None:
        gray = (norm_arr * 255.0 + 0.5).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)
    cm = mpl_cm.get_cmap(cmap)
    colored = cm(norm_arr)[..., :3]
    return (colored * 255.0 + 0.5).astype(np.uint8)


def _normalize_heatmap(
    tensor_2d: torch.Tensor,
    finite_mask: torch.Tensor,
    qmin: float,
    qmax: float,
) -> Optional[torch.Tensor]:
    values = tensor_2d[finite_mask]
    if values.numel() == 0:
        return None
    qmin = max(0.0, min(100.0, float(qmin))) / 100.0
    qmax = max(qmin + 1e-3, min(100.0, float(qmax))) / 100.0
    vmin = torch.quantile(values, qmin)
    vmax = torch.quantile(values, qmax)
    denom = (vmax - vmin).clamp_min(1e-6)
    norm = (tensor_2d - vmin) / denom
    norm = torch.clamp(norm, 0.0, 1.0)
    norm = norm.masked_fill(~finite_mask, 0.0)
    return norm


def _get_external_keep_mask(
    inputs: Dict[Any, Any],
    scale: int,
    device: torch.device,
    dtype: torch.dtype,
    target_hw: Tuple[int, int],
) -> Optional[torch.Tensor]:
    key = ("mask", 0, 0)
    mask = inputs.get(key, None)
    if mask is None or (not torch.is_tensor(mask)):
        key = ("mask", 0, scale)
        mask = inputs.get(key, None)
    if mask is None or (not torch.is_tensor(mask)):
        return None
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=device, dtype=dtype)
    th, tw = target_hw
    if mask.shape[-2] != th or mask.shape[-1] != tw:
        mask = F.interpolate(mask, size=(th, tw), mode="nearest")
    return mask > 0.5


def _run_model_on_split(
    opt,
    model_name: str,
    weights_folder: str,
    split: str,
    output_dir: Path,
    use_split_files: bool,
    use_train_aug: bool,
    mask_scale: int,
    save_masks: str,
    invert_mask: bool,
    mask_kind: str,
    save_mask_viz: bool,
    save_heatmap: bool,
    save_leak_viz: bool,
    heatmap_cmap: str,
    heatmap_qmin: float,
    heatmap_qmax: float,
    overlay_alpha: float,
    output_height: int,
    output_width: int,
    hr_percentile: float,
    hr_scope: str,
    max_batches: Optional[int],
) -> Dict[str, Any]:
    loader = _build_loader(opt, split, use_split_files=use_split_files, use_train_aug=use_train_aug)
    runner = EvalRunner(opt)
    load_model_weights_only(runner, weights_folder)
    handler = get_forward_handler(opt.methods)

    split_out = output_dir / model_name / split
    mask_kind = str(mask_kind).lower()
    if mask_kind not in {"identity", "final"}:
        raise ValueError(f"mask_kind must be identity or final, got {mask_kind}")
    mask_dir_name = "automask" if mask_kind == "identity" else "finalmask"
    mask_dir = split_out / mask_dir_name
    viz_dir = split_out / "viz"
    stats_path = split_out / "stats.jsonl"
    stats_desc_path = split_out / "stats_desc.json"
    split_out.mkdir(parents=True, exist_ok=True)

    need_png = save_masks in {"png", "both"} or save_mask_viz or save_heatmap or save_leak_viz
    if need_png and Image is None:
        raise RuntimeError("PIL not available; install pillow to save PNG visuals")

    total_pixels = 0
    mask_pixels = 0
    high_pixels_all = 0
    high_pixels_in_mask = 0
    num_images = 0
    warned_no_external = False
    mask_kind_fallback = False

    stats_desc = {
        "seq": "sequence name from dataset",
        "center_idx": "center frame index from dataset",
        "keep_ratio": "kept pixels / valid pixels (based on mask_kind)",
        "automask_ratio": "masked pixels / valid pixels (1 - keep_ratio)",
        "high_res_ratio_all": "high-residual pixels / valid pixels (bad_mask ratio)",
        "high_res_ratio_in_keep": "high-residual & kept pixels / kept pixels",
        "bad_keep_ratio": "high-residual & kept pixels / high-residual pixels",
        "min_reproj_pXX": "per-image percentile threshold of min_reproj (XX = hr_percentile)",
        "mask_kind": "mask used for stats/visualization: identity (automask) or final (external & identity)",
    }
    with open(stats_desc_path, "w") as f_desc:
        json.dump(stats_desc, f_desc, indent=2, ensure_ascii=False)

    with open(stats_path, "w") as f_stats, torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{model_name}/{split}")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = move_batch_to_device(batch, runner.device)
            batch_size = inputs[("color", 0, 0)].shape[0]
            _ensure_geometry_for_batch(runner, batch_size)

            outputs, _ = handler(runner, inputs)

            reproj_losses = _compute_reprojection_losses(runner, inputs, outputs, mask_scale)
            min_reproj = _compute_min_reproj(reproj_losses, avg_reprojection=opt.avg_reprojection)
            min_reproj = min_reproj.squeeze(1)

            mask = outputs.get(f"identity_selection/{mask_scale}", None)
            if mask is None:
                mask = _compute_identity_selection(runner, inputs, reproj_losses, mask_scale)
            mask_identity = mask.squeeze(1)
            mask_identity_bool = mask_identity > 0.5
            mask_bool = mask_identity_bool
            ext_keep = None
            if mask_kind == "final":
                ext_keep = _get_external_keep_mask(
                    inputs,
                    mask_scale,
                    device=runner.device,
                    dtype=mask_identity.dtype,
                    target_hw=(mask_identity.shape[-2], mask_identity.shape[-1]),
                )
                if ext_keep is None:
                    if not warned_no_external:
                        print("[diag] external keep mask not found; fallback to identity_selection")
                        warned_no_external = True
                    mask_kind_fallback = True
                else:
                    ext_keep = ext_keep.squeeze(1)
                    mask_bool = mask_identity_bool & ext_keep
            mask_kind_effective = mask_kind if (mask_kind == "identity" or ext_keep is not None) else "identity"
            mask_to_save = mask_identity if mask_kind_effective == "identity" else mask_bool.float()

            seq_list, idx_list = _extract_meta(inputs)
            if not seq_list:
                seq_list = ["seq"] * batch_size
            if not idx_list:
                idx_list = [None] * batch_size

            for b in range(batch_size):
                seq = _safe_name(seq_list[b] if b < len(seq_list) else "seq")
                idx = idx_list[b] if b < len(idx_list) else None
                idx_str = f"{idx:06d}" if isinstance(idx, int) else f"b{num_images:06d}"
                stem = f"{seq}_{idx_str}"

                m = mask_to_save[b]
                if output_height > 0 and output_width > 0:
                    target_hw = (int(output_height), int(output_width))
                else:
                    target_hw = (int(m.shape[-2]), int(m.shape[-1]))
                mr = min_reproj[b]
                finite = torch.isfinite(mr)
                total = int(finite.sum().item())
                high = None
                if total == 0:
                    threshold = float("nan")
                    high_all = 0
                    high_in_mask = 0
                    bad_keep_ratio = None
                else:
                    if hr_scope == "mask" and mask_bool[b].any():
                        values = mr[mask_bool[b] & finite]
                    else:
                        values = mr[finite]
                    if values.numel() == 0:
                        threshold = float("nan")
                        high_all = 0
                        high_in_mask = 0
                        bad_keep_ratio = None
                        high = None
                    else:
                        threshold = float(torch.quantile(values, hr_percentile / 100.0).item())
                        high = (mr > threshold) & finite
                        high_all = int(high.sum().item())
                        high_in_mask = int((high & mask_bool[b]).sum().item())
                        bad_total = high_all
                        bad_keep = int((high & mask_bool[b]).sum().item())
                        bad_keep_ratio = (bad_keep / bad_total) if bad_total > 0 else None

                keep_count = int((mask_bool[b] & finite).sum().item())
                masked_count = int((finite.sum().item() - keep_count)) if total > 0 else 0

                total_pixels += total
                mask_pixels += masked_count
                high_pixels_all += high_all
                high_pixels_in_mask += high_in_mask
                num_images += 1

                record = {
                    "seq": seq_list[b] if b < len(seq_list) else "seq",
                    "center_idx": idx,
                    "keep_ratio": (keep_count / total) if total > 0 else None,
                    "automask_ratio": (masked_count / total) if total > 0 else None,
                    "high_res_ratio_all": (high_all / total) if total > 0 else None,
                    "high_res_ratio_in_keep": (high_in_mask / keep_count) if keep_count > 0 else None,
                    "bad_keep_ratio": bad_keep_ratio,
                    "min_reproj_p{:.0f}".format(hr_percentile): threshold,
                    "mask_kind": mask_kind_effective,
                }
                f_stats.write(json.dumps(record, ensure_ascii=False) + "\n")

                if save_masks != "none":
                    rel_dir = mask_dir / seq
                    if save_masks in ("png", "both"):
                        m_save = m
                        if m_save.shape[-2:] != target_hw:
                            m_save = _resize_tensor_to_hw(m_save, target_hw[0], target_hw[1], "nearest")
                        _save_mask_image(m_save, rel_dir / f"{idx_str}.png", invert_mask)
                    if save_masks in ("npy", "both"):
                        m_save = m
                        if m_save.shape[-2:] != target_hw:
                            m_save = _resize_tensor_to_hw(m_save, target_hw[0], target_hw[1], "nearest")
                        _save_mask_npy(m_save, rel_dir / f"{idx_str}.npy", invert_mask)

                if save_mask_viz or save_heatmap or save_leak_viz:
                    rel_dir = viz_dir / seq
                    rel_dir.mkdir(parents=True, exist_ok=True)
                    base_img_t = inputs[("color", 0, 0)][b]
                    if base_img_t.shape[-2:] != target_hw:
                        base_img_t = _resize_tensor_to_hw(base_img_t, target_hw[0], target_hw[1], "bilinear")
                    base_rgb = _tensor_to_uint8_rgb(base_img_t)
                    base_h, base_w = base_rgb.shape[0], base_rgb.shape[1]

                    mask_vis = mask_bool[b]
                    if mask_vis.shape[-2:] != (base_h, base_w):
                        mask_vis = _resize_tensor_to_hw(mask_vis.float(), base_h, base_w, "nearest") > 0.5
                    mask_vis_np = mask_vis.detach().cpu().numpy().astype(np.bool_)

                    if save_mask_viz:
                        show_mask = mask_vis_np if invert_mask else (~mask_vis_np)
                        mask_color = (0, 255, 255) if not invert_mask else (0, 255, 0)
                        mask_overlay = _overlay_mask_rgb(base_rgb, show_mask, mask_color, overlay_alpha)
                        Image.fromarray(mask_overlay).save(str(rel_dir / f"{idx_str}_mask_overlay_{mask_kind_effective}.png"))

                    if save_heatmap:
                        mr_vis = min_reproj[b]
                        if mr_vis.shape[-2:] != (base_h, base_w):
                            mr_vis = _resize_tensor_to_hw(mr_vis, base_h, base_w, "bilinear")
                        finite_vis = torch.isfinite(mr_vis)
                        norm = _normalize_heatmap(mr_vis, finite_vis, heatmap_qmin, heatmap_qmax)
                        if norm is not None:
                            heat_rgb = _apply_colormap(norm.detach().cpu().numpy(), heatmap_cmap)
                            Image.fromarray(heat_rgb).save(str(rel_dir / f"{idx_str}_minreproj_heat.png"))

                    if save_leak_viz:
                        leak_mask = None
                        if high is not None:
                            leak_mask = high & mask_bool[b]
                            if leak_mask.shape[-2:] != (base_h, base_w):
                                leak_mask = _resize_tensor_to_hw(leak_mask.float(), base_h, base_w, "nearest") > 0.5
                        if leak_mask is not None:
                            leak_np = leak_mask.detach().cpu().numpy().astype(np.bool_)
                            leak_overlay = _overlay_mask_rgb(base_rgb, leak_np, (255, 0, 0), overlay_alpha)
                            Image.fromarray(leak_overlay).save(str(rel_dir / f"{idx_str}_leak.png"))

    summary = {
        "model": model_name,
        "method": opt.methods,
        "split": split,
        "num_images": num_images,
        "pixels_total": total_pixels,
        "automask_ratio": (mask_pixels / total_pixels) if total_pixels > 0 else None,
        "keep_ratio": (1.0 - (mask_pixels / total_pixels)) if total_pixels > 0 else None,
        "high_res_ratio_all": (high_pixels_all / total_pixels) if total_pixels > 0 else None,
        "high_res_ratio_in_keep": (high_pixels_in_mask / (total_pixels - mask_pixels)) if (total_pixels - mask_pixels) > 0 else None,
        "bad_keep_ratio": (high_pixels_in_mask / high_pixels_all) if high_pixels_all > 0 else None,
        "high_res_percentile": hr_percentile,
        "high_res_scope": hr_scope,
        "mask_scale": mask_scale,
        "save_masks": save_masks,
        "invert_mask": invert_mask,
        "mask_kind": mask_kind,
        "mask_kind_used": "identity" if mask_kind_fallback and mask_kind == "final" else mask_kind,
        "save_mask_viz": save_mask_viz,
        "save_heatmap": save_heatmap,
        "save_leak_viz": save_leak_viz,
        "heatmap_cmap": heatmap_cmap,
        "heatmap_qmin": heatmap_qmin,
        "heatmap_qmax": heatmap_qmax,
        "overlay_alpha": overlay_alpha,
        "output_height": output_height,
        "output_width": output_width,
    }
    summary["descriptions"] = {
        "automask_ratio": "masked pixels / valid pixels",
        "keep_ratio": "kept pixels / valid pixels",
        "high_res_ratio_all": "high-residual pixels / valid pixels (bad_mask ratio)",
        "high_res_ratio_in_keep": "high-residual & kept pixels / kept pixels",
        "bad_keep_ratio": "high-residual & kept pixels / high-residual pixels",
        "high_res_percentile": "percentile used to define bad_mask (e.g. 90 means top 10%)",
        "high_res_scope": "pixel scope for percentile threshold: all or mask (unmasked)",
        "mask_scale": "scale index used for automask",
        "mask_kind": "mask used for stats/visualization: identity or final (external & identity)",
    }
    with open(split_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(split_out / "summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    return summary


def main():
    base = MonodepthOptions()
    parser: argparse.ArgumentParser = base.parser
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return True
        if s in {"0", "false", "f", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")

    parser.add_argument("--output_dir", type=str, default="results/diag_automask")
    parser.add_argument("--eval_splits", nargs="+", type=str, default=["train"], choices=["train", "val"])
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--mask_scale", type=int, default=0, help="Scale index used for automask")
    parser.add_argument("--mask_kind", type=str, default="identity", choices=["identity", "final"],
                        help="Mask used for stats/visualization: identity (automask) or final (external & identity)")
    parser.add_argument("--save_masks", type=str, default="png", choices=["none", "png", "npy", "both"])
    parser.add_argument("--invert_mask", type=_str2bool, default=False,
                        help="Save inverted mask (identity=1). Use True/False.")
    parser.add_argument("--save_mask_viz", type=_str2bool, default=True,
                        help="Save mask overlay visualization. Use True/False.")
    parser.add_argument("--save_heatmap", type=_str2bool, default=True,
                        help="Save min reprojection heatmap. Use True/False.")
    parser.add_argument("--save_leak_viz", type=_str2bool, default=True,
                        help="Save leak overlay (high-residual & kept). Use True/False.")
    parser.add_argument("--heatmap_cmap", type=str, default="magma", help="Colormap for heatmap")
    parser.add_argument("--heatmap_qmin", type=float, default=2.0, help="Lower percentile for heatmap normalization")
    parser.add_argument("--heatmap_qmax", type=float, default=98.0, help="Upper percentile for heatmap normalization")
    parser.add_argument("--overlay_alpha", type=float, default=0.45, help="Overlay alpha for masks/leaks")
    parser.add_argument("--output_height", type=int, default=576,
                        help="Output visualization height when full_res not available")
    parser.add_argument("--output_width", type=int, default=1024,
                        help="Output visualization width when full_res not available")
    parser.add_argument("--use_train_aug", type=_str2bool, default=False,
                        help="Use training-time color aug/flip on train split. Use True/False.")
    parser.add_argument("--ignore_split", type=_str2bool, default=False,
                        help="Do not filter by split files. Use True/False.")
    parser.add_argument("--hr_percentile", type=float, default=90.0, help="High-residual percentile, e.g. 90")
    parser.add_argument("--hr_scope", type=str, default="all", choices=["mask", "all"],
                        help="Pixel scope used to compute percentile threshold")

    parser.add_argument("--posegt_weights", type=str, 
                        #default="/home/yxt/文档/mono_result/weights/UAVid_China/md2_vggt_posegt_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_9yjq/models/weights_36", 
                        #default="/home/yxt/文档/mono_result/weights/UAVula_R1/md2_vggt_posegt_uavula_tridataset_512x288_bs8_lr1e-04_e40_step20_rkbq/models/weights_24", 
                        default="/home/yxt/文档/mono_result/weights/UAVid_Germany/md2_vggt_posegt_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_rzsm/models/weights_23", 
                        help="PoseGT weights folder")
    parser.add_argument("--posegt_method", type=str, default="MD2_VGGT_PoseGT")
    parser.add_argument("--posegt_name", type=str, default="posegt")

    parser.add_argument("--md2_weights", type=str, 
                        #default="/home/yxt/文档/mono_result/weights/UAVid_China/md2_vggt_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_fwg0/models/weights_28", 
                        #default="/home/yxt/文档/mono_result/weights/UAVula_R1/md2_vggt_uavula_tridataset_512x288_bs8_lr1e-04_e40_step20_ge0y/models/weights_35", 
                        default="/home/yxt/文档/mono_result/weights/UAVid_Germany/md2_vggt_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_hnrb/models/weights_26", 
                        help="MD2 weights folder")
    parser.add_argument("--md2_method", type=str, default="Monodepth2")
    parser.add_argument("--md2_name", type=str, default="md2")

    # Dataset defaults (fill in your own values)
    # dataset options:
    #   triplet: UAVula_TriDataset, UAVid_TriDataset
    #   mono:    UAVula_Dataset, UAVid2020, WildUAV
    #   kitti:   kitti, kitti_odom
    # split options (see methods/splits/<split>/):
    #   UAVula, UAVid2020_China, UAVid2020_Germany, wilduav (depending on your repo)
    parser.set_defaults(
        #dataset="UAVid_TriDataset",
        #split="UAVid2020_Germany",
        dataset="UAVula_TriDataset",
        split="UAVula_Dataset",

        #data_path="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China",
        #triplet_root='/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China_tri/tri_win20_disp0.03_lap0_rot20d_U0.5_S0',
        #data_path="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany",
        #triplet_root='/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_win10_disp0.03_lap0_rot20d_U0.5_S0',
        data_path="/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_dataset",
        triplet_root='/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_tri/tri_win5_disp0.05_lap0_rot20d_U0.5_S0',

        height=288,
        width=512,
    )

    opt = parser.parse_args()

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = []
    if opt.posegt_weights:
        model_specs.append((opt.posegt_name, opt.posegt_method, opt.posegt_weights))
    if opt.md2_weights:
        model_specs.append((opt.md2_name, opt.md2_method, opt.md2_weights))

    if not model_specs:
        raise ValueError("Provide at least --posegt_weights or --md2_weights")

    summaries = []
    for model_name, method, weights in model_specs:
        for split in opt.eval_splits:
            model_opt = copy.deepcopy(opt)
            model_opt.methods = method
            summary = _run_model_on_split(
                model_opt,
                model_name=model_name,
                weights_folder=weights,
                split=split,
                output_dir=output_dir,
                use_split_files=not opt.ignore_split,
                use_train_aug=opt.use_train_aug,
                mask_scale=opt.mask_scale,
                save_masks=opt.save_masks,
                invert_mask=opt.invert_mask,
                mask_kind=opt.mask_kind,
                save_mask_viz=opt.save_mask_viz,
                save_heatmap=opt.save_heatmap,
                    save_leak_viz=opt.save_leak_viz,
                    heatmap_cmap=opt.heatmap_cmap,
                    heatmap_qmin=opt.heatmap_qmin,
                    heatmap_qmax=opt.heatmap_qmax,
                    overlay_alpha=opt.overlay_alpha,
                    output_height=opt.output_height,
                    output_width=opt.output_width,
                    hr_percentile=opt.hr_percentile,
                    hr_scope=opt.hr_scope,
                    max_batches=opt.max_batches,
                )
            summaries.append(summary)

    with open(output_dir / "summary_all.json", "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

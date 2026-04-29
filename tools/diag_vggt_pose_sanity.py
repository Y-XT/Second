#!/usr/bin/env python3
"""
VGGT pose sanity check:
  1) photometric(warp(I_s, T_vggt, D_t), I_t) vs photometric(I_s, I_t)
  2) repeat with different depth sources (MD2 depthnet, VGGT depth)

Example:
  python tools/diag_vggt_pose_sanity.py \
    --dataset UAVid_TriDataset \
    --split UAVid2020_China \
    --data_path /path/to/data \
    --triplet_root /path/to/triplets \
    --load_weights_depth_folder /path/to/md2/weights \
    --depth_sources pred vggt
"""
import argparse
import json
import numbers
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 关闭 W&B 相关行为，避免离线/无网环境报错
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

# 确保可以找到项目根目录的模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from options import MonodepthOptions
from trainer import Trainer
from trainer_init.data_init import _infer_norm_cfg
from trainer_init.geometry_init import init_geometry
from trainer_init.loss_init import init_losses
from trainer_init.model_init import init_models
from methods.datasets import UAVTripletJsonDataset
from layers import disp_to_depth


def load_model_weights_only(runner: "EvalRunner", folder: str, model_names=None):
    """仅加载模型权重，不触及 optimizer。"""
    if not folder:
        raise ValueError("load_model_weights_only: 需要提供权重目录")
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"找不到权重目录：{folder}")

    if model_names is None:
        model_names = list(runner.models.keys())

    for n in model_names:
        if n not in runner.models:
            continue
        path = os.path.join(folder, f"{n}.pth")
        if not os.path.isfile(path):
            print(f"[warn] 未找到权重文件 {path}，跳过加载")
            continue
        model_dict = runner.models[n].state_dict()
        pretrained_dict = torch.load(path, map_location="cpu")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        runner.models[n].load_state_dict(model_dict)


def _build_loader(opt, split: str) -> DataLoader:
    """构建指定 split(train/val) 的 DataLoader，并强制 use_triplet_pose=True 以获得外部位姿。"""
    assert split in {"train", "val"}
    dataset_kwargs = _infer_norm_cfg(opt)
    manifest_glob = "triplets_uniform_t.jsonl" if "UniformT" in str(opt.methods) else "triplets.jsonl"
    root = "Train" if split == "train" else "Validation"
    dataset = UAVTripletJsonDataset(
        data_path=os.path.join(opt.data_path, root),
        triplet_root=opt.triplet_root,
        height=opt.height,
        width=opt.width,
        frame_idxs=opt.frame_ids,
        num_scales=len(opt.scales),
        is_train=False,
        img_ext=".png" if opt.png else ".jpg",
        allow_flip=False,
        vggt_target_width=getattr(opt, "vggt_target_width", 518),
        use_triplet_pose=True,
        triplet_manifest_glob=manifest_glob,
        **dataset_kwargs,
    )
    _filter_samples_for_diag(
        dataset,
        data_root=os.path.join(opt.data_path, root),
        require_pose=True,
        frame_ids=opt.frame_ids,
    )

    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def _filter_samples_for_diag(
    dataset: UAVTripletJsonDataset,
    data_root: str,
    require_pose: bool,
    frame_ids: List[Any],
) -> None:
    if not dataset.samples:
        return
    keep = []
    missing_img = 0
    missing_pose = 0
    examples = []
    for rec in dataset.samples:
        ok = True
        for key in ("center", "prev", "next"):
            path = rec.get(key)
            if path and not os.path.isabs(path):
                path = os.path.join(data_root, path)
            if not path or not os.path.exists(path):
                ok = False
                missing_img += 1
                if len(examples) < 3:
                    examples.append({"seq": rec.get("seq"), "center_idx": rec.get("center_idx"), "path": path})
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
        for ex in examples:
            print(f"[diag]   missing sample: seq={ex['seq']} idx={ex['center_idx']} path={ex['path']}")
    if require_pose and missing_pose:
        print(f"[diag] filtered {missing_pose} samples without external pose under {data_root}")
    dataset.samples = keep


class EvalRunner(Trainer):
    """复用 Trainer 的前向/warp/损失逻辑，但精简初始化。"""

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

        init_models(self)
        init_losses(self)
        init_geometry(self)

        self.set_eval()


def move_batch_to_device(batch: Dict[Any, Any], device: torch.device) -> Dict[Any, Any]:
    out: Dict[Any, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def extract_meta(inputs: Dict[Any, Any]) -> Dict[str, Any]:
    seq = inputs.get("seq", "")
    if isinstance(seq, list):
        seq = seq[0]
    center_idx = inputs.get("center_idx", None)
    if torch.is_tensor(center_idx):
        center_idx = int(center_idx[0].item())
    return {"seq": seq, "center_idx": center_idx}


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


def _select_depthnet_input(inputs: Dict[Any, Any]) -> torch.Tensor:
    if ("color_norm", 0, 0) in inputs:
        return inputs[("color_norm", 0, 0)]
    return inputs[("color_aug", 0, 0)]


def _predict_depthnet_depth(runner: "EvalRunner", inputs: Dict[Any, Any]) -> torch.Tensor:
    if "encoder" not in runner.models or "depth" not in runner.models:
        raise RuntimeError("DepthNet 未初始化，无法预测深度")
    enc_in = _select_depthnet_input(inputs)
    features = runner.models["encoder"](enc_in)
    depth_out = runner.models["depth"](features)
    if isinstance(depth_out, dict):
        disp = depth_out.get(("disp", 0), None)
    else:
        disp = depth_out
    if disp is None or not torch.is_tensor(disp):
        raise RuntimeError("DepthNet 输出不包含 ('disp', 0)")
    disp = F.interpolate(disp, [runner.opt.height, runner.opt.width], mode="bilinear", align_corners=False)
    _, depth = disp_to_depth(disp, runner.opt.min_depth, runner.opt.max_depth)
    return depth


def _warp_with_pose(
    runner: "EvalRunner",
    inputs: Dict[Any, Any],
    depth: torch.Tensor,
    frame_id: int,
) -> torch.Tensor:
    T = inputs[("external_cam_T_cam", 0, frame_id)]
    cam_points = runner.backproject_depth[0](depth, inputs[("inv_K", 0)])
    pix_coords = runner.project_3d[0](cam_points, inputs[("K", 0)], T)
    source_img = inputs[("color", frame_id, 0)]
    return F.grid_sample(source_img, pix_coords, padding_mode="border", align_corners=True)


def _compute_photometric_pair(
    runner: "EvalRunner",
    inputs: Dict[Any, Any],
    depth: torch.Tensor,
    frame_ids: List[int],
) -> Tuple[float, float]:
    target = inputs[("color", 0, 0)]

    warp_errors: List[float] = []
    id_errors: List[float] = []
    for fid in frame_ids:
        warped = _warp_with_pose(runner, inputs, depth, fid)
        source = inputs[("color", fid, 0)]

        warp_loss = runner.loss.compute_reprojection_loss(warped, target).mean().item()
        id_loss = runner.loss.compute_reprojection_loss(source, target).mean().item()
        warp_errors.append(warp_loss)
        id_errors.append(id_loss)

    warp_mean = sum(warp_errors) / max(1, len(warp_errors))
    id_mean = sum(id_errors) / max(1, len(id_errors))
    return warp_mean, id_mean


def _summary_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
            "p05": float("nan"),
            "p95": float("nan"),
        }
    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = var ** 0.5

    def _percentile(q: float) -> float:
        if q <= 0:
            return float(vals[0])
        if q >= 1:
            return float(vals[-1])
        pos = (len(vals) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(vals) - 1)
        if lo == hi:
            return float(vals[lo])
        frac = pos - lo
        return float(vals[lo] * (1 - frac) + vals[hi] * frac)

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": vals[0],
        "max": vals[-1],
        "median": _percentile(0.5),
        "p05": _percentile(0.05),
        "p95": _percentile(0.95),
    }


def main():
    base = MonodepthOptions()
    parser: argparse.ArgumentParser = base.parser
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/diag_vggt_pose_sanity",
        help="Where to save JSONL metrics and summary.txt",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional cap on number of val batches to process",
    )
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        type=str,
        default=["val"],
        choices=["train", "val"],
        help="Which splits to run (train/val)",
    )
    parser.add_argument(
        "--load_weights_depth_folder",
        type=str,
        #default="/home/yxt/文档/mono_result/weights/UAVid_China/monodepth2_uavid2020_512x288_bs8_lr1e-04_e40_step20_u1ki/models/weights_18",
        default="/home/yxt/文档/mono_result/weights/UAVid_Germany/monodepth2_uavid2020_512x288_kdef_noapp_bs8_lr1e-04_e40_step20_mtau/models/weights_35",
        #default="/home/yxt/文档/mono_result/weights/UAVula_R1/monodepth2_uavula_tridataset_512x288_bs8_lr1e-04_e40_step20_9gdo/models/weights_26",
        help="Path to MD2 depth weights (encoder/depth)",
    )
    parser.add_argument(
        "--depth_sources",
        nargs="+",
        type=str,
        default=["pred", "vggt"],
        help="Depth sources to evaluate: pred, vggt",
    )
    parser.set_defaults(
        methods="MD2_VGGT",
        dataset="UAVid_TriDataset",
        #dataset="UAVula_TriDataset",
        #split="UAVid2020_China",
        split="UAVid2020_Germany",
        #split="UAVula",
        #data_path="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China",
        data_path="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany",
        #data_path="/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_dataset",
        #triplet_root="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China_tri/tri_win20_disp0.03_lap0_rot20d_U0.5_S0",
        triplet_root="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_win10_disp0.03_lap0_rot20d_U0.5_S0",
        #triplet_root="/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_tri/tri_win5_disp0.05_lap0_rot20d_U0.5_S0",
        height=288,
        width=512,
        batch_size=4,
    )

    opt = base.parse()
    os.makedirs(opt.output_dir, exist_ok=True)

    runner = EvalRunner(opt)
    load_model_weights_only(runner, opt.load_weights_depth_folder, ["encoder", "depth"])
    runner.set_eval()

    depth_sources = [s.lower() for s in opt.depth_sources]
    use_pred = "pred" in depth_sources
    use_vggt = "vggt" in depth_sources

    if not use_pred and not use_vggt:
        raise ValueError("depth_sources must include at least one of: pred, vggt")

    splits = list(dict.fromkeys(opt.eval_splits))
    loaders = {sp: _build_loader(opt, sp) for sp in splits}

    records: List[Dict[str, Any]] = []
    stats: Dict[str, Dict[str, List[float]]] = {
        "pred": {
            "warp_full": [],
            "id_full": [],
            "better_full": [],
        },
        "vggt": {
            "warp_full": [],
            "id_full": [],
            "better_full": [],
        },
    }
    warned_vggt_depth = False

    frame_ids = [fid for fid in opt.frame_ids[1:] if fid != "s"]
    if not frame_ids:
        raise ValueError("frame_ids must include at least one source frame (e.g. -1 or 1)")

    with torch.no_grad():
        for split_name, loader in loaders.items():
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f"{split_name} splits", ncols=100)
            for step, inputs_cpu in pbar:
                if opt.max_batches is not None and step >= opt.max_batches:
                    break

                inputs = move_batch_to_device(inputs_cpu, runner.device)
                meta = extract_meta(inputs)
                meta["split"] = split_name

                record = {**meta}

                pred_depth = None
                if use_pred:
                    pred_depth = _predict_depthnet_depth(runner, inputs)

                vggt_depth = None
                if use_vggt:
                    vggt_depth = inputs.get("vggt_depth", None)
                    if torch.is_tensor(vggt_depth):
                        if vggt_depth.ndim == 3:
                            vggt_depth = vggt_depth.unsqueeze(1)
                        vggt_depth = F.interpolate(
                            vggt_depth, [opt.height, opt.width], mode="bilinear", align_corners=False
                        )
                    else:
                        vggt_depth = None
                        if not warned_vggt_depth:
                            print("[diag][warn] vggt_depth missing in inputs; skip vggt branch")
                            warned_vggt_depth = True

                if use_pred and pred_depth is not None:
                    warp_full, id_full = _compute_photometric_pair(
                        runner, inputs, pred_depth, frame_ids
                    )
                    stats["pred"]["warp_full"].append(warp_full)
                    stats["pred"]["id_full"].append(id_full)
                    stats["pred"]["better_full"].append(float(warp_full < id_full))
                    record.update({
                        "pred_warp_full": warp_full,
                        "pred_id_full": id_full,
                    })

                if use_vggt and vggt_depth is not None:
                    warp_full, id_full = _compute_photometric_pair(
                        runner, inputs, vggt_depth, frame_ids
                    )
                    stats["vggt"]["warp_full"].append(warp_full)
                    stats["vggt"]["id_full"].append(id_full)
                    stats["vggt"]["better_full"].append(float(warp_full < id_full))
                    record.update({
                        "vggt_warp_full": warp_full,
                        "vggt_id_full": id_full,
                    })

                records.append(record)

    ds_token = _safe_name(opt.dataset)
    split_token = _safe_name(opt.split)
    metrics_path = os.path.join(opt.output_dir, f"metrics_{ds_token}_{split_token}.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    lines = [
        "VGGT pose sanity check (warp vs identity)",
        f"dataset : {opt.dataset}",
        f"split   : {opt.split}",
        f"batches : {len(records)}",
        "",
        "Definitions:",
        "  warp_mean      = mean photometric( warp(I_s, T_vggt, D_t), I_t ) over all source frames",
        "  id_mean        = mean photometric( I_s, I_t ) (identity / no-warp)",
        "  delta          = warp_mean - id_mean (negative means VGGT pose helps)",
        "  better_ratio   = fraction of samples where warp_mean < id_mean",
        "",
        "---- Results (full image) ----",
    ]

    for tag in ("pred", "vggt"):
        if tag == "pred" and not use_pred:
            continue
        if tag == "vggt" and not use_vggt:
            continue
        warp_full_stats = _summary_stats(stats[tag]["warp_full"])
        id_full_stats = _summary_stats(stats[tag]["id_full"])
        better_full = (
            sum(stats[tag]["better_full"]) / max(1, len(stats[tag]["better_full"]))
        )

        depth_desc = "MD2 depthnet (pred)" if tag == "pred" else "VGGT depth (vggt)"
        delta = warp_full_stats["mean"] - id_full_stats["mean"]
        lines.extend([
            f"[{tag}] depth_source = {depth_desc}",
            f"  warp_mean    : {_fmt(warp_full_stats['mean'])}",
            f"  id_mean      : {_fmt(id_full_stats['mean'])}",
            f"  delta        : {_fmt(delta)}",
            f"  better_ratio : {_fmt(better_full)}",
        ])

    summary_path = os.path.join(opt.output_dir, f"summary_{ds_token}_{split_token}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[diag] done")
    for ln in lines:
        print(f"[diag] {ln}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compare depth quality between Monodepth2 and VGGT prior against GT depth.

Evaluation uses UAVid-style: per-sample median scaling + 5/95 percentile clip.
Outputs:
  - metrics_<dataset>_<split>.jsonl
  - summary_<dataset>_<split>.txt
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import statistics

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from options import MonodepthOptions
from trainer_init.data_init import _infer_norm_cfg
from trainer_init.model_init import init_models
from methods.datasets import UAVTripletJsonDataset
from layers import disp_to_depth, compute_depth_errors


METRIC_NAMES = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]


def load_model_weights_only(runner: "DepthEvalRunner", folder: str, model_names=None):
    if not folder:
        raise ValueError("load_model_weights_only: missing weights folder")
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
            print(f"[warn] missing weights: {path}, skip")
            continue
        model_dict = runner.models[n].state_dict()
        pretrained_dict = torch.load(path, map_location="cpu")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        runner.models[n].load_state_dict(model_dict)


class DepthEvalRunner:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.models: Dict[str, torch.nn.Module] = {}
        self.parameters_to_train: List[torch.nn.Parameter] = []
        self.training = False
        init_models(self)
        self.set_eval()

    def set_eval(self):
        for m in self.models.values():
            m.eval()
        return self


def _path_exists(path_value: Optional[str], *bases: Optional[str]) -> bool:
    if not path_value:
        return False
    p = Path(path_value)
    if p.is_absolute():
        return p.exists()
    for base in bases:
        if base and Path(base, path_value).exists():
            return True
    return False


def _filter_samples_for_diag(
    dataset: UAVTripletJsonDataset,
    data_root: str,
    triplet_root: str,
    require_depth: bool,
    require_vggt: bool,
    frame_ids: List[Any],
) -> None:
    if not dataset.samples:
        return
    keep = []
    missing_img = 0
    missing_depth = 0
    missing_vggt = 0
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
        if ok and require_depth:
            depth_path = rec.get("depth_gt_path")
            if depth_path and not os.path.isabs(depth_path):
                depth_path = os.path.join(data_root, depth_path)
            if not depth_path or not os.path.exists(depth_path):
                ok = False
                missing_depth += 1
        if ok and require_vggt:
            vggt_path = rec.get("vggt_depth_path")
            if not _path_exists(vggt_path, data_root, triplet_root):
                ok = False
                missing_vggt += 1
        if ok:
            keep.append(rec)
    if missing_img:
        print(f"[diag] filtered {missing_img} samples without images under {data_root}")
        for ex in examples:
            print(f"[diag]   missing sample: seq={ex['seq']} idx={ex['center_idx']} path={ex['path']}")
    if require_depth and missing_depth:
        print(f"[diag] filtered {missing_depth} samples without depth_gt under {data_root}")
    if require_vggt and missing_vggt:
        print(f"[diag] filtered {missing_vggt} samples without vggt_depth under {triplet_root}")
    dataset.samples = keep


def _build_loader(opt, split: str) -> DataLoader:
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
        use_triplet_pose=False,
        triplet_manifest_glob=manifest_glob,
        **dataset_kwargs,
    )
    _filter_samples_for_diag(
        dataset,
        data_root=os.path.join(opt.data_path, root),
        triplet_root=opt.triplet_root,
        require_depth=True,
        require_vggt=True,
        frame_ids=opt.frame_ids,
    )
    if not dataset.samples:
        raise RuntimeError(f"[diag] no samples found for split={split} after filtering")

    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def move_batch_to_device(batch: Dict[Any, Any], device: torch.device) -> Dict[Any, Any]:
    out: Dict[Any, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _merge_masks(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a & b


def _compute_uavid_metrics_per_sample(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
) -> List[Tuple[int, Dict[str, float]]]:
    results: List[Tuple[int, Dict[str, float]]] = []
    if valid_mask is not None:
        valid_mask = valid_mask.view(-1).to(torch.bool)
    for b in range(pred.shape[0]):
        if valid_mask is not None and not bool(valid_mask[b].item()):
            continue
        pred_map = pred[b:b + 1]
        gt_map = gt[b:b + 1]
        pred_resized = F.interpolate(pred_map, gt_map.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        gt_resized = gt_map[0, 0].to(pred_resized.device, dtype=pred_resized.dtype)

        base_mask = (pred_resized > 0) & torch.isfinite(gt_resized)
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
        metrics = {name: float(depth_errors[i].detach().cpu().item()) for i, name in enumerate(METRIC_NAMES)}
        metrics["scale_ratio"] = float(scale_ratio.detach().cpu().item())
        results.append((b, metrics))
    return results


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


def _extract_batch_meta(inputs: Dict[Any, Any], batch_size: int) -> Tuple[List[str], List[int]]:
    seqs = inputs.get("seq", [])
    if not isinstance(seqs, list):
        seqs = [seqs] * batch_size
    center_idx = inputs.get("center_idx", None)
    if torch.is_tensor(center_idx):
        center_idx = [int(x) for x in center_idx.view(-1).tolist()]
    elif isinstance(center_idx, list):
        center_idx = [int(x) for x in center_idx]
    else:
        center_idx = [int(center_idx)] * batch_size if center_idx is not None else [None] * batch_size
    return seqs, center_idx


def main():
    base = MonodepthOptions()
    parser: argparse.ArgumentParser = base.parser
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/diag_depth_compare",
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
        #default="/home/yxt/文档/mono_result/weights/UAVid_China/monodepth2_uavid2020_512x288_kdef_noapp_bs8_lr1e-04_e40_step20_d1ly/models/weights_35",
        #default="/home/yxt/文档/mono_result/weights/UAVid_Germany/monodepth2_uavid2020_512x288_kdef_noapp_bs8_lr1e-04_e40_step20_mtau/models/weights_35",
        #default="/home/yxt/文档/mono_result/weights/UAVula_R1/monodepth2_uavula_tridataset_512x288_bs8_lr1e-04_e40_step20_9gdo/models/weights_26",
        default="/home/yxt/文档/mono_result/weights/UAVula_R1/md2_vggt_uavula_tridataset_512x288_bs8_lr1e-04_e40_step20_dvsm/models/weights_26",
        help="Path to depth weights (encoder/depth)",
    )

    parser.set_defaults(
        data_path="/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_dataset",
        #data_path="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany",
        #data_path="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China",
        triplet_root="/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_tri/tri_win5_disp0.05_lap0_rot20d_U0.5_S0",
        #triplet_root="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_win10_disp0.03_lap0_rot20d_U0.5_S0",
        #triplet_root="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China_tri/tri_win20_disp0.03_lap0_rot20d_U0.5_S0",
        methods="MD2_VGGT_ResPose_RT",
        #dataset="UAVid-China",
        #split="UAVid-germany",
        split="UAVula-md2",
    )

    opt = base.parse()

    opt.scales = [0]

    os.makedirs(opt.output_dir, exist_ok=True)

    if not opt.load_weights_depth_folder:
        raise ValueError("Please set --load_weights_depth_folder")

    runner = DepthEvalRunner(opt)
    load_model_weights_only(runner, opt.load_weights_depth_folder, ["encoder", "depth"])
    runner.set_eval()

    splits = list(dict.fromkeys(opt.eval_splits))
    loaders = {sp: _build_loader(opt, sp) for sp in splits}

    records: List[Dict[str, Any]] = []
    error_metrics = ["abs_rel", "sq_rel", "rmse", "rmse_log"]
    acc_metrics = ["a1", "a2", "a3"]
    stats = {
        "md2": {
            "sum": {k: 0.0 for k in METRIC_NAMES},
            "count": 0,
            "scales": [],
            "errors": {k: [] for k in error_metrics},
        },
        "vggt": {
            "sum": {k: 0.0 for k in METRIC_NAMES},
            "count": 0,
            "scales": [],
            "errors": {k: [] for k in error_metrics},
        },
    }

    device = runner.device

    with torch.no_grad():
        for split_name, loader in loaders.items():
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f"{split_name} splits", ncols=100)
            for step, inputs_cpu in pbar:
                if opt.max_batches is not None and step >= opt.max_batches:
                    break
                inputs = move_batch_to_device(inputs_cpu, device)
                batch_size = inputs.get(("color", 0, 0)).shape[0]
                seqs, center_idxs = _extract_batch_meta(inputs, batch_size)

                depth_gt = inputs.get("depth_gt", None)
                if depth_gt is None or not torch.is_tensor(depth_gt):
                    continue
                if depth_gt.ndim == 3:
                    depth_gt = depth_gt.unsqueeze(1)

                vggt_depth = inputs.get("vggt_depth", None)
                if vggt_depth is not None and torch.is_tensor(vggt_depth) and vggt_depth.ndim == 3:
                    vggt_depth = vggt_depth.unsqueeze(1)

                depth_has_valid = inputs.get("depth_has_valid", None)
                vggt_has_valid = inputs.get("vggt_depth_has_valid", None)
                md2_mask = depth_has_valid
                vggt_mask = _merge_masks(depth_has_valid, vggt_has_valid)

                feats = runner.models["encoder"](inputs[("color_aug", 0, 0)])
                depth_out = runner.models["depth"](feats)
                disp = depth_out[("disp", 0)]
                _, depth_md2 = disp_to_depth(disp, opt.min_depth, opt.max_depth)

                md2_metrics = _compute_uavid_metrics_per_sample(depth_md2, depth_gt, md2_mask)
                for b, metrics in md2_metrics:
                    for k in METRIC_NAMES:
                        stats["md2"]["sum"][k] += metrics[k]
                    stats["md2"]["count"] += 1
                    stats["md2"]["scales"].append(metrics["scale_ratio"])
                    for k in error_metrics:
                        stats["md2"]["errors"][k].append(metrics[k])
                    records.append({
                        "seq": seqs[b],
                        "center_idx": center_idxs[b],
                        "split": split_name,
                        "method": "md2",
                        **metrics,
                    })

                if vggt_depth is not None and torch.is_tensor(vggt_depth):
                    vggt_metrics = _compute_uavid_metrics_per_sample(vggt_depth, depth_gt, vggt_mask)
                    for b, metrics in vggt_metrics:
                        for k in METRIC_NAMES:
                            stats["vggt"]["sum"][k] += metrics[k]
                        stats["vggt"]["count"] += 1
                        stats["vggt"]["scales"].append(metrics["scale_ratio"])
                        for k in error_metrics:
                            stats["vggt"]["errors"][k].append(metrics[k])
                        records.append({
                            "seq": seqs[b],
                            "center_idx": center_idxs[b],
                            "split": split_name,
                            "method": "vggt",
                            **metrics,
                        })

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

    def _mean(stats_dict: Dict[str, Any]) -> Dict[str, float]:
        out = {}
        count = max(1, stats_dict["count"])
        for k in METRIC_NAMES:
            out[k] = stats_dict["sum"][k] / count
        return out

    def _percentiles(values: List[float], qs: List[float]) -> Dict[float, float]:
        if not values:
            return {q: float("nan") for q in qs}
        tensor = torch.tensor(values, dtype=torch.float32)
        return {q: float(torch.quantile(tensor, q / 100.0).item()) for q in qs}

    md2_mean = _mean(stats["md2"])
    vggt_mean = _mean(stats["vggt"])
    md2_scale_med = statistics.median(stats["md2"]["scales"]) if stats["md2"]["scales"] else float("nan")
    vggt_scale_med = statistics.median(stats["vggt"]["scales"]) if stats["vggt"]["scales"] else float("nan")

    lines = [
        f"dataset : {opt.dataset}",
        f"split   : {opt.split}",
        f"records : {len(records)}",
        "---- mean metrics (lower is better except a1/a2/a3) ----",
        "md2  : " + ", ".join(f"{k}={_fmt(md2_mean[k])}" for k in METRIC_NAMES),
        "vggt : " + ", ".join(f"{k}={_fmt(vggt_mean[k])}" for k in METRIC_NAMES),
        "---- md2 - vggt (error metrics: negative is better for md2) ----",
        ", ".join(f"{k}={_fmt(md2_mean[k] - vggt_mean[k])}" for k in error_metrics),
        "---- md2 - vggt (accuracy metrics: positive is better for md2) ----",
        ", ".join(f"{k}={_fmt(md2_mean[k] - vggt_mean[k])}" for k in acc_metrics),
    ]

    lines.append("---- error percentiles p50/p90/p99 (lower is better) ----")
    for method in ("md2", "vggt"):
        for k in error_metrics:
            p = _percentiles(stats[method]["errors"][k], [50.0, 90.0, 99.0])
            lines.append(
                f"{method}.{k} : p50={_fmt(p[50.0])}, p90={_fmt(p[90.0])}, p99={_fmt(p[99.0])}"
            )

    lines.extend([
        "---- sample counts ----",
        f"md2_samples  : {stats['md2']['count']}",
        f"vggt_samples : {stats['vggt']['count']}",
        f"md2_scale_ratio_median  : {_fmt(md2_scale_med)}",
        f"vggt_scale_ratio_median : {_fmt(vggt_scale_med)}",
    ])

    summary_path = os.path.join(opt.output_dir, f"summary_{ds_token}_{split_token}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[diag] done")
    for ln in lines:
        print(f"[diag] {ln}")


if __name__ == "__main__":
    main()

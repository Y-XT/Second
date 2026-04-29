#!/usr/bin/env python3
"""
UAVula 验证 / 测试 / 三元组验证集深度分布统计脚本。

核心能力：
  1. 按 `methods/splits/UAVula/{val,test}_files.txt` 解析样本列表；
  2. 自动匹配对应的深度文件：
       - 验证集使用 dataset 根目录下的 `.npy/.npz/.png/.tif/.tiff` 深度文件；
       - 测试集直接使用 `image_02/data/stereo/depth_maps/*.jpg.geometric.bin` 中的深度文件；
       - 三元组验证集（UAVula_TriDataset）根据 split 的 (seq, idx) 白名单过滤 triplets.jsonl，
         并读取 `depth_gt_path` 或推断中心帧同目录的 `depth/*.npy`（默认使用
         `/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_win10_0.03` 作为三元组根目录，以及
         `/mnt/data_nvme3n1p1/dataset/UAV_ula/dataset` 中的原始图像，可通过 CLI 覆盖）；
  3. 汇总整集合与按场景（同一序列）统计：
       - 基础信息：样本数、像素数量、有效像素比例；
       - 全局指标：最小值、最大值、均值、标准差；
       - 分位点：1% / 5% / 10% / 25% / 50% / 75% / 90% / 95% / 99%；
       - 5%-95% 裁剪区间内的均值 / 标准差 / 极值；
  4. 支持 `--limit` 仅分析前 N 条样本，用于快速冒烟测试；
  5. 支持将结果写入 JSON 文件，便于后续可视化或版本留存。

示例：
    python tools/uavula/analyze_val_depth_distribution.py \\
        --data-path /mnt/data_nvme3n1p1/dataset/UAV_ula/dataset
"""

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# -----------------------------
# 常量 / 简易配置
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
DEPTH_EXTS = (".npy", ".npz", ".png", ".tif", ".tiff")
SCENE_ALIASES = {
    "DJI_0166": "0166",
    "DJI_0415": "0415",
    "DJI_0502": "0502",
    "DJI_0502a": "0502",
    "DJI_0502b": "0502",
    "DJI_0502c": "0502",
    "DJI_0477a": "0477",
    "DJI_0477b": "0477",
}
GT_SEQUENCE_GROUPS: Dict[str, List[str]] = {
    "DJI_0502": ["DJI_0502a", "DJI_0502b"],
}
MEMMAP_THRESHOLD = 20_000_000  # 超过该数量的有效像素时使用 memmap（约 80 MB 浮点）


@dataclass
class DepthRecord:
    folder: str
    frame_idx: int
    depth_path: str
    loader: str  # "array" | "bin"
    scene: str


class StatsAccumulator:
    """用于累积基础统计量（像素计数、均值、方差、极值）。"""

    def __init__(self) -> None:
        self.total_pixels = 0
        self.valid_pixels = 0
        self.invalid_pixels = 0
        self.sum_depth = 0.0
        self.sum_depth_sq = 0.0
        self.min_depth = math.inf
        self.max_depth = -math.inf

    def update(self, depth: np.ndarray) -> None:
        total = depth.size
        valid = _filter_valid(depth)
        valid_count = int(valid.size)

        self.total_pixels += total
        self.valid_pixels += valid_count
        self.invalid_pixels += total - valid_count

        if valid_count == 0:
            return

        vals64 = valid.astype(np.float64)
        self.sum_depth += float(vals64.sum())
        self.sum_depth_sq += float((vals64 ** 2).sum())

        vmin = float(valid.min())
        vmax = float(valid.max())
        if vmin < self.min_depth:
            self.min_depth = vmin
        if vmax > self.max_depth:
            self.max_depth = vmax

    def finalize(self) -> Dict[str, float]:
        if self.valid_pixels == 0:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }

        mean = self.sum_depth / self.valid_pixels
        var = max(0.0, (self.sum_depth_sq / self.valid_pixels) - mean * mean)
        std = math.sqrt(var)
        return {
            "min": self.min_depth,
            "max": self.max_depth,
            "mean": mean,
            "std": std,
        }


# -----------------------------
# 文件解析与路径推断
# -----------------------------
def _read_split_lines(split_path: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    with open(split_path, "r", encoding="utf-8") as fp:
        for raw in fp:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            folder = parts[0]
            try:
                frame_idx = int(parts[1])
            except ValueError:
                continue
            samples.append((folder, frame_idx))
    return samples


def _normalize_seq(raw: str) -> str:
    raw = raw.replace("\\", "/").strip()
    if raw.startswith("Train/"):
        raw = raw[len("Train/") :]
    if raw.startswith("Validation/"):
        raw = raw[len("Validation/") :]
    if raw.startswith("./"):
        raw = raw[2:]
    return raw


def parse_split_pairs(lines: Sequence[str]) -> List[Tuple[str, int]]:
    pair_whitelist = []
    for ln in lines:
        if not ln:
            continue
        ln = ln.split("#", 1)[0].strip()
        if not ln:
            continue
        ln = ln.replace("\\", "/")
        try:
            seq_part, idx_str = ln.rsplit(maxsplit=1)
            seq = _normalize_seq(seq_part)
            idx = int(idx_str)
            pair_whitelist.append((seq, idx))
        except ValueError:
            continue
    return pair_whitelist


def _infer_scene_name(folder: str) -> str:
    parts = folder.replace("\\", "/").split("/")
    if not parts:
        return "unknown"
    category = parts[0]
    seq = parts[1] if len(parts) > 1 else category
    alias = SCENE_ALIASES.get(seq, seq)
    return f"{category}/{alias}"


def _resolve_depth_path_array(data_root: str, folder: str, frame_idx: int) -> Optional[str]:
    data_dir = os.path.normpath(os.path.join(data_root, folder))
    base_dir = os.path.dirname(data_dir)
    depth_dir = os.path.join(base_dir, "depth")
    if not os.path.isdir(depth_dir):
        return None

    stem = f"{frame_idx:010d}"
    for ext in DEPTH_EXTS:
        candidate = os.path.join(depth_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _resolve_depth_stereo_bin(data_root: str, folder: str, frame_idx: int) -> Optional[str]:
    data_dir = os.path.normpath(os.path.join(data_root, folder))
    stereo_dir = os.path.join(data_dir, "stereo", "depth_maps")
    if not os.path.isdir(stereo_dir):
        return None

    stem = f"{frame_idx:010d}.jpg.geometric.bin"
    candidate = os.path.join(stereo_dir, stem)
    return candidate if os.path.isfile(candidate) else None


    parts = folder.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None
    seq = parts[1]

    stem = f"{frame_idx:010d}.jpg.geometric.bin"

    candidates: List[str] = []
    group = GT_SEQUENCE_GROUPS.get(seq, [])
    if group:
        candidates.extend(group)
    candidates.append(seq)

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        candidate = os.path.join(
            gt_root,
            cand,
            "image_02",
            "data",
            "images",
            "stereo",
            "depth_maps",
            stem,
        )
        if os.path.isfile(candidate):
            return candidate

    return None


def _frame_idx_from_name(filename: str) -> int:
    stem = Path(filename).stem
    return int(stem.lstrip("0") or "0")


def _infer_triplet_depth_path(
    image_root: str,
    seq: str,
    center_file: str,
) -> Optional[str]:
    center_abs = os.path.join(image_root, seq, center_file)
    data_dir = os.path.dirname(center_abs)
    cam_dir = os.path.dirname(data_dir) if data_dir else None
    if not cam_dir or not os.path.isdir(cam_dir):
        return None

    stem = Path(center_abs).stem
    for depth_dir_name in ("depth", "Depth"):
        depth_dir = os.path.join(cam_dir, depth_dir_name)
        if not os.path.isdir(depth_dir):
            continue
        for ext in DEPTH_EXTS:
            candidate = os.path.join(depth_dir, stem + ext)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    return None


def _resolve_depth_with_loader(
    data_root: str,
    folder: str,
    frame_idx: int,
) -> Optional[Tuple[str, str]]:
    array_path = _resolve_depth_path_array(data_root, folder, frame_idx)
    if array_path is not None:
        return array_path, "array"

    stereo_path = _resolve_depth_stereo_bin(data_root, folder, frame_idx)
    if stereo_path is not None:
        return stereo_path, "bin"

    return None


# -----------------------------
# 深度读取与过滤
# -----------------------------
def _load_depth_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        depth = np.load(path)
    elif ext == ".npz":
        with np.load(path) as data:
            depth = data["arr_0"]
    elif ext in (".png", ".tif", ".tiff"):
        depth = np.array(Image.open(path))
    else:
        raise ValueError(f"Unsupported depth format: {ext} ({path})")

    depth = np.asarray(depth, dtype=np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _read_geometric_bin(path: str) -> np.ndarray:
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        while True:
            byte = fid.read(1)
            if not byte:
                raise ValueError(f"Invalid geometric.bin header: {path}")
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
        array = np.fromfile(fid, np.float32)

    array = array.reshape((width, height, channels), order="F")
    depth = np.transpose(array, (1, 0, 2)).squeeze()
    depth = np.asarray(depth, dtype=np.float32)
    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0] = 0.0
    return depth


def _load_depth(record: DepthRecord) -> np.ndarray:
    if record.loader == "array":
        return _load_depth_array(record.depth_path)
    if record.loader == "bin":
        return _read_geometric_bin(record.depth_path)
    raise ValueError(f"Unknown loader type: {record.loader}")


def _filter_valid(depth: np.ndarray) -> np.ndarray:
    mask = depth > 0
    if not np.any(mask):
        return np.empty((0,), dtype=np.float32)
    return depth[mask].astype(np.float32, copy=False)


def _quantile(data: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    q_array = np.asarray(qs, dtype=np.float64)
    try:
        return np.quantile(data, q_array, method="linear")
    except TypeError:
        return np.quantile(data, q_array, interpolation="linear")


# -----------------------------
# 统计构建辅助
# -----------------------------
def _collect_values(
    records: Sequence[DepthRecord],
    total_valid: int,
    threshold: int = MEMMAP_THRESHOLD,
) -> Tuple[np.ndarray, Optional[str]]:
    if total_valid == 0:
        return np.empty((0,), dtype=np.float32), None

    use_memmap = total_valid > threshold
    if use_memmap:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="uavula_depth_", suffix=".bin")
        os.close(tmp_fd)
        buffer: Union[np.ndarray, np.memmap] = np.memmap(
            tmp_path, mode="w+", dtype=np.float32, shape=(total_valid,)
        )
    else:
        tmp_path = None
        buffer = np.empty((total_valid,), dtype=np.float32)

    offset = 0
    for rec in records:
        depth = _load_depth(rec)
        valid = _filter_valid(depth)
        count = int(valid.size)
        if count == 0:
            continue
        buffer[offset : offset + count] = valid
        offset += count

    if isinstance(buffer, np.memmap):
        buffer.flush()
    if offset != total_valid:
        raise RuntimeError(f"Expected {total_valid} valid pixels, got {offset}")
    return buffer, tmp_path


def _compute_trimmed_stats(
    values: np.ndarray,
    low: float,
    high: float,
) -> Dict[str, Optional[float]]:
    if values.size == 0 or not np.isfinite(low) or not np.isfinite(high):
        return {
            "low": float(low),
            "high": float(high),
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    if isinstance(values, np.memmap):
        block = 5_000_000
        count = 0
        total = 0.0
        total_sq = 0.0
        tmin = math.inf
        tmax = -math.inf
        length = values.shape[0]
        for start in range(0, length, block):
            end = min(length, start + block)
            chunk = np.asarray(values[start:end])
            mask = (chunk >= low) & (chunk <= high)
            if not np.any(mask):
                continue
            inside = chunk[mask].astype(np.float64)
            cnt = inside.size
            count += cnt
            total += float(inside.sum())
            total_sq += float((inside ** 2).sum())
            tmin = min(tmin, float(inside.min()))
            tmax = max(tmax, float(inside.max()))
        if count == 0:
            mean = std = None
            tmin = tmax = None
        else:
            mean = total / count
            var = max(0.0, (total_sq / count) - mean * mean)
            std = math.sqrt(var)
        return {
            "low": float(low),
            "high": float(high),
            "count": count,
            "mean": mean,
            "std": std,
            "min": None if count == 0 else tmin,
            "max": None if count == 0 else tmax,
        }

    mask = (values >= low) & (values <= high)
    if not np.any(mask):
        return {
            "low": float(low),
            "high": float(high),
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    inside = values[mask].astype(np.float64)
    mean = float(inside.mean())
    std = float(inside.std(ddof=0))
    return {
        "low": float(low),
        "high": float(high),
        "count": inside.size,
        "mean": mean,
        "std": std,
        "min": float(inside.min()),
        "max": float(inside.max()),
    }


def _build_summary(
    stats: StatsAccumulator,
    records: Sequence[DepthRecord],
    listed: int,
    missing_depth: int,
) -> Dict[str, object]:
    processed = len(records)
    base = {
        "files": {
            "listed": listed,
            "processed": processed,
            "missing_depth": missing_depth,
        },
        "pixels": {
            "total": stats.total_pixels,
            "valid": stats.valid_pixels,
            "invalid": stats.invalid_pixels,
            "valid_ratio": (
                stats.valid_pixels / stats.total_pixels if stats.total_pixels > 0 else None
            ),
        },
    }

    depth_stats = stats.finalize()
    base["depth"] = depth_stats

    if stats.valid_pixels == 0:
        base["quantiles"] = {}
        base["trim_05_95"] = {
            "low": None,
            "high": None,
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
        return base

    values, tmp_path = _collect_values(records, stats.valid_pixels)
    try:
        quantile_levels = np.array(
            [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
            dtype=np.float64,
        )
        quantiles = _quantile(values, quantile_levels)
        base["quantiles"] = {
            f"q{int(level * 100):02d}": float(val)
            for level, val in zip(quantile_levels, quantiles)
        }

        trim_low = float(quantiles[1])  # 5%
        trim_high = float(quantiles[7])  # 95%
        base["trim_05_95"] = _compute_trimmed_stats(values, trim_low, trim_high)
    finally:
        if isinstance(values, np.memmap):
            del values
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return base


# -----------------------------
# 主流程：按 split 统计
# -----------------------------
def analyze_split(
    name: str,
    entries: Sequence[Tuple[str, int]],
    data_root: str,
    limit: Optional[int],
) -> Dict[str, object]:
    if limit is not None:
        entries = entries[:limit]

    stats_total = StatsAccumulator()
    per_scene_stats: Dict[str, StatsAccumulator] = defaultdict(StatsAccumulator)
    scene_listed = defaultdict(int)
    scene_missing = defaultdict(int)

    records: List[DepthRecord] = []

    total_listed = 0
    missing_depth = 0

    total_entries = len(entries)
    progress_step = max(total_entries // 20, 1) if total_entries else 1

    for folder, frame_idx in entries:
        total_listed += 1
        scene = _infer_scene_name(folder)
        scene_listed[scene] += 1

        resolved = _resolve_depth_with_loader(data_root, folder, frame_idx)
        if resolved is None:
            missing_depth += 1
            scene_missing[scene] += 1
            if total_listed % progress_step == 0 or total_listed == total_entries:
                print(f"[Progress][{name}] {total_listed}/{total_entries} | missing={missing_depth}")
            continue

        path, loader = resolved
        record = DepthRecord(folder=folder, frame_idx=frame_idx, depth_path=path, loader=loader, scene=scene)

        depth = _load_depth(record)
        stats_total.update(depth)
        per_scene_stats[scene].update(depth)
        records.append(record)

        if total_listed % progress_step == 0 or total_listed == total_entries:
            print(f"[Progress][{name}] {total_listed}/{total_entries}")

    summary = _build_summary(stats_total, records, total_listed, missing_depth)

    scene_summaries = {}
    for scene_name in sorted(per_scene_stats.keys()):
        scene_records = [rec for rec in records if rec.scene == scene_name]
        scene_stats = per_scene_stats[scene_name]
        scene_summary = _build_summary(
            scene_stats,
            scene_records,
            scene_listed[scene_name],
            scene_missing.get(scene_name, 0),
        )
        scene_summaries[scene_name] = scene_summary

    return {
        "dataset": name,
        "overall": summary,
        "scenes": scene_summaries,
    }


def _find_triplet_json(triplet_root: str, seq: str) -> Optional[str]:
    candidates = [
        os.path.join(triplet_root, seq, "triplets.jsonl"),
        os.path.join(triplet_root, "Validation", seq, "triplets.jsonl"),
        os.path.join(triplet_root, "Train", seq, "triplets.jsonl"),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    return None


def analyze_triplet_split(
    name: str,
    split_file: str,
    data_root: str,
    triplet_root: str,
    data_subdir: str,
    limit: Optional[int],
) -> Dict[str, object]:
    with open(split_file, "r", encoding="utf-8") as fp:
        raw_lines = fp.readlines()
    pairs = parse_split_pairs(raw_lines)
    if not pairs:
        raise RuntimeError(f"未能从 {split_file} 解析到有效的 (seq, idx) 列表")

    if limit is not None:
        pairs = pairs[:limit]

    whitelist = set(pairs)
    listed_count = len(pairs)

    seq_to_indices: Dict[str, set] = defaultdict(set)
    scene_listed = defaultdict(int)
    for seq, idx in pairs:
        clean_seq = _normalize_seq(seq)
        seq_to_indices[clean_seq].add(idx)
        scene = _infer_scene_name(f"{data_subdir}/{clean_seq}")
        scene_listed[scene] += 1

    image_root = os.path.join(data_root, data_subdir)
    records_map: Dict[Tuple[str, int], DepthRecord] = {}

    for seq in sorted(seq_to_indices.keys()):
        json_path = _find_triplet_json(triplet_root, seq)
        if not json_path:
            continue
        base_dir = os.path.dirname(json_path)
        print(f"[Progress][{name}] loading {json_path}")
        with open(json_path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rec_seq = _normalize_seq(rec.get("seq", seq))
                if rec_seq != seq:
                    continue

                center_info = rec.get("center", {})
                center_file = center_info.get("file")
                if not center_file:
                    continue
                center_idx = _frame_idx_from_name(center_file)
                key = (seq, center_idx)
                if key not in whitelist:
                    continue
                if key in records_map:
                    continue

                depth_path = rec.get("depth_gt_path")
                if depth_path:
                    if not os.path.isabs(depth_path):
                        depth_path = os.path.normpath(os.path.join(base_dir, depth_path))
                if not depth_path:
                    depth_path = _infer_triplet_depth_path(image_root, seq, center_file)
                if not depth_path or not os.path.isfile(depth_path):
                    continue

                folder = f"{data_subdir}/{seq}"
                scene_name = _infer_scene_name(folder)
                records_map[key] = DepthRecord(
                    folder=folder,
                    frame_idx=center_idx,
                    depth_path=depth_path,
                    loader="array",
                    scene=scene_name,
                )

    keys_sorted = [(seq, idx) for seq, idx in pairs]
    records: List[DepthRecord] = []
    stats_total = StatsAccumulator()
    per_scene_stats: Dict[str, StatsAccumulator] = defaultdict(StatsAccumulator)
    scene_records: Dict[str, List[DepthRecord]] = defaultdict(list)

    total_pairs = len(keys_sorted)
    progress_step = max(total_pairs // 20, 1) if total_pairs else 1

    for seq, idx in keys_sorted:
        record = records_map.get((seq, idx))
        if record is None:
            continue
        depth = _load_depth(record)
        stats_total.update(depth)
        per_scene_stats[record.scene].update(depth)
        records.append(record)
        scene_records[record.scene].append(record)

        processed_count = len(records)
        if processed_count % progress_step == 0 or processed_count == total_pairs:
            print(f"[Progress][{name}] {processed_count}/{total_pairs}")

    processed_count = len(records)
    missing_pairs = [key for key in keys_sorted if key not in records_map]
    missing_depth_count = len(missing_pairs)

    scene_missing = defaultdict(int)
    for seq, _ in missing_pairs:
        scene = _infer_scene_name(f"{data_subdir}/{seq}")
        scene_missing[scene] += 1

    overall_summary = _build_summary(
        stats_total,
        records,
        listed_count,
        missing_depth_count,
    )

    scene_summaries = {}
    for scene in sorted(scene_listed.keys()):
        scene_stat = per_scene_stats.get(scene, StatsAccumulator())
        scene_summary = _build_summary(
            scene_stat,
            scene_records.get(scene, []),
            scene_listed[scene],
            scene_missing.get(scene, 0),
        )
        scene_summaries[scene] = scene_summary

    return {
        "dataset": name,
        "overall": overall_summary,
        "scenes": scene_summaries,
    }


# -----------------------------
# CLI
# -----------------------------
def _guess_triplet_root(data_path: str) -> Optional[str]:
    env = os.environ.get("UAVULA_TRI_ROOT")
    if env and os.path.isdir(env):
        return env

    parent = os.path.dirname(os.path.abspath(data_path))
    candidates = []
    try:
        for name in os.listdir(parent):
            full = os.path.join(parent, name)
            if not os.path.isdir(full):
                continue
            lname = name.lower()
            if lname.startswith("tri") or "triplet" in lname or "tri_win" in lname:
                candidates.append(full)
    except FileNotFoundError:
        candidates = []

    def has_triplets(path: str) -> bool:
        for root, _, files in os.walk(path):
            for fname in files:
                if fname.endswith("triplets.jsonl"):
                    return True
        return False

    for path in sorted(candidates):
        if has_triplets(path):
            return path
    return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计 UAVula 验证 / 测试集的深度分布（含按场景细分）。"
    )
    parser.add_argument(
        "--data-path",
        default="/mnt/data_nvme3n1p1/dataset/UAV_ula/dataset",
        help="UAVula dataset 根目录（默认 /mnt/data_nvme3n1p1/dataset/UAV_ula/dataset）。",
    )
    parser.add_argument(
        "--val-split",
        default="methods/splits/UAVula/val_files.txt",
        help="验证集 split 文件路径。",
    )
    parser.add_argument(
        "--test-split",
        default="methods/splits/UAVula/test_files.txt",
        help="测试集 split 文件路径（可选）。",
    )
    parser.add_argument(
        "--tri-triplet-root",
        default="/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_win1_disp0.03_lap0.05_U0.07_R0_Qn0_S0",
        help="三元组 JSON 根目录（默认指向 /mnt/data_nvme3n1p1/dataset/UAV_ula/tri_win10_0.03，可按需覆盖）。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅分析前 N 条样本，用于快速检测。",
    )
    parser.add_argument(
        "--output-json",
        default=os.path.join(PROJECT_ROOT, "tools", "uavula", "uavula_depth_stats.json"),
        help=(
            "若提供路径，则将统计结果写入该 JSON 文件。"
            "默认保存到工程内的 tools/uavula/uavula_depth_stats.json，可按需覆盖。"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    data_path = os.path.abspath(args.data_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"data-path 不存在：{data_path}")

    results = {}

    if args.val_split:
        val_entries = _read_split_lines(args.val_split)
        if val_entries:
            results["val"] = analyze_split(
                "val",
                val_entries,
                data_path,
                args.limit,
            )
        else:
            print(f"[WARN] 验证集 split 为空：{args.val_split}", file=sys.stderr)

    if args.test_split:
        test_entries = _read_split_lines(args.test_split)
        if test_entries:
            results["test"] = analyze_split(
                "test",
                test_entries,
                data_path,
                args.limit,
            )
        else:
            print(f"[WARN] 测试集 split 为空：{args.test_split}", file=sys.stderr)

    tri_split = (args.val_split or "").strip()
    tri_root_arg = (args.tri_triplet_root or "").strip()
    if tri_split:
        tri_subdir = "Validation"
        if tri_root_arg.lower() == "auto" or not tri_root_arg:
            tri_root = _guess_triplet_root(data_path)
            if tri_root is None:
                print("[WARN] 未能自动推断三元组根目录，跳过 tri_val 统计。"
                      "（可通过 --tri-triplet-root 手动指定，或设置环境变量 UAVULA_TRI_ROOT）",
                      file=sys.stderr)
            else:
                results["tri_val"] = analyze_triplet_split(
                    "tri_val",
                    split_file=tri_split,
                    data_root=data_path,
                    triplet_root=tri_root,
                    data_subdir=tri_subdir,
                    limit=args.limit,
                )
        else:
            tri_root = os.path.abspath(tri_root_arg)
            if not os.path.isdir(tri_root):
                print(f"[WARN] 指定的三元组根目录不存在：{tri_root}，跳过 tri_val 统计。", file=sys.stderr)
            else:
                results["tri_val"] = analyze_triplet_split(
                    "tri_val",
                    split_file=tri_split,
                    data_root=data_path,
                    triplet_root=tri_root,
                    data_subdir=tri_subdir,
                    limit=args.limit,
                )
    elif tri_root_arg and tri_root_arg.lower() != "auto":
        print("[WARN] 已指定 --tri-triplet-root 但未提供 --tri-split，跳过 tri_val 统计。", file=sys.stderr)

    if not results:
        print("未执行任何统计任务。", file=sys.stderr)
        return 1

    json_text = json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True)
    print(json_text)

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except PermissionError:
                print(f"[WARN] 无法创建输出目录：{output_dir}（权限不足），请手动创建或指定其他路径。",
                      file=sys.stderr)
            except OSError as exc:
                print(f"[WARN] 创建输出目录失败：{output_dir}（{exc}），请检查路径。",
                      file=sys.stderr)
        try:
            with open(args.output_json, "w", encoding="utf-8") as fp:
                fp.write(json_text)
            print(f"[INFO] 统计结果已保存到：{args.output_json}")
        except OSError as exc:
            print(f"[WARN] 写入输出文件失败：{args.output_json}（{exc}）",
                  file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

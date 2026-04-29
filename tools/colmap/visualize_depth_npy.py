#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将目录中的 .npy 深度图按 5-95 百分位裁剪并可视化为 PNG。

示例：
  python tools/colmap/visualize_depth_npy.py --input_dir /path/to/depth
  python tools/colmap/visualize_depth_npy.py --input_dir /path/to/depth --recursive
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
except Exception as exc:
    raise RuntimeError("需要 matplotlib 用于可视化，请先安装 matplotlib") from exc


def _iter_npy_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob("*.npy")
    else:
        yield from root.glob("*.npy")


def _prepare_depth(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # 常见形状：(1, H, W) 或 (H, W, 1)
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            # 若是多通道，默认取第一个通道
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Unsupported depth shape: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _compute_percentiles(
    depth: np.ndarray,
    q: Tuple[float, float],
    mask_positive: bool,
) -> Tuple[float, float]:
    valid = np.isfinite(depth)
    if mask_positive:
        valid &= depth > 0
    vals = depth[valid]
    if vals.size == 0:
        raise ValueError("No valid pixels for percentile computation")
    lo, hi = np.percentile(vals, [q[0], q[1]])
    return float(lo), float(hi)


def _normalize_clip(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if hi <= lo:
        # 避免除以 0，退化为常数图
        return np.zeros_like(depth, dtype=np.float32)
    clipped = np.clip(depth, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32, copy=False)


def _save_png(norm: np.ndarray, out_path: Path, cmap_name: str) -> None:
    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(norm) * 255.0).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(out_path), rgba)


def visualize_one(
    npy_path: Path,
    input_root: Path,
    output_root: Path,
    q: Tuple[float, float],
    cmap_name: str,
    mask_positive: bool,
) -> None:
    depth = _prepare_depth(np.load(npy_path))
    lo, hi = _compute_percentiles(depth, q, mask_positive)
    norm = _normalize_clip(depth, lo, hi)

    rel = npy_path.relative_to(input_root)
    out_path = output_root / rel.with_suffix(".png")
    _save_png(norm, out_path, cmap_name)

    print(f"[OK] {npy_path} -> {out_path}  (p{q[0]}={lo:.4f}, p{q[1]}={hi:.4f})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize .npy depth maps with percentile clipping.")
    p.add_argument(
        "--input_dir",
        type=str,
        default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China/Test/UAV_seq1/depth",
        help="含 .npy 深度图的目录",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China/Test/UAV_seq1/color",
        help="输出目录，默认与 input_dir 相同",
    )
    p.add_argument("--recursive", action="store_true", help="递归子目录")
    p.add_argument("--percentiles", nargs=2, type=float, default=[5.0, 95.0],
                   help="裁剪分位数，如 5 95")
    p.add_argument("--cmap", type=str, default="magma", help="matplotlib colormap 名称")
    p.add_argument("--keep_zero", action="store_true", help="计算分位数时包含 0/负值")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir).resolve()
    if not input_root.is_dir():
        raise ValueError(f"input_dir 不存在或不是目录: {input_root}")

    if args.output_dir:
        output_root = Path(args.output_dir).resolve()
    else:
        output_root = input_root / "vis_5_95"

    q = (float(args.percentiles[0]), float(args.percentiles[1]))
    if q[0] >= q[1]:
        raise ValueError("percentiles 需要满足 p_low < p_high")

    files = list(_iter_npy_files(input_root, args.recursive))
    if not files:
        print(f"[WARN] 未找到 .npy 文件：{input_root}")
        return

    for npy_path in files:
        try:
            visualize_one(
                npy_path=npy_path,
                input_root=input_root,
                output_root=output_root,
                q=q,
                cmap_name=args.cmap,
                mask_positive=not args.keep_zero,
            )
        except Exception as exc:
            print(f"[FAIL] {npy_path}: {exc}")


if __name__ == "__main__":
    main()

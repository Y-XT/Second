#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 UAVula 原始数据集与 tri_images 目录中的帧数量。

默认假设：
  data-root/
    ├── Train/...
    └── Validation/...
  tri-root/
    └── <seq>/image_02/data/*.jpg

用法示例：
  python tools/uavula/compare_sequence_counts.py \
      --data-root /mnt/data_nvme3n1p1/dataset/UAV_ula/dataset \
      --tri-root /mnt/data_nvme3n1p1/dataset/UAV_ula/tri_images \
      --output logs/uavula_sequence_compare.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare per-sequence frame counts between dataset and tri_images.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data_nvme3n1p1/dataset/UAV_ula/dataset"),
        help="Root directory of the original dataset (expected to contain Train/ and Validation/).",
    )
    parser.add_argument(
        "--tri-root",
        type=Path,
        default=Path("/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_images"),
        help="Root directory containing tri_images sequences.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".jpg",
        help="Image suffix to count (default: .jpg).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/uavula_sequence_compare.csv"),
        help="Optional CSV file to store the comparison table.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="只输出总计摘要，不逐序列列出。",
    )
    return parser


def count_dataset_frames(data_root: Path, suffix: str) -> Dict[str, Dict[str, int]]:
    counts = defaultdict(lambda: {"Train": 0, "Validation": 0})
    for phase in ("Train", "Validation"):
        phase_dir = data_root / phase
        if not phase_dir.exists():
            continue
        for img_path in phase_dir.rglob(f"*{suffix}"):
            rel = img_path.relative_to(phase_dir)
            parts = rel.parts
            if len(parts) < 3:
                continue
            seq = "/".join(parts[:3])
            counts[seq][phase] += 1
    return counts


def count_tri_frames(tri_root: Path, suffix: str) -> Dict[str, int]:
    counts = defaultdict(int)
    if not tri_root.exists():
        return counts
    for img_path in tri_root.rglob(f"*{suffix}"):
        rel = img_path.relative_to(tri_root)
        parts = rel.parts
        if len(parts) < 3:
            continue
        seq = "/".join(parts[:3])
        counts[seq] += 1
    return counts


def format_row(
    seq: str, train_count: int, val_count: int, tri_count: int
) -> Tuple[str, int, int, int, int]:
    dataset_total = train_count + val_count
    diff = tri_count - dataset_total
    return seq, train_count, val_count, dataset_total, tri_count, diff


def main():
    parser = build_parser()
    args = parser.parse_args()

    data_root = args.data_root
    tri_root = args.tri_root
    suffix = args.suffix

    if not data_root.exists():
        print(f"[ERROR] data-root 不存在: {data_root}", file=sys.stderr)
        sys.exit(1)
    if not tri_root.exists():
        print(f"[警告] tri-root 不存在: {tri_root}", file=sys.stderr)

    dataset_counts = count_dataset_frames(data_root, suffix)
    tri_counts = count_tri_frames(tri_root, suffix)

    all_seqs = sorted(set(dataset_counts.keys()) | set(tri_counts.keys()))
    rows = [
        format_row(
            seq,
            dataset_counts[seq]["Train"],
            dataset_counts[seq]["Validation"],
            tri_counts[seq],
        )
        for seq in all_seqs
    ]

    total_train = sum(dataset_counts[seq]["Train"] for seq in all_seqs)
    total_val = sum(dataset_counts[seq]["Validation"] for seq in all_seqs)
    total_dataset = total_train + total_val
    total_tri = sum(tri_counts[seq] for seq in all_seqs)
    total_diff = total_tri - total_dataset

    print("=== Summary ===")
    print(f"data_root: {data_root}")
    print(f"tri_root:  {tri_root}")
    print(f"Image suffix: {suffix}")
    print(f"Total Train frames:       {total_train}")
    print(f"Total Validation frames:  {total_val}")
    print(f"Total Dataset frames:     {total_dataset}")
    print(f"Total Tri frames:         {total_tri}")
    print(f"Tri - Dataset difference: {total_diff}")

    if not args.summary_only:
        header = ["seq", "train_count", "val_count", "dataset_total", "tri_count", "diff"]
        print("\n" + ",".join(header))
        for row in rows:
            print(",".join(str(item) for item in row))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        header = ["seq", "train_count", "val_count", "dataset_total", "tri_count", "diff"]
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["# data_root", str(data_root)])
            writer.writerow(["# tri_root", str(tri_root)])
            writer.writerow(["# suffix", suffix])
            writer.writerow([])
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)
        print(f"\n[INFO] Comparison table saved to: {args.output}")


if __name__ == "__main__":
    main()

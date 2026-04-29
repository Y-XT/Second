#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定将 COLMAP PatchMatchStereo 生成的 depth_maps/*.bin 转换为 depth/*.npy

输入文件名形如：
  0000003605.jpg.geometric.bin
  0000003605.jpg.photometric.bin

固定使用 geometric 深度：
  /mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany/Test/UAV_seq30

输出：
  <seq_root>/depth/0000003605.npy  （float32，保持原量纲）
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import Tuple  # 兼容 Py<3.9

def _read_header_and_seek(fid) -> Tuple[int, int, int]:
    header_vals = []
    token = bytearray()
    while len(header_vals) < 3:
        b = fid.read(1)
        if not b:
            raise ValueError("Unexpected EOF while reading COLMAP header")
        if b == b'&':
            if not token:
                raise ValueError("Empty token in COLMAP header")
            header_vals.append(int(token.decode('ascii')))
            token.clear()
        else:
            token.extend(b)
    return header_vals[0], header_vals[1], header_vals[2]  # (W, H, C)

def read_colmap_depth(bin_path: Path) -> np.ndarray:
    with open(bin_path, "rb") as fid:
        width, height, channels = _read_header_and_seek(fid)
        arr = np.fromfile(fid, dtype=np.float32)
    try:
        arr = arr.reshape((width, height, channels), order="F").transpose(1, 0, 2)
    except ValueError as e:
        raise ValueError(f"{bin_path}: reshape 失败，头部为 (W={width}, H={height}, C={channels})，"
                         f"数据长度={arr.size}") from e
    arr = arr.squeeze()
    if arr.shape[:2] != (height, width):
        raise ValueError(f"Shape mismatch: got {arr.shape}, expected ({height},{width})")
    return arr.astype(np.float32, copy=False)

def convert_one_sequence(
    seq_root: Path,
    src_subdir: str = "stereo/depth_maps",
    dst_subdir: str = "depth",
    prefer: str = "geometric",
):
    src_dir = seq_root / src_subdir
    dst_dir = seq_root / dst_subdir
    dst_dir.mkdir(parents=True, exist_ok=True)

    suffix = f".{prefer}.bin"
    bins = sorted([p for p in src_dir.glob("*.bin") if p.name.endswith(suffix)])
    if not bins:
        print(f"[WARN] {src_dir} 中未找到 *{suffix}；请检查路径/类型。")
        return

    for i, p in enumerate(bins, 1):
        base = re.sub(r"\.(geometric|photometric)\.bin$", "", p.name)
        stem, _ = os.path.splitext(base)
        D = read_colmap_depth(p)
        np.save(dst_dir / f"{stem}.npy", D, allow_pickle=False)
        if i % 20 == 0 or i == len(bins):
            dmin = float(np.nanmin(D)) if D.size else float("nan")
            dmax = float(np.nanmax(D)) if D.size else float("nan")
            print(f"[{i}/{len(bins)}] {stem}.npy  shape={D.shape}  min={dmin:.4f} max={dmax:.4f}")

def main():
    seq_root = Path("/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China/Validation/sfm/seq41").resolve()
    prefer = "geometric"
    convert_one_sequence(seq_root, prefer=prefer)

if __name__ == "__main__":
    main()

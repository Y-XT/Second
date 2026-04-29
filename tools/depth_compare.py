
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

/mnt/data_nvme3n1p1/dataset/UAV_ula/compare

------------------------
Depth Comparison Toolkit
------------------------
功能：
- 扫描数据目录，按同名文件（stem）对齐：RGB、GT 深度、各方法预测深度
- 支持 PNG(8/16bit)、NPY 深度文件
- 可选单目深度的中位数尺度对齐（median scaling）
- 可选 KITTI Eigen 裁剪
- 计算常见指标：AbsRel、SqRel、RMSE、RMSE_log、SILog、δ1/δ2/δ3
- 导出：
  - 每个样本的对比图（RGB、GT、每个方法的深度 + 误差图）
  - metrics_summary.csv（按方法汇总与逐样本指标）
  - report.html（简易浏览页面，便于人工对比）

目录假设：
data_root/
  images/               # RGB，可为 jpg/png，文件名与深度图同名
  gt_depth/             # GT 深度（png 或 npy），与 RGB 同名
  methods/
    methodA/
    methodB/
  （可选）mask/         # 无效像素掩码，白色=有效，黑色=无效；或 PNG 单通道 0/255；文件名同名

使用示例：
python depth_compare.py \
  --data-root /path/to/dataset \
  --methods methodA methodB \
  --out /path/to/out \
  --depth-scale 256 \
  --scale median \
  --eigen-crop
"""
import argparse
import os
import os.path as osp
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import imageio.v2 as imageio
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 后端不弹窗
import matplotlib.pyplot as plt
import csv
import html
from collections import defaultdict

# ----------------------------- IO 工具 -----------------------------
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
DEPTH_EXTS = {'.png', '.npy', '.tif', '.tiff'}

def _stem(p: str) -> str:
    return osp.splitext(osp.basename(p))[0]

def read_rgb(path: str) -> np.ndarray:
    im = Image.open(path).convert('RGB')
    return np.array(im)

def read_depth(path: str, depth_scale: float = 1.0) -> np.ndarray:
    ext = osp.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path).astype(np.float32)
    else:
        arr = imageio.imread(path)
        if arr.dtype == np.uint16 or arr.dtype == np.uint32:
            arr = arr.astype(np.float32) / float(depth_scale)
        elif arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / float(depth_scale)
        else:
            arr = arr.astype(np.float32)
    return arr

def read_mask(path: str) -> np.ndarray:
    # 返回布尔有效掩码（True=有效）
    arr = imageio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1:
        mask = arr > 127
    else:
        mask = arr.astype(bool)
    return mask

# ----------------------------- 评估工具 -----------------------------
def eigen_crop_mask(h: int, w: int) -> np.ndarray:
    # KITTI Eigen crop，参考常用设置
    top = int(0.40810811 * h)
    bottom = int(0.99189189 * h)
    left = int(0.03594771 * w)
    right = int(0.96405229 * w)
    mask = np.zeros((h, w), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask

def compute_metrics(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    # 屏蔽非法
    m = mask & np.isfinite(gt) & np.isfinite(pred) & (gt > 0)
    if m.sum() == 0:
        return {k: float('nan') for k in [
            'abs_rel','sq_rel','rmse','rmse_log','silog','delta1','delta2','delta3'
        ]}
    p = pred[m].astype(np.float64)
    g = gt[m].astype(np.float64)

    # AbsRel / SqRel
    abs_rel = np.mean(np.abs(p - g) / g)
    sq_rel = np.mean(((p - g) ** 2) / g)

    # RMSE / RMSE_log
    rmse = np.sqrt(np.mean((p - g) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(np.maximum(p,1e-6)) - np.log(np.maximum(g,1e-6))) ** 2))

    # SILog（scale-invariant log RMSE）
    d = np.log(np.maximum(p,1e-6)) - np.log(np.maximum(g,1e-6))
    silog = np.sqrt(np.mean(d**2) - (np.mean(d)**2)) * 100.0

    # delta thresholds
    max_ratio = np.maximum(p / g, g / p)
    delta1 = np.mean(max_ratio < 1.25)
    delta2 = np.mean(max_ratio < 1.25 ** 2)
    delta3 = np.mean(max_ratio < 1.25 ** 3)

    return {
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'silog': float(silog),
        'delta1': float(delta1),
        'delta2': float(delta2),
        'delta3': float(delta3),
    }

def median_scale(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    m = mask & np.isfinite(gt) & np.isfinite(pred) & (gt > 0)
    if m.sum() == 0:
        return 1.0
    scale = np.median(gt[m]) / np.median(np.maximum(pred[m], 1e-8))
    return float(scale)

# ----------------------------- 可视化 -----------------------------
def percentile_minmax(arr: np.ndarray, mask: np.ndarray, lo: float=2.0, hi: float=98.0) -> Tuple[float,float]:
    valid = arr[mask & np.isfinite(arr)]
    if valid.size < 10:
        return float(np.nanmin(valid)) if valid.size>0 else 0.0, float(np.nanmax(valid)) if valid.size>0 else 1.0
    vmin = float(np.percentile(valid, lo))
    vmax = float(np.percentile(valid, hi))
    if not np.isfinite(vmax) or not np.isfinite(vmin) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(valid)), float(np.nanmax(valid))
    return vmin, vmax

def make_figure(rgb: np.ndarray,
                gt: np.ndarray,
                preds: Dict[str, np.ndarray],
                mask: np.ndarray,
                vmin: float,
                vmax: float,
                out_png: str):
    n_methods = len(preds)
    # 布局：
    # 第1行：RGB | GT
    # 之后每行：方法深度 | 误差热力图(|pred - gt|)
    rows = 1 + n_methods
    cols = 2
    fig_h = 3 * rows
    fig_w = 6 * cols / 2
    plt.figure(figsize=(fig_w, fig_h), dpi=150)

    # row 1
    ax = plt.subplot(rows, cols, 1)
    ax.imshow(rgb)
    ax.set_title('RGB')
    ax.axis('off')

    ax = plt.subplot(rows, cols, 2)
    im = ax.imshow(gt, vmin=vmin, vmax=vmax)
    ax.set_title('GT depth')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # methods
    r = 2
    for name, dp in preds.items():
        ax = plt.subplot(rows, cols, (r-1)*cols + 1)
        im1 = ax.imshow(dp, vmin=vmin, vmax=vmax)
        ax.set_title(f'{name}')
        ax.axis('off')
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        ax = plt.subplot(rows, cols, (r-1)*cols + 2)
        err = np.abs(dp - gt)
        # 误差图范围用 GT 分位差范围的一定比例，避免奇异值影响
        emax = np.nanpercentile(err[mask], 98) if np.isfinite(err[mask]).any() else 1.0
        im2 = ax.imshow(err, vmin=0.0, vmax=emax)
        ax.set_title(f'{name} | |pred-gt|')
        ax.axis('off')
        plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

        r += 1

    plt.tight_layout()
    os.makedirs(osp.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

# ----------------------------- 主流程 -----------------------------
def collect_stems(img_dir: str, gt_dir: str) -> List[str]:
    img_paths = { _stem(p): p for ext in IMG_EXTS for p in glob(osp.join(img_dir, f'*{ext}')) }
    gt_paths = { _stem(p): p for ext in DEPTH_EXTS for p in glob(osp.join(gt_dir, f'*{ext}')) }
    stems = sorted(set(img_paths.keys()) & set(gt_paths.keys()))
    return stems

def main():
    parser = argparse.ArgumentParser(description='Depth comparison & reporting')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--methods', type=str, nargs='+', required=True, help='methods 子目录名（位于 data_root/methods 下）')
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--depth-scale', type=float, default=256.0, help='读取整数深度时的缩放因子，常见 256/1000/1')
    parser.add_argument('--scale', type=str, default='none', choices=['none','median'], help='预测与 GT 的尺度对齐方式')
    parser.add_argument('--eigen-crop', action='store_true', help='是否使用 KITTI Eigen 裁剪')
    parser.add_argument('--mask-dir', type=str, default=None, help='可选：有效像素掩码目录（data_root/mask）')
    args = parser.parse_args()

    img_dir = osp.join(args.data_root, 'images')
    gt_dir = osp.join(args.data_root, 'gt_depth')
    mtd_root = osp.join(args.data_root, 'methods')
    if args.mask_dir is None:
        mask_dir = osp.join(args.data_root, 'mask')
    else:
        mask_dir = args.mask_dir

    stems = collect_stems(img_dir, gt_dir)
    if len(stems) == 0:
        raise RuntimeError('未在 images/ 与 gt_depth/ 中发现可对齐的同名样本')

    # 统计结构
    per_sample_rows = []
    per_method_metrics = defaultdict(lambda: defaultdict(list))

    figs_dir = osp.join(args.out, 'figs')
    os.makedirs(figs_dir, exist_ok=True)

    for i, sid in enumerate(stems, 1):
        rgb_path = None
        for ext in IMG_EXTS:
            p = osp.join(img_dir, sid + ext)
            if osp.exists(p):
                rgb_path = p
                break
        gt_path = None
        for ext in DEPTH_EXTS:
            p = osp.join(gt_dir, sid + ext)
            if osp.exists(p):
                gt_path = p
                break
        if rgb_path is None or gt_path is None:
            continue

        rgb = read_rgb(rgb_path)
        gt = read_depth(gt_path, depth_scale=args.depth_scale)

        h, w = gt.shape[:2]
        valid_mask = np.ones((h, w), dtype=bool)
        if args.eigen_crop:
            valid_mask &= eigen_crop_mask(h, w)
        # 禁用 gt 中的零/无穷
        valid_mask &= np.isfinite(gt) & (gt > 0)

        # 读取可选 mask
        if osp.isdir(mask_dir):
            mpath = None
            for ext in ('.png', '.jpg', '.jpeg', '.bmp'):
                p = osp.join(mask_dir, sid + ext)
                if osp.exists(p):
                    mpath = p
                    break
            if mpath is not None:
                valid_mask &= read_mask(mpath)

        # 读取各方法预测
        preds = {}
        for mname in args.methods:
            mprefix = osp.join(mtd_root, mname, sid)
            ppath = None
            for ext in DEPTH_EXTS:
                p = mprefix + ext
                if osp.exists(p):
                    ppath = p
                    break
            if ppath is None:
                continue
            dp = read_depth(ppath, depth_scale=args.depth_scale)
            if dp.shape != gt.shape:
                # 简单 resize 到 GT 尺寸（双线性），保持数值不变
                dp_img = Image.fromarray(dp.astype(np.float32))
                dp = np.array(dp_img.resize((w, h), resample=Image.BILINEAR))
            preds[mname] = dp

        if not preds:
            continue

        # 尺度对齐 & 指标
        scaled_preds = {}
        for mname, dp in preds.items():
            if args.scale == 'median':
                s = median_scale(dp, gt, valid_mask)
            else:
                s = 1.0
            dps = dp * s
            scaled_preds[mname] = dps

        vmin, vmax = percentile_minmax(gt, valid_mask, 2.0, 98.0)

        # 出图
        out_png = osp.join(figs_dir, f'{sid}.png')
        make_figure(rgb, gt, scaled_preds, valid_mask, vmin, vmax, out_png)

        # 逐方法指标
        for mname, dps in scaled_preds.items():
            m = compute_metrics(dps, gt, valid_mask)
            row = {'id': sid, 'method': mname, **m}
            per_sample_rows.append(row)
            for k, v in m.items():
                per_method_metrics[mname][k].append(v)

        print(f'[{i}/{len(stems)}] {sid} done.')

    # 导出 CSV
    os.makedirs(args.out, exist_ok=True)
    csv_path = osp.join(args.out, 'metrics_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['id','method','abs_rel','sq_rel','rmse','rmse_log','silog','delta1','delta2','delta3']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_sample_rows:
            writer.writerow(r)

    # 汇总均值
    summary_lines = []
    summary_lines.append('method,abs_rel,sq_rel,rmse,rmse_log,silog,delta1,delta2,delta3')
    for mname, d in per_method_metrics.items():
        means = {k: np.nanmean(v) if len(v)>0 else float('nan') for k, v in d.items()}
        summary_lines.append(','.join([mname] + [f'{means[k]:.6f}' for k in ['abs_rel','sq_rel','rmse','rmse_log','silog','delta1','delta2','delta3']]))

    with open(osp.join(args.out, 'method_means.csv'), 'w') as f:
        f.write('\n'.join(summary_lines))

    # 生成 HTML 报告
    html_path = osp.join(args.out, 'report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('<!doctype html><meta charset="utf-8"><title>Depth Comparison Report</title>')
        f.write('<style>body{font-family: sans-serif;max-width:1200px;margin:20px auto;padding:0 16px}')
        f.write('.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}')
        f.write('.card{border:1px solid #ddd;border-radius:8px;padding:8px} .card img{width:100%;height:auto;border-radius:6px}')
        f.write('table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:6px 8px;text-align:right} th{text-align:center}')
        f.write('</style>')
        f.write('<h1>Depth Comparison Report</h1>')
        f.write('<p>对齐规则：按同名文件（RGB/GT/各方法）。度量在有效掩码上计算。色条范围基于 GT 的 2%-98% 分位。</p>')

        # 方法均值表
        f.write('<h2>方法总体均值</h2><table><tr><th>method</th><th>abs_rel</th><th>sq_rel</th><th>rmse</th><th>rmse_log</th><th>silog</th><th>δ1</th><th>δ2</th><th>δ3</th></tr>')
        for line in summary_lines[1:]:
            parts = line.split(',')
            f.write('<tr>' + ''.join([f'<td>{html.escape(p)}</td>' for p in parts]) + '</tr>')
        f.write('</table>')

        # 样本墙
        f.write('<h2>样本对比</h2><div class="grid">')
        fig_files = sorted(glob(osp.join(figs_dir, '*.png')))
        for fp in fig_files:
            sid = _stem(fp)
            f.write('<div class="card">')
            f.write(f'<div style="font-weight:600;margin:4px 0 8px">{html.escape(sid)}</div>')
            rel = osp.relpath(fp, args.out).replace('\\','/')
            f.write(f'<img loading="lazy" src="{rel}" alt="{html.escape(sid)}">')
            f.write('</div>')
        f.write('</div>')

        f.write('<p>CSV 输出：<code>metrics_summary.csv</code>（逐样本），<code>method_means.csv</code>（各方法均值）。</p>')

    print(f'完成。HTML: {html_path}')
    print(f'逐样本指标: {csv_path}')
    print(f'均值指标: {osp.join(args.out, "method_means.csv")}')

if __name__ == "__main__":
    main()

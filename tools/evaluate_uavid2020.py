#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAV 单目深度推理与评测（去除视图合成；支持“depth 基准配对”；统一输出目录）

本版特性
--------
- **仅保留**传统深度指标：AbsRel / SqRel / RMSE / RMSE_log / δ<1.25^i / SI-Log / Edge-F1
- **删除**视图合成（PSNR/SSIM）相关全部代码与参数
- **支持以 depth 为基准配对**：depth 位于 <root>/<depth_subdir>，image 位于 <root>/<img_subdir>
  - depth 支持 .npy / .tif / .tiff（可通过 --depth_exts 扩展）
  - image 扩展由 --ext（首选）与 --img_exts（候选列表）决定
  - 仅对“有 GT 的图像”参与评测
- **统一输出目录策略**：始终写入 `<output_base>/<exp_name>_<weights_name>/{npy,vis,eval_result.txt}`
  - 若 `--output_base` 为空，则回落到 `./outputs`
- **RMSE_log 采用标准定义**：`sqrt(mean((log(gt)-log(pred))^2))`
"""

import os
import glob
import argparse
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from torchvision import transforms

import cv2  # 用于 Edge-F1 与 TIFF 读取
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)  # 确保可直接 import methods 包

from methods import networks
from layers import disp_to_depth
import csv

# ==============================
# 常量定义（用于输出结构）
# ==============================
ERROR_METRICS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "si_log"]
METRIC_NAMES = ERROR_METRICS + ["edgeF1"]
CSV_COLUMNS  = ["image_id"] + METRIC_NAMES + ["scale_ratio", "scale_ratio_median", "scale_ratio_rel_std"]
DEFAULT_FEED_HEIGHT = 288
DEFAULT_FEED_WIDTH = 512


# ==============================
# Argument Parsing
# ==============================

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='Predict and Evaluate Depth (UAV)')

    # 数据根 & 子目录
    parser.add_argument('--gt_folder', type=str,
                        #default='/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany/Test',
                        default='/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China/Test',
                        help='Test 根目录，下面为各序列 UAV_seqXX/')

    parser.add_argument('--img_subdir', type=str, default='images',
                        help='每个序列内的图像子目录（默认 images）')

    parser.add_argument('--depth_subdir', type=str, default='depth',
                        help='每个序列内的深度 GT 子目录（默认 depth）')

    # 可选扩展名（image / depth）
    parser.add_argument('--ext', type=str, default='jpg',
                        help='首选图像扩展名（如 jpg 或 png）')

    parser.add_argument('--img_exts', nargs='+', type=str, default=['jpg', 'png', 'jpeg'],
                        help='额外允许的图像扩展名（不带点）')

    parser.add_argument('--depth_exts', nargs='+', type=str, default=['.npy'],
                        help='允许的深度扩展名（带点）；默认仅 .npy')

    # 输出与实验（统一输出目录策略）
    parser.add_argument('--output_base', type=str, default='/home/yxt/文档/mono_result/eval',
                        help='集中输出根目录；为空则回落 ./outputs')
    parser.add_argument('--exp_dir', type=str,
                        default='/mnt/data_nvme3n1p1/mono_weights/weights/UAVid_China/monovit_vggt_rflow_tinj_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20',
                        help='实验目录：包含 models/weights_x 的根目录')
    parser.add_argument('--weights', type=str, default='weights_29',
                        help="'latest' | 整数（如 7）| 目录名（如 weights_7）")
    parser.add_argument('--model_name', type=str, default='auto',
                        help='模型分支（auto | MonoViT* | MonoDepth2*）')
    
    # 可选评估 Mask（黑=不评估，白=评估）China need mask
    parser.add_argument('--mask_subdir', type=str, 
                        default='colmap_masks',
                        help='每个序列内的 mask 子目录（空则不使用 mask）')
    parser.add_argument('--mask_exts', nargs='+', type=str, default=['png'],
                        help='mask 扩展名（不带点），优先按 depth 同名；若不存在则按 image 文件名（含扩展）匹配')
    
    # 深度范围与基础选项
    parser.add_argument('--min_depth', type=float, default=0.1)
    parser.add_argument('--max_depth', type=float, default=150.0)

    parser.add_argument('--batch_size', type=int, default=8,
                        help='一次前向处理的图像数量 (>=1)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'auto'],
                        help="推理设备：'cuda'（默认，需可用 GPU）、'cpu' 或 'auto'")

    # 评测与可视化开关
    parser.add_argument('--do_evaluation', action='store_false', default=True,
                        help='是否基于 GT 计算误差（默认 True；出现该开关则置 False）')
    parser.add_argument('--visualize', action='store_false', default=True,
                        help='是否导出可视化图（默认 True；出现该开关则置 False）')
    parser.add_argument('--gt_clip_percentile', nargs=2, type=float, default=[5.0, 95.0],
                        help='误差计算前对 GT 深度进行分位裁剪以抑制极端值，格式 [low, high]')



    return parser.parse_args()


# ==============================
# 输出目录工具（统一）
# ==============================

def prepare_output_dirs(output_base: str, run_dir_name: str, visualize: bool) -> Tuple[str, str, Optional[str]]:
    """创建输出目录：<output_base>/<run_dir_name>/{npy,vis}"""
    root = (output_base or '').strip() or './outputs'
    output_dir = os.path.join(root, run_dir_name)
    npy_dir = os.path.join(output_dir, 'npy')
    os.makedirs(npy_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'vis') if visualize else None
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    return output_dir, npy_dir, vis_dir


# ==============================
# I/O 与可视化
# ==============================

def save_colormap_img(array: np.ndarray, path: str,
                      cmap: str = 'plasma',
                      vmin: Optional[float] = None,
                      vmax: Optional[float] = None,
                      mask: Optional[np.ndarray] = None,
                      colorbar_suffix: str = "_colorbar",
                      colorbar_ticks: bool = True) -> None:
    if vmin is None:
        vmin = float(np.nanmin(array))
    if vmax is None:
        vmax = float(np.nanmax(array))

    display = np.array(array, dtype=np.float32)
    if mask is not None:
        display = np.where(mask.astype(bool), display, float(vmin))
    display = np.nan_to_num(display, nan=float(vmin), posinf=float(vmax), neginf=float(vmin))

    # 强制保存为 RGB（无 alpha），避免 masked/nan 导致透明区域。
    span = max(float(vmax) - float(vmin), 1e-12)
    normed = np.clip((display - float(vmin)) / span, 0.0, 1.0)
    cmap_fn = mpl.colormaps[cmap]
    rgb = cmap_fn(normed, bytes=True)[..., :3]
    Image.fromarray(rgb, mode='RGB').save(path)

    base, ext = os.path.splitext(path)
    bar_path = f"{base}{colorbar_suffix}{ext}"
    width = max(display.shape[1], 2)
    bar_height = max(12, display.shape[0] // 20)

    fig_width = max(width / 200.0, 1.5)
    fig_height = max(bar_height / 200.0, 0.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.45)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cbar.set_label('')
    cbar.outline.set_linewidth(0)

    if colorbar_ticks:
        ticks = np.linspace(vmin, vmax, num=5)
        cbar.set_ticks(ticks)
        if vmax - vmin >= 100:
            labels = [f"{t:.0f}" for t in ticks]
        elif vmax - vmin >= 10:
            labels = [f"{t:.1f}" for t in ticks]
        else:
            labels = [f"{t:.2f}" for t in ticks]
        cbar.ax.set_xticklabels(labels)
        cbar.ax.tick_params(axis='x', length=3, pad=2, labelsize=9)
    else:
        cbar.set_ticks([])

    fig.savefig(bar_path, dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def process_image(img_path: str,
                  feed_width: int,
                  feed_height: int,
                  norm_cfg: Optional[Dict[str, object]] = None) -> torch.Tensor:
    input_img = Image.open(img_path).convert('RGB')
    resample = getattr(Image, 'Resampling', Image).LANCZOS
    input_img = input_img.resize((feed_width, feed_height), resample)
    tensor = transforms.ToTensor()(input_img)

    if norm_cfg and norm_cfg.get('normalize', False):
        mode = norm_cfg.get('norm_mode', 'imagenet')
        if mode == 'imagenet':
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean = norm_cfg.get('norm_mean')
            std = norm_cfg.get('norm_std')
            if mean is None or std is None:
                raise ValueError('custom/sat 模式需要提供 mean/std')
        tensor = transforms.Normalize(mean=mean, std=std)(tensor)

    return tensor.unsqueeze(0)


def read_depth_file(path: str) -> np.ndarray:
    """读取 .npy / .tif / .tiff 深度为 float32（保持原始量纲）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        d = np.load(path)
    elif ext in ('.tif', '.tiff'):
        d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if d is None:
            raise FileNotFoundError(f'无法读取深度文件：{path}')
    else:
        raise ValueError(f'不支持的深度格式：{ext}（仅支持 .npy/.tif/.tiff）')
    return d.astype(np.float32)


def read_mask_file(path: str) -> np.ndarray:
    """读取 mask 为二维数组（保持原始灰度/二值）。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        m = np.load(path)
    else:
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f'无法读取 mask 文件：{path}')
    if m.ndim == 3:
        m = m[..., 0]
    return m


# ==============================
# 指标计算
# ==============================

def compute_silog(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> float:
    e = np.log(np.clip(pred, eps, None)) - np.log(np.clip(gt, eps, None))
    m1 = np.mean(e ** 2)
    m2 = (np.mean(e)) ** 2
    return float(np.sqrt(max(m1 - m2, 0.0)))


def compute_errors(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, ...]:
    """传统指标计算；pred 假定已与 gt 尺度对齐。rmse_log 为标准定义。"""
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    si_log = compute_silog(gt, pred)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, si_log


# ---------- Edge-F1（边缘一致性） ----------

def _depth_edge_binary(depth: np.ndarray, q: int = 95) -> np.ndarray:
    d = depth.astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag[np.isfinite(mag)], q)
    return mag >= thr


def _binary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray, dilate: int = 1) -> float:
    return _binary_f1_masked(pred_bin, gt_bin, dilate=dilate, mask=None)


def _binary_f1_masked(pred_bin: np.ndarray,
                      gt_bin: np.ndarray,
                      dilate: int = 1,
                      mask: Optional[np.ndarray] = None) -> float:
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        pred_bin = cv2.dilate(pred_bin.astype(np.uint8), k) > 0
        gt_bin   = cv2.dilate(gt_bin.astype(np.uint8), k) > 0
    if mask is not None:
        pred_bin = np.logical_and(pred_bin, mask)
        gt_bin = np.logical_and(gt_bin, mask)
    tp = np.logical_and(pred_bin, gt_bin).sum()
    pp = pred_bin.sum()
    gp = gt_bin.sum()
    if pp == 0 or gp == 0:
        return 0.0
    prec = tp / float(pp + 1e-12)
    rec  = tp / float(gp + 1e-12)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def compute_edge_f1(pred_depth: np.ndarray,
                    gt_depth: Optional[np.ndarray] = None,
                    img_rgb: Optional[np.ndarray] = None,
                    q: int = 95,
                    mask: Optional[np.ndarray] = None) -> float:
    pred_edge = _depth_edge_binary(pred_depth, q=q)
    if gt_depth is not None:
        gt_edge = _depth_edge_binary(gt_depth, q=q)
    else:
        assert img_rgb is not None, 'Edge-F1 退化模式需要提供 img_rgb'
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gt_edge = cv2.Canny(gray, 80, 160) > 0
    return float(_binary_f1_masked(pred_edge, gt_edge, dilate=1, mask=mask))


# ==============================
# 模型加载
# ==============================

def _select_weights_dir(exp_dir: str, weights_sel: str) -> Tuple[str, str]:
    exp_dir = os.path.abspath(exp_dir)
    models_dir = os.path.join(exp_dir, 'models')
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f'未找到 models 目录：{models_dir}')

    if weights_sel == 'latest':
        cands = [d for d in glob.glob(os.path.join(models_dir, 'weights_*')) if os.path.isdir(d)]
        if not cands:
            raise FileNotFoundError(f'未找到 {models_dir}/weights_*')
        def suf(p):
            b = os.path.basename(p)
            try:
                return int(b.split('_')[-1])
            except Exception:
                return -1
        weights_dir = max(cands, key=suf)
    else:
        if isinstance(weights_sel, str) and weights_sel.startswith('weights_'):
            weights_dir = os.path.join(models_dir, weights_sel)
        else:
            idx = int(weights_sel)
            weights_dir = os.path.join(models_dir, f'weights_{idx}')
        if not os.path.isdir(weights_dir):
            raise FileNotFoundError(f'指定权重目录不存在：{weights_dir}')

    return weights_dir, os.path.basename(weights_dir)


def _strip_prefix_if_present(state_dict: Dict[str, object], prefix: str) -> Dict[str, object]:
    """移除 state_dict key 的公共前缀（常见于 DDP 的 module. 前缀）。"""
    out: Dict[str, object] = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def _load_state_dict_file(path: str, device: torch.device) -> Dict[str, object]:
    sd = torch.load(path, map_location=device)
    if not isinstance(sd, dict):
        raise TypeError(f'权重文件不是字典：{path}')
    # 常见兼容：DataParallel 保存的 "module." 前缀。
    return _strip_prefix_if_present(sd, 'module.')


def _resolve_ckpt_path(model_path: str, filenames: List[str], tag: str) -> str:
    """按候选文件名顺序查找 checkpoint 文件并返回首个存在路径。"""
    tried = []
    for name in filenames:
        path = os.path.join(model_path, name)
        tried.append(path)
        if os.path.isfile(path):
            return path
    tried_text = '\n  - '.join(tried)
    raise FileNotFoundError(f'[{tag}] 未找到可用权重文件，尝试了:\n  - {tried_text}')


def _infer_model_branch_from_encoder_sd(enc_sd: Dict[str, object]) -> Optional[str]:
    keys = [k for k in enc_sd.keys() if isinstance(k, str)]
    if not keys:
        return None

    if any(k.startswith('stem.') for k in keys) and any(k.startswith('patch_embed_stages.') for k in keys):
        return 'MonoViT'

    if any(k.startswith('encoder.conv1.') for k in keys) and any(k.startswith('encoder.layer1.') for k in keys):
        return 'MonoDepth2'

    return None


def _infer_model_branch_from_opt_json(weights_dir: str) -> Optional[str]:
    """
    从 <exp>/models/opt.json 读取 methods 字段。
    weights_dir 形如 .../models/weights_x。
    """
    model_root = os.path.dirname(os.path.abspath(weights_dir))
    opt_path = os.path.join(model_root, "opt.json")
    if not os.path.isfile(opt_path):
        return None
    try:
        with open(opt_path, "r", encoding="utf-8") as f:
            opt = json.load(f)
    except Exception as e:
        print(f"[Warning] 读取 opt.json 失败，忽略 methods 提示: {e}")
        return None

    method = str(opt.get("methods", "")).strip().lower()
    if not method:
        return None
    if method.startswith("monovit"):
        return "MonoViT"
    if method.startswith("monodepth2") or method.startswith("md2"):
        return "MonoDepth2"
    return None


def _resolve_model_branch(model_name: str) -> str:
    name = (model_name or '').strip().lower()
    if not name or name == 'auto':
        return 'auto'
    if name.startswith('monovit'):
        return 'MonoViT'
    if name.startswith('monodepth2') or name.startswith('md2'):
        return 'MonoDepth2'
    raise ValueError(f"Unsupported model_name for paper eval: {model_name}")


def _decode_disp_to_eval_depth(disp: torch.Tensor, args: argparse.Namespace, model_branch: str) -> torch.Tensor:
    _, depth = disp_to_depth(disp, args.min_depth, args.max_depth)
    return depth


def _load_partial_state_dict(module: torch.nn.Module,
                             ckpt_sd: Dict[str, object],
                             tag: str) -> None:
    """仅加载 key/shape 同时匹配的参数，并打印匹配统计。"""
    model_sd = module.state_dict()
    matched = {}
    shape_mismatch = 0
    for k, v in ckpt_sd.items():
        if k not in model_sd:
            continue
        try:
            if tuple(v.shape) != tuple(model_sd[k].shape):
                shape_mismatch += 1
                continue
        except Exception:
            # 非 Tensor 类型，直接跳过
            continue
        matched[k] = v

    if not matched:
        sample = ', '.join(list(ckpt_sd.keys())[:8])
        raise RuntimeError(
            f'[{tag}] 未找到可加载的参数（0/{len(model_sd)}）。'
            f'通常是 model_name 与 checkpoint 不匹配。'
            f'checkpoint 示例 key: {sample}'
        )

    module.load_state_dict(matched, strict=False)
    print(f'[{tag}] matched {len(matched)}/{len(model_sd)} keys'
          + (f' (shape mismatch: {shape_mismatch})' if shape_mismatch else ''))


def load_model(args: argparse.Namespace, device: torch.device):
    assert args.model_name is not None, 'You must specify --model_name'
    model_path = args.weights_dir
    requested_branch = _resolve_model_branch(args.model_name)
    inferred_branch_from_opt = _infer_model_branch_from_opt_json(model_path)
    encoder_path = os.path.join(model_path, 'encoder.pth')
    depth_path = os.path.join(model_path, 'depth.pth')
    encoder_sd = _load_state_dict_file(encoder_path, device)
    inferred_branch = _infer_model_branch_from_encoder_sd(encoder_sd)

    if requested_branch == 'auto':
        model_branch = inferred_branch_from_opt or inferred_branch or 'MonoDepth2'
    else:
        model_branch = requested_branch
        preferred_inferred = inferred_branch_from_opt or inferred_branch
        if preferred_inferred is not None and model_branch != preferred_inferred:
            print(f"[Warning] --model_name={args.model_name} 与 checkpoint 推断分支({preferred_inferred})不一致，已自动采用 {preferred_inferred}")
            model_branch = preferred_inferred

    if model_branch not in {"MonoViT", "MonoDepth2"}:
        raise ValueError(f"Unsupported model branch for paper eval: {model_branch}")

    print(f'[model_name]  requested={args.model_name} | resolved={model_branch}'
          + (f' | inferred={inferred_branch}' if inferred_branch else '')
          + (f' | inferred_from_opt={inferred_branch_from_opt}' if inferred_branch_from_opt else ''))

    if model_branch == 'MonoViT':
        decoder_path = _resolve_ckpt_path(model_path, ['decoder.pth', 'depth.pth'], 'MonoViT.decoder')
        print('Loading MonoViT encoder:', encoder_path)
        print('Loading MonoViT decoder:', decoder_path)

        model = networks.DeepNet(type='mpvitnet')
        encoder = model.encoder
        decoder = model.decoder

        enc_sd = encoder_sd if encoder_sd is not None else _load_state_dict_file(encoder_path, device)
        _load_partial_state_dict(encoder, enc_sd, 'MonoViT.encoder')
        encoder.to(device).eval()
        feed_height = int(enc_sd.get('height', DEFAULT_FEED_HEIGHT))
        feed_width  = int(enc_sd.get('width', DEFAULT_FEED_WIDTH))

        dec_sd = _load_state_dict_file(decoder_path, device)
        _load_partial_state_dict(decoder, dec_sd, 'MonoViT.decoder')
        decoder.to(device).eval()

    else:
        decoder_path = depth_path

        encoder = networks.ResnetEncoder(18, False)
        _load_partial_state_dict(encoder, encoder_sd, 'MonoDepth2.encoder')
        encoder.to(device).eval()
        feed_height, feed_width = encoder_sd['height'], encoder_sd['width']

        decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        dec_sd = _load_state_dict_file(decoder_path, device)
        _load_partial_state_dict(decoder, dec_sd, 'MonoDepth2.decoder')
        decoder.to(device).eval()

    print(f'Loaded weights from: {model_path}')
    return encoder, decoder, feed_height, feed_width, model_branch


# ==============================
# 数据对齐与配对
# ==============================

def _norm_img_exts(primary_ext: str, extras: List[str]) -> List[str]:
    seq = [primary_ext.lower().lstrip('.')]
    seq += [e.lower().lstrip('.') for e in (extras or [])]
    # 去重且保持顺序
    dedup = []
    for e in seq:
        if e and e not in dedup:
            dedup.append(e)
    return dedup


def _norm_depth_exts(exts: List[str]) -> List[str]:
    out = []
    for e in exts or []:
        e = e.lower()
        if not e.startswith('.'):
            e = '.' + e
        if e not in out:
            out.append(e)
    return out


def _norm_mask_exts(exts: List[str]) -> List[str]:
    out = []
    for e in exts or []:
        e = e.lower().lstrip('.')
        if e and e not in out:
            out.append(e)
    return out


def build_pairs_from_depth(root: str,
                           img_subdir: str,
                           depth_subdir: str,
                           img_exts: List[str],
                           depth_exts: List[str],
                           mask_subdir: str = '',
                           mask_exts: Optional[List[str]] = None
                           ) -> Tuple[List[Tuple[str, str, str, Optional[str]]], int, int]:
    """
    新版：遍历 <root> 下的每个序列目录（如 UAV_seqXX），序列内按 depth 为基准配对：
      - depth 位于 <seq_dir>/<depth_subdir>
      - image 位于 <seq_dir>/<img_subdir>
      - mask（可选）位于 <seq_dir>/<mask_subdir>，优先与 depth 同名；若无则按 image 文件名（含扩展）匹配
    返回：pairs[(img_path, depth_path, image_id, mask_path)]，以及 miss_img/miss_mask 计数
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f'未找到 Test 根目录：{root}')

    # 收集一级子目录作为序列目录（例如 UAV_seq29 等）
    seq_dirs = sorted([d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d)])
    if not seq_dirs:
        raise FileNotFoundError(f'未发现序列目录：{root}/*')

    pairs: List[Tuple[str, str, str, Optional[str]]] = []
    miss_img = 0
    miss_mask = 0
    use_mask = bool((mask_subdir or '').strip())
    mask_exts = mask_exts or []

    for seq_dir in seq_dirs:
        seq_name = os.path.basename(os.path.normpath(seq_dir))
        depth_dir = os.path.join(seq_dir, depth_subdir)
        image_dir = os.path.join(seq_dir, img_subdir)
        mask_dir = os.path.join(seq_dir, mask_subdir) if use_mask else None

        if not os.path.isdir(depth_dir) or not os.path.isdir(image_dir):
            # 非标准序列结构则跳过
            continue

        # 列出该序列中所有 depth（按允许扩展名）
        depth_files: List[str] = []
        for ext in depth_exts:
            depth_files.extend(sorted(glob.glob(os.path.join(depth_dir, f'**/*{ext}'), recursive=True)))

        for dpath in depth_files:
            rel_stem = os.path.splitext(os.path.relpath(dpath, depth_dir))[0]  # 相对 depth_dir 的干名（可包含子层级）
            # 在同序列的 images 下按相同干名 + 允许图像扩展匹配
            found_img = None
            for iext in img_exts:
                cand = os.path.join(image_dir, rel_stem + '.' + iext)
                if os.path.exists(cand):
                    found_img = cand
                    break
            if found_img is None:
                miss_img += 1
                continue

            mask_path = None
            if use_mask:
                if not mask_dir or not os.path.isdir(mask_dir):
                    miss_mask += 1
                else:
                    stem_candidates = [rel_stem, os.path.basename(found_img)]
                    for stem in stem_candidates:
                        for mext in mask_exts:
                            cand = os.path.join(mask_dir, stem + '.' + mext)
                            if os.path.exists(cand):
                                mask_path = cand
                                break
                        if mask_path is not None:
                            break
                    if mask_path is None:
                        miss_mask += 1

            # image_id 加序列前缀，避免跨序列重名
            image_id = f"{seq_name}_{rel_stem.replace(os.sep, '_')}"
            pairs.append((found_img, dpath, image_id, mask_path))

    if not pairs:
        raise FileNotFoundError(
            f'未配到任何样本，请检查路径与扩展名；root={root}, img_subdir={img_subdir}, depth_subdir={depth_subdir}, '
            f'img_exts={img_exts}, depth_exts={depth_exts}'
        )

    return pairs, miss_img, miss_mask


# ==============================
# 单图评估（误差、可视化、边缘 F1）
# ==============================

def evaluate_image(depth_np: np.ndarray,
                   gt: np.ndarray,
                   args: argparse.Namespace,
                   img_path: str,
                   vis_dir: Optional[str],
                   image_id: str,
                   mask: Optional[np.ndarray] = None
                   ) -> Tuple[Optional[Tuple[float, ...]],
                              Optional[float],
                              Optional[float]]:
    from cv2 import resize, INTER_LINEAR
    from cv2 import INTER_NEAREST

    # 将预测深度 resize 到 GT 尺寸（仅用于误差评估与可视化）
    if gt.shape != depth_np.shape:
        depth_np = resize(depth_np, (gt.shape[1], gt.shape[0]), interpolation=INTER_LINEAR)

    mask_bool = None
    if mask is not None:
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != gt.shape:
            mask = resize(mask, (gt.shape[1], gt.shape[0]), interpolation=INTER_NEAREST)
        mask_bool = mask > 0

    valid_pred_mask = depth_np > 0
    if mask_bool is not None:
        valid_pred_mask = np.logical_and(valid_pred_mask, mask_bool)
    if valid_pred_mask.sum() == 0:
        return None, None, None

    # 仅用于误差计算：对 GT 进行分位裁剪（抑制极端值）
    min_d, max_d = np.percentile(gt[valid_pred_mask], args.gt_clip_percentile)
    gt_clipped = np.clip(gt, min_d, max_d)

    eval_mask = (gt_clipped > 0) & valid_pred_mask
    if eval_mask.sum() == 0:
        return None, None, None

    pred_depth = depth_np[eval_mask]
    gt_depth = gt_clipped[eval_mask]

    # 中值对齐（median scaling）
    ratio = np.median(gt_depth) / np.median(pred_depth + 1e-12)
    pred_depth_scaled = np.clip(pred_depth * ratio, min_d, max_d)

    # 计算指标
    metrics = compute_errors(gt_depth, pred_depth_scaled)

    # （可选）保存伪彩：GT / Pred（已缩放）/ Diff
    # 可视化统一“无数据”语义：无效深度（<=0）与 mask 掉区域都不着色。
    if args.visualize and vis_dir is not None:
        pred_scaled_full = depth_np * ratio  # HxW
        diff_map = np.abs(gt - pred_scaled_full)
        vis_mask = (gt > 0) & (depth_np > 0)
        if mask_bool is not None:
            vis_mask = np.logical_and(vis_mask, mask_bool)
        save_colormap_img(gt, os.path.join(vis_dir, f'{image_id}_gt.png'),
                          cmap='plasma', vmin=0, vmax=max_d, mask=vis_mask)
        save_colormap_img(pred_scaled_full, os.path.join(vis_dir, f'{image_id}_pred.png'),
                          cmap='plasma', vmin=0, vmax=max_d, mask=vis_mask)
        save_colormap_img(diff_map, os.path.join(vis_dir, f'{image_id}_diff.png'),
                          cmap='inferno', vmin=0, vmax=(max_d - 0) * 0.3, mask=vis_mask)

    # Edge-F1：与“裁剪后的 GT 边缘”比对
    pred_scaled_full = depth_np * ratio
    edge_f1 = compute_edge_f1(pred_scaled_full, gt_depth=gt_clipped, img_rgb=None, q=95, mask=mask_bool)

    return metrics, float(ratio), float(edge_f1)


# ==============================
# 归一化策略
# ==============================

def _infer_norm_cfg(opt: argparse.Namespace) -> Dict[str, object]:
    """Paper evaluation uses raw [0, 1] tensors."""
    return {}


def _log_norm_decision(opt: argparse.Namespace, cfg: Dict[str, object]) -> None:
    """打印归一化判定结果，便于复现。"""
    if not cfg:
        print('[Normalize] OFF | 使用原始 [0,1] 张量输入')
        return


# ==============================
# 主流程：推理 + 评估
# ==============================

def test_simple(args: argparse.Namespace) -> None:
    # —— 权重解析与输出目录确定 ——
    weights_dir, weights_name = _select_weights_dir(args.exp_dir, args.weights)
    args.weights_dir = weights_dir

    exp_name = os.path.basename(os.path.normpath(args.exp_dir))
    run_dir_name = f'{exp_name}_{weights_name}'

    output_dir, npy_dir, vis_dir = prepare_output_dirs(args.output_base, run_dir_name, args.visualize)

    print(f'[exp_dir]    {os.path.abspath(args.exp_dir)}')
    print(f'[weights]    {weights_name} -> {weights_dir}')
    print(f'[outputs]    {output_dir}')

    if args.batch_size < 1:
        raise ValueError('--batch_size must be >= 1')

    # 统一的设备解析逻辑：默认使用 CUDA，若显式指定 cpu 或自动回退则按需切换。
    device_req = (args.device or 'auto').lower()
    if device_req == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，但 --device 设置为 'cuda'")
        device = torch.device('cuda')
    elif device_req == 'cpu':
        device = torch.device('cpu')
    elif device_req == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError(f"未知设备类型：{args.device}")

    print(f'[device]     {device}')

    encoder, decoder, feed_height, feed_width, model_branch = load_model(args, device)

    norm_cfg = _infer_norm_cfg(args)
    _log_norm_decision(args, norm_cfg)

    # —— 以 depth 为基准进行配对 ——
    img_exts = _norm_img_exts(args.ext, args.img_exts)
    depth_exts = _norm_depth_exts(args.depth_exts)

    use_mask = bool((args.mask_subdir or '').strip())
    mask_exts = _norm_mask_exts(args.mask_exts) if use_mask else []
    pairs, miss_img, miss_mask = build_pairs_from_depth(
        root=args.gt_folder,
        img_subdir=args.img_subdir,
        depth_subdir=args.depth_subdir,
        img_exts=img_exts,
        depth_exts=depth_exts,
        mask_subdir=args.mask_subdir,
        mask_exts=mask_exts,
    )
    msg = f'-> Found {len(pairs)} paired samples; skipped {miss_img} depth files without matching image'
    if use_mask:
        msg += f'; missing mask for {miss_mask} samples'
    print(msg)
    if use_mask:
        print(f'[mask]       ON  | subdir={args.mask_subdir} | missing={miss_mask}')
    else:
        print('[mask]       OFF')
    metrics_list: List[Tuple[float, ...]] = []
    ratios: List[float] = []
    edge_f1_list: List[float] = []
    per_image_lines: List[str] = []

    total_pairs = len(pairs)
    batch_size = args.batch_size

    for start_idx in range(0, total_pairs, batch_size):
        batch_pairs = pairs[start_idx:start_idx + batch_size]

        # ---------- 前向推理（批处理） ----------
        # 将一个批次的图像读入并堆叠后送入网络，充分利用 GPU 吞吐。
        batch_tensors = [
            process_image(img_path, feed_width, feed_height, norm_cfg=norm_cfg)
            for (img_path, _, _, _) in batch_pairs
        ]
        input_batch = torch.cat(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            features = encoder(input_batch)
            disp = decoder(features)[('disp', 0)]
            depth_pred = _decode_disp_to_eval_depth(disp, args, model_branch)

        # depth_pred 形状为 (B,1,H,W)，取出单通道深度并转至 NumPy 方便后续处理。
        depth_np_batch = depth_pred[:, 0].cpu().numpy()

        for offset, ((img_path, depth_gt_path, image_id, mask_path), depth_np) in enumerate(zip(batch_pairs, depth_np_batch)):
            np.save(os.path.join(npy_dir, f'{image_id}_depth.npy'), depth_np)

            # 逐图记录容器：指标/尺度/EdgeF1 仍以单样本粒度汇总，保持原有 CSV 结构。
            row_vals = {'image_id': image_id, 'edgeF1': None, 'scale_ratio': None}
            metric_tuple: Optional[Tuple[float, ...]] = None
            ratio: Optional[float] = None

            # ---------- 评估（若启用且存在 GT） ----------
            if args.do_evaluation:
                try:
                    gt = read_depth_file(depth_gt_path)
                except Exception as e:
                    print(f'[!] Failed to read GT: {depth_gt_path} ({e})')
                    gt = None
                if gt is not None:
                    mask_arr = None
                    if (args.mask_subdir or '').strip():
                        if mask_path is None:
                            print(f'[!] Missing mask, fallback to no-mask eval: {image_id}')
                        else:
                            try:
                                mask_arr = read_mask_file(mask_path)
                            except Exception as e:
                                print(f'[!] Failed to read mask, fallback to no-mask eval: {mask_path} ({e})')
                        metrics, ratio, edgef1 = evaluate_image(
                            depth_np, gt, args, img_path, vis_dir, image_id, mask=mask_arr
                        )
                    else:
                        metrics, ratio, edgef1 = evaluate_image(depth_np, gt, args, img_path, vis_dir, image_id)
                    if metrics is not None:
                        metrics_list.append(metrics)
                        metric_tuple = metrics
                    if ratio is not None:
                        ratios.append(ratio)
                        row_vals['scale_ratio'] = float(ratio)
                    if edgef1 is not None:
                        edge_f1_list.append(edgef1)
                        row_vals['edgeF1'] = float(edgef1)

            # ---------- 记录逐图 CSV 行 ----------
            # 与原逻辑一致：每张图生成一行字符串，后续在写入 CSV 时拆分。
            # 列顺序：image_id + 8 指标 + edgeF1 + scale_ratio + scale_ratio_median + scale_ratio_rel_std
            row = [row_vals['image_id']]
            if metric_tuple is not None:
                row += [f'{m:.4f}' for m in metric_tuple]
            else:
                row += [''] * len(ERROR_METRICS)
            row.append('' if row_vals['edgeF1'] is None else f"{row_vals['edgeF1']:.4f}")
            row.append('' if row_vals['scale_ratio'] is None else f"{row_vals['scale_ratio']:.6f}")
            row.extend(['', ''])  # median / rel_std 占位（仅汇总行使用）
            per_image_lines.append(','.join(row))

            global_idx = start_idx + offset
            print(f'[{global_idx + 1}/{total_pairs}] Processed: {image_id}')

    # ---------- 终端汇总 ----------
    if args.do_evaluation and metrics_list:
        mean_metrics = np.array(metrics_list).mean(0)  # 8 项
        print('==== Evaluation Results ====')
        print('abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
              'a1: {:.3f}, a2: {:.3f}, a3: {:.3f}, si_log: {:.3f}'.format(*mean_metrics))
        if edge_f1_list:
            print('edge_F1: {:.3f}'.format(np.mean(edge_f1_list)))
        if ratios:
            ratios_np = np.array(ratios)
            med = np.median(ratios_np)
            std = np.std(ratios_np / (med + 1e-12))
            print(f'Scale ratios | median: {med:.3f}, std: {std:.3f}')

    # ---------- 写入结果文本（含头注/表头/明细/汇总） ----------
    # ---------- 写入结果文本（含参数/明细/汇总，统一写到一个 CSV） ----------
    if per_image_lines:
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, 'eval_results.csv')

        # 统一表头：kind + 原明细列 + 汇总两列（metric,value）
        header = ['kind'] + CSV_COLUMNS + ['metric', 'value']

        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # 0) 参数行（kind=param）：参数名写 metric，参数值写 value，其余列留空
            #    注意：这里直接遍历 vars(args)，并将值转为字符串，csv.writer 会负责必要转义
            empty_cols = [''] * len(CSV_COLUMNS)
            for k in sorted(vars(args).keys()):
                v = getattr(args, k)
                writer.writerow(['param'] + empty_cols + [k, str(v)])
            writer.writerow(['param'] + empty_cols + ['use_mask', str(use_mask)])
            if use_mask:
                writer.writerow(['param'] + empty_cols + ['missing_mask_count', str(miss_mask)])

            # 1) 明细行（kind=detail）
            #    per_image_lines 的内容与 CSV_COLUMNS 对齐：image_id + 8 指标 + edgeF1 + scale_ratio
            #    末尾再补空的 metric,value 两列
            for line in per_image_lines:
                writer.writerow(['detail'] + line.split(',') + ['', ''])

            # 2) 汇总行（kind=summary，横向展开）
            if args.do_evaluation and metrics_list:
                mean_metrics = np.array(metrics_list).mean(0)  # 8 项（不含 edgeF1）
                summary_map = {name: '' for name in METRIC_NAMES}
                for name, val in zip(ERROR_METRICS, mean_metrics):
                    summary_map[name] = f'{val:.6f}'
                if 'edgeF1' in summary_map:
                    summary_map['edgeF1'] = f'{np.mean(edge_f1_list):.6f}' if edge_f1_list else ''

                ratio_median = ''
                ratio_rel_std = ''
                if ratios:
                    ratios_np = np.array(ratios)
                    med = np.median(ratios_np)
                    std = np.std(ratios_np / (med + 1e-12))
                    ratio_median = f'{med:.6f}'
                    ratio_rel_std = f'{std:.6f}'

                summary_row = []
                for col in CSV_COLUMNS:
                    if col == 'image_id':
                        summary_row.append('summary_mean')
                    elif col in METRIC_NAMES:
                        summary_row.append(summary_map.get(col, ''))
                    elif col == 'scale_ratio':
                        summary_row.append('')
                    elif col == 'scale_ratio_median':
                        summary_row.append(ratio_median)
                    elif col == 'scale_ratio_rel_std':
                        summary_row.append(ratio_rel_std)
                    else:
                        summary_row.append('')

                label_row = []
                for col in CSV_COLUMNS:
                    if col == 'image_id':
                        label_row.append('summary_label')
                    elif col in METRIC_NAMES:
                        label_row.append(f'{col}_mean')
                    elif col == 'scale_ratio':
                        label_row.append('')
                    elif col == 'scale_ratio_median':
                        label_row.append('scale_ratio_median')
                    elif col == 'scale_ratio_rel_std':
                        label_row.append('scale_ratio_rel_std')
                    else:
                        label_row.append('')

                writer.writerow(['summary'] + label_row + ['', ''])
                writer.writerow(['summary'] + summary_row + ['', ''])

        print(f'-> Results saved to: {out_csv}')


# ==============================
# Entry Point
# ==============================
if __name__ == '__main__':
    args = parse_args()
    test_simple(args)

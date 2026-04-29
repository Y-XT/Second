"""
test_simple.py — UAV 单目深度推理与评测（增强版，强化注释版）

本版仅增强“可读性与注释”，不改变原有默认行为与指标计算逻辑，以保持与既有结果的可比性。

功能概述
--------
1) 传统单目深度指标（AbsRel, SqRel, RMSE, RMSE_log, δ<1.25^i）
   - 采用“逐图像中值对齐（median scaling）”
   - 注意：当前 rmse_log 的实现为 mean(abs(log(gt)-log(pred)))（详见 compute_errors 注释）

2) 尺度不变对数误差（SI-Log RMSE）
   - 公式：sqrt( mean(e^2) - (mean(e))^2 ), e = log(pred) - log(gt)

3) Edge-F1（深度边缘与 GT 边缘一致性）
   - Sobel 取梯度幅值，按分位数阈值二值化；再与 GT 边缘（由深度或 Canny）计算 F1

4) 可选：基于相邻帧的视图合成（PSNR/SSIM，掩码内计算）
   - 用 ORB + Essential 矩阵估计相对位姿（OpenCV）
   - 以“目标帧预测深度”将源帧重投影至目标帧
   - SSIM/PSNR 在“有效投影掩码(valid)”内统计
   - 需要 K（以 W/H 归一化形式提供，内部换算至像素）

保持的设计
----------
- CLI 参数与默认值保持不变（包括 `--eval_recon` 的 store_false 语义：默认启用，出现该开关则关闭）
- 输出目录结构/文件名保持不变
- 结果聚合与文本写入格式保持不变
"""

import os
import glob
import argparse
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
from torchvision import transforms

import cv2  # 用于 Edge-F1 与视图合成/位姿估计

import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)  # 确保可直接 import methods 包

import methods
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
    """
    解析命令行参数。注意 `--eval_recon` 的语义：
    - 使用 action='store_false'：默认 True
    - 一旦命令行出现 `--eval_recon`，其值会被置为 False（即关闭重建评估）
    选择该写法是为了“默认启用视图合成评估”，但语义比较反直觉，保留以保持兼容。
    """
    parser = argparse.ArgumentParser(description='Predict and Evaluate Depth (UAV)')

    # 数据与输出
    """    
    parser.add_argument('--gt_folder', type=str, default='/mnt/data_nvme3n1p1/dataset/UAV_ula/GT_ula',
                        help='包含各序列的 GT 与图像根目录')
    """
    parser.add_argument('--gt_folder', type=str, default='/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_dataset/Test',
                        help='包含各序列的 GT 与图像根目录')
    parser.add_argument('--output_base', type=str, default='/home/yxt/文档/mono_result/eval',
                        help='输出根目录（会在其中创建 model_name 子目录）')
    parser.add_argument('--exp_dir', type=str, default='/mnt/data_nvme3n1p1/mono_weights/weights/UAVula_R1/monovit_vggt_rflow_tinj_uavula_tridataset_512x288_bs8_lr1e-04_e40_step20_9id0',
                        help='实验目录：包含 models/weights_x 的根目录')
    parser.add_argument('--weights', type=str, default='weights_17',
                        help="'latest' | 整数（如 7）| 目录名（如 weights_7）")
    parser.add_argument('--model_name', type=str, default='auto',
                        help='模型分支（auto | GasMono* | Madhuanand* | MRFEDepth* | MonoViT* | LiteMono* | SPIDepth* | monodepth2_dino* | MonoDepth2*）')

    # 深度范围与基础选项
    parser.add_argument('--min_depth', type=float, default=0.1)
    parser.add_argument('--max_depth', type=float, default=150.0)
    parser.add_argument('--ext', type=str, default="jpg",
                        help='测试图像扩展名（每个序列 image_02/data/images/*.ext）')
    parser.add_argument('--image_rel_path', type=str, default=None,
                        help='相对序列根路径的图像目录（留空则依次尝试 image_02/data/images 与 image_02/data）')
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
# I/O 与可视化工具
# ==============================
def read_bin_safe(path: str) -> np.ndarray:
    """
    读取自定义 .geometric.bin 稠密深度文件。

    文件格式（简述）：
    - 文件头以 "width&height&channels" 形式出现，后续为 float32 数据
    - 数据列主序（Fortran order）存储，因此需要 reshape 时使用 order='F'
    - 返回 shape 为 (H, W) 或 (H, W, 1) squeeze 后的二维数组

    参数
    ----
    path: 文件路径

    返回
    ----
    depth: (H, W) float32
    """
    with open(path, "rb") as fid:
        # 首行形如：W&H&C
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        # 回到文件头，读到第三个分隔符'&'之后即为原始数组
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)

    # 文件内部为 (W,H,C) 的列主序存储，需转为 (H,W,C) 再 squeeze
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def save_colormap_img(array: np.ndarray, path: str,
                      cmap: str = "plasma",
                      vmin: Optional[float] = None,
                      vmax: Optional[float] = None,
                      mask: Optional[np.ndarray] = None,
                      colorbar_suffix: str = "_colorbar",
                      colorbar_ticks: bool = True) -> None:
    """
    将标量场保存为伪彩色图。若提供 mask，则仅保留有效像素。
    同时额外生成一张对应的色条图像，文件名追加 `colorbar_suffix`。
    将 colorbar_ticks 设为 False 可隐藏刻度。
    """
    if vmin is None:
        vmin = float(np.nanmin(array))
    if vmax is None:
        vmax = float(np.nanmax(array))
    display = np.array(array, dtype=np.float32)
    if mask is not None:
        display = np.ma.array(display, mask=~mask)

    plt.imsave(path, display, cmap=cmap, vmin=vmin, vmax=vmax)

    base, ext = os.path.splitext(path)
    bar_path = f"{base}{colorbar_suffix}{ext}"
    width = max(display.shape[1], 2)
    bar_height = max(12, display.shape[0] // 20)

    fig_width = max(width / 200.0, 1.5)
    fig_height = max(bar_height / 200.0, 0.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.45)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="horizontal")
    cbar.set_label("")
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
        cbar.ax.tick_params(axis="x", length=3, pad=2, labelsize=9)
    else:
        cbar.set_ticks([])

    fig.savefig(bar_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def process_image(img_path: str, feed_width: int, feed_height: int,
                  norm_cfg: Optional[dict] = None) -> torch.Tensor:
    """
    读取并 resize 到网络输入大小，返回形状为 (1,3,H,W) 的 Tensor（RGB，[0,1]），
    若提供 norm_cfg 且开启 normalize，则在数据侧进行均值方差归一化。
    """
    input_img = Image.open(img_path).convert('RGB')
    resample = getattr(Image, "Resampling", Image).LANCZOS
    input_img = input_img.resize((feed_width, feed_height), resample)
    tensor = transforms.ToTensor()(input_img)  # (3,H,W), 值域[0,1]

    if norm_cfg and norm_cfg.get("normalize", False):
        mode = norm_cfg.get("norm_mode", "imagenet")
        if mode == "imagenet":
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        else:
            mean = norm_cfg.get("norm_mean")
            std  = norm_cfg.get("norm_std")
            assert mean is not None and std is not None, "custom/sat 模式需要提供 mean/std"
        tensor = transforms.Normalize(mean=mean, std=std)(tensor)

    return tensor.unsqueeze(0)  # (1,3,H,W)


# ==============================
# 指标计算
# ==============================

def compute_silog(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> float:
    """
    Scale-invariant log RMSE (Eigen et al.)
    si-log = sqrt( mean(e^2) - (mean(e))^2 ), 其中 e = log(pred) - log(gt)
    """
    e = np.log(np.clip(pred, eps, None)) - np.log(np.clip(gt, eps, None))
    m1 = np.mean(e ** 2)
    m2 = (np.mean(e)) ** 2
    return float(np.sqrt(max(m1 - m2, 0.0)))

def compute_errors(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, ...]:
    """
    传统指标的计算（假设 pred 已与 gt 尺度对齐）。
    注意：
      - 这里 rmse_log 的实现与常见定义不同：
        代码为  mean( sqrt( (log(gt)-log(pred))^2 ) ) == mean( |log(gt)-log(pred)| )
        常见定义为 sqrt( mean( (log(gt)-log(pred))^2 ) )
      - 出于与历史结果/脚本兼容性考虑，保持现有实现不变。
    """
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
    """
    以 Sobel 计算梯度幅值，取 q 分位数为阈值进行二值化，返回 bool 边缘图。
    """
    d = depth.astype(np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(mag[np.isfinite(mag)], q)
    return mag >= thr

def _binary_f1(pred_bin: np.ndarray, gt_bin: np.ndarray, dilate: int = 1) -> float:
    """
    计算二值边缘图的 F1。可选形态学膨胀以提升容错。
    """
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        pred_bin = cv2.dilate(pred_bin.astype(np.uint8), k) > 0
        gt_bin   = cv2.dilate(gt_bin.astype(np.uint8), k) > 0

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

    meta = {k: sd[k] for k in ('height', 'width') if k in sd}
    for nested_key in ('state_dict', 'model_state_dict', 'model'):
        nested = sd.get(nested_key)
        if isinstance(nested, dict):
            sd = dict(nested)
            for k, v in meta.items():
                sd.setdefault(k, v)
            break

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

    if any(k.startswith('encoder.encoder.layer1.') for k in keys) and any(k.startswith('decoder.up1.net.') for k in keys):
        return 'SPIDepth'

    if any(k.startswith('backbone.downsample_layers.') for k in keys) and \
            any(k.startswith('backbone.stages.') for k in keys):
        return 'monodepth2_dino'

    if any(k.startswith('stem.') for k in keys) and any(k.startswith('patch_embed_stages.') for k in keys):
        return 'MonoViT'

    if any(k.startswith('downsample_layers.') for k in keys) and any(k.startswith('stages.') for k in keys):
        return 'LiteMono'

    if any(k.startswith('encoder.conv1.') for k in keys) and any(k.startswith('encoder.layer1.') for k in keys):
        return 'MonoDepth2'

    return None


def _infer_md2_family_from_depth_sd(depth_sd: Dict[str, object]) -> Optional[str]:
    """区分 Monodepth2 与 Madhuanand（3D decoder）家族。"""
    conv_w = depth_sd.get("decoder.0.conv.conv.weight", None)
    if torch.is_tensor(conv_w):
        if conv_w.ndim == 5:
            return "Madhuanand"
        if conv_w.ndim == 4:
            return "MonoDepth2"
    return None


def _infer_model_branch_from_opt_json(weights_dir: str) -> Optional[str]:
    """
    从 <exp>/models/opt.json 读取 methods 字段，辅助区分同 backbone 的分支（如 MonoViT vs GasMono）。
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
    if method == "gasmono":
        return "GasMono"
    if method.startswith("monovit"):
        return "MonoViT"
    if method == "litemono":
        return "LiteMono"
    if method == "spidepth":
        return "SPIDepth"
    if method == "mrfedepth":
        return "MRFEDepth"
    if method == "madhuanand":
        return "Madhuanand"
    if method.startswith("monodepth2_dino"):
        return "monodepth2_dino"
    if method.startswith("monodepth2") or method.startswith("md2"):
        return "MonoDepth2"
    return None


def _looks_like_gasmono_depth_sd(depth_sd: Dict[str, object]) -> bool:
    """基于 depth.pth key 结构识别 GasMonoFSDepthDecoder。"""
    needed = (
        "decoder.0.conv.conv.weight",
        "decoder.2.conv_se.weight",
        "decoder.30.2.weight",
    )
    return all(k in depth_sd for k in needed)


def _resolve_model_branch(model_name: str) -> str:
    name = (model_name or '').strip().lower()
    if not name or name == 'auto':
        return 'auto'
    if name.startswith('gasmono'):
        return 'GasMono'
    if name.startswith('madhuanand'):
        return 'Madhuanand'
    if name.startswith('mrfedepth'):
        return 'MRFEDepth'
    if name.startswith('monovit'):
        return 'MonoViT'
    if name.startswith('litemono'):
        return 'LiteMono'
    if name.startswith('spidepth'):
        return 'SPIDepth'
    if name.startswith('monodepth2_dino'):
        return 'monodepth2_dino'
    return 'MonoDepth2'


def _infer_spidepth_num_layers_from_encoder_sd(enc_sd: Dict[str, object]) -> int:
    keys = [k for k in enc_sd.keys() if isinstance(k, str)]

    def _layer_block_count(layer_idx: int) -> int:
        prefix = f"encoder.encoder.layer{layer_idx}."
        block_ids = []
        for k in keys:
            if not k.startswith(prefix):
                continue
            tail = k[len(prefix):]
            block = tail.split(".", 1)[0]
            if block.isdigit():
                block_ids.append(int(block))
        return (max(block_ids) + 1) if block_ids else 0

    layout = tuple(_layer_block_count(i) for i in (1, 2, 3, 4))
    has_bottleneck = any(k.startswith("encoder.encoder.layer") and ".conv3." in k for k in keys)
    if layout == (2, 2, 2, 2):
        return 18
    if layout == (3, 4, 6, 3):
        return 50 if has_bottleneck else 34
    if layout == (3, 4, 23, 3):
        return 101
    if layout == (3, 8, 36, 3):
        return 152
    return 50 if has_bottleneck else 18


def _infer_spidepth_encoder_cfg(enc_sd: Dict[str, object]) -> Tuple[bool, int, int, int]:
    conv2_w = enc_sd.get("decoder.conv2.weight", None)
    conv3_w = enc_sd.get("decoder.conv3.weight", None)
    if not torch.is_tensor(conv2_w) or not torch.is_tensor(conv3_w):
        raise RuntimeError("无法从 encoder.pth 推断 SPIDepth 配置：缺少 decoder.conv2/conv3 权重")

    num_features = int(conv2_w.shape[0])
    model_dim = int(conv3_w.shape[0])
    is_lite = (num_features == 256)
    if is_lite:
        num_layers = 18
    else:
        num_layers = _infer_spidepth_num_layers_from_encoder_sd(enc_sd)
    return is_lite, num_layers, num_features, model_dim


def _infer_spidepth_decoder_cfg(depth_sd: Dict[str, object]) -> Tuple[int, int, int, int, int, int]:
    embed_w = depth_sd.get("embedding_convPxP.weight", None)
    conv3x3_w = depth_sd.get("conv3x3.weight", None)
    prob_w = depth_sd.get("convert_to_prob.0.weight", None)
    ff_w = depth_sd.get("transformer_encoder.layers.0.linear1.weight", None)
    if (not torch.is_tensor(embed_w) or not torch.is_tensor(conv3x3_w)
            or not torch.is_tensor(prob_w) or not torch.is_tensor(ff_w)):
        raise RuntimeError("无法从 depth.pth 推断 SPIDepth 配置：缺少 QueryTransformer 关键权重")

    in_channels = int(conv3x3_w.shape[1])
    embedding_dim = int(conv3x3_w.shape[0])
    patch_size = int(embed_w.shape[-1])
    dim_out = int(prob_w.shape[0])
    query_nums = int(prob_w.shape[1])
    transformer_ff_dim = int(ff_w.shape[0])
    return in_channels, embedding_dim, patch_size, query_nums, dim_out, transformer_ff_dim


def _decode_disp_to_eval_depth(disp: torch.Tensor, args: argparse.Namespace, model_branch: str) -> torch.Tensor:
    if model_branch == "SPIDepth":
        return disp
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


def _count_shape_matched_keys(module: torch.nn.Module,
                              ckpt_sd: Dict[str, object]) -> Tuple[int, int]:
    """统计 checkpoint 与 module 的 key+shape 匹配数量。"""
    model_sd = module.state_dict()
    matched = 0
    shape_mismatch = 0
    for k, v in ckpt_sd.items():
        if k not in model_sd:
            continue
        try:
            if tuple(v.shape) == tuple(model_sd[k].shape):
                matched += 1
            else:
                shape_mismatch += 1
        except Exception:
            continue
    return matched, shape_mismatch


def _read_litemono_variant_from_opt_json(weights_dir: str) -> Optional[str]:
    """
    从 <exp>/models/opt.json 读取训练时的 litemono_variant。
    weights_dir 形如 .../models/weights_x。
    """
    model_root = os.path.dirname(os.path.abspath(weights_dir))
    opt_path = os.path.join(model_root, "opt.json")
    if not os.path.isfile(opt_path):
        return None
    try:
        with open(opt_path, "r", encoding="utf-8") as f:
            opt = json.load(f)
    except Exception:
        return None
    variant = opt.get("litemono_variant")
    if isinstance(variant, str) and variant.strip():
        return variant.strip()
    return None


def _resolve_litemono_variant(weights_dir: str,
                              encoder_sd: Dict[str, object],
                              feed_height: int,
                              feed_width: int) -> str:
    """确定 LiteMono 变体：优先 opt.json，其次按 key+shape 自动匹配。"""
    valid_variants = ("lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m")

    variant_from_opt = _read_litemono_variant_from_opt_json(weights_dir)
    if variant_from_opt in valid_variants:
        print(f"[LiteMono] variant from opt.json: {variant_from_opt}")
        return variant_from_opt

    if variant_from_opt is not None:
        print(f"[LiteMono] opt.json 中变体无效：{variant_from_opt}，将自动匹配")

    best_variant = "lite-mono-8m"
    best_matched = -1
    best_shape_mismatch = 10 ** 9
    stats = []

    for variant in valid_variants:
        probe = networks.LiteMono(model=variant, width=feed_width, height=feed_height)
        matched, shape_mismatch = _count_shape_matched_keys(probe, encoder_sd)
        stats.append((variant, matched, shape_mismatch))
        if matched > best_matched or (matched == best_matched and shape_mismatch < best_shape_mismatch):
            best_variant = variant
            best_matched = matched
            best_shape_mismatch = shape_mismatch

    stat_str = ", ".join([f"{v}:matched={m},shape_mismatch={s}" for v, m, s in stats])
    print(f"[LiteMono] auto variant match stats -> {stat_str}")
    print(f"[LiteMono] resolved variant: {best_variant}")
    return best_variant


def compute_edge_f1(pred_depth: np.ndarray,
                    gt_depth: Optional[np.ndarray] = None,
                    img_rgb: Optional[np.ndarray] = None,
                    q: int = 95) -> float:
    """
    计算 Edge-F1。优先与 GT 深度边缘匹配；无 GT 时退化为与 RGB Canny 边缘匹配。
    """
    pred_edge = _depth_edge_binary(pred_depth, q=q)
    if gt_depth is not None:
        gt_edge = _depth_edge_binary(gt_depth, q=q)
    else:
        assert img_rgb is not None, "Edge-F1 退化模式需要提供 img_rgb"
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gt_edge = cv2.Canny(gray, 80, 160) > 0
    return float(_binary_f1(pred_edge, gt_edge, dilate=1))

# ==============================
# 模型加载
# ==============================
def load_model(args: argparse.Namespace, device: torch.device):
    """
    根据 model_name 加载相应编码器/解码器与输入分辨率。
    自动兼容 checkpoint 前缀、常见封装格式，以及默认实验目录/模型分支不一致的情况。
    """
    assert args.model_name is not None, "You must specify --model_name"
    model_path = args.weights_dir
    requested_branch = _resolve_model_branch(args.model_name)
    inferred_branch = None
    inferred_branch_from_opt = _infer_model_branch_from_opt_json(model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    encoder_sd: Optional[Dict[str, object]] = None
    mrfe_encoder_path = os.path.join(model_path, "DepthEncoder.pth")
    depth_path = os.path.join(model_path, "depth.pth")
    depth_sd_probe: Optional[Dict[str, object]] = None

    if os.path.isfile(encoder_path):
        encoder_sd = _load_state_dict_file(encoder_path, device)
        inferred_branch = _infer_model_branch_from_encoder_sd(encoder_sd)
    elif os.path.isfile(mrfe_encoder_path):
        inferred_branch = "MRFEDepth"

    # Encoder key 形态无法区分 Monodepth2 vs Madhuanand 时，借助 depth 权重维度判别。
    if inferred_branch == "MonoDepth2" and os.path.isfile(depth_path):
        try:
            depth_sd_probe = _load_state_dict_file(depth_path, device)
            md2_family = _infer_md2_family_from_depth_sd(depth_sd_probe)
            if md2_family is not None:
                inferred_branch = md2_family
        except Exception as e:
            print(f"[Warning] 读取 depth.pth 进行分支判别失败，将按 {inferred_branch} 继续: {e}")

    # MPViT backbone 下，MonoViT 与 GasMono 仅看 encoder key 容易冲突，尝试用 depth 结构补判。
    if inferred_branch == "MonoViT" and os.path.isfile(depth_path):
        try:
            if depth_sd_probe is None:
                depth_sd_probe = _load_state_dict_file(depth_path, device)
            if _looks_like_gasmono_depth_sd(depth_sd_probe):
                inferred_branch = "GasMono"
        except Exception as e:
            print(f"[Warning] 读取 depth.pth 进行 GasMono 判别失败，将按 {inferred_branch} 继续: {e}")

    if requested_branch == 'auto':
        model_branch = inferred_branch_from_opt or inferred_branch or 'MonoDepth2'
    else:
        model_branch = requested_branch
        preferred_inferred = inferred_branch_from_opt or inferred_branch
        if preferred_inferred is not None and model_branch != preferred_inferred and model_branch not in {'MRFEDepth', 'SPIDepth', 'Madhuanand', 'GasMono'}:
            print(f"[Warning] --model_name={args.model_name} 与 checkpoint 推断分支({preferred_inferred})不一致，已自动采用 {preferred_inferred}")
            model_branch = preferred_inferred

    print(f'[model_name]  requested={args.model_name} | resolved={model_branch}'
          + (f' | inferred={inferred_branch}' if inferred_branch else '')
          + (f' | inferred_from_opt={inferred_branch_from_opt}' if inferred_branch_from_opt else ''))

    if model_branch == "MRFEDepth":

        depth_encoder_path = os.path.join(model_path, "DepthEncoder.pth")
        depth_decoder_path = os.path.join(model_path, "DepthDecoder.pth")

        print("Loading pretrained encoder")
        encoder = methods.networks.MRFE_depth_encoder.hrnet18(False)
        encoder.num_ch_enc = [64, 18, 36, 72, 144]
        loaded_dict_enc = _load_state_dict_file(depth_encoder_path, device)
        _load_partial_state_dict(encoder, loaded_dict_enc, "MRFEDepth.encoder")
        encoder.to(device).eval()
        para_sum_encoder = sum(p.numel() for p in encoder.parameters())

        decoder = networks.MRFEDepthDecoder(encoder.num_ch_enc, range(4))
        loaded_dict = _load_state_dict_file(depth_decoder_path, device)
        _load_partial_state_dict(decoder, loaded_dict, "MRFEDepth.decoder")
        decoder.to(device).eval()
        para_sum_decoder = sum(p.numel() for p in decoder.parameters())
        print(f"depth_encoder: {para_sum_encoder} params, depth_decoder: {para_sum_decoder} params, total: {para_sum_encoder+para_sum_decoder}")
        # Keep input resolution consistent with training/export metadata.
        # Fallback is the project default: H=288, W=512.
        feed_height = int(loaded_dict_enc.get("height", DEFAULT_FEED_HEIGHT))
        feed_width = int(loaded_dict_enc.get("width", DEFAULT_FEED_WIDTH))

    elif model_branch == "MonoViT":
        decoder_path = _resolve_ckpt_path(model_path, ["decoder.pth", "depth.pth"], "MonoViT.decoder")
        print("Loading MonoViT encoder from", encoder_path)
        print("Loading MonoViT decoder from", decoder_path)
        model = networks.DeepNet(type='mpvitnet')
        encoder = model.encoder
        decoder = model.decoder
        encoder_dict = encoder_sd if encoder_sd is not None else _load_state_dict_file(encoder_path, device)
        _load_partial_state_dict(encoder, encoder_dict, "MonoViT.encoder")

        encoder.to(device).eval()
        feed_height = int(encoder_dict.get('height', DEFAULT_FEED_HEIGHT))
        feed_width  = int(encoder_dict.get('width', DEFAULT_FEED_WIDTH))
        decoder_dict = _load_state_dict_file(decoder_path, device)
        _load_partial_state_dict(decoder, decoder_dict, "MonoViT.decoder")

        decoder.to(device).eval()

    elif model_branch == "GasMono":
        decoder_path = _resolve_ckpt_path(model_path, ["depth.pth", "decoder.pth"], "GasMono.depth")
        print("Loading GasMono encoder from", encoder_path)
        print("Loading GasMono decoder from", decoder_path)

        encoder = networks.mpvit_small()
        encoder.num_ch_enc = [64, 128, 216, 288, 288]
        encoder_dict = encoder_sd if encoder_sd is not None else _load_state_dict_file(encoder_path, device)
        _load_partial_state_dict(encoder, encoder_dict, "GasMono.encoder")
        encoder.to(device).eval()
        feed_height = int(encoder_dict.get('height', DEFAULT_FEED_HEIGHT))
        feed_width = int(encoder_dict.get('width', DEFAULT_FEED_WIDTH))

        decoder = networks.GasMonoFSDepthDecoder(encoder.num_ch_enc, range(4))
        if depth_sd_probe is not None and os.path.abspath(decoder_path) == os.path.abspath(depth_path):
            decoder_dict = depth_sd_probe
        else:
            decoder_dict = _load_state_dict_file(decoder_path, device)
        _load_partial_state_dict(decoder, decoder_dict, "GasMono.depth")
        decoder.to(device).eval()

    elif model_branch == "LiteMono":
        decoder_path = os.path.join(model_path, "depth.pth")

        encoder_dict = encoder_sd if encoder_sd is not None else _load_state_dict_file(encoder_path, device)
        feed_height = int(encoder_dict.get("height", DEFAULT_FEED_HEIGHT))
        feed_width = int(encoder_dict.get("width", DEFAULT_FEED_WIDTH))
        litemono_variant = _resolve_litemono_variant(model_path, encoder_dict, feed_height, feed_width)

        encoder = networks.LiteMono(model=litemono_variant, width=feed_width, height=feed_height)
        _load_partial_state_dict(encoder, encoder_dict, f"LiteMono.encoder({litemono_variant})")
        encoder.to(device).eval()

        decoder = networks.DepthDecoder_litemono(np.array(encoder.num_ch_enc), [0, 1, 2])
        decoder_dict = _load_state_dict_file(decoder_path, device)
        _load_partial_state_dict(decoder, decoder_dict, f"LiteMono.decoder({litemono_variant})")
        decoder.to(device).eval()

    elif model_branch == "SPIDepth":
        decoder_path = os.path.join(model_path, "depth.pth")
        encoder_dict = encoder_sd if encoder_sd is not None else _load_state_dict_file(encoder_path, device)
        is_lite, num_layers, num_features, model_dim = _infer_spidepth_encoder_cfg(encoder_dict)
        if is_lite:
            encoder = networks.SPILiteResnetEncoderDecoder(model_dim=model_dim, pretrained=False)
            print(f"[SPIDepth] encoder: lite num_layers=18 num_features={num_features} model_dim={model_dim}")
        else:
            encoder = networks.SPIResnetEncoderDecoder(
                num_layers=num_layers,
                num_features=num_features,
                model_dim=model_dim,
                pretrained=False,
            )
            print(f"[SPIDepth] encoder: resnet num_layers={num_layers} num_features={num_features} model_dim={model_dim}")
        _load_partial_state_dict(encoder, encoder_dict, "SPIDepth.encoder")
        encoder.to(device).eval()
        feed_height = int(encoder_dict.get("height", DEFAULT_FEED_HEIGHT))
        feed_width = int(encoder_dict.get("width", DEFAULT_FEED_WIDTH))

        decoder_dict = _load_state_dict_file(decoder_path, device)
        in_channels, embedding_dim, patch_size, query_nums, dim_out, transformer_ff_dim = _infer_spidepth_decoder_cfg(decoder_dict)
        decoder = networks.SPIDepthDecoderQueryTr(
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            patch_size=patch_size,
            query_nums=query_nums,
            dim_out=dim_out,
            transformer_ff_dim=transformer_ff_dim,
            min_val=args.min_depth,
            max_val=args.max_depth,
        )
        print(
            "[SPIDepth] decoder: "
            f"in_channels={in_channels} embedding_dim={embedding_dim} patch={patch_size} "
            f"query_nums={query_nums} dim_out={dim_out} ff_dim={transformer_ff_dim}"
        )
        _load_partial_state_dict(decoder, decoder_dict, "SPIDepth.depth")
        decoder.to(device).eval()

    elif model_branch == "monodepth2_dino":
        decoder_path = depth_path
        encoder = networks.DinoConvNeXtMultiScale(arch="dinov3_convnext_small")
        encoder_dict = encoder_sd if encoder_sd is not None else _load_state_dict_file(encoder_path, device)
        _load_partial_state_dict(encoder, encoder_dict, "monodepth2_dino.encoder")
        encoder.to(device).eval()
        feed_height, feed_width = encoder_dict['height'], encoder_dict['width']

        decoder = networks.UPerDispHead(in_channels=[96, 192, 384, 768], compat="monodepth2")
        decoder_dict = _load_state_dict_file(decoder_path, device)
        _load_partial_state_dict(decoder, decoder_dict, "monodepth2_dino.decoder")
        decoder.to(device).eval()

    elif model_branch == "Madhuanand":
        decoder_path = depth_path

        encoder = networks.ResnetEncoder(18, False)
        encoder_dict = encoder_sd if encoder_sd is not None else _load_state_dict_file(encoder_path, device)
        _load_partial_state_dict(encoder, encoder_dict, "Madhuanand.encoder")
        encoder.to(device).eval()
        feed_height, feed_width = encoder_dict['height'], encoder_dict['width']

        decoder = networks.DepthDecoder_3d(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        decoder_dict = depth_sd_probe if depth_sd_probe is not None else _load_state_dict_file(decoder_path, device)
        _load_partial_state_dict(decoder, decoder_dict, "Madhuanand.decoder")
        decoder.to(device).eval()

    else:
        decoder_path = depth_path

        encoder = networks.ResnetEncoder(18, False)
        encoder_dict = encoder_sd if encoder_sd is not None else _load_state_dict_file(encoder_path, device)
        _load_partial_state_dict(encoder, encoder_dict, "MonoDepth2.encoder")
        encoder.to(device).eval()
        feed_height, feed_width = encoder_dict['height'], encoder_dict['width']

        decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
        decoder_dict = _load_state_dict_file(decoder_path, device)
        _load_partial_state_dict(decoder, decoder_dict, "MonoDepth2.decoder")
        decoder.to(device).eval()

    print(f'Loaded weights from: {model_path}')
    return encoder, decoder, feed_height, feed_width, model_branch


# ==============================
# 数据集辅助
# ==============================
SCENE_ALIASES = {
    "DJI_0166": "0166",
    "DJI_0415": "0415",
    "DJI_0502a": "502",
    "DJI_0502b": "502",
}

DEFAULT_IMAGE_DIR_CANDIDATES = [
    os.path.join("image_02", "data", "images"),
    os.path.join("image_02", "data"),
]


def _resolve_image_dir(seq_root: str, rel_candidates: List[str]) -> Optional[str]:
    """Return the first existing RGB目录 under seq_root based on候选相对路径."""
    for rel in rel_candidates:
        if not rel:
            continue
        rel_norm = rel.strip().strip("\\/")
        if not rel_norm:
            continue
        candidate = os.path.join(seq_root, rel_norm)
        if os.path.isdir(candidate):
            return candidate
    return None


def group_image_paths_by_seq(gt_folder: str, ext: str,
                             image_dir_candidates: List[str]) -> Dict[str, List[str]]:
    """
    按固定测试序列名搜集图像路径，并以 {seq_name: [paths]} 返回。
    image_dir_candidates 允许在不同数据布局之间切换（例如 data/images vs data）。
    """
    test_sequences = ["DJI_0166", "DJI_0415", "DJI_0502a", "DJI_0502b"]
    #test_sequences = ["DJI_0166", "DJI_0415", "DJI_0477a", "DJI_0477b", "DJI_0502a", "DJI_0502b"]
    #test_sequences = ["DJI_0166", "DJI_0415", "DJI_0477a", "DJI_0477b", "DJI_0502a", "DJI_0502b", "DJI_0478", "DJI_0486"]
    groups: Dict[str, List[str]] = {}
    for seq in test_sequences:
        seq_root = os.path.join(gt_folder, seq)
        image_dir = _resolve_image_dir(seq_root, image_dir_candidates)
        if image_dir is None:
            print(f"[跳过] 未找到图像目录：{seq_root}")
            continue
        paths = sorted(glob.glob(os.path.join(image_dir, f"*.{ext}")))
        if not paths:
            print(f"[跳过] 空图像目录：{image_dir}")
            continue
        groups[seq] = paths
    return groups


def _build_next_frame_map_from_groups(groups: Dict[str, List[str]]) -> Dict[str, str]:
    """为每张图构建同序列下一帧路径；末帧回退为自身。"""
    next_map: Dict[str, str] = {}
    for paths in groups.values():
        if not paths:
            continue
        for idx, path in enumerate(paths):
            next_map[path] = paths[idx + 1] if idx + 1 < len(paths) else path
    return next_map


def get_seq_from_path(img_path: str, gt_folder: str) -> str:
    """从完整路径还原序列名（路径形如 <seq>/image_02/.../xxx.jpg）。"""
    rel = os.path.relpath(img_path, gt_folder)
    return rel.split(os.sep)[0]


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
# 单图评估（误差、可视化、边缘 F1）
# ==============================

def evaluate_image(depth_np: np.ndarray,
                   gt: np.ndarray,
                   args: argparse.Namespace,
                   img_path: str,
                   vis_dir: Optional[str],
                   image_id: str
                   ) -> Tuple[Optional[Tuple[float, ...]],
                              Optional[float],
                              Optional[float]]:
    """
    对单张样本计算：
      - 传统 7 指标 + si_log（在中值对齐后）
      - 可视化（GT / 预测 / 差异）
      - Edge-F1（基于裁剪后的 GT）

    返回 (metrics_tuple_or_None, scale_ratio_or_None, edgeF1_or_None)
    """
    from cv2 import resize, INTER_LINEAR

    # 将预测深度 resize 到 GT 尺寸（仅用于误差评估与可视化）
    if gt.shape != depth_np.shape:
        depth_np = resize(depth_np, (gt.shape[1], gt.shape[0]), interpolation=INTER_LINEAR)

    valid_pred_mask = depth_np > 0
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
    if args.visualize and vis_dir is not None:
        pred_scaled_full = depth_np * ratio  # HxW
        diff_map = np.abs(gt - pred_scaled_full)
        save_colormap_img(gt, os.path.join(vis_dir, f"{image_id}_gt.png"),
                          cmap="plasma", vmin=0, vmax=max_d, mask=None)
        save_colormap_img(pred_scaled_full, os.path.join(vis_dir, f"{image_id}_pred.png"),
                          cmap="plasma", vmin=0, vmax=max_d, mask=None)
        save_colormap_img(diff_map, os.path.join(vis_dir, f"{image_id}_diff.png"),
                          cmap="inferno", vmin=0, vmax=(max_d - 0) * 0.3, mask=None)

    # Edge-F1：与“裁剪后的 GT 边缘”比对
    pred_scaled_full = depth_np * ratio
    edge_f1 = compute_edge_f1(pred_scaled_full, gt_depth=gt_clipped, img_rgb=None, q=95)

    return metrics, float(ratio), float(edge_f1)


# ==============================
# 主流程：推理 + 评估 + （可选）视图合成
# ==============================

def test_simple(args: argparse.Namespace) -> None:
    """
    核心流程：
      1) 加载模型
      2) 遍历测试图像：前向推理 -> 保存深度 .npy
      3) 若启用评估：计算指标、可视化、Edge-F1
      4) 若启用视图合成：估计相对位姿 -> 重投影 -> 掩码内 PSNR/SSIM
      5) 逐图 CSV 行缓存，最终在 eval_result.txt 中输出头注 + 表头 + 明细 + 汇总
    """

    # —— 权重解析与输出目录确定 ——
    weights_dir, weights_name = _select_weights_dir(args.exp_dir, args.weights)
    args.weights_dir = weights_dir

    exp_name = os.path.basename(os.path.normpath(args.exp_dir))
    run_dir_name = f'{exp_name}_{weights_name}'

    output_dir, npy_dir, vis_dir = prepare_output_dirs(args.output_base, run_dir_name, args.visualize)

    print(f'[exp_dir]    {os.path.abspath(args.exp_dir)}')
    print(f'[weights]    {weights_name} -> {weights_dir}')
    print(f'[outputs]    {output_dir}')
    data_root = os.path.abspath(args.gt_folder)
    print(f'[data_root]  {data_root}')

    if args.batch_size < 1:
        raise ValueError('--batch_size must be >= 1')

    # 统一的设备解析策略：默认走 CUDA，若不可用则根据用户选择报错或自动回退。
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

    print(f'[device]    {device}')

    encoder, decoder, feed_height, feed_width, model_branch = load_model(args, device)

    # —— 归一化判定（一次即可）——
    norm_cfg = _infer_norm_cfg(args)   # 自动判断是否需要均值方差归一化
    _log_norm_decision(args, norm_cfg)

    rel_candidates: List[str] = []
    if args.image_rel_path:
        rel_candidates.append(args.image_rel_path)
    rel_candidates.extend(DEFAULT_IMAGE_DIR_CANDIDATES)
    seen_candidates = set()
    image_dir_candidates: List[str] = []
    for rel in rel_candidates:
        rel_norm = (rel or '').strip().strip("\\/")
        if not rel_norm or rel_norm in seen_candidates:
            continue
        seen_candidates.add(rel_norm)
        image_dir_candidates.append(rel_norm)

    groups = group_image_paths_by_seq(data_root, args.ext, image_dir_candidates)
    image_paths = [p for seq in groups for p in groups[seq]]
    print(f"-> Found {len(image_paths)} test images")
    source_path_map: Dict[str, str] = {}
    if model_branch == "Madhuanand":
        source_path_map = _build_next_frame_map_from_groups(groups)
        fallback_self = sum(1 for p in image_paths if source_path_map.get(p, p) == p)
        print(f"[Madhuanand] using paired source frame for inference (self-fallback: {fallback_self}/{len(image_paths)})")

    metrics_list: List[Tuple[float, ...]] = []
    ratios: List[float] = []
    edge_f1_list: List[float] = []
    per_image_lines: List[str] = []
    scene_metrics = defaultdict(list)
    scene_ratios = defaultdict(list)
    scene_edge_f1 = defaultdict(list)


    total_images = len(image_paths)
    batch_size = args.batch_size

    for start_idx in range(0, total_images, batch_size):
        batch_paths = image_paths[start_idx:start_idx + batch_size]

        # ---------- 前向推理（批处理） ----------
        batch_tensors = [
            process_image(img_path, feed_width, feed_height, norm_cfg=norm_cfg)
            for img_path in batch_paths
        ]
        input_batch = torch.cat(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            if model_branch == "Madhuanand":
                source_tensors = [
                    process_image(source_path_map.get(img_path, img_path), feed_width, feed_height, norm_cfg=norm_cfg)
                    for img_path in batch_paths
                ]
                source_batch = torch.cat(source_tensors, dim=0).to(device)
                features_ref = encoder(input_batch)
                features_src = encoder(source_batch)
                fused_features = [
                    torch.stack([feat_ref, feat_src], dim=2)
                    for feat_ref, feat_src in zip(features_ref, features_src)
                ]
                disp = decoder(fused_features)[("disp", 0)]
            else:
                features = encoder(input_batch)
                disp = decoder(features)[("disp", 0)]
            depth_batch = _decode_disp_to_eval_depth(disp, args, model_branch)

        # depth_batch 形状为 (B,1,H,W)，此处拿到第 0 个通道的距离并移到 CPU 参与后续 NumPy 计算。
        depth_np_batch = depth_batch[:, 0].cpu().numpy()

        for offset, (img_path, depth_np) in enumerate(zip(batch_paths, depth_np_batch)):
            rel_path = os.path.relpath(img_path, data_root)
            image_id = os.path.splitext(rel_path)[0].replace(os.sep, "_")
            seq_name = get_seq_from_path(img_path, data_root)
            scene_key = SCENE_ALIASES.get(seq_name, seq_name)

            # 每张图仍单独保存预测，便于后续可视化/调试使用。
            np.save(os.path.join(npy_dir, f"{image_id}_depth.npy"), depth_np)

            # 逐图记录容器（初始化）
            row_vals = {
                "image_id": image_id,
                "edgeF1": None,
                "scale_ratio": None,
            }
            metric_tuple: Optional[Tuple[float, ...]] = None
            ratio: Optional[float] = None

            # ---------- 评估（若有 GT） ----------
            if args.do_evaluation:
                gt_path = os.path.join(os.path.dirname(img_path), "stereo", "depth_maps",
                                       os.path.basename(img_path) + ".geometric.bin")
                if not os.path.exists(gt_path):
                    print(f"[!] Missing GT: {gt_path}")
                else:
                    gt = read_bin_safe(gt_path)
                    metrics, ratio, edgef1 = evaluate_image(depth_np, gt, args, img_path, vis_dir, image_id)
                    if metrics is not None:
                        metrics_list.append(metrics)
                        metric_tuple = metrics
                        scene_metrics[scene_key].append(metrics)
                    if ratio is not None:
                        ratios.append(ratio)
                        row_vals["scale_ratio"] = float(ratio)
                        scene_ratios[scene_key].append(ratio)
                    if edgef1 is not None:
                        edge_f1_list.append(edgef1)
                        row_vals["edgeF1"] = float(edgef1)
                        scene_edge_f1[scene_key].append(edgef1)

            # ---------- 记录逐图 CSV 行 ----------
            row = [row_vals["image_id"]]
            if metric_tuple is not None:
                row += [f"{m:.4f}" for m in metric_tuple]
            else:
                row += [""] * len(ERROR_METRICS)
            row.append("" if row_vals["edgeF1"] is None else f"{row_vals['edgeF1']:.4f}")
            row.append("" if row_vals["scale_ratio"] is None else f"{row_vals['scale_ratio']:.6f}")
            row.extend(["", ""])  # median / rel_std 占位
            per_image_lines.append(",".join(row))

            global_idx = start_idx + offset
            print(f"[{global_idx + 1}/{total_images}] Processed: {image_id}")

    # ---------- 终端汇总 ----------
    if args.do_evaluation and metrics_list:
        mean_metrics = np.array(metrics_list).mean(0)  # 8 项
        print("\n==== Evaluation Results ====")
        print("abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, "
              "a1: {:.3f}, a2: {:.3f}, a3: {:.3f}, si_log: {:.3f}".format(*mean_metrics))
        if edge_f1_list:
            print("edge_F1: {:.3f}".format(np.mean(edge_f1_list)))
        if ratios:
            ratios_np = np.array(ratios)
            med = np.median(ratios_np)
            std = np.std(ratios_np / (med + 1e-12))
            print(f"Scale ratios | median: {med:.3f}, std: {std:.3f}")

    #    # ---------- 写入 CSV（参数/明细/汇总合并到一个文件） ----------
    if per_image_lines:
        os.makedirs(output_dir, exist_ok=True)
        out_csv = os.path.join(output_dir, "eval_results.csv")

        # 表头：kind + 原明细列 + 汇总两列（metric, value）
        header = ['kind'] + CSV_COLUMNS + ['metric', 'value']

        with open(out_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # 0) 参数行（kind=param）
            #    参数名写入 metric 列，参数值写入 value 列，其余列留空
            empty_cols = [''] * len(CSV_COLUMNS)
            for k in sorted(vars(args).keys()):
                v = getattr(args, k)
                writer.writerow(['param'] + empty_cols + [k, str(v)])

            # 1) 明细行（kind=detail）
            #    per_image_lines 与 CSV_COLUMNS 对齐：image_id + 各指标 + edgeF1 + scale_ratio
            #    末尾补空 metric,value 两列
            for line in per_image_lines:
                writer.writerow(['detail'] + line.split(',') + ['', ''])

            # 2) 汇总行（kind=summary，横向展开）
            if args.do_evaluation and metrics_list:
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

                summary_rows: List[List[str]] = []

                def build_summary_row(row_id: str,
                                      metrics_mean: np.ndarray,
                                      edge_vals: List[float],
                                      ratio_vals: List[float]) -> None:
                    summary_map = {name: '' for name in METRIC_NAMES}
                    if metrics_mean is not None:
                        for name, val in zip(ERROR_METRICS, metrics_mean):
                            summary_map[name] = f'{val:.6f}'
                    if edge_vals:
                        summary_map['edgeF1'] = f'{np.mean(edge_vals):.6f}'

                    ratio_median = ''
                    ratio_rel_std = ''
                    if ratio_vals:
                        ratios_np = np.array(ratio_vals)
                        med = np.median(ratios_np)
                        std = np.std(ratios_np / (med + 1e-12))
                        ratio_median = f'{med:.6f}'
                        ratio_rel_std = f'{std:.6f}'

                    row = []
                    for col in CSV_COLUMNS:
                        if col == 'image_id':
                            row.append(row_id)
                        elif col in METRIC_NAMES:
                            row.append(summary_map.get(col, ''))
                        elif col == 'scale_ratio':
                            row.append('')
                        elif col == 'scale_ratio_median':
                            row.append(ratio_median)
                        elif col == 'scale_ratio_rel_std':
                            row.append(ratio_rel_std)
                        else:
                            row.append('')
                    summary_rows.append(row)

                overall_mean = np.array(metrics_list).mean(0)
                build_summary_row('summary_mean', overall_mean, edge_f1_list, ratios)

                for scene_name, scene_values in sorted(scene_metrics.items()):
                    if not scene_values:
                        continue
                    scene_mean = np.array(scene_values).mean(0)
                    row_id = f'summary_scene_{scene_name}_mean'
                    build_summary_row(
                        row_id,
                        scene_mean,
                        scene_edge_f1.get(scene_name, []),
                        scene_ratios.get(scene_name, [])
                    )

                writer.writerow(['summary'] + label_row + ['', ''])
                for row in summary_rows:
                    writer.writerow(['summary'] + row + ['', ''])

        print(f"-> Results saved to: {out_csv}")

def _infer_norm_cfg(opt):
    """
    基于 `opt` 中的模型/权重关键词自动推断归一化策略。
    返回：
        {}                     -> 不做归一化（保持旧行为）
        {'normalize': True, ...} -> 开启归一化的配置
    """
    # 收集可能包含骨干/权重信息的字段，统一小写匹配
    names = []
    for key in [
        "encoder", "backbone", "model", "model_name", "teacher",
        "ckpt", "encoder_ckpt", "backbone_ckpt"
    ]:
        if hasattr(opt, key) and getattr(opt, key):
            names.append(str(getattr(opt, key)))
    joined = " ".join(names).lower()

    if "dino" in joined:  # 匹配 monodepth2_dino / dinov2 / dinov3 等
        if ("sat" in joined) or ("satellite" in joined):
            return dict(normalize=True, norm_mode="custom",
                        norm_mean=(0.430, 0.411, 0.296),
                        norm_std=(0.213, 0.156, 0.143))
        return dict(normalize=True, norm_mode="imagenet")

    # 其它情况：关闭
    return {}

def _log_norm_decision(opt, cfg: dict):
    """打印归一化决策与原因，便于记录可重复性。"""
    names = []
    for k in ["encoder", "backbone", "model", "model_name", "teacher",
              "ckpt", "encoder_ckpt", "backbone_ckpt"]:
        if hasattr(opt, k) and getattr(opt, k):
            names.append(str(getattr(opt, k)))
    joined = " ".join(names).lower()

    if not cfg:
        print("[Normalize] OFF | 维持原始 [0,1] 张量输入（数据侧不做归一化）")
        return

    if cfg.get("norm_mode") == "imagenet":
        print("[Normalize] ON  | mode=ImageNet | reason=自动启用（检测到 *dino* 关键词）")
    else:
        mean = cfg.get("norm_mean"); std = cfg.get("norm_std")
        sat_mean = (0.430, 0.411, 0.296); sat_std = (0.213, 0.156, 0.143)
        mode = "Satellite" if (mean == sat_mean and std == sat_std) else "Custom"
        print(f"[Normalize] ON  | mode={mode} | mean={mean} | std={std} | reason=自动启用（检测到 *dino* 关键词）")


# ==============================
# Entry Point
# ==============================
if __name__ == '__main__':
    args = parse_args()
    test_simple(args)

# -*- coding: utf-8 -*-
"""
UAVula 双数据集测试 + 可视化（按你的统计口径）：
- automask_ratio：被 automask 覆盖的像素 / 全图像素
- prev_ratio_on_whole：未被 automask 的像素里被 prev(-1) 选中的数量 / 全图像素
- next_ratio_on_whole：未被 automask 的像素里被 next(+1) 选中的数量 / 全图像素

落盘（仅 PNG，不保存 NPY）：
- 原图：center.png、prevsrc_*.png、nextsrc_*.png
- warp 图：prev_*_warp.png、next_*_warp.png
- 误差热力图：prev_*_err.png、next_*_err.png、err_min.png
- automask 01 PNG：automask01.png（0=未mask，1=mask）
- 指派图（含图例）：assign.png（未mask：红=prev，蓝=next；mask=黑）

tag：当前样本所属的数据集/配置标签（如 jsonK、defK）。

center_path / prev_path / next_path：中心帧、前一帧、后一帧图像的原始文件路径。

center_seq / center_idx：中心帧所在序列名与帧索引（从文件名解析）。

prev_seq / prev_idx：前一帧的序列名与帧索引。

next_seq / next_idx：后一帧的序列名与帧索引。

prev_offset / next_offset：相对中心帧的索引偏移。理想相邻应分别为 -1 与 +1；若不是，则说明非相邻帧（可用于筛查异常配对）。

err_min：对每个像素在所有源帧重投影误差中取最小值后的整图平均值（纯“最小重投影误差”的均值），单位为误差值（非百分比）。

loss_photometric：启用 automask 的光度损失（Monodepth2 风格：把重投影误差与 identity 误差拼接后逐像素取最小，再对全图取均值）。

automask_ratio：被 automask 屏蔽的像素比例（被判为 identity 更优的像素占全图像素的比例），取值 ∈ [0,1]。

prev_ratio_on_whole：只统计未被 automask 的像素中，被 前一帧（-1） 作为“最小重投影来源”选中的像素数，再除以全图像素数得到的比例（∈ [0,1]）。

next_ratio_on_whole：同上，但来源是 后一帧（+1）。

说明：prev_ratio_on_whole + next_ratio_on_whole 理论上应 ≈ 1 - automask_ratio（数值上允许极小误差）。

"""

import os
import csv
from typing import List, Set, Tuple as Tup

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from layers import transformation_from_parameters
from methods.datasets.UAVula_tri_dataset import UAVTripletJsonDataset
from methods import networks

# =========================
# 配置
# =========================
DATA_ROOT_IMAGES_TRAIN = "/mnt/data_nvme3n1p1/dataset/UAV_ula/dataset/Train"
TRIPLET_JSONL_TRAIN    = "/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_win10"
SPLIT_DIR              = "/mnt/data_nvme3n1p1/PycharmProjects/monodepth2/methods/splits/UAVula"

IMG_EXT                = ".png"
HEIGHT, WIDTH          = 288, 512
FRAME_IDS              = [0, -1, 1]
NUM_SCALES             = 4
NUM_WORKERS            = 4

# 权重
CKPT_ENCODER       = "/home/yxt/文档/mono_result/weights/UAVula_new/md2_vggt_xapp_bs8_lr1e-04_512x288_e40_step20/models/weights_39/encoder.pth"
CKPT_DEPTH         = "/home/yxt/文档/mono_result/weights/UAVula_new/md2_vggt_xapp_bs8_lr1e-04_512x288_e40_step20/models/weights_39/depth.pth"
CKPT_POSE_ENCODER  = "/home/yxt/文档/mono_result/weights/UAVula_new/md2_vggt_xapp_bs8_lr1e-04_512x288_e40_step20/models/weights_39/pose_encoder.pth"
CKPT_POSE          = "/home/yxt/文档/mono_result/weights/UAVula_new/md2_vggt_xapp_bs8_lr1e-04_512x288_e40_step20/models/weights_39/pose.pth"
STRICT_LOAD        = False

# 损失/统计
ENABLE_AUTOMASK = True
ADD_ID_NOISE    = 1e-5
# 差异很大阈值（用于 prev/next 的 masked→whole 口径差值）
LARGE_GAP_THRESH = 0.15
# 输出
RESULTS_DIR = "./"
CSV_ALL     = os.path.join(RESULTS_DIR, "results_all.csv")
CSV_NON_NEIGHBOR = os.path.join(RESULTS_DIR, "results_non_neighbors.csv")
CSV_LARGE_GAP = os.path.join(RESULTS_DIR, "results_large_gap.csv")

SAVE_VIZ = True
OUT_DIR  = "./viz_dual"
VIZ_LIMIT = 8202

# =========================
# 小工具
# =========================
def _as_str_path(x):
    if isinstance(x, (list, tuple)) and x: x = x[0]
    return str(x)

def _seq_and_idx(p: str):
    p = _as_str_path(p)
    parts = p.replace("\\", "/").split("/")
    seq = ""
    for q in parts:
        if q.startswith("DJI_"): seq = q
    if not seq:
        try:
            im_idx = parts.index("image_02")
            seq = parts[im_idx - 1]
        except Exception:
            seq = parts[-4] if len(parts) >= 4 else parts[-2]
    stem = os.path.splitext(os.path.basename(p))[0]
    idx_int = int(stem.lstrip("0") or "0")
    return seq, idx_int

def _ensure_dirs_for_csv():
    os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# CSV 初始化
# =========================
def _init_csv_if_needed(csv_path: str):
    """
    若文件不存在则写入表头。三类 CSV（all / non_neighbor / large_gap）统一使用同一套表头，
    方便后续合并或比对。
    """
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "tag",
                "center_path", "prev_path", "next_path",
                "center_seq", "center_idx",
                "prev_seq", "prev_idx",
                "next_seq", "next_idx",
                "prev_offset", "next_offset",
                "err_min",
                # —— 全图口径（未应用 mask，仅对照）
                "prev_ratio_all", "next_ratio_all",
                # —— 训练使用/展示口径：仅计未被 mask 的像素，但分母=全图像素
                "prev_ratio_masked_on_whole", "next_ratio_masked_on_whole",
                # 其它指标
                "automask_ratio", "loss_photometric",
                # 标记
                "non_neighbor_prev", "non_neighbor_next",
                "large_gap_flag"
            ])

def _row_from_stats(stats: dict) -> list:
    """把 run_batch 返回的 stats 映射成 CSV 行。"""
    return [
        stats["tag"],
        str(stats["center"]), str(stats["prev"]), str(stats["next"]),
        stats["center_seq"], stats["center_idx"],
        stats["prev_seq"],   stats["prev_idx"],
        stats["next_seq"],   stats["next_idx"],
        stats["prev_offset"], stats["next_offset"],
        stats["err_min"],
        stats["prev_ratio_all"], stats["next_ratio_all"],
        stats["prev_ratio_masked_on_whole"], stats["next_ratio_masked_on_whole"],
        stats["automask_ratio"], stats["loss_photometric"],
        int(stats["non_neighbor_prev"]), int(stats["non_neighbor_next"]),
        int(stats["large_gap_flag"]),
    ]


def _append_csv_row(csv_path: str, row: list):
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

# =========================
# 模型
# =========================
def init_depth_and_pose(num_layers=18, depth_scales=(0,1,2,3),
                        pose_model_input="pairs", weights_init="pretrained",
                        device=None):
    device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pretrained = (str(weights_init).lower() == "pretrained")

    depth_encoder = networks.ResnetEncoder(num_layers, use_pretrained).to(device_obj)
    depth_decoder = networks.DepthDecoder(depth_encoder.num_ch_enc, list(depth_scales)).to(device_obj)

    num_input_images = 2 if pose_model_input == "pairs" else len(FRAME_IDS)
    pose_encoder = networks.ResnetEncoder(num_layers, use_pretrained, num_input_images=num_input_images).to(device_obj)
    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2).to(device_obj)

    return {"encoder": depth_encoder, "depth": depth_decoder,
            "pose_encoder": pose_encoder, "pose": pose_decoder}, device_obj

def load_weights(models, device):
    for name, path in [("encoder", CKPT_ENCODER), ("depth", CKPT_DEPTH),
                       ("pose_encoder", CKPT_POSE_ENCODER), ("pose", CKPT_POSE)]:
        if path and os.path.isfile(path):
            state = torch.load(path, map_location=device)
            if "state_dict" in state: state = state["state_dict"]
            try:
                models[name].load_state_dict(state, strict=STRICT_LOAD)
                print(f"[Load OK] {name} <- {path}")
            except Exception as e:
                print(f"[Load FAIL] {name} {e}")

# =========================
# 几何
# =========================
class BackprojectDepth(torch.nn.Module):
    def __init__(self, b, h, w):
        super().__init__()
        mesh = torch.meshgrid(torch.arange(w, dtype=torch.float32),
                              torch.arange(h, dtype=torch.float32),
                              indexing="xy")
        self.register_buffer("pix", torch.stack([mesh[0].reshape(-1), mesh[1].reshape(-1)], 0))
    def forward(self, depth, invK):
        B, _, H, W = depth.shape
        pix = torch.cat([self.pix, torch.ones_like(self.pix[:1])], 0).unsqueeze(0).repeat(B, 1, 1).to(depth.device)
        cam = torch.matmul(invK[:, :3, :3], pix)
        cam = depth.view(B, 1, -1) * cam
        return torch.cat([cam, torch.ones(B, 1, cam.shape[-1], device=depth.device)], 1)

class Project3D(torch.nn.Module):
    def __init__(self, b, h, w, eps=1e-7):
        super().__init__(); self.h=h; self.w=w; self.eps=eps
    def forward(self, pts, K, T):
        P = torch.matmul(K, T)[:, :3, :]
        cam = torch.matmul(P, pts)
        pix = cam[:, :2, :] / (cam[:, 2:3, :] + self.eps)
        x = 2.0 * (pix[:, 0, :] / (self.w - 1.0)) - 1.0
        y = 2.0 * (pix[:, 1, :] / (self.h - 1.0)) - 1.0
        return torch.stack([x, y], 2).view(-1, self.h, self.w, 2)

# =========================
# Photometric 与辅助
# =========================
def disp_to_depth(disp, min_depth=0.1, max_depth=150):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def infer_depth(models, img):
    feats = models["encoder"](img)
    outs = models["depth"](feats)
    disp = outs.get(("disp", 0), next(v for v in outs.values() if torch.is_tensor(v)))
    _, depth = disp_to_depth(disp)
    return depth

def predict_pose(models, batch, device):
    tgt = batch[("color_aug", 0, 0)].to(device)
    out = {}
    for s in [sid for sid in FRAME_IDS if sid != 0]:
        src = batch[("color_aug", s, 0)].to(device)
        feats = models["pose_encoder"](torch.cat([tgt, src], 1))
        axis, trans = models["pose"]([feats])
        out[s] = transformation_from_parameters(axis[:, 0], trans[:, 0], invert=False)
    return out

def ssim(x, y):
    C1, C2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x  = F.avg_pool2d(x*x, 3, 1, 1) - mu_x*mu_x
    sigma_y  = F.avg_pool2d(y*y, 3, 1, 1) - mu_y*mu_y
    sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
    num = (2*mu_x*mu_y + C1) * (2*sigma_xy + C2)
    den = (mu_x*mu_x + mu_y*mu_y + C1) * (sigma_x + sigma_y + C2)
    return torch.clamp((1 - num / (den + 1e-7)) / 2, 0, 1)

def photometric_err(a, b, alpha=0.85):
    ssim_term = ssim(a, b).mean(1, keepdim=True)
    l1_term   = (a - b).abs().mean(1, keepdim=True)
    return alpha * ssim_term + (1 - alpha) * l1_term

def compute_identity_errors(sources: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    return torch.stack([photometric_err(src, target) for src in sources], dim=1)  # [B,N,1,H,W]

def reduce_photometric_loss_with_automask(reproj_errs: torch.Tensor,
                                          id_errs: torch.Tensor,
                                          add_noise: float = 1e-5) -> torch.Tensor:
    if add_noise and add_noise > 0:
        id_errs = id_errs + torch.randn_like(id_errs) * add_noise
    all_errs = torch.cat([reproj_errs, id_errs], dim=1)   # [B,2N,1,H,W]
    min_errs, _ = all_errs.min(dim=1, keepdim=True)       # [B,1,1,H,W]
    return min_errs.mean()

def warp_and_photometric_error(source_img, target_img, depth_t, K_s, invK_t, T_t2s, backproj, project):
    cam_points = backproj(depth_t, invK_t)
    grid = project(cam_points, K_s, T_t2s)
    source_warped = F.grid_sample(source_img, grid, padding_mode="border", align_corners=True)
    pe = photometric_err(source_warped, target_img)  # [B,1,H,W]
    return pe, source_warped

# =========================
# 可视化与保存（仅 PNG）
# =========================
def _save_rgb(img_t, save_path):
    """保存 [B,3,H,W] 或 [3,H,W] -> PNG"""
    if img_t.dim() == 4: img_t = img_t[0]
    arr = np.clip(img_t.detach().cpu().numpy().transpose(1,2,0), 0, 1)
    plt.figure(); plt.imshow(arr); plt.axis("off"); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()

def _save_heatmap_png(tensor_2d, save_png, title="", cmap="magma"):
    arr = tensor_2d.detach().cpu().numpy()
    plt.figure(); plt.imshow(arr, cmap=cmap); plt.colorbar()
    if title: plt.title(title)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(save_png, dpi=200); plt.close()

def _save_mask01(mask_bool_t, save_png, ratio: float = None):
    """
    显示并保存 0/1 二值 PNG（0=未mask，1=mask），在标题中显示 automask 百分比。
    视觉上：白=mask，黑=未mask（与 m*255 存盘一致）。
    """
    m = mask_bool_t.detach().cpu().numpy().astype(np.uint8)  # 0/1
    plt.figure()
    plt.imshow(m, cmap="gray", vmin=0, vmax=1)
    if ratio is not None:
        plt.title(f"automask = {ratio*100:.1f}%")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_png, dpi=200)
    plt.close()

def _save_assignment_mask(min_idx, save_png, title="", automask=None):
    mp = min_idx.cpu().numpy().astype(np.uint8)
    H, W = mp.shape
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[mp == 0] = [1, 0, 0]   # prev
    rgb[mp == 1] = [0, 0, 1]   # next
    if automask is not None:
        am = automask.detach().cpu().numpy().astype(bool)
        rgb[am] = [0, 0, 0]    # masked -> black
    plt.figure(figsize=(6, 4))
    plt.imshow(rgb)
    if title: plt.title(title)
    plt.axis("off")
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=[1, 0, 0], label="Prev (-1)"),
        mpatches.Patch(color=[0, 0, 1], label="Next (+1)"),
        mpatches.Patch(color=[0, 0, 0], label="Masked"),
    ]
    plt.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.75)
    plt.tight_layout()
    plt.savefig(save_png, dpi=200)
    plt.close()

# =========================
# 核心：单 batch
# =========================
def run_batch(models, batch, device, use_pose, tag, idx):
    B, _, H, W = batch[("color_aug", 0, 0)].shape
    tgt  = batch[("color_aug", 0, 0)].to(device)
    K    = batch[("K", 0)].to(device).float()
    invK = batch[("inv_K", 0)].to(device).float()

    depth   = infer_depth(models, tgt)
    backproj = BackprojectDepth(B, H, W).to(device)
    proj     = Project3D(B, H, W).to(device)

    if use_pose:
        T_map = predict_pose(models, batch, device)
    else:
        T_map = {s: torch.linalg.inv(batch[("T", s)].to(device).float())
                 for s in FRAME_IDS if s != 0}

    sid_list = [s for s in FRAME_IDS if s != 0]  # [-1, +1]

    # --- 重投影误差 / warp ---
    per_err, warped, src_imgs = [], [], []
    for s in sid_list:
        src = batch[("color_aug", s, 0)].to(device)
        e, w = warp_and_photometric_error(src, tgt, depth, K, invK, T_map[s], backproj, proj)
        per_err.append(e)      # [B,1,H,W]
        warped.append(w)       # [B,3,H,W]
        src_imgs.append(src)   # [B,3,H,W]

    reproj_errs = torch.stack(per_err, dim=1)  # [B,N,1,H,W]

    # --- automask 与 photometric loss ---
    if ENABLE_AUTOMASK:
        id_errs = compute_identity_errors(src_imgs, tgt)         # [B,N,1,H,W]
        loss_photometric = reduce_photometric_loss_with_automask(
            reproj_errs, id_errs, add_noise=ADD_ID_NOISE
        )
    else:
        min_reproj, _ = reproj_errs.min(dim=1, keepdim=True)
        loss_photometric = min_reproj.mean()

    # automask 像素：identity 获胜即 masked
    if 'id_errs' not in locals():
        id_errs = compute_identity_errors(src_imgs, tgt)
    chosen = torch.cat([reproj_errs, id_errs], dim=1).argmin(dim=1, keepdim=True)  # [B,1,1,H,W]
    N_src = reproj_errs.shape[1]
    automask_pixels = (chosen >= N_src)                    # bool, [B,1,1,H,W]
    automask_ratio  = float(automask_pixels.float().mean().item())

    # --- prev/next 统计 ---
    errs = reproj_errs
    minv, mini = errs.min(1, keepdim=True)                 # [B,1,1,H,W], [B,1,1,H,W]
    prev_pos = sid_list.index(-1) if -1 in sid_list else None
    next_pos = sid_list.index(+1) if +1 in sid_list else None

    prev_ratio_all = next_ratio_all = float("nan")
    prev_ratio_masked_on_whole = next_ratio_masked_on_whole = float("nan")

    if prev_pos is not None and next_pos is not None:
        sel = mini[0, 0, 0]                                 # [H,W]，最小重投影来源索引
        total_whole = float(sel.numel())                    # 分母=全图像素
        unmasked = (~automask_pixels[0, 0, 0])              # True=未mask

        # 全图口径（未排除 mask，只用来对照）
        prev_ratio_all = (sel.eq(prev_pos).sum().item()) / total_whole
        next_ratio_all = (sel.eq(next_pos).sum().item()) / total_whole

        # 你的统计口径：仅计未被 mask 的像素，但分母=全图像素
        prev_ratio_masked_on_whole = ((sel.eq(prev_pos) & unmasked).sum().item()) / total_whole
        next_ratio_masked_on_whole = ((sel.eq(next_pos) & unmasked).sum().item()) / total_whole

        # 自检：两者和应≈ 1 - automask_ratio
        s = prev_ratio_masked_on_whole + next_ratio_masked_on_whole
        target = 1.0 - automask_ratio
        if abs(s - target) > 0.02:
            print(f"[WARN] (prev+next)={s:.3f} vs 1-automask={target:.3f}")

    # --- 大差异样本标记（用 masked→whole 口径） ---
    LARGE_GAP = abs(prev_ratio_masked_on_whole - next_ratio_masked_on_whole) \
                if (not np.isnan(prev_ratio_masked_on_whole) and not np.isnan(next_ratio_masked_on_whole)) else 0.0
    large_gap_flag = (LARGE_GAP >= LARGE_GAP_THRESH)

    # --- 路径/索引与偏移 ---
    c_seq, c_idx = _seq_and_idx(batch["center"])
    p_seq, p_idx = _seq_and_idx(batch["prev"])
    n_seq, n_idx = _seq_and_idx(batch["next"])
    prev_offset  = p_idx - c_idx
    next_offset  = n_idx - c_idx
    non_neighbor_prev = (prev_offset != -1)
    non_neighbor_next = (next_offset != +1)

    # --- 可视化与落盘 ---
    if SAVE_VIZ and idx < VIZ_LIMIT and prev_pos is not None and next_pos is not None:
        os.makedirs(OUT_DIR, exist_ok=True)
        base = f"{c_seq}_{c_idx}"

        # 原图
        _save_rgb(tgt, os.path.join(OUT_DIR, f"{base}.png"))
        _save_rgb(src_imgs[prev_pos], os.path.join(OUT_DIR, f"{base}_prevsrc_{p_seq}_{p_idx}.png"))
        _save_rgb(src_imgs[next_pos], os.path.join(OUT_DIR, f"{base}_nextsrc_{n_seq}_{n_idx}.png"))

        # warp
        _save_rgb(warped[prev_pos], os.path.join(OUT_DIR, f"{base}_prev_{p_seq}_{p_idx}_warp.png"))
        _save_rgb(warped[next_pos], os.path.join(OUT_DIR, f"{base}_next_{n_seq}_{n_idx}_warp.png"))

        # automask 01（标题写比例）
        _save_mask01(automask_pixels[0, 0, 0], os.path.join(OUT_DIR, f"{base}_automask01.png"),
                     ratio=automask_ratio)

        # 误差热力图
        _save_heatmap_png(per_err[prev_pos][0, 0], os.path.join(OUT_DIR, f"{base}_prev_{p_seq}_{p_idx}_err.png"))
        _save_heatmap_png(per_err[next_pos][0, 0], os.path.join(OUT_DIR, f"{base}_next_{n_seq}_{n_idx}_err.png"))
        _save_heatmap_png(minv[0, 0, 0], os.path.join(OUT_DIR, f"{base}_err_min.png"))

        # 指派图（mask=黑；prev=红；next=蓝），标题显示三种比例
        title = (f"prev(mask→whole)={prev_ratio_masked_on_whole*100:.1f}%, "
                 f"next(mask→whole)={next_ratio_masked_on_whole*100:.1f}%, "
                 f"mask={automask_ratio*100:.1f}%")
        _save_assignment_mask(mini[0, 0, 0].int(),
                              os.path.join(OUT_DIR, f"{base}_assign.png"),
                              title,
                              automask=automask_pixels[0, 0, 0])

    # --- 汇总返回 ---
    stats = {
        "tag": tag,
        "center": batch["center"], "prev": batch["prev"], "next": batch["next"],
        "center_seq": c_seq, "center_idx": c_idx,
        "prev_seq": p_seq, "prev_idx": p_idx,
        "next_seq": n_seq, "next_idx": n_idx,
        "prev_offset": prev_offset, "next_offset": next_offset,
        "non_neighbor_prev": non_neighbor_prev, "non_neighbor_next": non_neighbor_next,

        "err_min": float(minv.mean().item()),
        "loss_photometric": float(loss_photometric.item()),

        # 比例
        "automask_ratio": automask_ratio,
        "prev_ratio_all": prev_ratio_all, "next_ratio_all": next_ratio_all,
        "prev_ratio_masked_on_whole": prev_ratio_masked_on_whole,
        "next_ratio_masked_on_whole": next_ratio_masked_on_whole,

        # 大差异标记
        "large_gap_flag": large_gap_flag,
    }
    return stats


# =========================
# 主流程
# =========================
def _normalize_seq(raw:str)->str:
    raw=raw.replace("\\","/").strip()
    if raw.startswith("Train/"): raw=raw[len("Train/"):]
    if raw.startswith("Validation/"): raw=raw[len("Validation/"):]
    if raw.startswith("./"): raw=raw[2:]
    return raw

def parse_split_pairs(lines:List[str])->Set[Tup[str,int]]:
    out=set()
    for ln in lines:
        ln=ln.split("#",1)[0].strip()
        if not ln: continue
        try:
            seq,idx_str=ln.rsplit(maxsplit=1)
            out.add((_normalize_seq(seq),int(idx_str)))
        except: continue
    return out

def filter_dataset_by_split(dataset:UAVTripletJsonDataset,pairs:Set[Tup[str,int]],verbose=True):
    before=len(dataset.samples)
    dataset.samples=[s for s in dataset.samples if (s["seq"],s.get("center_idx")) in pairs]
    after=len(dataset.samples)
    if verbose: print(f"[Filter] {before}->{after}")

def main():
    # 目录 & CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    _ensure_dirs_for_csv()
    _init_csv_if_needed(CSV_ALL)
    _init_csv_if_needed(CSV_NON_NEIGHBOR)
    _init_csv_if_needed(CSV_LARGE_GAP)

    # 读取 split 并过滤
    split_file = os.path.join(SPLIT_DIR, "train_files.txt")
    with open(split_file, "r", encoding="utf-8") as f:
        pairs = parse_split_pairs([ln.strip() for ln in f])

    datasets = {
        "poseNet": UAVTripletJsonDataset(
            DATA_ROOT_IMAGES_TRAIN, TRIPLET_JSONL_TRAIN,
            HEIGHT, WIDTH, FRAME_IDS, NUM_SCALES, is_train=False,
            img_ext=IMG_EXT, allow_flip=False,
            normalize=False, norm_mode="imagenet"
        ),
    }
    for ds in datasets.values():
        filter_dataset_by_split(ds, pairs)

    loaders = {
        k: DataLoader(
            v, batch_size=1, shuffle=False, sampler=SequentialSampler(v),
            num_workers=NUM_WORKERS, pin_memory=True
        )
        for k, v in datasets.items()
    }

    # 模型
    models, device = init_depth_and_pose()[0:2]
    for m in models.values():
        m.eval()
    load_weights(models, device)

    # 推理
    for tag, loader in loaders.items():
        print(f"\n====== 测试 {tag} 数据集 ======")
        use_pose = True

        for b_idx, batch in enumerate(loader):
            with torch.no_grad():
                stats = run_batch(models, batch, device, use_pose, tag, b_idx)

            # 安全取值（有默认，避免 KeyError）
            center_path = _as_str_path(stats.get("center", ""))
            err_min     = float(stats.get("err_min", float("nan")))
            loss_ph     = float(stats.get("loss_photometric", float("nan")))
            prev_all    = float(stats.get("prev_ratio_all", float("nan")))
            next_all    = float(stats.get("next_ratio_all", float("nan")))
            prev_mw     = float(stats.get("prev_ratio_masked_on_whole", float("nan")))
            next_mw     = float(stats.get("next_ratio_masked_on_whole", float("nan")))
            am_ratio    = float(stats.get("automask_ratio", float("nan")))
            nn_prev     = bool(stats.get("non_neighbor_prev", False))
            nn_next     = bool(stats.get("non_neighbor_next", False))
            lg_flag     = bool(stats.get("large_gap_flag", False))

            flag_nn = " NN" if (nn_prev or nn_next) else ""
            flag_lg = " LG" if lg_flag else ""

            print(
                f"[{tag}] {center_path} | "
                f"min误差={err_min:.6f} | "
                f"prev(all)={prev_all * 100:.1f}% "
                f"next(all)={next_all * 100:.1f}% | "
                f"prev(mask→whole)={prev_mw * 100:.1f}% "
                f"next(mask→whole)={next_mw * 100:.1f}% | "
                f"autoMask={am_ratio * 100:.1f}% | "
                f"loss={loss_ph:.6f}{flag_nn}{flag_lg}"
            )

            # 写 CSV：全量 + 非邻接 + 大差异
            row = _row_from_stats(stats)
            _append_csv_row(CSV_ALL, row)
            if nn_prev or nn_next:
                _append_csv_row(CSV_NON_NEIGHBOR, row)
            if lg_flag:
                _append_csv_row(CSV_LARGE_GAP, row)

            if b_idx + 1 >= VIZ_LIMIT:
                break



if __name__ == "__main__":
    main()

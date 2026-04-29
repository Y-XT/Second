import copy
import os
import numpy as np
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import torch
import wandb

def _to_numpy_hwc_uint8(x):
    """将 tensor/ndarray/PIL 统一成 numpy uint8, HWC."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        # CHW -> HWC
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0)
        x = x.numpy()
    elif hasattr(x, "numpy"):  # e.g. PIL.Image -> numpy 由 wandb 也能接，但我们统一
        try:
            x = np.array(x)
        except Exception:
            x = np.array(x)

    # 单通道 -> 3 通道
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=2)
    elif x.ndim == 3 and x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)

    # 归一化到 uint8
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 1) if x.max() <= 1.0 else np.clip(x / 255.0, 0, 1)
        x = (x * 255.0 + 0.5).astype(np.uint8)
    return x

def _log_row_images_to_wandb(images, titles, key, step, save_path=None, log_to_wandb=True):
    """
    images: list[np.ndarray/PIL/torch.Tensor]  会被统一为 HWC uint8
    titles: list[str]  与 images 对应
    key:    wandb 面板的键名
    step:   日志 step
    """
    ims = [_to_numpy_hwc_uint8(im) for im in images]
    n = len(ims)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 3), dpi=150)
    if n == 1:
        axes = [axes]
    for ax, im, title in zip(axes, ims, titles):
        ax.imshow(im)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.tight_layout(pad=0.1, w_pad=0.1)
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    if log_to_wandb:
        wandb.log({key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def _to_numpy_hw_float(x):
    """将 tensor/ndarray 统一成 numpy float32, HW."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().squeeze().numpy()
    else:
        x = np.asarray(x).squeeze().astype(np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected a 2D scalar map, got shape={x.shape}")
    return x


def _log_scalar_row_with_colorbar_to_wandb(
    scalar_images,
    scalar_titles,
    key,
    step,
    vmin,
    vmax,
    cmap="magma",
    extra_images=None,
    extra_titles=None,
    scalar_vmins=None,
    scalar_vmaxs=None,
    colorbar_groups=None,
    scalar_cmaps=None,
    scalar_valid_masks=None,
    masked_color=(1.0, 1.0, 1.0, 1.0),
    save_path=None,
    log_to_wandb=True,
):
    """
    记录一行图像：前半是标量热力图（带 colorbar），后半可追加普通 RGB 图（不带 colorbar）。
    """
    extra_images = extra_images or []
    extra_titles = extra_titles or []
    if len(scalar_images) != len(scalar_titles):
        raise ValueError("scalar_images and scalar_titles must have the same length")
    if len(extra_images) != len(extra_titles):
        raise ValueError("extra_images and extra_titles must have the same length")

    scalar_maps = [_to_numpy_hw_float(im) for im in scalar_images]
    n_scalar = len(scalar_maps)
    if scalar_vmins is None:
        scalar_vmins = [float(vmin)] * n_scalar
    else:
        if len(scalar_vmins) != n_scalar:
            raise ValueError("scalar_vmins and scalar_images must have the same length")
        scalar_vmins = [float(v) for v in scalar_vmins]
    if scalar_vmaxs is None:
        scalar_vmaxs = [float(vmax)] * n_scalar
    else:
        if len(scalar_vmaxs) != n_scalar:
            raise ValueError("scalar_vmaxs and scalar_images must have the same length")
        scalar_vmaxs = [float(v) for v in scalar_vmaxs]
    if colorbar_groups is None:
        colorbar_groups = [0] * n_scalar
    else:
        if len(colorbar_groups) != n_scalar:
            raise ValueError("colorbar_groups and scalar_images must have the same length")
        colorbar_groups = [int(g) for g in colorbar_groups]
    if scalar_cmaps is None:
        scalar_cmaps = [cmap] * n_scalar
    else:
        if len(scalar_cmaps) != n_scalar:
            raise ValueError("scalar_cmaps and scalar_images must have the same length")
    if scalar_valid_masks is None:
        scalar_valid_masks = [None] * n_scalar
    else:
        if len(scalar_valid_masks) != n_scalar:
            raise ValueError("scalar_valid_masks and scalar_images must have the same length")
    rgb_maps = [_to_numpy_hwc_uint8(im) for im in extra_images]

    total = len(scalar_maps) + len(rgb_maps)
    if total == 0:
        return

    fig, axes = plt.subplots(1, total, figsize=(3 * total, 2.5), dpi=150)
    if total == 1:
        axes = [axes]

    idx = 0
    scalar_axes_by_group = {}
    color_mappable_by_group = {}
    for i, (val_np, title) in enumerate(zip(scalar_maps, scalar_titles)):
        ax = axes[idx]
        idx += 1
        lo = scalar_vmins[i]
        hi = scalar_vmaxs[i]
        if hi <= lo:
            hi = lo + 1e-6
        gid = colorbar_groups[i]
        valid_mask = scalar_valid_masks[i]
        cmap_obj = copy.copy(mpl_cm.get_cmap(scalar_cmaps[i]))
        if hasattr(cmap_obj, "set_bad"):
            cmap_obj.set_bad(color=masked_color)
        if valid_mask is not None:
            mask_np = _to_numpy_hw_float(valid_mask) > 0.5
            val_to_show = np.ma.array(val_np, mask=~mask_np)
        else:
            val_to_show = val_np
        mappable = ax.imshow(val_to_show, cmap=cmap_obj, vmin=lo, vmax=hi)
        ax.set_title(title, fontsize=5, pad=2)
        ax.axis("off")
        scalar_axes_by_group.setdefault(gid, []).append(ax)
        if gid not in color_mappable_by_group:
            color_mappable_by_group[gid] = mappable

    for rgb_np, title in zip(rgb_maps, extra_titles):
        ax = axes[idx]
        idx += 1
        ax.imshow(rgb_np)
        ax.set_title(title, fontsize=5, pad=2)
        ax.axis("off")

    num_groups = len(scalar_axes_by_group)
    bar_h = 0.024
    bar_gap = 0.012
    bar_bottom_base = 0.06
    if num_groups > 0:
        bars_total_h = num_groups * bar_h + max(0, num_groups - 1) * bar_gap
        subplot_bottom = bar_bottom_base + bars_total_h + 0.02
    else:
        subplot_bottom = 0.12
    fig.subplots_adjust(left=0.004, right=0.996, top=0.92, bottom=subplot_bottom, wspace=0.01)

    if scalar_axes_by_group and color_mappable_by_group:
        # colorbar 仅对 scalar_images 生效，避免视觉上与 extra_images（如 fid/mask）混淆。
        for rank, gid in enumerate(sorted(scalar_axes_by_group.keys())):
            scalar_axes = scalar_axes_by_group[gid]
            color_mappable = color_mappable_by_group[gid]
            x0 = min(ax.get_position().x0 for ax in scalar_axes)
            x1 = max(ax.get_position().x1 for ax in scalar_axes)
            span = max(0.01, x1 - x0)
            bar_width = span * 0.88
            bar_x = x0 + (span - bar_width) * 0.5
            bar_bottom = bar_bottom_base + (num_groups - 1 - rank) * (bar_h + bar_gap)
            cax = fig.add_axes([bar_x, bar_bottom, bar_width, bar_h])
            cbar = fig.colorbar(
                color_mappable,
                cax=cax,
                orientation="horizontal",
            )
            cbar.ax.tick_params(labelsize=5, pad=1, length=2)
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    if log_to_wandb:
        wandb.log({key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def _disp_to_colormap(disp_tensor, to_tensor=False, cmap: str = "magma"):
    """
    将 1 通道 disp 转为彩色图。
    - disp_tensor: torch.Tensor, 形状 [1,H,W] 或 [B,1,H,W] 中取出的单张
    - to_tensor=True: 返回 float[3,H,W] (0~1)，用于 TensorBoard
      to_tensor=False: 返回 uint8[H,W,3]，用于 W&B 的 Image
    - cmap 默认 'magma'，若 matplotlib 不可用则回退为灰度三通道
    """
    disp = disp_tensor.detach().float().squeeze()  # [H,W]
    # 按每张图自适应 0~1 归一化
    dmin = torch.min(disp)
    dmax = torch.max(disp)
    disp_norm = (disp - dmin) / (dmax - dmin + 1e-6)
    disp_np = disp_norm.clamp(0, 1).cpu().numpy()  # [H,W] in [0,1]

    if mpl_cm is not None:
        colored = mpl_cm.get_cmap(cmap)(disp_np)[..., :3]  # [H,W,3] float 0~1
    else:
        # 回退：将灰度复制为 3 通道
        colored = np.stack([disp_np, disp_np, disp_np], axis=-1)

    if to_tensor:
        # 返回 [3,H,W] float 0~1
        return torch.from_numpy(colored).permute(2, 0, 1)
    else:
        # 返回 [H,W,3] uint8
        return (colored * 255).astype(np.uint8)


def _scalar_to_colormap_fixed(value_tensor, to_tensor=False, cmap: str = "magma", vmin: float = 0.0, vmax: float = 1.0):
    """
    将任意单通道标量图按固定范围 [vmin, vmax] 映射到彩色图。
    适用于需要跨样本可比的可视化（如 P_derot）。
    """
    val = value_tensor.detach().float().squeeze()  # [H,W]
    lo = float(vmin)
    hi = float(vmax)
    if hi <= lo:
        hi = lo + 1e-6

    val_norm = ((val - lo) / (hi - lo)).clamp(0, 1)
    val_np = val_norm.cpu().numpy()

    if mpl_cm is not None:
        colored = mpl_cm.get_cmap(cmap)(val_np)[..., :3]
    else:
        colored = np.stack([val_np, val_np, val_np], axis=-1)

    if to_tensor:
        return torch.from_numpy(colored).permute(2, 0, 1)
    return (colored * 255).astype(np.uint8)


def _scalar_to_colormap_masked(
    value_tensor,
    valid_mask_tensor,
    to_tensor=False,
    cmap: str = "magma",
    vmin: float = 0.0,
    vmax: float = 1.0,
    invalid_color=(0, 0, 0),
):
    val = value_tensor.detach().float().squeeze()
    valid = valid_mask_tensor.detach().float().squeeze() > 0.5
    lo = float(vmin)
    hi = float(vmax)
    if hi <= lo:
        hi = lo + 1e-6

    val_norm = ((val - lo) / (hi - lo)).clamp(0, 1)
    val_np = val_norm.cpu().numpy()
    valid_np = valid.cpu().numpy()

    if mpl_cm is not None:
        colored = mpl_cm.get_cmap(cmap)(val_np)[..., :3]
    else:
        colored = np.stack([val_np, val_np, val_np], axis=-1)

    invalid_rgb = np.array(invalid_color, dtype=np.float32).reshape(1, 1, 3) / 255.0
    colored[~valid_np] = invalid_rgb

    if to_tensor:
        return torch.from_numpy(colored).permute(2, 0, 1)
    return (colored * 255).astype(np.uint8)


def _source_fid_to_rgb(
    source_tensor,
    prev_fid: int,
    next_fid: int,
    to_tensor: bool = False,
    color_prev=(59, 130, 246),   # blue
    color_next=(245, 158, 11),   # orange
    color_none=(0, 0, 0),        # black
):
    """
    将来源图（像素值为 frame_id）转成离散 RGB 图，便于看出 -1/+1 的主导区域。
    """
    src = source_tensor.detach().float().squeeze().cpu().numpy()  # [H,W]
    h, w = src.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = np.array(color_none, dtype=np.uint8)

    prev_mask = np.isclose(src, float(prev_fid))
    next_mask = np.isclose(src, float(next_fid))
    rgb[prev_mask] = np.array(color_prev, dtype=np.uint8)
    rgb[next_mask] = np.array(color_next, dtype=np.uint8)

    if to_tensor:
        return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return rgb

# -*- coding: utf-8 -*-
"""Resize (if needed) and center-crop utilities for images/depth."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import skimage.transform
from PIL import Image


def compute_resize_crop_params(orig_w: int,
                               orig_h: int,
                               target_w: int,
                               target_h: int) -> Tuple[float, int, int, int, int]:
    """Compute isotropic upscaling (if needed) + center crop parameters."""
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError(f"Invalid image size: ({orig_w}, {orig_h})")

    scale = 1.0
    if orig_w < target_w or orig_h < target_h:
        scale = max(target_w / float(orig_w), target_h / float(orig_h))

    new_w = int(math.ceil(orig_w * scale))
    new_h = int(math.ceil(orig_h * scale))

    left = max(0, (new_w - target_w) // 2)
    top = max(0, (new_h - target_h) // 2)
    return scale, new_w, new_h, left, top


def resize_and_center_crop_pil(img: Image.Image,
                               target_w: int,
                               target_h: int,
                               params: Tuple[float, int, int, int, int] | None = None,
                               interp=None) -> Tuple[Image.Image, Tuple[float, int, int, int, int]]:
    if params is None:
        params = compute_resize_crop_params(img.width, img.height, target_w, target_h)
    scale, new_w, new_h, left, top = params

    if scale != 1.0:
        if interp is None:
            if hasattr(Image, "Resampling"):
                interp = Image.Resampling.LANCZOS
            else:
                interp = Image.ANTIALIAS
        img = img.resize((new_w, new_h), resample=interp)

    if (new_w, new_h) != (target_w, target_h):
        img = img.crop((left, top, left + target_w, top + target_h))

    return img, params


def resize_and_center_crop_array(arr: np.ndarray,
                                 target_w: int,
                                 target_h: int,
                                 params: Tuple[float, int, int, int, int],
                                 order: int = 1) -> np.ndarray:
    scale, new_w, new_h, left, top = params
    out = arr
    if scale != 1.0:
        out = skimage.transform.resize(
            out,
            (new_h, new_w),
            order=order,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        ).astype(np.float32)

    if (new_w, new_h) != (target_w, target_h):
        out = out[top:top + target_h, left:left + target_w]

    return out


def resize_array(arr: np.ndarray, target_h: int, target_w: int, order: int = 1) -> np.ndarray:
    """Resize a HxW array to target size."""
    out = skimage.transform.resize(
        arr,
        (target_h, target_w),
        order=order,
        preserve_range=True,
        mode="constant",
        anti_aliasing=False,
    ).astype(np.float32)
    return out


def adjust_K_after_resize_crop(K4: np.ndarray,
                               orig_w: int,
                               orig_h: int,
                               target_w: int,
                               target_h: int,
                               params: Tuple[float, int, int, int, int]) -> np.ndarray:
    """Adjust normalized K (4x4) after isotropic resize + center crop."""
    scale, _, _, left, top = params
    K = np.array(K4, dtype=np.float32, copy=True)

    fx = K[0, 0] * float(orig_w)
    fy = K[1, 1] * float(orig_h)
    cx = K[0, 2] * float(orig_w)
    cy = K[1, 2] * float(orig_h)

    fx *= scale
    fy *= scale
    cx *= scale
    cy *= scale
    cx -= float(left)
    cy -= float(top)

    K[0, 0] = fx / float(target_w)
    K[1, 1] = fy / float(target_h)
    K[0, 2] = cx / float(target_w)
    K[1, 2] = cy / float(target_h)

    return K

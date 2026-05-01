# methods/datasets/tri_triplet_base.py
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional
import numbers

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .mono_dataset import MonoDataset, pil_loader
from .crop_utils import (
    compute_resize_crop_params,
    resize_and_center_crop_pil,
    resize_and_center_crop_array,
    adjust_K_after_resize_crop,
    resize_array,
)


def _K3_to_K4(K3: np.ndarray) -> np.ndarray:
    K4 = np.eye(4, dtype=np.float32)
    K4[:3, :3] = K3.astype(np.float32)
    return K4


def _make_K_pyramid(K0_3x3: np.ndarray, W_net: int, H_net: int, num_scales: int):
    """
    根据输入分辨率 (W_net, H_net) 下的 3x3 内参，生成各尺度金字塔（含 inv_K）。
    """
    Ks, invKs = {}, {}
    K0 = K0_3x3.astype(np.float32)
    for s in range(num_scales):
        sw = (W_net // (2 ** s)) / float(W_net)
        sh = (H_net // (2 ** s)) / float(H_net)
        S = np.diag([sw, sh, 1.0]).astype(np.float32)
        Ks3 = (S @ K0).astype(np.float32)
        K4 = _K3_to_K4(Ks3)
        Ks[s] = torch.from_numpy(K4)
        invKs[s] = torch.from_numpy(np.linalg.pinv(K4).astype(np.float32))
    return Ks, invKs


class BaseTripletDataset(MonoDataset):
    """
    针对三元组（prev, center, next）的通用数据集基类，封装：
      - triplets.jsonl 读取与归档
      - 默认 K 金字塔生成
      - 颜色增广、可选归一化
      - GT 深度探测、读取、重采样、翻转
    子类只需提供：
      - 默认 K（归一化形式）
      - 可选的 seq 正规化逻辑
      - 根据 center 图像定位 GT 深度文件的方法
    """

    def __init__(self,
                 data_path: str,
                 triplet_root: str,
                 height: int,
                 width: int,
                 frame_idxs: List[int],
                 num_scales: int,
                 *,
                 triplet_manifest_glob: str = "triplets.jsonl",
                 is_train: bool = False,
                 img_ext: str = ".jpg",
                 allow_flip: bool = False,
                 default_K_4x4: np.ndarray,
                 fallback_vggt_hw: Optional[int] = None,
                 normalize: bool = False,
                 norm_mode: str = "imagenet",
                 norm_mean: tuple = (0.485, 0.456, 0.406),
                 norm_std: tuple = (0.229, 0.224, 0.225),
                 normalization_mode: str = "copy",
                 norm_source_prefix: str = "color_aug",
                 norm_target_prefix: str = "color_norm",
                 use_triplet_pose: bool = False,
                 use_vggt_depth: bool = False):

        self._depth_exts = ('.npy', '.npz', '.png', '.tif', '.tiff', '.exr')
        self._depth_path_cache: Dict[str, Optional[str]] = {}
        self._depth_target_shape: Optional[tuple] = None
        self._missing_depth_logged = False
        self._missing_vggt_logged = False
        self.load_depth = False

        self.image_root = Path(data_path)
        self.triplets_root = Path(triplet_root)
        self._triplet_manifest_glob = str(triplet_manifest_glob)
        self.use_triplet_pose = bool(use_triplet_pose)
        self.use_vggt_depth = bool(use_vggt_depth)
        self._warned_missing_pose = False
        self._warned_singular_pose = False
        self._pose_flip_matrix = np.diag([-1.0, 1.0, 1.0, 1.0]).astype(np.float32)

        self.allow_flip = bool(allow_flip)
        if self.use_triplet_pose and self.allow_flip:
            print(f"[{self.__class__.__name__}] use_triplet_pose=True，启用水平翻转时将同步镜像外部位姿")

        self._default_K_4x4 = np.asarray(default_K_4x4, dtype=np.float32)
        self._fallback_hw = fallback_vggt_hw
        self.full_res_shape = (1024, 576)
        self._depth_target_shape = (self.full_res_shape[1], self.full_res_shape[0])

        super().__init__(data_path=data_path,
                         filenames=[],
                         height=height,
                         width=width,
                         frame_idxs=list(frame_idxs),
                         num_scales=num_scales,
                         is_train=is_train,
                         img_ext=img_ext,
                         allow_flip=self.allow_flip)
        split_name = "train" if self.is_train else "val"
        flip_status = "ON" if self.allow_flip else "OFF"
        print(f"[{self.__class__.__name__}] horizontal_flip={flip_status} ({split_name})")

        self.normalize = bool(normalize)
        if norm_mode == "imagenet":
            self._norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self._norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        elif norm_mode == "custom":
            self._norm_mean = np.array(norm_mean, dtype=np.float32)
            self._norm_std = np.array(norm_std, dtype=np.float32)
        else:
            raise ValueError("norm_mode must be 'imagenet' or 'custom'")

        self._norm_apply_mode = str(normalization_mode)
        self._norm_source_prefix = norm_source_prefix
        self._norm_target_prefix = norm_target_prefix

        self.samples: List[Dict] = self._load_all_triplets(self.triplets_root)
        self.load_depth = self._probe_depth_availability()

    # ------- triplets.jsonl 读取 -------
    def _load_all_triplets(self, root: Path) -> List[Dict]:
        out: List[Dict] = []
        pattern = self._triplet_manifest_glob or "triplets.jsonl"
        manifest_files = sorted(root.rglob(pattern))
        if not manifest_files:
            raise FileNotFoundError(f"未在 {root} 下找到 {pattern}")

        for p in manifest_files:
            base = p.parent
            with p.open("r", encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    rec = json.loads(ln)
                    item = self._parse_triplet_record(rec, base)
                    if item is None:
                        continue
                    out.append(item)

        if not out:
            raise FileNotFoundError(f"在 {root} 下读取 {pattern} 结果为空")
        return out

    def _parse_triplet_record(self, rec: Dict, json_dir: Path) -> Optional[Dict]:
        seq_raw = rec.get("seq", "")
        seq = self._resolve_sequence(seq_raw)

        def _img_path(fname: str) -> str:
            return str(self.image_root / seq / fname)

        def _frame_idx(fname: str) -> int:
            return self._frame_idx_from_name(Path(fname).stem)

        center_info = rec.get("center", {})
        prev_info = rec.get("prev", {})
        next_info = rec.get("next", {})

        center_file = center_info.get("file")
        prev_file = prev_info.get("file")
        next_file = next_info.get("file")
        if not center_file or not prev_file or not next_file:
            return None

        center_path = _img_path(center_file)
        prev_path = _img_path(prev_file)
        next_path = _img_path(next_file)

        H, W = self._extract_hw(rec)

        item = dict(
            seq=seq,
            center=center_path,
            prev=prev_path,
            next=next_path,
            center_idx=self._frame_idx_from_name(Path(center_file).stem),
            prev_idx=_frame_idx(prev_file),
            next_idx=_frame_idx(next_file),
            H=H,
            W=W
        )

        def _resolve_optional_path(path_value):
            if not path_value:
                return None
            p = Path(path_value)
            if not p.is_absolute():
                p = (json_dir / p).resolve()
            return str(p)

        depth_gt_path = rec.get("depth_gt_path", None)
        depth_gt_path = _resolve_optional_path(depth_gt_path)
        if not depth_gt_path:
            depth_gt_path = self._infer_depth_path_from_center(center_path)
        item["depth_gt_path"] = depth_gt_path

        vggt_depth_path = rec.get("depth_vggt_path") or rec.get("depth_t_npy") or rec.get("depth_npy")
        vggt_conf_path = rec.get("depth_vggt_conf_path") or rec.get("depth_conf_t_npy")
        item["vggt_depth_path"] = _resolve_optional_path(vggt_depth_path)
        item["vggt_conf_path"] = _resolve_optional_path(vggt_conf_path)

        if self.use_triplet_pose:
            pose_prev = rec.get("T_prev_to_t", None)
            pose_next = rec.get("T_next_to_t", None)
            if pose_prev is not None or pose_next is not None:
                item["_external_pose"] = {
                    "prev_to_center": np.asarray(pose_prev, dtype=np.float32) if pose_prev is not None else None,
                    "next_to_center": np.asarray(pose_next, dtype=np.float32) if pose_next is not None else None,
                }

        return item

    def _resolve_sequence(self, seq_raw: str) -> str:
        """子类可覆盖：用于规范化 JSON 中的 seq 字段。"""
        return seq_raw.replace("\\", "/").lstrip("./")

    def _extract_hw(self, rec: Dict) -> tuple:
        H = rec.get("H", self._fallback_hw)
        W = rec.get("W", self._fallback_hw)
        if H is None or W is None:
            raise ValueError("triplets.jsonl 缺少 H/W 字段，且未提供 fallback_vggt_hw")
        return int(H), int(W)

    def _frame_idx_from_name(self, stem: str) -> int:
        return int(stem.lstrip("0") or "0")

    # ------- MonoDataset 接口 -------
    def get_color(self, *args, **kwargs):
        raise NotImplementedError("Triplet 数据集不使用 get_color")

    def check_depth(self):
        return self.load_depth

    def get_depth(self, *args, **kwargs):
        raise NotImplementedError("Triplet 数据集不提供 GT 深度")

    # ------- 采样 -------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        rec = self.samples[index]
        inputs: Dict = {}

        do_color_aug = self.is_train and (random.random() > 0.5)
        do_flip = False if not self.allow_flip else (self.is_train and (random.random() > 0.5))

        center_path = self._resolve_image_path(rec["center"])
        prev_path = self._resolve_image_path(rec["prev"])
        next_path = self._resolve_image_path(rec["next"])

        if center_path is None or prev_path is None or next_path is None:
            raise FileNotFoundError(f"找不到图像文件（尝试过 Train/Validation 互换）：{rec}")

        img_c = pil_loader(center_path)
        img_p = pil_loader(prev_path)
        img_n = pil_loader(next_path)

        if do_flip:
            img_c = img_c.transpose(Image.FLIP_LEFT_RIGHT)
            img_p = img_p.transpose(Image.FLIP_LEFT_RIGHT)
            img_n = img_n.transpose(Image.FLIP_LEFT_RIGHT)

        if img_c.size != img_p.size or img_c.size != img_n.size:
            if not hasattr(self, "_warned_triplet_size"):
                self._warned_triplet_size = False
            worker_info = torch.utils.data.get_worker_info()
            should_log = (worker_info is None) or (worker_info.id == 0)
            if should_log and not self._warned_triplet_size:
                split_name = "train" if self.is_train else "val"
                print(
                    f"[{self.__class__.__name__}] 警告({split_name})：三元组图像尺寸不一致，"
                    f"将 prev/next 等比缩放并中心裁剪到 center 尺寸，"
                    f"seq={rec.get('seq')} idx={rec.get('center_idx')} "
                    f"sizes={img_c.size},{img_p.size},{img_n.size}"
                )
                self._warned_triplet_size = True

            ref_w, ref_h = img_c.size
            if img_p.size != (ref_w, ref_h):
                params_p = compute_resize_crop_params(img_p.size[0], img_p.size[1], ref_w, ref_h)
                img_p, _ = resize_and_center_crop_pil(img_p, ref_w, ref_h, params=params_p)
            if img_n.size != (ref_w, ref_h):
                params_n = compute_resize_crop_params(img_n.size[0], img_n.size[1], ref_w, ref_h)
                img_n, _ = resize_and_center_crop_pil(img_n, ref_w, ref_h, params=params_n)

        target_w, target_h = self.full_res_shape
        orig_w, orig_h = img_c.size
        crop_params = compute_resize_crop_params(
            orig_w, orig_h, target_w, target_h
        )

        img_c, _ = resize_and_center_crop_pil(
            img_c, target_w, target_h, params=crop_params
        )
        img_p, _ = resize_and_center_crop_pil(
            img_p, target_w, target_h, params=crop_params
        )
        img_n, _ = resize_and_center_crop_pil(
            img_n, target_w, target_h, params=crop_params
        )

        inputs[("color", 0, -1)] = img_c
        inputs[("color", -1, -1)] = img_p
        inputs[("color", 1, -1)] = img_n

        K_norm = adjust_K_after_resize_crop(
            self._default_K_4x4,
            orig_w,
            orig_h,
            target_w,
            target_h,
            crop_params,
        )
        K_net3 = self._build_intrinsics(K_norm)
        Ks, invKs = _make_K_pyramid(K_net3, self.width, self.height, self.num_scales)
        for s in range(self.num_scales):
            inputs[("K", s)] = Ks[s]
            inputs[("inv_K", s)] = invKs[s]

        color_aug = transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue
        ) if do_color_aug else (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            inputs.pop(("color", i, -1), None)
            inputs.pop(("color_aug", i, -1), None)

        self._apply_normalization(inputs)

        offset_prev = float(rec["prev_idx"] - rec["center_idx"])
        offset_next = float(rec["next_idx"] - rec["center_idx"])
        for fid in self.frame_idxs:
            if fid == 0 or fid == "s":
                continue
            if isinstance(fid, numbers.Integral) and fid < 0:
                inputs[("frame_offset", fid)] = torch.tensor(offset_prev, dtype=torch.float32)
            elif isinstance(fid, numbers.Integral) and fid > 0:
                inputs[("frame_offset", fid)] = torch.tensor(offset_next, dtype=torch.float32)

        if self.load_depth:
            expected_shape = (orig_h, orig_w)
            depth_map, conf_map = self._load_depth_map(
                rec, do_flip, expected_shape=expected_shape, crop_params=crop_params
            )
            if depth_map is None:
                target_shape = self._depth_target_shape or (self.height, self.width)
                depth_map = np.zeros(target_shape, dtype=np.float32)
                conf_map = None
                has_depth = False
            else:
                has_depth = True

            depth_tensor = torch.from_numpy(np.expand_dims(depth_map, 0).astype(np.float32))
            inputs["depth_gt"] = depth_tensor
            inputs["depth_has_valid"] = torch.tensor(has_depth, dtype=torch.bool)

            if conf_map is not None:
                conf_tensor = torch.from_numpy(np.expand_dims(conf_map, 0).astype(np.float32))
                inputs["depth_conf"] = conf_tensor

        if self.use_vggt_depth:
            vggt_depth, vggt_conf = self._load_vggt_depth_map(
                rec, do_flip, expected_shape=(orig_h, orig_w), crop_params=crop_params
            )
            if vggt_depth is not None:
                vggt_tensor = torch.from_numpy(np.expand_dims(vggt_depth, 0).astype(np.float32))
                inputs["vggt_depth"] = vggt_tensor
                inputs["vggt_depth_has_valid"] = torch.tensor(True, dtype=torch.bool)
                if vggt_conf is not None:
                    vggt_conf_tensor = torch.from_numpy(np.expand_dims(vggt_conf, 0).astype(np.float32))
                    inputs["vggt_conf"] = vggt_conf_tensor
            else:
                inputs["vggt_depth_has_valid"] = torch.tensor(False, dtype=torch.bool)
        else:
            inputs["vggt_depth_has_valid"] = torch.tensor(False, dtype=torch.bool)

        inputs["full_res"] = torch.tensor([target_h, target_w], dtype=torch.int32)
        inputs["center_idx"] = torch.tensor(rec["center_idx"], dtype=torch.int32)
        inputs["seq"] = rec["seq"]

        self._attach_external_pose(inputs, rec, do_flip=do_flip)

        return inputs

    def _build_intrinsics(self, K_norm_4x4: Optional[np.ndarray] = None) -> np.ndarray:
        K_src = K_norm_4x4 if K_norm_4x4 is not None else self._default_K_4x4
        K3 = np.asarray(K_src, dtype=np.float32)[:3, :3]
        fx = K3[0, 0] * float(self.width)
        fy = K3[1, 1] * float(self.height)
        cx = K3[0, 2] * float(self.width)
        cy = K3[1, 2] * float(self.height)
        return np.array([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=np.float32)

    # ------- 归一化 -------
    def _apply_normalization(self, inputs: Dict):
        if not self.normalize:
            return

        mean = torch.tensor(self._norm_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self._norm_std, dtype=torch.float32).view(3, 1, 1)

        if self._norm_apply_mode == "copy":
            for i in self.frame_idxs:
                for s in range(self.num_scales):
                    src_key = (self._norm_source_prefix, i, s)
                    if src_key not in inputs:
                        continue
                    tensor = inputs[src_key]
                    normed = self._normalize_tensor(tensor, mean, std)
                    dest_key = (self._norm_target_prefix, i, s)
                    inputs[dest_key] = normed
        elif self._norm_apply_mode == "inplace":
            for i in self.frame_idxs:
                for s in range(self.num_scales):
                    for prefix in ("color", "color_aug"):
                        key = (prefix, i, s)
                        if key not in inputs:
                            continue
                        tensor = inputs[key]
                        inputs[key] = self._normalize_tensor(tensor, mean, std)
        else:
            raise ValueError(f"Unsupported normalization_mode: {self._norm_apply_mode}")

    @staticmethod
    def _normalize_tensor(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        if tensor.dim() != 3 or tensor.shape[0] != 3:
            return tensor
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        mm = mean.to(tensor.device)
        ss = std.to(tensor.device)
        return (tensor - mm) / ss

    # ------- 数据集路径兜底：Train/Validation 互换 -------
    def _resolve_image_path(self, path: str) -> Optional[str]:
        """
        优先使用原路径；若不存在，尝试将路径中的 Train/Validation 互换后再检查。
        用于处理部分样本仅存在于 Train 或 Validation 目录的情况。
        """
        if path and os.path.exists(path):
            return path
        if not path:
            return None
        p = Path(path)
        parts = list(p.parts)
        if "Train" in parts:
            idx = parts.index("Train")
            parts[idx] = "Validation"
        elif "Validation" in parts:
            idx = parts.index("Validation")
            parts[idx] = "Train"
        else:
            return None
        alt_path = Path(*parts)
        if alt_path.exists():
            return str(alt_path)
        return None

    # ------- 深度相关 -------
    def _probe_depth_availability(self) -> bool:
        if not self.samples:
            return False
        for rec in self.samples:
            depth_path = rec.get("depth_gt_path")
            if depth_path and os.path.exists(depth_path):
                rec["_depth_gt_exists"] = True
                return True
        return False

    def _infer_depth_path_from_center(self, center_path: str) -> Optional[str]:
        raise NotImplementedError

    def _read_depth_file(self, path: str):
        ext = Path(path).suffix.lower()
        conf = None
        if ext == ".npy":
            depth = np.load(path)
        elif ext == ".npz":
            with np.load(path) as data:
                if "depth" in data:
                    depth = data["depth"]
                elif "arr_0" in data:
                    depth = data["arr_0"]
                else:
                    raise KeyError(f"{path} 缺少 depth 数组")
                if "confidence" in data:
                    conf = data["confidence"].astype(np.float32)
        elif ext in (".png", ".tif", ".tiff", ".exr"):
            depth = np.array(Image.open(path))
        else:
            raise ValueError(f"Unsupported depth format: {ext} ({path})")

        depth = depth.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0] = 0.0

        if conf is not None:
            conf = conf.astype(np.float32)
            conf[~np.isfinite(conf)] = 0.0
            conf = np.clip(conf, 0.0, None)

        return depth, conf

    def _load_depth_map(self,
                        rec: Dict,
                        do_flip: bool,
                        expected_shape: Optional[tuple] = None,
                        crop_params: Optional[tuple] = None):
        depth_path = rec.get("depth_gt_path")
        conf_map = None
        if not depth_path or not os.path.exists(depth_path):
            if not self._missing_depth_logged:
                print(f"[{self.__class__.__name__}] 警告：未找到深度文件 seq={rec.get('seq')} idx={rec.get('center_idx')}")
                self._missing_depth_logged = True
            return None, None

        depth, conf = self._read_depth_file(depth_path)
        if expected_shape is not None and depth.shape != expected_shape:
            if not hasattr(self, "_warned_depth_size_count"):
                self._warned_depth_size_count = 0
            worker_info = torch.utils.data.get_worker_info()
            should_log = (worker_info is None) or (worker_info.id == 0)
            if should_log and self._warned_depth_size_count < 3:
                split_name = "train" if self.is_train else "val"
                print(
                    f"[{self.__class__.__name__}] 警告({split_name})：depth 与图像尺寸不一致，"
                    f"将先缩放到图像尺寸，再执行中心裁剪。"
                    f"seq={rec.get('seq')} idx={rec.get('center_idx')} "
                    f"depth={depth.shape} image={expected_shape}"
                )
                self._warned_depth_size_count += 1
            depth = resize_array(depth, expected_shape[0], expected_shape[1], order=1)

        if expected_shape is not None and conf is not None and conf.shape != expected_shape:
            conf = resize_array(conf, expected_shape[0], expected_shape[1], order=1)

        if crop_params is not None:
            target_w, target_h = self.full_res_shape
            depth = resize_and_center_crop_array(
                depth, target_w, target_h, crop_params, order=1
            )
            if conf is not None:
                conf_map = resize_and_center_crop_array(
                    conf, target_w, target_h, crop_params, order=1
                )
        else:
            conf_map = conf

        if do_flip:
            depth = np.ascontiguousarray(np.fliplr(depth))
            if conf_map is not None:
                conf_map = np.ascontiguousarray(np.fliplr(conf_map))

        return depth, conf_map

    def _load_vggt_depth_map(self,
                             rec: Dict,
                             do_flip: bool,
                             expected_shape: Optional[tuple] = None,
                             crop_params: Optional[tuple] = None):
        depth_path = rec.get("vggt_depth_path")
        if not depth_path:
            return None, None
        if not os.path.exists(depth_path):
            if not self._missing_vggt_logged:
                print(f"[{self.__class__.__name__}] 警告：未找到 VGGT 深度文件 seq={rec.get('seq')} idx={rec.get('center_idx')} path={depth_path}")
                self._missing_vggt_logged = True
            return None, None

        depth, depth_embedded_conf = self._read_depth_file(depth_path)

        conf_map = depth_embedded_conf
        conf_path = rec.get("vggt_conf_path")
        if conf_path and os.path.exists(conf_path):
            conf_data = np.load(conf_path)
            if isinstance(conf_data, np.lib.npyio.NpzFile):
                if "confidence" in conf_data:
                    conf_map = conf_data["confidence"]
                elif "arr_0" in conf_data:
                    conf_map = conf_data["arr_0"]
                conf_data.close()
            else:
                conf_map = conf_data
            if conf_map is not None:
                conf_map = conf_map.astype(np.float32)
                conf_map[~np.isfinite(conf_map)] = 0.0

        if expected_shape is not None and depth.shape != expected_shape:
            if not hasattr(self, "_warned_vggt_size_count"):
                self._warned_vggt_size_count = 0
            if self._warned_vggt_size_count < 3:
                split_name = "train" if self.is_train else "val"
                print(
                    f"[{self.__class__.__name__}] 警告({split_name})：vggt_depth 尺寸与图像不一致，"
                    f"将先缩放到图像尺寸，再执行中心裁剪。"
                    f"seq={rec.get('seq')} idx={rec.get('center_idx')} "
                    f"depth={depth.shape} image={expected_shape}"
                )
                self._warned_vggt_size_count += 1
            depth = resize_array(depth, expected_shape[0], expected_shape[1], order=1)

        if expected_shape is not None and conf_map is not None and conf_map.shape != expected_shape:
            conf_map = resize_array(conf_map, expected_shape[0], expected_shape[1], order=1)

        if crop_params is not None:
            target_w, target_h = self.full_res_shape
            depth = resize_and_center_crop_array(
                depth, target_w, target_h, crop_params, order=1
            )
            if conf_map is not None:
                conf_map = resize_and_center_crop_array(
                    conf_map, target_w, target_h, crop_params, order=1
                )

        if do_flip:
            depth = np.ascontiguousarray(np.fliplr(depth))
            if conf_map is not None:
                conf_map = np.ascontiguousarray(np.fliplr(conf_map))

        return depth, conf_map

    # ------- 位姿注入 -------
    def _attach_external_pose(self, inputs: Dict, rec: Dict, do_flip: bool = False):
        if not self.use_triplet_pose:
            return

        pose_pack = rec.get("_external_pose")
        if not pose_pack:
            if not self._warned_missing_pose:
                print(f"[{self.__class__.__name__}] 警告：外部位姿缺失 seq={rec.get('seq')} idx={rec.get('center_idx')}")
                self._warned_missing_pose = True
            return

        has_valid = False
        for fid in self.frame_idxs:
            if fid == 0 or fid == "s" or not isinstance(fid, numbers.Integral):
                continue

            if fid < 0:
                mat = pose_pack.get("prev_to_center")
            else:
                mat = pose_pack.get("next_to_center")

            if mat is None:
                continue

            try:
                T = np.linalg.inv(mat).astype(np.float32)
            except np.linalg.LinAlgError:
                if not self._warned_singular_pose:
                    print(f"[{self.__class__.__name__}] 警告：位姿矩阵不可逆 seq={rec.get('seq')} idx={rec.get('center_idx')}")
                    self._warned_singular_pose = True
                continue

            if do_flip:
                T = self._mirror_pose_matrix(T)

            inputs[("external_cam_T_cam", 0, fid)] = torch.from_numpy(T)
            has_valid = True

        if has_valid:
            inputs["external_pose_available"] = torch.tensor(True, dtype=torch.bool)

    def _mirror_pose_matrix(self, T: np.ndarray) -> np.ndarray:
        if T.shape != (4, 4):
            return T
        F = self._pose_flip_matrix
        return (F @ T @ F).astype(np.float32)

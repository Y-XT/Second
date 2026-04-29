from .mono_dataset import MonoDataset
from .crop_utils import (
    compute_resize_crop_params,
    resize_and_center_crop_pil,
    resize_and_center_crop_array,
    adjust_K_after_resize_crop,
    resize_array,
)
import numpy as np
import random
import torch
from torchvision import transforms
import PIL.Image as pil
import os

class UAVid2020_Dataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
        Args:
        data_path from option
        filenames train_filenames = readlines(fpath.format("train")) Train/seq10/data 0
        height
        width
        frame_idxs default=[0, -1, 1]
        num_scales 4
        is_train True
        img_ext img_ext = '.png' if self.opt.png else '.jpg'
    """
    _K_CHINA_4x4 = np.array(
        [[0.6399, 0.0, 0.5, 0.0],
         [0.0, 1.1376, 0.5, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
    )
    _K_GERMANY_4x4 = np.array(
        [[0.6314, 0.0, 0.5, 0.0],
         [0.0, 1.1227, 0.5, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
    )
    _DEFAULT_K_4x4 = _K_GERMANY_4x4

    @classmethod
    def _choose_default_k(cls, k_region: str, data_path: str):
        region = (k_region or "auto").strip().lower()
        if region in {"china", "cn"}:
            return cls._K_CHINA_4x4, "china"
        if region in {"germany", "de"}:
            return cls._K_GERMANY_4x4, "germany"

        if region == "auto":
            hay = (data_path or "").lower()
            if "china" in hay:
                return cls._K_CHINA_4x4, "china"
            if "germany" in hay:
                return cls._K_GERMANY_4x4, "germany"

        return cls._DEFAULT_K_4x4, "default"

    def __init__(self, *args,
                 normalize=False,               # ← 开关：是否做归一化（就地替换）
                 norm_mode="imagenet",          # "imagenet" 或 "custom"
                 norm_mean=(0.485, 0.456, 0.406),
                 norm_std=(0.229, 0.224, 0.225),
                 k_region="auto",
                 **kwargs):
        # 深度文件缓存需在父类初始化（check_depth）前构建
        self._depth_path_cache = {}
        self._depth_exts = ('.npy', '.npz', '.png', '.tif', '.tiff', '.exr')
        self._depth_target_shape = None
        self._missing_depth_logged = False
        self._warned_image_size_count = 0
        self._warned_depth_size_count = 0
        super(UAVid2020_Dataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.

        # 归一化配置
        self.normalize = bool(normalize)
        if norm_mode == "imagenet":
            self._norm_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self._norm_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        elif norm_mode == "custom":
            self._norm_mean = np.array(norm_mean, dtype=np.float32)
            self._norm_std = np.array(norm_std, dtype=np.float32)
        else:
            raise ValueError("norm_mode must be 'imagenet' or 'custom'")

        #self.K = np.array([[0.58, 0, 0.5, 0],
        #                   [0, 1.92, 0.5, 0],
        #                   [0, 0, 1, 0],
        #                   [0, 0, 0, 1]], dtype=np.float32)
        default_k_4x4, k_source = self._choose_default_k(k_region, self.data_path)
        if k_source == "default" and (k_region or "auto").strip().lower() == "auto":
            print(f"[{self.__class__.__name__}] 未能从路径推断内参区域，使用默认 K")
        else:
            print(f"[{self.__class__.__name__}] 使用 {k_source} 内参")
        print(f"[{self.__class__.__name__}] K={default_k_4x4.tolist()}")
        self.K = default_k_4x4.copy()
        self.missing_images = []
        self.full_res_shape = (1024, 576)
        self._depth_target_shape = (self.full_res_shape[1], self.full_res_shape[0])
        self._pending_image_size = None
        self._pending_crop_params = None
        #China & Germany have different k or resolution
        #self.full_res_shape = (5280, 3956)
        #self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def _should_log_warning(self):
        worker_info = torch.utils.data.get_worker_info()
        return (worker_info is None) or (worker_info.id == 0)

    def _warn_image_size_mismatch(self, folder, center_index, target_index, size, expected):
        if self._warned_image_size_count >= 3 or not self._should_log_warning():
            return
        split_name = "train" if self.is_train else "val"
        print(
            f"[{self.__class__.__name__}] 警告({split_name})：邻帧图像尺寸不一致，"
            f"将先对齐到中心帧尺寸。folder={folder} center={center_index:010d} "
            f"target={target_index:010d} size={size} expected={expected}"
        )
        self._warned_image_size_count += 1

    def _warn_depth_size_mismatch(self, folder, frame_index, depth_shape, image_shape):
        if self._warned_depth_size_count >= 3 or not self._should_log_warning():
            return
        split_name = "train" if self.is_train else "val"
        print(
            f"[{self.__class__.__name__}] 警告({split_name})：depth 与图像尺寸不一致，"
            f"将先缩放到图像尺寸，再执行中心裁剪。folder={folder} "
            f"index={frame_index:010d} depth={depth_shape} image={image_shape}"
        )
        self._warned_depth_size_count += 1

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps.

        <frame_id> 是一个整数（例如 0、-1 或 1），表示相对于当前样本的时间步。

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and self.allow_flip and random.random() > 0.5

        line = self.filenames[index].split()
        # line = ['Train/seq10/data', '0']
        #line = self.filenames[index].split('/')
        #line = ['seq03', '000001.jpg']  00001.PNG NOW
        folder = line[0]
        frame_index = int(line[1])
        side = None  # UAVid2020 为单目数据集，保留参数占位
        #if len(line) == 3:
        #    frame_index = int(line[1])
        #else:
        #    frame_index = 0

        #if len(line) == 3:
        #    side = line[2]
        #else:
        #    side = None

        #for i in self.frame_idxs:
        #    if i == "s":
        #        other_side = {"r": "l", "l": "r"}[side]
        #        inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
        #    else:
        #        inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        target_w, target_h = self.full_res_shape
        ref_frame_id = 0 if 0 in self.frame_idxs else self.frame_idxs[0]
        raw_colors = {}

        for i in self.frame_idxs:
            target_index = frame_index + i
            fname = f"{target_index:010d}{self.img_ext}"
            image_path = os.path.join(self.data_path, folder, fname)

            if not os.path.exists(image_path):
                #self.missing_images.append(image_path)
                #print(f"[跳过] 缺失图像: {image_path}")
                return self.__getitem__((index + 1) % len(self.filenames))

            raw_colors[i] = self.loader(image_path)

        ref_color = raw_colors[ref_frame_id]
        orig_size = ref_color.size  # (W, H)
        crop_params = compute_resize_crop_params(
            orig_size[0], orig_size[1], target_w, target_h
        )
        K_base = adjust_K_after_resize_crop(
            self.K, orig_size[0], orig_size[1], target_w, target_h, crop_params
        )

        for i in self.frame_idxs:
            target_index = frame_index + i
            color = raw_colors[i]
            if color.size != orig_size:
                self._warn_image_size_mismatch(
                    folder, frame_index, target_index, color.size, orig_size
                )
                align_params = compute_resize_crop_params(
                    color.size[0], color.size[1], orig_size[0], orig_size[1]
                )
                color, _ = resize_and_center_crop_pil(
                    color, orig_size[0], orig_size[1], params=align_params
                )
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)

            color, _ = resize_and_center_crop_pil(
                color, target_w, target_h, params=crop_params
            )

            inputs[("color", i, -1)] = color

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = (K_base if K_base is not None else self.K).copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            #color_aug = transforms.ColorJitter.get_params(
            #    self.brightness, self.contrast, self.saturation, self.hue)
            color_aug = transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

            # ===== 新增：按需就地归一化（几何/增强之后，返回前） =====
            if self.normalize:
                mean = torch.tensor(self._norm_mean).view(3, 1, 1)
                std = torch.tensor(self._norm_std).view(3, 1, 1)
                for i in self.frame_idxs:
                    for s in range(self.num_scales):
                        for prefix in ("color", "color_aug"):
                            key = (prefix, i, s)
                            t = inputs.get(key, None)
                            if t is None:
                                continue
                            # 保险：确保 float32 且归一化前在 [0,1]
                            if t.dtype != torch.float32:
                                t = t.float()
                            if t.max() > 1.0:  # 若上游没 /255，这里兜底
                                t = t / 255.0
                            mm = mean.to(t.device)
                            ss = std.to(t.device)
                            inputs[key] = (t - mm) / ss
            # ===== 归一化结束 =====

        if self.load_depth:
            self._pending_image_size = (orig_size[1], orig_size[0]) if orig_size else None
            self._pending_crop_params = crop_params
            try:
                depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            finally:
                self._pending_image_size = None
                self._pending_crop_params = None
            if depth_gt is None:
                target_shape = self._depth_target_shape or (self.height, self.width)
                depth_gt = np.zeros(target_shape, dtype=np.float32)
                has_depth = False
            else:
                has_depth = True

            depth_tensor = torch.from_numpy(np.expand_dims(depth_gt, 0).astype(np.float32))
            inputs["depth_gt"] = depth_tensor
            inputs["depth_has_valid"] = torch.tensor(has_depth, dtype=torch.bool)

        return inputs


    def check_depth(self):
        if not self.filenames:
            return False

        total = len(self.filenames)
        for idx in range(total):
            line = self.filenames[idx].split()
            if len(line) < 2:
                continue
            folder = line[0]
            try:
                frame_index = int(line[1])
            except ValueError:
                continue
            depth_path = self._resolve_depth_path(folder, frame_index)
            if depth_path is not None and os.path.isfile(depth_path):
                if self._depth_target_shape is None:
                    depth = self._read_depth_file(depth_path)
                    self._depth_target_shape = depth.shape
                return True
        return False

    def _resolve_depth_path(self, folder, frame_index):
        key = (folder, frame_index)
        if key in self._depth_path_cache:
            return self._depth_path_cache[key]

        base_dir = os.path.normpath(os.path.join(self.data_path, folder))
        seq_dir = os.path.dirname(base_dir)
        if not os.path.isdir(seq_dir):
            self._depth_path_cache[key] = None
            return None

        stem = f"{frame_index:010d}"
        depth_dir = os.path.join(seq_dir, "depth")
        found_path = None
        for ext in self._depth_exts:
            cand = os.path.join(depth_dir, stem + ext)
            if os.path.isfile(cand):
                found_path = cand
                break
        self._depth_path_cache[key] = found_path
        return found_path

    def _read_depth_file(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            depth = np.load(path)
        elif ext == '.npz':
            data = np.load(path)
            depth = data['arr_0']
        elif ext in ('.png', '.tif', '.tiff', '.exr'):
            with pil.open(path) as img:
                depth = np.array(img)
        else:
            raise ValueError(f"Unsupported depth format: {ext} ({path})")
        depth = depth.astype(np.float32)
        # 统一清理异常值，负值和 NaN 视为无效深度
        depth[~np.isfinite(depth)] = 0.0
        depth[depth < 0] = 0.0
        return depth

    def get_depth(self, folder, frame_index, side=None, do_flip=False):
        depth_path = self._resolve_depth_path(folder, frame_index)
        if depth_path is None:
            if not self._missing_depth_logged:
                print(f"[UAVid2020_Dataset] 警告：未找到深度文件，folder={folder}, index={frame_index:010d}")
                self._missing_depth_logged = True
            return None

        depth = self._read_depth_file(depth_path)
        expected_hw = self._pending_image_size
        if expected_hw is not None and depth.shape != expected_hw:
            self._warn_depth_size_mismatch(folder, frame_index, depth.shape, expected_hw)
            depth = resize_array(depth, expected_hw[0], expected_hw[1], order=1)

        crop_params = self._pending_crop_params
        if crop_params is not None:
            target_w, target_h = self.full_res_shape
            depth = resize_and_center_crop_array(
                depth, target_w, target_h, crop_params, order=1
            )
        if do_flip:
            depth = np.ascontiguousarray(np.fliplr(depth))
        return depth

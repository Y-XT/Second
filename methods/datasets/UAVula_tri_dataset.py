# methods/datasets/uav_triplet_json_dataset.py
from pathlib import Path
from typing import List

import numpy as np

from .tri_triplet_base import BaseTripletDataset


class UAVTripletJsonDataset(BaseTripletDataset):
    """
    UAVula 三元组数据集：读取 triplets.jsonl，输出三帧图像及默认 K 金字塔。
    仅依赖 PoseNet 预测相对位姿，忽略 JSON 中的 VGGT K/T。
    """

    _DEFAULT_K_4x4 = np.array(
        [[0.78913, 0.0, 0.49802, 0.0],
         [0.0, 1.40643, 0.45859, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
    )

    def __init__(self,
                 data_path: str,
                 triplet_root: str,
                 height: int,
                 width: int,
                 frame_idxs: List[int],
                 num_scales: int,
                 is_train: bool = False,
                 img_ext: str = ".jpg",
                 allow_flip: bool = False,
                 vggt_target_width: int = 518,
                 normalize: bool = False,
                 norm_mode: str = "imagenet",
                 norm_mean: tuple = (0.485, 0.456, 0.406),
                 norm_std: tuple = (0.229, 0.224, 0.225),
                 use_triplet_pose: bool = False,
                 use_vggt_depth: bool = False,
                 triplet_manifest_glob: str = "triplets.jsonl",
                 use_external_mask: bool = False,
                 external_mask_dir: str = "mask",
                 external_mask_ext: str = ".png",
                 external_mask_thresh: float = 0.5):

        super().__init__(
            data_path=data_path,
            triplet_root=triplet_root,
            height=height,
            width=width,
            frame_idxs=frame_idxs,
            num_scales=num_scales,
            is_train=is_train,
            img_ext=img_ext,
            allow_flip=allow_flip,
            default_K_4x4=self._DEFAULT_K_4x4,
            fallback_vggt_hw=int(vggt_target_width),
            normalize=normalize,
            norm_mode=norm_mode,
            norm_mean=norm_mean,
            norm_std=norm_std,
            use_triplet_pose=use_triplet_pose,
            use_vggt_depth=use_vggt_depth,
            triplet_manifest_glob=triplet_manifest_glob,
            use_external_mask=use_external_mask,
            external_mask_dir=external_mask_dir,
            external_mask_ext=external_mask_ext,
            external_mask_thresh=external_mask_thresh,
        )

    def _resolve_sequence(self, seq_raw: str) -> str:
        seq_s = seq_raw.replace("\\", "/").lstrip("./")
        root_s = str(self.image_root).replace("\\", "/").rstrip("/")

        if root_s.endswith("/Train") and seq_s.startswith("Train/"):
            return seq_s[len("Train/"):]
        if root_s.endswith("/Validation") and seq_s.startswith("Validation/"):
            return seq_s[len("Validation/"):]
        return seq_s

    def _infer_depth_path_from_center(self, center_path: str):
        if not center_path:
            return None
        if center_path in self._depth_path_cache:
            return self._depth_path_cache[center_path]

        center_p = Path(center_path)
        data_dir = center_p.parent
        cam_dir = data_dir.parent if data_dir is not None else None

        found_path = None
        if cam_dir and cam_dir.is_dir():
            stem = center_p.stem
            depth_dirs = [cam_dir / "depth", cam_dir / "Depth"]
            for depth_dir in depth_dirs:
                if not depth_dir.is_dir():
                    continue
                for ext in self._depth_exts:
                    candidate = depth_dir / f"{stem}{ext}"
                    if candidate.is_file():
                        found_path = str(candidate.resolve())
                        break
                if found_path:
                    break

        self._depth_path_cache[center_path] = found_path
        return found_path

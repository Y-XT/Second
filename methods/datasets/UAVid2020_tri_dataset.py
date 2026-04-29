# methods/datasets/uavid2020_triplet_json_dataset.py
from pathlib import Path
from typing import List

import numpy as np

from .tri_triplet_base import BaseTripletDataset


class UAVid2020TripletJsonDataset(BaseTripletDataset):
    """
    UAVid2020 三元组数据集：同样采用默认内参 + PoseNet，相对位姿不再依赖 JSON。
    """
    #uavid2020_China
    #self.K = np.array([[0.6399, 0, 0.5, 0],
    #                   [0, 1.1376, 0.5, 0],
    #                   [0, 0, 1, 0],
    #                   [0, 0, 0, 1]], dtype=np.float32)
    #uavid2020_germany
    #self.K = np.array([[0.6314, 0, 0.5, 0],
    #                   [0, 1.1227, 0.5, 0],
    #                   [0, 0, 1, 0],
    #                   [0, 0, 0, 1]], dtype=np.float32)
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
    def _choose_default_k(cls, k_region: str, data_path: str, triplet_root: str):
        region = (k_region or "auto").strip().lower()
        if region in {"china", "cn"}:
            return cls._K_CHINA_4x4, "china"
        if region in {"germany", "de"}:
            return cls._K_GERMANY_4x4, "germany"

        if region == "auto":
            hay = " ".join([data_path or "", triplet_root or ""]).lower()
            if "china" in hay:
                return cls._K_CHINA_4x4, "china"
            if "germany" in hay:
                return cls._K_GERMANY_4x4, "germany"

        return cls._DEFAULT_K_4x4, "default"

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
                 k_region: str = "auto",
                 use_external_mask: bool = False,
                 external_mask_dir: str = "mask",
                 external_mask_ext: str = ".png",
                 external_mask_thresh: float = 0.5):

        default_k_4x4, k_source = self._choose_default_k(k_region, data_path, triplet_root)
        if k_source == "default" and (k_region or "auto").strip().lower() == "auto":
            print(f"[{self.__class__.__name__}] 未能从路径推断内参区域，使用默认 K")
        else:
            print(f"[{self.__class__.__name__}] 使用 {k_source} 内参")
        print(f"[{self.__class__.__name__}] K={default_k_4x4.tolist()}")

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
            default_K_4x4=default_k_4x4,
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

    def _infer_depth_path_from_center(self, center_path: str):
        if not center_path:
            return None
        if center_path in self._depth_path_cache:
            return self._depth_path_cache[center_path]

        center_p = Path(center_path)
        seq_dir = center_p.parent.parent if center_p.parent is not None else None
        found_path = None
        if seq_dir and seq_dir.is_dir():
            stem = center_p.stem
            depth_dir_candidates = [
                seq_dir / "depth",
                seq_dir / "Depth",
            ]
            for depth_dir in depth_dir_candidates:
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

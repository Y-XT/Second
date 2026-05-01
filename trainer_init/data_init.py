import functools
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from methods.datasets import UAVTripletJsonDataset, UAVid2020TripletJsonDataset
from utils import readlines


TRIPLET_POSE_METHODS = {
    "MonoViT_VGGT_RFlow_TInj",
}


def _seed_worker(worker_id, base_seed):
    worker_seed = (base_seed + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def init_dataloaders(self):
    """Initialize the triplet datasets used by the paper experiments."""
    if self.opt.dataset not in {"UAVula_TriDataset", "UAVid_TriDataset"}:
        raise ValueError(
            "Paper data_init only supports UAVula_TriDataset and UAVid_TriDataset, "
            f"got {self.opt.dataset!r}"
        )

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    split_pattern = os.path.join(project_root, "methods/splits", self.opt.split, "{}_files.txt")
    train_lines = readlines(split_pattern.format("train"))
    val_lines = readlines(split_pattern.format("val"))
    train_pairs = parse_split_pairs(train_lines)
    val_pairs = parse_split_pairs(val_lines)

    img_ext = ".png" if self.opt.png else ".jpg"
    train_allow_flip = bool(getattr(self.opt, "enable_flip", False))
    use_triplet_pose = str(getattr(self.opt, "methods", "")) in TRIPLET_POSE_METHODS
    triplet_manifest_glob = "triplets.jsonl"

    common_kwargs = dict(
        triplet_root=self.opt.triplet_root,
        height=self.opt.height,
        width=self.opt.width,
        frame_idxs=self.opt.frame_ids,
        num_scales=4,
        img_ext=img_ext,
        use_triplet_pose=use_triplet_pose,
        triplet_manifest_glob=triplet_manifest_glob,
    )

    if self.opt.dataset == "UAVula_TriDataset":
        train_dataset = UAVTripletJsonDataset(
            data_path=os.path.join(self.opt.data_path, "Train"),
            is_train=True,
            allow_flip=train_allow_flip,
            **common_kwargs,
        )
        val_dataset = UAVTripletJsonDataset(
            data_path=os.path.join(self.opt.data_path, "Validation"),
            is_train=False,
            allow_flip=False,
            **common_kwargs,
        )
        dataset_name = "UAVula_TriDataset"
    else:
        k_region = _infer_uavid_k_region(self.opt)
        train_dataset = UAVid2020TripletJsonDataset(
            data_path=os.path.join(self.opt.data_path, "Train"),
            is_train=True,
            allow_flip=train_allow_flip,
            vggt_target_width=getattr(self.opt, "vggt_target_width", 518),
            k_region=k_region,
            **common_kwargs,
        )
        val_dataset = UAVid2020TripletJsonDataset(
            data_path=os.path.join(self.opt.data_path, "Validation"),
            is_train=False,
            allow_flip=False,
            vggt_target_width=getattr(self.opt, "vggt_target_width", 518),
            k_region=k_region,
            **common_kwargs,
        )
        dataset_name = "UAVid_TriDataset"

    train_dataset.samples = [
        s for s in train_dataset.samples if (s["seq"], s.get("center_idx")) in train_pairs
    ]
    val_dataset.samples = [
        s for s in val_dataset.samples if (s["seq"], s.get("center_idx")) in val_pairs
    ]

    if len(train_dataset) == 0:
        print(f"[warning] {dataset_name} train split is empty after filtering")
    if len(val_dataset) == 0:
        print(f"[warning] {dataset_name} val split is empty after filtering")

    self.dataset = train_dataset.__class__
    self.normalize = {"enabled": False, "mode": None}
    self.num_total_steps = max(1, len(train_dataset) // self.opt.batch_size) * self.opt.num_epochs

    seed = getattr(self.opt, "seed", None)
    generator = None
    worker_init_fn = None
    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None
    if seed is not None and seed >= 0:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = functools.partial(_seed_worker, base_seed=seed)

    self.train_loader = DataLoader(
        train_dataset,
        self.opt.batch_size,
        shuffle=True,
        num_workers=self.opt.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    self.val_loader = DataLoader(
        val_dataset,
        self.opt.batch_size,
        shuffle=True,
        num_workers=self.opt.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    self.val_iter = iter(self.val_loader)


def _infer_uavid_k_region(opt) -> str:
    split = str(getattr(opt, "split", "")).lower()
    if "china" in split:
        return "china"
    if "germany" in split:
        return "germany"

    data_path = str(getattr(opt, "data_path", ""))
    triplet_root = str(getattr(opt, "triplet_root", ""))
    haystack = f"{data_path} {triplet_root}".lower()
    if "china" in haystack:
        return "china"
    if "germany" in haystack:
        return "germany"
    return "auto"


def _normalize_seq(raw: str) -> str:
    raw = raw.replace("\\", "/").strip()
    if raw.startswith("Train/"):
        raw = raw[len("Train/"):]
    if raw.startswith("Validation/"):
        raw = raw[len("Validation/"):]
    if raw.startswith("./"):
        raw = raw[2:]
    return raw


def parse_split_pairs(lines):
    pair_whitelist = set()
    for line in lines:
        if not line:
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        line = line.replace("\\", "/")
        try:
            seq_part, idx_str = line.rsplit(maxsplit=1)
            pair_whitelist.add((_normalize_seq(seq_part), int(idx_str)))
        except ValueError:
            continue
    return pair_whitelist

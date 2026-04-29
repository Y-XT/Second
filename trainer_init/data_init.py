# -*- coding: utf-8 -*-
"""
带有详细中文注释的 `init_dataloaders` 模块。

功能概览
--------
1) 根据 `self.opt.dataset` 选择并构造对应的数据集对象（包含 UAVid/UAVula 系列与 KITTI）。
2) 读取 train/val 划分文件（`methods/splits/<split>/{train,val}_files.txt`）。
3) 自动注入“是否归一化及其均值/方差”的配置到数据集构造函数中：
   - 若检测到骨干/权重包含 "dino"（如 DINOv2/DINOv3），默认开启 **ImageNet** 统计归一化；
   - 若权重名包含 "sat/satellite"，切换为 **卫星** 统计归一化；
   - 也支持从命令行/配置显式覆盖（`--normalize --norm_mode custom --norm_mean ... --norm_std ...`）。
4) 统一构建 `DataLoader`（训练集 shuffle=True 且 drop_last=True；验证集 shuffle=False 且 drop_last=False）。
5) 提供 `_normalize_seq` 与 `parse_split_pairs` 以支持按 (seq, center_idx) 过滤 triplet 数据集。

使用前提
--------
- `self.opt` 对象应包含：`dataset`、`data_path`、`split`、`height`、`width`、`frame_ids`、`batch_size`、`num_workers`、`num_epochs`、`png` 等字段；
- 若使用 triplet 数据集，还需 `triplet_root`；
- 数据集类（如 `UAVid2020_Dataset`、`UAVula_Dataset`、`UAVTripletJsonDataset`、`UAVid2020TripletJsonDataset`）需支持可选的构造参数：
  `normalize: bool`、`norm_mode: str`、`norm_mean: tuple`、`norm_std: tuple`，以便接收本模块自动注入的归一化配置。
"""

from utils import *  # 提供 readlines 等常用工具（项目内自定义）
from methods.datasets import (
    UAVid2020_Dataset,
    UAVula_Dataset,
    UAVTripletJsonDataset,
    UAVid2020TripletJsonDataset,
    wilduav_dataset
)
from methods import datasets  # 包含 KITTI 等其它官方数据集封装
from torch.utils.data import DataLoader
import functools
import os
import random
import numpy as np
import torch


def _seed_worker(worker_id, base_seed):
    """Ensure each DataLoader worker has a deterministic RNG state."""
    worker_seed = (base_seed + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def init_dataloaders(self):
    """根据配置初始化 train/val 数据集与 DataLoader。

    关键步骤：
    1) 选择数据集类；
    2) 解析 split 文件；
    3) 根据骨干/权重推断归一化配置，并通过 **kwargs 统一注入各数据集构造；
    4) 构建 DataLoader，并设置 `self.num_total_steps` 供进度/调度使用。
    """

    # 1) 根据标识字符串选择具体数据集类（字符串来自命令行/配置）
    datasets_dict = {
        "kitti": datasets.KITTIRAWDataset,
        "kitti_odom": datasets.KITTIOdomDataset,
        "UAVid2020": UAVid2020_Dataset,
        "UAVula_Dataset": UAVula_Dataset,
        "UAVula_TriDataset": UAVTripletJsonDataset,
        "UAVid_TriDataset": UAVid2020TripletJsonDataset,
        "WildUAV": wilduav_dataset.WildUAVDataset,
    }
    self.dataset = datasets_dict[self.opt.dataset]
    method_name = str(getattr(self.opt, "methods", ""))
    external_pose_methods = {
        "MD2_VGGT_NoPose", "MD2_VGGT_NoPose_UniformT", "MD2_VGGT_NoPose_SAlign",
        "MD2_VGGT_NoPose_TScale",
    }
    pose_prior_methods = external_pose_methods | {
        "GasMono",
        "MD2_VGGT_ResPose_RT", "MD2_VGGT_ResPose_RT_Reg", "MD2_VGGT_ResPose_RT_RMul",
        "MD2_VGGT_ResPose_T", "MD2_VGGT_ResPose_T_Reg",
        "MD2_VGGT_Gated", "MD2_VGGT_Teacher", "MD2_VGGT_Teacher_Distill",
        "MD2_VGGT_Teacher_Photo",
        "MD2_VGGT_ResPose_Decay", "MD2_VGGT_PoseToRes",
        "MD2_VGGT_TDir_PoseMag", "MD2_VGGT_TPrior_Alpha",
        "MD2_VGGT_TPrior_AlignRes",
        "MD2_VGGT_RPrior_TPose",
        "MD2_VGGT_PoseGT", "MD2_VGGT_PoseGT_DepthCycleViz",
        "MD2_VGGT_PoseGT_DepthSensitivityViz",
        "MD2_VGGT_PoseGT_DepthSensViz",
        "MD2_VGGT_PoseGT_DepthSensWeight",
        "MD2_VGGT_PoseGT_BadScoreWeight",
        "MD2_VGGT_PoseGT_BadScoreLocalWeight",
        "MD2_VGGT_PoseGT_HRMask", "MD2_VGGT_PoseGT_Mask",
        "MD2_VGGT_PoseGT_DeRotHardMask",
        "MD2_VGGT_PoseGT_DeRotSigmoidWeight",
        "MonoViT_VGGT_RDistill",
        "MonoViT_VGGT_RMaskSwitch",
        "MonoViT_VGGT_PreWarp",
        "MonoViT_VGGT_RPrior_ResR_TPose",
        "MonoViT_VGGT_RFlow_Pose",
        "MonoViT_VGGT_RFlow_ResR_TPose",
        "MonoViT_VGGT_RFlow_ResR_TPose_SingleHead",
        "MonoViT_VGGT_RFlow_TInj",
        "MonoViT_PoseGT",
        "MonoViT_PoseGT_Mask",
        "MonoViT_PoseGT_HRMask",
        "MonoViT_VGGT_PoseGT_BadScoreWeight",
    }
    uniform_triplet_methods = {"MD2_VGGT_NoPose_UniformT"}
    use_triplet_pose = method_name in pose_prior_methods
    triplet_manifest_glob = "triplets_uniform_t.jsonl" if method_name in uniform_triplet_methods else "triplets.jsonl"

    # 2) 读取 train/val 划分文件（每行通常是 "<相对路径> <索引>" 或仅序列路径）
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # 注意：你的原始代码是 "{}_files.txt"，如果项目里实际文件名就是该形式，请改回去。
    fpath = os.path.join(PROJECT_ROOT, "methods/splits", self.opt.split, "{}_files.txt")
    train_lines = readlines(fpath.format("train"))  # 逐行读取 train 划分
    val_lines = readlines(fpath.format("val"))      # 逐行读取 val 划分

    # 按命令行开关决定图像后缀（常见于 KITTI/UAV 类数据混合）
    img_ext = ".png" if self.opt.png else ".jpg"
    train_allow_flip = bool(getattr(self.opt, "enable_flip", False))

    # 3) 根据骨干/权重推断归一化配置（是否 normalize、使用何种 mean/std）
    #    返回的 dict 将被 `**dataset_kwargs_common` 注入到数据集构造函数中。
    dataset_kwargs_common = _infer_norm_cfg(self.opt)
    # 例如：
    #   {'normalize': True, 'norm_mode': 'imagenet'}
    # 或：
    #   {'normalize': True, 'norm_mode': 'custom',
    #    'norm_mean': (0.430,0.411,0.296), 'norm_std': (0.213,0.156,0.143)}
    _log_norm_decision(self.opt, dataset_kwargs_common)

    # 4) 构造不同类型的数据集
    mask_kwargs_train = dict(
        use_external_mask=bool(getattr(self.opt, "use_external_mask", False)),
        external_mask_dir=getattr(self.opt, "external_mask_dir", "mask"),
        external_mask_ext=getattr(self.opt, "external_mask_ext", ".png"),
        external_mask_thresh=float(getattr(self.opt, "external_mask_thresh", 0.5)),
    )
    mask_kwargs_val = dict(mask_kwargs_train)
    mask_kwargs_val["use_external_mask"] = False
    if self.opt.dataset == "UAVula_TriDataset":
        # 4.1) Triplet（UAVula）数据集：支持在 split 中用 (seq, center_idx) 精确过滤
        train_pairs = parse_split_pairs(train_lines)
        val_pairs = parse_split_pairs(val_lines)

        # 训练集：is_train=True 开启颜色增广；allow_flip 可按需启用
        train_dataset = UAVTripletJsonDataset(
            data_path=os.path.join(self.opt.data_path, "Train"),  # 指向图像根
            triplet_root=self.opt.triplet_root,                     # 指向 triplets.jsonl 根目录
            height=self.opt.height,
            width=self.opt.width,
            frame_idxs=self.opt.frame_ids,  # 一般为 [0, -1, 1]
            num_scales=4,
            is_train=True,
            img_ext=img_ext,
            allow_flip=train_allow_flip,
            use_triplet_pose=use_triplet_pose,
            triplet_manifest_glob=triplet_manifest_glob,
            **mask_kwargs_train,
            **dataset_kwargs_common,         # ← 自动注入归一化开关与统计
        )
        # 验证集：is_train=False 关闭增广
        val_dataset = UAVTripletJsonDataset(
            data_path=os.path.join(self.opt.data_path, "Validation"),
            triplet_root=self.opt.triplet_root,
            height=self.opt.height,
            width=self.opt.width,
            frame_idxs=self.opt.frame_ids,
            num_scales=4,
            is_train=False,
            img_ext=img_ext,
            allow_flip=False,
            use_triplet_pose=use_triplet_pose,
            triplet_manifest_glob=triplet_manifest_glob,
            **mask_kwargs_val,
            **dataset_kwargs_common,         # ← 自动注入归一化
        )
        # 过滤：仅保留 split 文本中列出的 (seq, center_idx)
        train_dataset.samples = [
            s for s in train_dataset.samples if (s["seq"], s.get("center_idx")) in train_pairs
        ]
        val_dataset.samples = [
            s for s in val_dataset.samples if (s["seq"], s.get("center_idx")) in val_pairs
        ]
        if len(train_dataset) == 0:
            print("[警告] UAVula_TriDataset训练集过滤后为 0，请检查 split 与 triplets.jsonl 的 seq/idx 对齐")
        if len(val_dataset) == 0:
            print("[警告] UAVula_TriDataset验证集过滤后为 0，请检查 split 与 triplets.jsonl 的 seq/idx 对齐")

        if mask_kwargs_train["use_external_mask"]:
            print(
                "[ExternalMask] train=ON "
                f"dir={mask_kwargs_train['external_mask_dir']} "
                f"ext={mask_kwargs_train['external_mask_ext']} "
                f"thresh={mask_kwargs_train['external_mask_thresh']}; val=OFF"
            )
        else:
            print("[ExternalMask] train=OFF; val=OFF")

        # 估算总步数：每 epoch 内的 step 数 × epoch 数（取整除并乘）
        self.num_total_steps = max(1, len(train_dataset) // self.opt.batch_size) * self.opt.num_epochs

    elif self.opt.dataset == "UAVid_TriDataset":
        # 4.2) Triplet（UAVid2020）数据集：逻辑与上面类似
        train_pairs = parse_split_pairs(train_lines)
        val_pairs = parse_split_pairs(val_lines)
        k_region = _infer_uavid_k_region(self.opt)

        train_dataset = UAVid2020TripletJsonDataset(
            data_path=os.path.join(self.opt.data_path, "Train"),
            triplet_root=self.opt.triplet_root,
            height=self.opt.height,
            width=self.opt.width,
            frame_idxs=self.opt.frame_ids,  # 包含 {0, -1, 1}
            num_scales=4,
            is_train=True,
            img_ext=img_ext,
            allow_flip=train_allow_flip,
            vggt_target_width=getattr(self.opt, "vggt_target_width", 518),  # VGGT 默认 518
            use_triplet_pose=use_triplet_pose,
            triplet_manifest_glob=triplet_manifest_glob,
            k_region=k_region,
            **mask_kwargs_train,
            **dataset_kwargs_common,                                            # ← 自动注入归一化
        )
        val_dataset = UAVid2020TripletJsonDataset(
            data_path=os.path.join(self.opt.data_path, "Validation"),
            triplet_root=self.opt.triplet_root,
            height=self.opt.height,
            width=self.opt.width,
            frame_idxs=self.opt.frame_ids,
            num_scales=4,
            is_train=False,
            img_ext=img_ext,
            allow_flip=False,
            vggt_target_width=getattr(self.opt, "vggt_target_width", 518),
            use_triplet_pose=use_triplet_pose,
            triplet_manifest_glob=triplet_manifest_glob,
            k_region=k_region,
            **mask_kwargs_val,
            **dataset_kwargs_common,                                            # ← 自动注入归一化
        )
        # 过滤样本（同上）
        train_dataset.samples = [
            s for s in train_dataset.samples if (s["seq"], s.get("center_idx")) in train_pairs
        ]
        val_dataset.samples = [
            s for s in val_dataset.samples if (s["seq"], s.get("center_idx")) in val_pairs
        ]
        if len(train_dataset) == 0:
            print("[警告] UAVid2020 训练集过滤后为 0，请检查 split 与 triplets.jsonl 的 seq/idx 对齐")
        if len(val_dataset) == 0:
            print("[警告] UAVid2020 验证集过滤后为 0，请检查 split 与 triplets.jsonl 的 seq/idx 对齐")

        if mask_kwargs_train["use_external_mask"]:
            print(
                "[ExternalMask] train=ON "
                f"dir={mask_kwargs_train['external_mask_dir']} "
                f"ext={mask_kwargs_train['external_mask_ext']} "
                f"thresh={mask_kwargs_train['external_mask_thresh']}; val=OFF"
            )
        else:
            print("[ExternalMask] train=OFF; val=OFF")

        self.num_total_steps = max(1, len(train_dataset) // self.opt.batch_size) * self.opt.num_epochs

    else:
        # 4.3) 其它非 triplet 数据集（如 KITTI/UAVid2020 帧索引样式），仍按行列表直接构造
        train_filenames = train_lines
        val_filenames = val_lines

        # 估算总步数（与上相同）
        self.num_total_steps = max(1, len(train_filenames) // self.opt.batch_size) * self.opt.num_epochs

        dataset_kwargs = dict(dataset_kwargs_common)
        if self.opt.dataset == "UAVid2020":
            dataset_kwargs["k_region"] = _infer_uavid_k_region(self.opt)

        # 将 dataset_kwargs_common 注入，确保若检测到 DINO 则自动归一化
        train_dataset = self.dataset(
            self.opt.data_path,
            train_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,                      # num_scales，通常与 Monodepth2 保持 4 层
            True,                   # is_train=True
            img_ext,
            allow_flip=train_allow_flip,
            **dataset_kwargs,
        )
        val_dataset = self.dataset(
            self.opt.data_path,
            val_filenames,
            self.opt.height,
            self.opt.width,
            self.opt.frame_ids,
            4,                      # num_scales
            False,                  # is_train=False
            img_ext,
            allow_flip=False,
            **dataset_kwargs,
        )

    # === 仅记录 normalize 配置到 self ===
    use_norm = bool(getattr(train_dataset, "normalize", False))
    norm_mode = getattr(train_dataset, "norm_mode", None)
    self.normalize = {
        "enabled": use_norm,
        "mode": norm_mode,
    }

    # 4.5) 设定 DataLoader 的 RNG（保证 shuffle 与数据增强可复现）
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

    # 5) 构建 DataLoader
    # 训练：shuffle=True 提升随机性；drop_last=True 保证每个 batch 尺寸一致，稳定 BN/同步策略
    self.train_loader = DataLoader(
        train_dataset,
        self.opt.batch_size,
        True,  # shuffle
        num_workers=self.opt.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    # 验证：建议 shuffle=False 以便复现/稳定；drop_last=False 以保留不足一个 batch 的样本
    self.val_loader = DataLoader(
        val_dataset,
        self.opt.batch_size,
        True,  # shuffle=False
        num_workers=self.opt.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    # 可在训练循环外预先构造一个 val 迭代器（按需使用）
    self.val_iter = iter(self.val_loader)


# --- 新增：根据模型/骨干名自动选择归一化配置 ---

def _infer_norm_cfg(opt):
    """基于 `opt` 推断数据归一化策略。

    返回值
    ------
    - `dict`：包含将要被注入数据集构造函数的关键字参数；
              若无需归一化则返回空 dict。

    判定规则
    --------
    1) **显式覆盖优先**：如果命令行/配置里写了 `normalize=True`，则无条件开启：
       - `norm_mode` 可选 `imagenet` / `sat` / `custom`：
         - `imagenet`：采用常规 ImageNet 统计（mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)）。
         - `sat`：采用卫星统计（mean=(0.430,0.411,0.296), std=(0.213,0.156,0.143)）。
         - `custom`：从 `opt.norm_mean` 与 `opt.norm_std` 读取用户自定义统计。
    2) **自动识别**：若 `opt.encoder`/`opt.backbone`/`opt.model`/`opt.teacher`/`opt.ckpt` 等字符串
       包含 "dino" 关键词（大小写不敏感），则认为使用 DINO/DINOv3 生态，自动开启归一化：
       - 若权重名包含 "sat"/"satellite" → 采用卫星统计；否则采用 ImageNet 统计。
    3) 其它情况：返回空 dict（即不在数据侧做归一化；保留 [0,1] 图像给网络或损失侧处理）。
    """

    # 将可能指示骨干/权重的字段收集起来并拼接为字符串（便于统一小写匹配）
    names = []
    for key in [
        "encoder",
        "backbone",
        "methods",
        "teacher",
        "ckpt",
        "encoder_ckpt",
        "backbone_ckpt",
    ]:
        if hasattr(opt, key) and getattr(opt, key):
            names.append(str(getattr(opt, key)))
    joined = " ".join(names).lower()

    # 1) 显式开关：若用户已指定 normalize=True，则尊重并读取模式/统计
    if getattr(opt, "normalize", None) is True:
        mode = getattr(opt, "norm_mode", "imagenet")
        if mode == "custom":
            mean = tuple(getattr(opt, "norm_mean"))
            std = tuple(getattr(opt, "norm_std"))
            return dict(normalize=True, norm_mode="custom", norm_mean=mean, norm_std=std)
        elif mode == "sat":
            return dict(
                normalize=True,
                norm_mode="custom",
                norm_mean=(0.430, 0.411, 0.296),
                norm_std=(0.213, 0.156, 0.143),
            )
        else:  # imagenet
            return dict(normalize=True, norm_mode="imagenet")

    # 2) 自动识别：遇到 dino/dinov2/dinov3 等关键字则默认开启归一化
    if "dino" in joined:
        if ("sat" in joined) or ("satellite" in joined):
            return dict(
                normalize=True,
                norm_mode="custom",
                norm_mean=(0.430, 0.411, 0.296),
                norm_std=(0.213, 0.156, 0.143),
            )
        else:
            return dict(normalize=True, norm_mode="imagenet")

    # 3) 非 DINO：返回空 dict → 数据集构造时不会注入归一化参数
    return {}

def _log_norm_decision(opt, cfg: dict):
    """打印最终是否开启归一化及其模式/统计与触发原因。"""
    # 判断触发原因：显式开关 > 自动识别 > 关闭
    names = []
    for k in ["encoder", "backbone", "methods", "teacher", "ckpt", "encoder_ckpt", "backbone_ckpt"]:
        if hasattr(opt, k) and getattr(opt, k):
            names.append(str(getattr(opt, k)))
    joined = " ".join(names).lower()

    if not cfg:
        print("[Normalize] OFF | 保持原始[0,1]张量输入（数据侧不做归一化）")
        return

    reason = "显式开启 (--normalize)" if getattr(opt, "normalize", None) is True \
        else ("自动启用（检测到 *dino* 关键词）" if "dino" in joined else "（未知来源）")

    # 识别模式
    sat_mean = (0.430, 0.411, 0.296)
    sat_std  = (0.213, 0.156, 0.143)

    mode = cfg.get("norm_mode", "custom")
    if mode == "imagenet":
        print(f"[Normalize] ON  | mode=ImageNet | reason={reason}")
    else:
        mean = cfg.get("norm_mean")
        std  = cfg.get("norm_std")
        if mean == sat_mean and std == sat_std:
            print(f"[Normalize] ON  | mode=Satellite | mean={mean} | std={std} | reason={reason}")
        else:
            print(f"[Normalize] ON  | mode=Custom    | mean={mean} | std={std} | reason={reason}")


def _infer_uavid_k_region(opt) -> str:
    """根据 split/data_path/triplet_root 推断 UAVid 区域（china/germany/auto）。"""
    split = str(getattr(opt, "split", "")).lower()
    if "china" in split:
        return "china"
    if "germany" in split:
        return "germany"

    data_path = str(getattr(opt, "data_path", ""))
    triplet_root = str(getattr(opt, "triplet_root", ""))
    hay = f"{data_path} {triplet_root}".lower()
    if "china" in hay:
        return "china"
    if "germany" in hay:
        return "germany"
    return "auto"


def _normalize_seq(raw: str) -> str:
    """规范化序列路径字符串，使不同写法等价（便于匹配过滤）。

    处理规则：
    - 统一斜杠方向为 `/`；
    - 去掉开头的 `Train/` 或 `Validation/` 前缀；
    - 去掉开头的 `./`；

    参数
    ----
    raw : str
        原始序列路径（可能包含平台特定的分隔符或前缀）。

    返回
    ----
    str : 规范化后的相对路径（与 triplets.jsonl 中的 `seq` 对齐）。
    """
    raw = raw.replace("\\", "/").strip()
    if raw.startswith("Train/"):
        raw = raw[len("Train/") :]
    if raw.startswith("Validation/"):
        raw = raw[len("Validation/") :]
    if raw.startswith("./"):
        raw = raw[2:]
    return raw


def parse_split_pairs(lines):
    """解析 split 行，生成 (seq, center_idx) 的白名单集合。

    输入行格式举例：
    - `Train/seq03 12`：表示序列 `seq03` 且仅保留中心帧索引 12；
    - `Validation/area_1/xxx 0`：同理；
    - 也容忍纯序列行（不带索引），此时 **当前实现会跳过**；如需把纯序列视为全量保留，
      可自行扩展逻辑（例如为该 seq 记录一个特殊标记，后续过滤时放行）。

    参数
    ----
    lines : List[str]
        从 `{train,val}_files.txt` 读取的行列表。

    返回
    ----
    Set[Tuple[str,int]]
        白名单集合，用于与样本 (seq, center_idx) 对比过滤。
    """
    pair_whitelist = set()
    for ln in lines:
        if not ln:
            continue
        ln = ln.split("#", 1)[0].strip()  # 去掉行内注释，并裁剪空白
        if not ln:
            continue
        ln = ln.replace("\\", "/")
        try:
            # 从末尾分割，取出整数索引；若失败将落到 except 并跳过该行
            seq_part, idx_str = ln.rsplit(maxsplit=1)
            seq = _normalize_seq(seq_part)
            idx = int(idx_str)
            pair_whitelist.add((seq, idx))
        except ValueError:
            # 行尾没有合法的整数索引：当前实现选择跳过
            # 如需“只给了 seq 就全保留”的语义，可在此处记录 seq 到一个 set，
            # 然后在过滤阶段放行该 seq 下的所有 center_idx。
            continue
    return pair_whitelist

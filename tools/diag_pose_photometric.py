#!/usr/bin/env python3
"""
快速对比“同一深度 + 不同位姿”下的 photometric 误差：
  - PoseNet 预测位姿（baseline monodepth2 路径）
  - PoseGT 估计位姿（使用 triplet 外部位姿 + trans 网络）
  - 外部 VGGT 位姿（直接用 triplet 外部位姿）

用法示例：
  python tools/diag_pose_photometric.py \\
    --methods MD2_VGGT_PoseGT \\
    --dataset UAVula_TriDataset \\
    --split UAVula \\
    --data_path /path/to/UAVula \\
    --triplet_root /path/to/uav_triplets \\
    --load_weights_pose_folder /path/to/baseline_ckpt \\
    --load_weights_posegt_folder /path/to/posegt_ckpt \\
    --batch_size 4 --height 320 --width 1024
"""
import argparse
import json
import numbers
import os
import sys
from pathlib import Path
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 关闭 W&B 相关行为，避免离线/无网环境报错
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

# 确保可以找到项目根目录的模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from options import MonodepthOptions
from trainer import Trainer
from trainer_init.data_init import _infer_norm_cfg
from trainer_init.geometry_init import init_geometry
from trainer_init.loss_init import init_losses
from trainer_init.model_init import init_models
from trainer_init.scale_alignment import DepthScaleAligner
from methods.datasets import UAVTripletJsonDataset
from layers import SSIM, disp_to_depth, transformation_from_gtmsrtpose, transformation_from_parameters


def load_model_weights_only(runner: "Trainer", folder: str, model_names=None):
    """
    仅加载模型权重，不触及 optimizer，避免 Eval 模式缺少 optimizer 报错。
    """
    if not folder:
        raise ValueError("load_model_weights_only: 需要提供权重目录")
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"找不到权重目录：{folder}")

    if model_names is None:
        model_names = list(runner.models.keys())

    for n in model_names:
        if n not in runner.models:
            continue
        path = os.path.join(folder, f"{n}.pth")
        if not os.path.isfile(path):
            print(f"[warn] 未找到权重文件 {path}，跳过加载")
            continue
        model_dict = runner.models[n].state_dict()
        pretrained_dict = torch.load(path, map_location="cpu")
        # 移除保存时附带的 height/width
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        runner.models[n].load_state_dict(model_dict)



def _build_loader(opt, split: str) -> DataLoader:
    """构建指定 split(train/val) 的 DataLoader，并强制 use_triplet_pose=True 以获得外部位姿。"""
    assert split in {"train", "val"}
    dataset_kwargs = _infer_norm_cfg(opt)
    manifest_glob = "triplets_uniform_t.jsonl" if "UniformT" in str(opt.methods) else "triplets.jsonl"
    root = "Train" if split == "train" else "Validation"
    dataset = UAVTripletJsonDataset(
        data_path=os.path.join(opt.data_path, root),
        triplet_root=opt.triplet_root,
        height=opt.height,
        width=opt.width,
        frame_idxs=opt.frame_ids,
        num_scales=len(opt.scales),
        is_train=False,
        img_ext=".png" if opt.png else ".jpg",
        allow_flip=False,
        vggt_target_width=getattr(opt, "vggt_target_width", 518),
        use_triplet_pose=True,  # 关键：确保 batch 含 external_cam_T_cam
        triplet_manifest_glob=manifest_glob,
        **dataset_kwargs,
    )
    _filter_samples_for_diag(
        dataset,
        data_root=os.path.join(opt.data_path, root),
        require_depth=True,
        require_pose=True,
        frame_ids=opt.frame_ids,
    )

    # 为了与 BackprojectDepth 的固定 batch_size 对齐，默认 drop_last=True
    loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


def _filter_samples_for_diag(
    dataset: UAVTripletJsonDataset,
    data_root: str,
    require_depth: bool,
    require_pose: bool,
    frame_ids: List[Any],
) -> None:
    """
    以 data_path 为基准过滤 triplet 样本，确保 center/prev/next 图像都存在。
    可选地要求 depth_gt 存在，以及 triplet 先验位姿完整。
    不使用 Train/Validation 互换兜底，避免跨 split 的不一致。
    """
    if not dataset.samples:
        return
    keep = []
    missing_img = 0
    missing_depth = 0
    missing_pose = 0
    examples = []
    for rec in dataset.samples:
        ok = True
        for key in ("center", "prev", "next"):
            path = rec.get(key)
            if path and not os.path.isabs(path):
                path = os.path.join(data_root, path)
            if not path or not os.path.exists(path):
                ok = False
                missing_img += 1
                if len(examples) < 3:
                    examples.append({"seq": rec.get("seq"), "center_idx": rec.get("center_idx"), "path": path})
                break
        if ok and require_depth:
            depth_path = rec.get("depth_gt_path")
            if depth_path and not os.path.isabs(depth_path):
                depth_path = os.path.join(data_root, depth_path)
            if not depth_path or not os.path.exists(depth_path):
                ok = False
                missing_depth += 1
        if ok and require_pose:
            pose = rec.get("_external_pose")
            if not pose:
                ok = False
                missing_pose += 1
            else:
                for fid in frame_ids[1:]:
                    if fid == "s":
                        continue
                    if isinstance(fid, numbers.Integral) and fid < 0:
                        if pose.get("prev_to_center") is None:
                            ok = False
                            missing_pose += 1
                            break
                    elif isinstance(fid, numbers.Integral) and fid > 0:
                        if pose.get("next_to_center") is None:
                            ok = False
                            missing_pose += 1
                            break
        if ok:
            keep.append(rec)
    if missing_img:
        print(f"[diag] filtered {missing_img} samples without images under {data_root}")
        for ex in examples:
            print(f"[diag]   missing sample: seq={ex['seq']} idx={ex['center_idx']} path={ex['path']}")
    if require_depth and missing_depth:
        print(f"[diag] filtered {missing_depth} samples without depth_gt under {data_root}")
    if require_pose and missing_pose:
        print(f"[diag] filtered {missing_pose} samples without external pose under {data_root}")
    dataset.samples = keep


class EvalRunner(Trainer):
    """
    复用 Trainer 的前向/warp/损失逻辑，但精简初始化，避免训练与日志相关开销。
    """

    def __init__(self, opt):
        # --------- 轻量初始化（不调 logging/dataloader/optimizer）---------
        self.opt = opt
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        self.models: Dict[str, torch.nn.Module] = {}
        self.parameters_to_train: List[torch.nn.Parameter] = []
        self.training = False

        assert self.opt.height % 32 == 0
        assert self.opt.width % 32 == 0
        assert self.opt.frame_ids[0] == 0

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        self.use_pose_net = True

        # 只做模型/损失/几何模块的构建
        init_models(self)
        init_losses(self)
        init_geometry(self)

        align_mode = str(getattr(self.opt, "scale_align_mode", "off"))
        self.scale_aligner = DepthScaleAligner(self.opt) if align_mode != "off" else None
        self._scale_align_monitor = {"train": {"attempts": 0, "success": 0}, "val": {"attempts": 0, "success": 0}}
        self._scale_align_warned = {"train": False, "val": False}

        if not self.opt.no_ssim:
            self.ssim = SSIM().to(self.device)

        self.depth_metric_names = [
            "de/abs_rel",
            "de/sq_rel",
            "de/rms",
            "de/log_rms",
            "da/a1",
            "da/a2",
            "da/a3",
        ]
        self.collect_debug_metrics = False
        self.pose_t_history = deque(maxlen=1)
        self._metric_buffers: Dict[str, Dict[str, list]] = {}

        self.set_eval()


def move_batch_to_device(batch: Dict[Any, Any], device: torch.device) -> Dict[Any, Any]:
    out: Dict[Any, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def extract_meta(inputs: Dict[Any, Any]) -> Dict[str, Any]:
    seq = inputs.get("seq", "")
    if isinstance(seq, list):
        seq = seq[0]
    center_idx = inputs.get("center_idx", None)
    if torch.is_tensor(center_idx):
        center_idx = int(center_idx[0].item())
    return {"seq": seq, "center_idx": center_idx}


def _safe_name(s: str) -> str:
    """将字符串清洗为文件名友好的 token。"""
    s = str(s)
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("-")
    name = "".join(out).strip("-")
    return name or "default"


def _select_depthnet_input(runner: "EvalRunner", inputs: Dict[Any, Any]) -> torch.Tensor:
    """选择 DepthNet 的输入张量。优先 color_norm，否则用 color_aug。"""
    if ("color_norm", 0, 0) in inputs:
        return inputs[("color_norm", 0, 0)]
    return inputs[("color_aug", 0, 0)]


def _predict_depthnet_depth(runner: "EvalRunner", inputs: Dict[Any, Any]) -> torch.Tensor:
    """使用 encoder+depth 预测深度，返回 [B,1,H,W]。"""
    if "encoder" not in runner.models or "depth" not in runner.models:
        raise RuntimeError("DepthNet 未初始化，无法预测深度")
    enc_in = _select_depthnet_input(runner, inputs)
    features = runner.models["encoder"](enc_in)
    depth_out = runner.models["depth"](features)
    if isinstance(depth_out, dict):
        disp = depth_out.get(("disp", 0), None)
    else:
        disp = depth_out
    if disp is None or not torch.is_tensor(disp):
        raise RuntimeError("DepthNet 输出不包含 ('disp', 0)")
    disp = F.interpolate(disp, [runner.opt.height, runner.opt.width], mode="bilinear", align_corners=False)
    _, depth = disp_to_depth(disp, runner.opt.min_depth, runner.opt.max_depth)
    return depth


def _median_between_percentiles(values: torch.Tensor, p_low: float, p_high: float) -> Optional[torch.Tensor]:
    """返回位于 [p_low, p_high] 分位区间内的中值。"""
    if values.numel() == 0:
        return None
    lo = torch.quantile(values, p_low / 100.0)
    hi = torch.quantile(values, p_high / 100.0)
    mid = values[(values >= lo) & (values <= hi)]
    if mid.numel() == 0:
        return None
    return torch.median(mid)


def _compute_depth_scale_median(
    anchor: torch.Tensor,
    target: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
    p_low: float,
    p_high: float,
    min_valid_pixels: int,
    eps: float,
    scale_min: float,
    scale_max: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算每个样本的尺度（anchor / target），使用 5-95% 区间的中值对齐。
    返回 scale [B,1,1,1] 与 valid_mask [B]。
    """
    if anchor.dim() == 3:
        anchor = anchor.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if valid_mask is not None and valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)

    bsz = anchor.shape[0]
    scales = torch.ones(bsz, device=anchor.device, dtype=anchor.dtype)
    usable = torch.zeros(bsz, device=anchor.device, dtype=torch.bool)

    for b in range(bsz):
        a = anchor[b, 0]
        t = target[b, 0]
        mask = torch.isfinite(a) & torch.isfinite(t) & (a > 0) & (t > 0)
        if valid_mask is not None:
            mask = mask & (valid_mask[b, 0] > 0)
        if mask.sum().item() < min_valid_pixels:
            continue

        a_vals = a[mask]
        t_vals = t[mask]
        a_med = _median_between_percentiles(a_vals, p_low, p_high)
        t_med = _median_between_percentiles(t_vals, p_low, p_high)
        if a_med is None or t_med is None:
            continue
        if float(torch.abs(t_med).item()) <= eps:
            continue

        scale = a_med / (t_med + eps)
        scale = torch.clamp(scale, scale_min, scale_max)
        scales[b] = scale
        usable[b] = True

    return scales.view(bsz, 1, 1, 1), usable


def _predict_poses_with_irw(
    runner: "EvalRunner",
    inputs: Dict[Any, Any],
    irw_imgs: Dict[int, torch.Tensor],
) -> Dict[Any, Any]:
    """
    使用 pose 网络预测位姿。若提供 irw_imgs，则用其替换对应 frame 的 pose 输入。
    仅覆盖 standard PoseNet 分支（axisangle + translation）。
    """
    outputs: Dict[Any, Any] = {}
    pose_feats = {f_i: inputs[("color_aug", f_i, 0)] for f_i in runner.opt.frame_ids}
    for f_i in runner.opt.frame_ids[1:]:
        if f_i == "s":
            continue
        if f_i < 0:
            pose_img = irw_imgs.get(f_i, pose_feats[f_i])
            pose_inputs = [pose_img, pose_feats[0]]
        else:
            pose_img = irw_imgs.get(f_i, pose_feats[f_i])
            pose_inputs = [pose_feats[0], pose_img]

        pose_inputs = [runner.models["pose_encoder"](torch.cat(pose_inputs, 1))]
        pose_out = runner.models["pose"](pose_inputs)
        if not (isinstance(pose_out, (tuple, list)) and len(pose_out) == 2):
            raise RuntimeError("Pose decoder output must be (axisangle, translation)")
        axisangle, translation = pose_out
        outputs[("axisangle", 0, f_i)] = axisangle
        outputs[("translation", 0, f_i)] = translation
        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
        )
    return outputs


def _generate_posegt_irw(
    runner: "EvalRunner",
    inputs: Dict[Any, Any],
    depth: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """
    使用 trans_encoder/trans 生成 posegt 的 irw_img（预对齐图）。
    depth 应为 [B,1,H,W]，已对齐到 opt.height/opt.width。
    """
    irw_imgs: Dict[int, torch.Tensor] = {}
    for frame_id in runner.opt.frame_ids[1:]:
        if frame_id == "s":
            continue
        raw_img = inputs[("color_aug", frame_id, 0)]
        tgt_img = inputs[("color_aug", 0, 0)]
        if frame_id < 0:
            pose_inputs = [raw_img, tgt_img]
        else:
            pose_inputs = [tgt_img, raw_img]

        pose_inputs = [runner.models["trans_encoder"](torch.cat(pose_inputs, 1))]
        translation, scale = runner.models["trans"](pose_inputs)

        key = ("external_cam_T_cam", 0, frame_id)
        T_prior = inputs.get(key, None)
        if T_prior is None:
            raise KeyError(f"Missing external pose for posegt, frame id={frame_id}")

        T = transformation_from_gtmsrtpose(
            T_prior.to(translation.device).clone(),
            scale[:, 0],
            translation[:, 0],
            invert=(frame_id < 0),
        )
        cam_points = runner.backproject_depth[0](depth, inputs[("inv_K", 0)])
        pix_coords = runner.project_3d[0](cam_points, inputs[("K", 0)], T)
        source_img = inputs[("color", frame_id, 0)]
        irw_imgs[frame_id] = F.grid_sample(
            source_img, pix_coords, padding_mode="border", align_corners=True
        )
    return irw_imgs


def _warp_with_pose(
    runner: "EvalRunner",
    inputs: Dict[Any, Any],
    depth: torch.Tensor,
    cam_T: Dict[int, torch.Tensor],
    source_imgs: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[Any, Any]:
    """
    用给定 cam_T_cam 将源帧 warp 到目标帧，返回包含 ("color", fid, scale) 的输出。
    depth 应为 [B,1,H,W]，已对齐到 opt.height/opt.width。
    """
    outputs: Dict[Any, Any] = {}
    for fid in runner.opt.frame_ids[1:]:
        if fid == "s":
            continue
        T = cam_T[fid]
        cam_points = runner.backproject_depth[0](depth, inputs[("inv_K", 0)])
        pix_coords = runner.project_3d[0](cam_points, inputs[("K", 0)], T)
        source_img = None
        if source_imgs is not None:
            source_img = source_imgs.get(fid, None)
        if source_img is None:
            source_img = inputs[("color", fid, 0)]
        outputs[("color", fid, 0)] = F.grid_sample(
            source_img, pix_coords, padding_mode="border", align_corners=True
        )
    if len(runner.opt.scales) > 1:
        for s in runner.opt.scales:
            if s == 0:
                continue
            for fid in runner.opt.frame_ids[1:]:
                if fid == "s":
                    continue
                outputs[("color", fid, s)] = outputs[("color", fid, 0)]
    return outputs


def _compute_photometric_only(
    runner: "EvalRunner",
    inputs: Dict[Any, Any],
    outputs: Dict[Any, Any],
) -> torch.Tensor:
    """
    仅计算重投影 photometric（SSIM+L1），不包含平滑项/posegt/teacher/reg。
    与训练的 automasking/min 逻辑一致。
    """
    opt = runner.opt
    total_loss = 0.0
    for scale in opt.scales:
        reprojection_losses = []
        target = inputs[("color", 0, 0)]
        for frame_id in opt.frame_ids[1:]:
            if frame_id == "s":
                continue
            pred = outputs[("color", frame_id, scale)]
            reproj_loss = runner.loss.compute_reprojection_loss(pred, target)
            reprojection_losses.append(reproj_loss)

        reprojection_losses = torch.cat(reprojection_losses, 1)

        identity_loss_tensor = None
        if not opt.disable_automasking:
            identity_losses = []
            for frame_id in opt.frame_ids[1:]:
                if frame_id == "s":
                    continue
                pred = inputs[("color", frame_id, 0)]
                identity_losses.append(runner.loss.compute_reprojection_loss(pred, target))
            identity_losses = torch.cat(identity_losses, 1)
            if opt.avg_reprojection:
                identity_loss_tensor = identity_losses.mean(1, keepdim=True)
            else:
                identity_loss_tensor = identity_losses

        if opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not opt.disable_automasking and identity_loss_tensor is not None:
            identity_loss_tensor = identity_loss_tensor + torch.randn(
                identity_loss_tensor.shape, device=runner.device
            ) * 1e-5
            combined = torch.cat((identity_loss_tensor, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        to_optimise = combined if combined.shape[1] == 1 else torch.min(combined, dim=1)[0]
        total_loss += to_optimise.mean()

    total_loss /= len(opt.scales)
    return total_loss


def main():
    base = MonodepthOptions()
    parser: argparse.ArgumentParser = base.parser
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/diag_pose_photometric",
        help="Where to save JSONL metrics and summary.txt",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional cap on number of val batches to process",
    )
    parser.add_argument(
        "--eval_splits",
        nargs="+",
        type=str,
        default=["val"],
        choices=["train", "val"],
        help="Which splits to run (train/val)",
    )
    parser.add_argument(
        "--load_weights_pose_folder",
        type=str,
        #default="/home/yxt/文档/mono_result/weights/UAVid_China/md2_vggt_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_fwg0/models/weights_28",
        #efault="/home/yxt/文档/mono_result/weights/UAVula_R1/md2_vggt_uavula_tridataset_512x288_bs8_lr1e-04_e40_step20_ge0y/models/weights_35",
        default="/home/yxt/文档/mono_result/weights/UAVid_Germany/md2_vggt_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_hnrb/models/weights_26",
        help="Path to monodepth2 baseline weights (encoder/depth/pose_encoder/pose)",
    )
    parser.add_argument(
        "--load_weights_posegt_folder",
        type=str,
        #default="/home/yxt/文档/mono_result/weights/UAVid_China/md2_vggt_posegt_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_9yjq/models/weights_36",
        #default="/home/yxt/文档/mono_result/weights/UAVula_R1/md2_vggt_posegt_uavula_tridataset_512x288_bs8_lr1e-04_e40_step20_rkbq/models/weights_24",
        default="/home/yxt/文档/mono_result/weights/UAVid_Germany/md2_vggt_posegt_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_rzsm/models/weights_23",
        help="Path to PoseGT weights (pose_encoder/pose/trans_encoder/trans)",
    )
    parser.add_argument(
        "--depth_align_source",
        type=str,
        default="branch",
        choices=["off", "pred", "vggt", "auto", "branch"],
        help="对齐 anchor 来源: pred(DepthNet) / vggt / auto(按分支) / branch(按分支) / off",
    )
    parser.add_argument(
        "--depth_align_p_low",
        type=float,
        default=5.0,
        help="中值对齐的低分位 (percentile)",
    )
    parser.add_argument(
        "--depth_align_p_high",
        type=float,
        default=95.0,
        help="中值对齐的高分位 (percentile)",
    )
    parser.add_argument(
        "--depth_align_min_pixels",
        type=int,
        default=2048,
        help="对齐时最少有效像素数",
    )
    parser.add_argument(
        "--depth_align_scale_min",
        type=float,
        default=0.05,
        help="对齐尺度最小值（防止异常）",
    )
    parser.add_argument(
        "--depth_align_scale_max",
        type=float,
        default=40.0,
        help="对齐尺度最大值（防止异常）",
    )
    parser.add_argument(
        "--depth_align_eps",
        type=float,
        default=1e-6,
        help="对齐时的 eps",
    )
    # Dataset defaults (fill in your own values)
    # dataset options:
    #   triplet: UAVula_TriDataset, UAVid_TriDataset
    #   mono:    UAVula_Dataset, UAVid2020, WildUAV
    #   kitti:   kitti, kitti_odom
    # split options (see methods/splits/<split>/):
    #   UAVula, UAVid2020_China, UAVid2020_Germany, wilduav (depending on your repo)
    parser.set_defaults(
        methods="MD2_VGGT_PoseGT",
        #dataset="UAVula_TriDataset",
        dataset="UAVid_TriDataset",
        #split="UAVula",
        split="UAVid2020_Germany",

        #data_path="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China",
        #triplet_root='/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China_tri/tri_win20_disp0.03_lap0_rot20d_U0.5_S0',
        data_path="/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany",
        triplet_root='/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_win10_disp0.03_lap0_rot20d_U0.5_S0',
        #data_path="/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_dataset",
        #triplet_root='/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_tri/tri_win5_disp0.05_lap0_rot20d_U0.5_S0',
        batch_size=4,

        height=288,
        width=512,
    )

    opt = base.parse()

    os.makedirs(opt.output_dir, exist_ok=True)

    # 1) 构建模型（含 PoseGT 的 trans 分支）
    runner = EvalRunner(opt)
    # 加载基线权重（encoder/depth/pose_encoder/pose），不加载 optimizer
    load_model_weights_only(runner, opt.load_weights_pose_folder, ["encoder", "depth", "pose_encoder", "pose"])
    runner.set_eval()

    if "trans_encoder" not in runner.models or "trans" not in runner.models:
        raise ValueError("PoseGT 模型未初始化，请使用 --methods MD2_VGGT_PoseGT")

    def _read_weights(folder: str, keep: List[str]):
        if not folder:
            return {}
        out = {}
        for n in keep:
            path = os.path.join(folder, f"{n}.pth")
            if not os.path.isfile(path):
                continue
            state = torch.load(path, map_location="cpu")
            for k in ("height", "width"):
                state.pop(k, None)
            out[n] = state
        return out

    def _apply_weights(weight_set: Dict[str, Dict[str, torch.Tensor]]):
        if not weight_set:
            return
        for n, state in weight_set.items():
            if n not in runner.models:
                continue
            model = runner.models[n]
            model_dict = model.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_dict}
            model_dict.update(filtered)
            model.load_state_dict(model_dict)

    weights_pose = _read_weights(opt.load_weights_pose_folder, ["pose_encoder", "pose"])
    weights_depth_base = _read_weights(opt.load_weights_pose_folder, ["encoder", "depth"])
    weights_posegt = _read_weights(
        opt.load_weights_posegt_folder, ["pose_encoder", "pose", "trans_encoder", "trans"]
    )
    weights_depth_posegt = _read_weights(opt.load_weights_posegt_folder, ["encoder", "depth"])
    missing_posegt = [k for k in ("pose_encoder", "pose", "trans_encoder", "trans") if k not in weights_posegt]
    if missing_posegt:
        raise FileNotFoundError(
            f"PoseGT 权重缺失: {missing_posegt} (folder={opt.load_weights_posegt_folder})"
        )

    # 2) 构造需要的 split（带 external_cam_T_cam）
    splits = list(dict.fromkeys(opt.eval_splits))  # 去重同时保持顺序
    loaders = {sp: _build_loader(opt, sp) for sp in splits}

    records: List[Dict[str, Any]] = []
    device = runner.device
    total_pose = 0.0
    total_posegt = 0.0
    total_vggt = 0.0
    count_pose = 0
    count_posegt = 0
    count_vggt = 0
    used_pose_images = 0
    used_posegt_images = 0
    used_vggt_images = 0
    loss_samples: Dict[str, List[float]] = {
        "pose": [],
        "posegt": [],
        "vggt_raw": [],
        "pose_aligned": [],
        "posegt_aligned": [],
        "vggt_aligned": [],
    }
    depth_align_source = str(getattr(opt, "depth_align_source", "off")).lower()
    depth_align_enabled = depth_align_source != "off"
    depth_align_warned = False
    depth_align_posegt_warned = False

    with torch.no_grad():
        for split_name, loader in loaders.items():
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f"{split_name} splits", ncols=100)
            for step, inputs_cpu in pbar:
                if opt.max_batches is not None and step >= opt.max_batches:
                    break

                inputs = move_batch_to_device(inputs_cpu, device)
                meta = extract_meta(inputs)
                meta["split"] = split_name
                batch_size = inputs.get(("color", 0, 0)).shape[0] if ("color", 0, 0) in inputs else opt.batch_size

                # 深度源：仅使用 depth_gt（脚本入口已过滤）
                depth_gt = inputs.get("depth_gt")
                if depth_gt is None or not torch.is_tensor(depth_gt):
                    raise RuntimeError("Missing depth_gt after filtering; check dataset paths")
                if depth_gt.ndim == 3:
                    depth_gt = depth_gt.unsqueeze(1)
                depth_ready = F.interpolate(
                    depth_gt, [opt.height, opt.width], mode="bilinear", align_corners=False
                )

                # ---- 深度尺度对齐（按分支选择 anchor 深度对齐 colmap depth_gt） ----
                depth_ready_pose_aligned = depth_ready
                depth_ready_posegt_aligned = depth_ready
                depth_ready_vggt_aligned = depth_ready
                align_used_pose = "off"
                align_used_posegt = "off"
                align_used_vggt = "off"
                align_factor_pose = None
                align_factor_posegt = None
                align_factor_vggt = None
                align_ratio_pose = None
                align_ratio_posegt = None
                align_ratio_vggt = None

                if depth_align_enabled:
                    mode = depth_align_source
                    if mode == "auto":
                        mode = "branch"

                    need_base_pred = mode in {"pred", "branch"}
                    need_posegt_pred = mode in {"branch"}
                    need_vggt = mode in {"vggt", "branch"}

                    base_pred_depth = None
                    posegt_pred_depth = None
                    vggt_depth = None
                    vggt_conf = None

                    if need_vggt:
                        vggt_depth = inputs.get("vggt_depth", None)
                        if torch.is_tensor(vggt_depth):
                            if vggt_depth.ndim == 3:
                                vggt_depth = vggt_depth.unsqueeze(1)
                            vggt_depth = F.interpolate(
                                vggt_depth, [opt.height, opt.width], mode="bilinear", align_corners=False
                            )
                            vggt_conf = inputs.get("vggt_conf", None)
                            if torch.is_tensor(vggt_conf):
                                if vggt_conf.ndim == 3:
                                    vggt_conf = vggt_conf.unsqueeze(1)
                                if vggt_conf.shape[-2:] != (opt.height, opt.width):
                                    vggt_conf = F.interpolate(
                                        vggt_conf, [opt.height, opt.width], mode="bilinear", align_corners=False
                                    )
                        else:
                            vggt_depth = None

                    if need_base_pred:
                        if weights_depth_base:
                            _apply_weights(weights_depth_base)
                        base_pred_depth = _predict_depthnet_depth(runner, inputs)

                    if need_posegt_pred:
                        if weights_depth_posegt:
                            _apply_weights(weights_depth_posegt)
                            posegt_pred_depth = _predict_depthnet_depth(runner, inputs)
                        else:
                            if not depth_align_posegt_warned:
                                print("[diag][warn] posegt depth weights not found; fallback to base depthnet")
                                depth_align_posegt_warned = True
                            posegt_pred_depth = None

                    def _align_depth_for_anchor(anchor_depth, anchor_name, valid_mask=None):
                        if anchor_depth is None:
                            return depth_ready, "off", None, None
                        scale_factor, scale_valid = _compute_depth_scale_median(
                            anchor_depth,
                            depth_ready,
                            valid_mask=valid_mask,
                            p_low=float(opt.depth_align_p_low),
                            p_high=float(opt.depth_align_p_high),
                            min_valid_pixels=int(opt.depth_align_min_pixels),
                            eps=float(opt.depth_align_eps),
                            scale_min=float(opt.depth_align_scale_min),
                            scale_max=float(opt.depth_align_scale_max),
                        )
                        depth_aligned = depth_ready * scale_factor
                        if not bool(scale_valid.all()):
                            depth_aligned = depth_ready.clone()
                            depth_aligned[scale_valid] = depth_ready[scale_valid] * scale_factor[scale_valid]
                        valid_ratio = float(scale_valid.float().mean().item())
                        factor_mean = None
                        if bool(scale_valid.any()):
                            factor_mean = float(scale_factor[scale_valid].mean().item())
                        return depth_aligned, anchor_name, factor_mean, valid_ratio

                    if mode == "pred":
                        depth_ready_pose_aligned, align_used_pose, align_factor_pose, align_ratio_pose = \
                            _align_depth_for_anchor(base_pred_depth, "pred_base")
                        depth_ready_posegt_aligned, align_used_posegt, align_factor_posegt, align_ratio_posegt = \
                            _align_depth_for_anchor(base_pred_depth, "pred_base")
                        depth_ready_vggt_aligned, align_used_vggt, align_factor_vggt, align_ratio_vggt = \
                            _align_depth_for_anchor(base_pred_depth, "pred_base")
                    elif mode == "vggt":
                        depth_ready_pose_aligned, align_used_pose, align_factor_pose, align_ratio_pose = \
                            _align_depth_for_anchor(vggt_depth, "vggt", valid_mask=vggt_conf)
                        depth_ready_posegt_aligned, align_used_posegt, align_factor_posegt, align_ratio_posegt = \
                            _align_depth_for_anchor(vggt_depth, "vggt", valid_mask=vggt_conf)
                        depth_ready_vggt_aligned, align_used_vggt, align_factor_vggt, align_ratio_vggt = \
                            _align_depth_for_anchor(vggt_depth, "vggt", valid_mask=vggt_conf)
                    else:  # branch
                        depth_ready_pose_aligned, align_used_pose, align_factor_pose, align_ratio_pose = \
                            _align_depth_for_anchor(base_pred_depth, "pred_base")

                        posegt_anchor = posegt_pred_depth if posegt_pred_depth is not None else base_pred_depth
                        posegt_name = "pred_posegt" if posegt_pred_depth is not None else "pred_base"
                        depth_ready_posegt_aligned, align_used_posegt, align_factor_posegt, align_ratio_posegt = \
                            _align_depth_for_anchor(posegt_anchor, posegt_name)

                        vggt_anchor = vggt_depth if vggt_depth is not None else base_pred_depth
                        vggt_name = "vggt" if vggt_depth is not None else "pred_base"
                        vggt_mask = vggt_conf if vggt_depth is not None else None
                        depth_ready_vggt_aligned, align_used_vggt, align_factor_vggt, align_ratio_vggt = \
                            _align_depth_for_anchor(vggt_anchor, vggt_name, valid_mask=vggt_mask)

                    if (align_used_pose == "off" and align_used_posegt == "off" and align_used_vggt == "off"):
                        if not depth_align_warned:
                            print("[diag][warn] depth_align_source enabled but no anchor depth found")
                            depth_align_warned = True

                # ---- 纯 PoseNet（Monodepth2 样式，使用基线 pose 权重） ----
                _apply_weights(weights_pose)
                pose_out = _predict_poses_with_irw(runner, inputs, {})
                pose_cam_T = {fid: pose_out[("cam_T_cam", 0, fid)] for fid in opt.frame_ids[1:] if fid != "s"}
                # raw: 不做对齐
                pose_warp = _warp_with_pose(runner, inputs, depth_ready, pose_cam_T)
                photometric_pose = float(_compute_photometric_only(runner, inputs, pose_warp).item())
                total_pose += photometric_pose
                count_pose += 1
                used_pose_images += batch_size
                loss_samples["pose"].append(photometric_pose)
                # aligned: 使用尺度校正后的 depth
                photometric_pose_aligned = None
                if depth_align_enabled and align_used_pose != "off":
                    pose_warp_aligned = _warp_with_pose(runner, inputs, depth_ready_pose_aligned, pose_cam_T)
                    photometric_pose_aligned = float(
                        _compute_photometric_only(runner, inputs, pose_warp_aligned).item()
                    )
                    loss_samples["pose_aligned"].append(photometric_pose_aligned)

                # ---- PoseGT（使用 trans 网络生成 irw_img，再预测位姿） ----
                _apply_weights(weights_posegt)
                # raw: 不做对齐
                irw_imgs_raw = _generate_posegt_irw(runner, inputs, depth_ready)
                posegt_out_raw = _predict_poses_with_irw(runner, inputs, irw_imgs_raw)
                posegt_cam_T_raw = {
                    fid: posegt_out_raw[("cam_T_cam", 0, fid)] for fid in opt.frame_ids[1:] if fid != "s"
                }
                posegt_warp = _warp_with_pose(
                    runner, inputs, depth_ready, posegt_cam_T_raw, source_imgs=irw_imgs_raw
                )
                photometric_posegt = float(_compute_photometric_only(runner, inputs, posegt_warp).item())
                total_posegt += photometric_posegt
                count_posegt += 1
                used_posegt_images += batch_size
                loss_samples["posegt"].append(photometric_posegt)
                # aligned: 使用尺度校正后的 depth（含 IRW）
                photometric_posegt_aligned = None
                if depth_align_enabled and align_used_posegt != "off":
                    irw_imgs_aligned = _generate_posegt_irw(runner, inputs, depth_ready_posegt_aligned)
                    posegt_out_aligned = _predict_poses_with_irw(runner, inputs, irw_imgs_aligned)
                    posegt_cam_T_aligned = {
                        fid: posegt_out_aligned[("cam_T_cam", 0, fid)] for fid in opt.frame_ids[1:] if fid != "s"
                    }
                    posegt_warp_aligned = _warp_with_pose(
                        runner,
                        inputs,
                        depth_ready_posegt_aligned,
                        posegt_cam_T_aligned,
                        source_imgs=irw_imgs_aligned,
                    )
                    photometric_posegt_aligned = float(
                        _compute_photometric_only(runner, inputs, posegt_warp_aligned).item()
                    )
                    loss_samples["posegt_aligned"].append(photometric_posegt_aligned)

                # ---- 纯 VGGT 位姿（直接使用外部位姿） ----
                ext_cam_T = {
                    fid: inputs[("external_cam_T_cam", 0, fid)] for fid in opt.frame_ids[1:] if fid != "s"
                }
                # raw: 不做对齐
                vggt_warp = _warp_with_pose(runner, inputs, depth_ready, ext_cam_T)
                photometric_vggt = float(_compute_photometric_only(runner, inputs, vggt_warp).item())
                total_vggt += photometric_vggt
                count_vggt += 1
                used_vggt_images += batch_size
                loss_samples["vggt_raw"].append(photometric_vggt)
                # aligned: 使用尺度校正后的 depth
                photometric_vggt_aligned = None
                if depth_align_enabled and align_used_vggt != "off":
                    vggt_warp_aligned = _warp_with_pose(runner, inputs, depth_ready_vggt_aligned, ext_cam_T)
                    photometric_vggt_aligned = float(
                        _compute_photometric_only(runner, inputs, vggt_warp_aligned).item()
                    )
                    loss_samples["vggt_aligned"].append(photometric_vggt_aligned)

                rec = {
                    **meta,
                    "depth_source": "gt",
                    "depth_align_source": depth_align_source,
                    "depth_align_source_pose": align_used_pose,
                    "depth_align_source_posegt": align_used_posegt,
                    "depth_align_source_vggt": align_used_vggt,
                    "depth_align_factor_mean_pose": align_factor_pose,
                    "depth_align_factor_mean_posegt": align_factor_posegt,
                    "depth_align_factor_mean_vggt": align_factor_vggt,
                    "depth_align_valid_ratio_pose": align_ratio_pose,
                    "depth_align_valid_ratio_posegt": align_ratio_posegt,
                    "depth_align_valid_ratio_vggt": align_ratio_vggt,
                    "photometric_pose": photometric_pose,
                    "photometric_posegt": photometric_posegt,
                    "photometric_vggt": photometric_vggt,
                    "photometric_pose_aligned": photometric_pose_aligned,
                    "photometric_posegt_aligned": photometric_posegt_aligned,
                    "photometric_vggt_aligned": photometric_vggt_aligned,
                }
                records.append(rec)

    # 4) 保存结果
    ds_token = _safe_name(opt.dataset)
    split_token = _safe_name(opt.split)
    metrics_path = os.path.join(opt.output_dir, f"metrics_{ds_token}_{split_token}.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _fmt(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v)

    def _percentile(sorted_vals: List[float], q: float) -> float:
        if not sorted_vals:
            return float("nan")
        if q <= 0:
            return float(sorted_vals[0])
        if q >= 1:
            return float(sorted_vals[-1])
        pos = (len(sorted_vals) - 1) * q
        lo = int(pos)
        hi = min(lo + 1, len(sorted_vals) - 1)
        if lo == hi:
            return float(sorted_vals[lo])
        frac = pos - lo
        return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)

    def _summary_stats(values: List[float]) -> Dict[str, float]:
        if not values:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
                "p05": float("nan"),
                "p95": float("nan"),
            }
        vals = sorted(values)
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        std = var ** 0.5
        return {
            "count": n,
            "mean": mean,
            "std": std,
            "min": vals[0],
            "max": vals[-1],
            "median": _percentile(vals, 0.5),
            "p05": _percentile(vals, 0.05),
            "p95": _percentile(vals, 0.95),
        }

    pose_mean = total_pose / max(1, count_pose)
    posegt_mean = total_posegt / max(1, count_posegt)
    vggt_raw_mean = total_vggt / max(1, count_vggt)

    lines = [
        f"dataset : {opt.dataset}",
        f"split   : {opt.split}",
        f"batches : {len(records)}",
        "---- photometric mean ----",
        f"pose_mean              (基线 PoseNet)          : {_fmt(pose_mean)}",
        f"posegt_mean            (PoseGT)               : {_fmt(posegt_mean)}",
    ]
    lines.extend([
        f"vggt_raw_mean          (原始 VGGT)             : {_fmt(vggt_raw_mean)}",
        "---- 相对 PoseNet (负值=更好) ----",
    ])
    lines.extend([
        f"posegt - pose                               : {_fmt(posegt_mean - pose_mean)}",
        f"vggt_raw - pose                               : {_fmt(vggt_raw_mean - pose_mean)}",
    ])
    lines.extend([
        "---- branch image counts ----",
        f"pose_images            (batches={count_pose}) : {used_pose_images}",
        f"posegt_images          (batches={count_posegt}) : {used_posegt_images}",
        f"vggt_images            (batches={count_vggt}) : {used_vggt_images}",
    ])

    aligned_has_any = any(
        len(loss_samples[key]) > 0 for key in ("pose_aligned", "posegt_aligned", "vggt_aligned")
    )
    stats = {
        "pose": _summary_stats(loss_samples["pose"]),
        "posegt": _summary_stats(loss_samples["posegt"]),
        "vggt_raw": _summary_stats(loss_samples["vggt_raw"]),
        "pose_aligned": _summary_stats(loss_samples["pose_aligned"]),
        "posegt_aligned": _summary_stats(loss_samples["posegt_aligned"]),
        "vggt_aligned": _summary_stats(loss_samples["vggt_aligned"]),
    }
    if aligned_has_any:
        lines.append("---- photometric mean (aligned depth) ----")
        lines.append(
            f"pose_aligned_mean      (PoseNet)             : {_fmt(stats['pose_aligned']['mean'])}"
        )
        lines.append(
            f"posegt_aligned_mean    (PoseGT)              : {_fmt(stats['posegt_aligned']['mean'])}"
        )
        lines.append(
            f"vggt_aligned_mean      (VGGT pose)           : {_fmt(stats['vggt_aligned']['mean'])}"
        )
    lines.append("---- photometric distribution ----")
    lines.append(
        "pose: count={count} mean={mean} std={std} min={min} p05={p05} "
        "median={median} p95={p95} max={max}".format(
            **{k: _fmt(v) if isinstance(v, float) else v for k, v in stats["pose"].items()}
        )
    )
    lines.append(
        "posegt: count={count} mean={mean} std={std} min={min} p05={p05} "
        "median={median} p95={p95} max={max}".format(
            **{k: _fmt(v) if isinstance(v, float) else v for k, v in stats["posegt"].items()}
        )
    )
    lines.append(
        "vggt_raw: count={count} mean={mean} std={std} min={min} p05={p05} "
        "median={median} p95={p95} max={max}".format(
            **{k: _fmt(v) if isinstance(v, float) else v for k, v in stats["vggt_raw"].items()}
        )
    )
    if aligned_has_any:
        lines.append("---- photometric distribution (aligned depth) ----")
        lines.append(
            "pose_aligned: count={count} mean={mean} std={std} min={min} p05={p05} "
            "median={median} p95={p95} max={max}".format(
                **{k: _fmt(v) if isinstance(v, float) else v for k, v in stats["pose_aligned"].items()}
            )
        )
        lines.append(
            "posegt_aligned: count={count} mean={mean} std={std} min={min} p05={p05} "
            "median={median} p95={p95} max={max}".format(
                **{k: _fmt(v) if isinstance(v, float) else v for k, v in stats["posegt_aligned"].items()}
            )
        )
        lines.append(
            "vggt_aligned: count={count} mean={mean} std={std} min={min} p05={p05} "
            "median={median} p95={p95} max={max}".format(
                **{k: _fmt(v) if isinstance(v, float) else v for k, v in stats["vggt_aligned"].items()}
            )
        )
    branch_counts = [used_pose_images, used_posegt_images, used_vggt_images]
    if len(set(branch_counts)) > 1:
        lines.append("[warn] branch image counts differ; check missing external poses or filtering rules")

    summary_path = os.path.join(opt.output_dir, f"summary_{ds_token}_{split_token}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    stats_path = os.path.join(opt.output_dir, f"stats_{ds_token}_{split_token}.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[diag] done")
    for ln in lines:
        print(f"[diag] {ln}")


if __name__ == "__main__":
    main()

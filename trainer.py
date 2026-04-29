# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import time
import math
from collections import deque, defaultdict
import torch
from utils import *
from kitti_utils import *
from layers import *
import wandb
from trainer_support import (
    accumulate_metrics as trainer_accumulate_metrics,
    compute_depth_losses as trainer_compute_depth_losses,
    compute_kitti_depth_metrics as trainer_compute_kitti_depth_metrics,
    compute_uavid_depth_metrics as trainer_compute_uavid_depth_metrics,
    load_model as trainer_load_model,
    log as trainer_log,
    log_epoch_metrics as trainer_log_epoch_metrics,
    log_time as trainer_log_time,
    reset_metrics as trainer_reset_metrics,
    save_model as trainer_save_model,
    save_opts as trainer_save_opts,
    update_pose_metrics as trainer_update_pose_metrics,
)

from trainer_init.model_init import init_models, get_forward_handler, SCALE_ALIGN_METHODS
from trainer_init.loss_init import init_losses
from trainer_init.data_init import init_dataloaders
from trainer_init.logging_init import init_logging
from trainer_init.optim_init import init_optimizers
from trainer_init.geometry_init import init_geometry
from trainer_init.scale_alignment import DepthScaleAligner


class Trainer:
    """封装模型构建、训练循环与日志记录的核心控制器。"""

    def __init__(self, options):
        self.opt = options
        # 在初始化早期就确定训练设备，后续所有张量搬运都以此为准。
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        # 先用初始值占位，后面覆盖
        self.models = {}
        self.parameters_to_train = []
        self.training = False

        assert self.opt.height % 32 == 0
        assert self.opt.width % 32 == 0
        assert self.opt.frame_ids[0] == 0

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        method_name = str(getattr(self.opt, "methods", ""))
        self.use_pose_net = True
        self.use_posegt = method_name in {
            "MD2_VGGT_PoseGT",
            "MD2_VGGT_PoseGT_DepthCycleViz",
            "MD2_VGGT_PoseGT_DepthSensitivityViz",
            "MD2_VGGT_PoseGT_DepthSensViz",
            "MD2_VGGT_PoseGT_DepthSensWeight",
            "MD2_VGGT_PoseGT_BadScoreWeight",
            "MD2_VGGT_PoseGT_BadScoreLocalWeight",
            "MD2_VGGT_PoseGT_HRMask",
            "MD2_VGGT_PoseGT_Mask",
            "MD2_VGGT_PoseGT_DeRotHardMask",
            "MD2_VGGT_PoseGT_DeRotSigmoidWeight",
            "GasMono",
            "MonoViT_PoseGT",
            "MonoViT_PoseGT_Mask",
            "MonoViT_PoseGT_HRMask",
            "MonoViT_VGGT_PoseGT_BadScoreWeight",
        }
        self.use_vggt_prewarp = method_name == "MonoViT_VGGT_PreWarp"
        self.use_rdistill = method_name == "MonoViT_VGGT_RDistill"
        self.use_rmask_switch = method_name == "MonoViT_VGGT_RMaskSwitch"
        self.use_derot_hardmask = method_name == "MD2_VGGT_PoseGT_DeRotHardMask"
        self.use_derot_sigmoid_weight = method_name == "MD2_VGGT_PoseGT_DeRotSigmoidWeight"
        self.use_gasmono_fixed = method_name == "GasMono"
        self.gasmono_iiters = max(
            1,
            int(getattr(self.opt, "iiters", getattr(self.opt, "gasmono_iiters", 2))),
        )
        self.gasmono_iiter_start_epoch = max(0, int(getattr(self.opt, "gasmono_iiter_start_epoch", 10)))

        # ========= 1) 提前初始化 W&B，并用 sweep 覆盖超参 =========
        # 注意：需要在 trainer_init/logging_init.py 中实现 sweep 感知逻辑（见下节）
        init_logging(self)                 # <-- 移到最前
        self._rebuild_model_name_and_paths()

        # ========= 2) 其余初始化按原流程 =========
        init_models(self)
        init_losses(self)
        init_dataloaders(self)
        if hasattr(self, "loss"):
            self.loss.total_steps = getattr(self, "num_total_steps", None)
            if hasattr(self, "train_loader"):
                self.loss.steps_per_epoch = len(self.train_loader)
        init_optimizers(self)
        init_geometry(self)
        self._configure_pose_residual_schedule()

        align_mode = str(getattr(self.opt, "scale_align_mode", "off"))
        align_enabled = str(getattr(self.opt, "methods", "")) in SCALE_ALIGN_METHODS
        self.scale_aligner = DepthScaleAligner(self.opt) if align_enabled and align_mode != "off" else None
        self._scale_align_monitor = {
            "train": {"attempts": 0, "success": 0},
            "val": {"attempts": 0, "success": 0},
        }
        self._scale_align_warned = {"train": False, "val": False}

        if not self.opt.no_ssim:
            self.ssim = SSIM().to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        self.collect_debug_metrics = bool(getattr(self.opt, "enable_debug_metrics", False))
        self.enable_depth_cycle_viz = bool(getattr(self.opt, "enable_depth_cycle_viz", False))
        if method_name in {"MD2_VGGT_PoseGT_DepthCycleViz", "MD2_VGGT_DepthCycleViz"}:
            self.enable_depth_cycle_viz = True
        self.depth_cycle_viz_scale = int(getattr(self.opt, "depth_cycle_viz_scale", 0))
        self._depth_cycle_viz_warned = False
        self.enable_depth_sensitivity_viz = bool(getattr(self.opt, "enable_depth_sensitivity_viz", False))
        if method_name in {
            "MD2_VGGT_PoseGT_DepthSensitivityViz",
            "MD2_VGGT_DepthSensitivityViz",
            "MD2_VGGT_PoseGT_DepthSensViz",
            "MD2_VGGT_PoseGT_DepthSensWeight",
        }:
            self.enable_depth_sensitivity_viz = True
        self.enable_depth_pixshift_viz = bool(getattr(self.opt, "enable_depth_pixshift_viz", False))
        if method_name in {"MD2_VGGT_PoseGT_DepthSensViz", "MD2_VGGT_PoseGT_DepthSensWeight"}:
            self.enable_depth_pixshift_viz = True
        self.depth_sensitivity_viz_scale = int(getattr(self.opt, "depth_sensitivity_viz_scale", 0))
        self.depth_sensitivity_factor = float(getattr(self.opt, "depth_sensitivity_factor", 1.1))
        self._depth_sensitivity_viz_warned = False
        self.pose_t_history = deque(maxlen=100) if self.collect_debug_metrics else deque(maxlen=1)
        self._metric_buffers = {
            "train": defaultdict(list),
            "val": defaultdict(list),
        } if self.collect_debug_metrics else {}

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and logs are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)
        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(self.train_loader.dataset), len(self.val_loader.dataset)))

        self.save_opts()

    def make_model_name(self, opt=None):
        """基于最终 opt 生成可读且稳定的 model_name。"""
        opt = opt or self.opt
        dataset = getattr(opt, "dataset", "dataset").lower()
        return (
            f"{opt.methods.lower()}_{dataset}_{opt.width}x{opt.height}_"
            f"bs{opt.batch_size}_lr{opt.learning_rate:.0e}_"
            f"e{opt.num_epochs}_step{opt.scheduler_step_size}"
        )

    def _rebuild_model_name_and_paths(self):
        # init_logging() 回灌 sweep/CLI 后，再生成“最终”名字
        self.opt.model_name = self.make_model_name(self.opt)

        # 可选：加 run.id 后缀以避免目录名覆盖（如并发运行）
        try:
            if wandb.run is not None and getattr(wandb.run, "id", None):
                self.opt.model_name = f"{self.opt.model_name}_{wandb.run.id[:4]}"
        except Exception:
            pass

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        os.makedirs(os.path.join(self.log_path, "models"), exist_ok=True)

        # 仅设置显示名（不写入 config）
        if getattr(self, "using_wandb", False):
            try:
                wandb.run.name = self.opt.model_name
                wandb.run.save()
            except Exception:
                pass

    def _configure_pose_residual_schedule(self):
        self.pose_residual_schedule = None
        self.pose_residual_scale = 1.0
        self.pose_residual_scale_start = float(getattr(self.opt, "pose_residual_scale_start", 1.0))
        self.pose_residual_scale_end = float(getattr(self.opt, "pose_residual_scale_end", 0.0))
        self.pose_residual_decay_epochs = int(getattr(self.opt, "pose_residual_decay_epochs", 0))
        self.pose_residual_switch_epoch = int(getattr(self.opt, "pose_residual_switch_epoch", -1))
        self.pose_residual_switch_mode = str(getattr(self.opt, "pose_residual_switch_mode", "rt"))
        self._pose_residual_switched = False

        if self.opt.methods == "MD2_VGGT_ResPose_Decay":
            self.pose_residual_schedule = "decay"
        elif self.opt.methods == "MD2_VGGT_PoseToRes":
            self.pose_residual_schedule = "switch"

    def _update_pose_residual_schedule(self):
        if self.pose_residual_schedule == "decay":
            decay_epochs = self.pose_residual_decay_epochs
            if decay_epochs <= 0:
                decay_epochs = int(getattr(self.opt, "num_epochs", 1))
            denom = max(1, decay_epochs - 1)
            progress = min(float(self.epoch) / float(denom), 1.0)
            self.pose_residual_scale = (
                self.pose_residual_scale_start
                + (self.pose_residual_scale_end - self.pose_residual_scale_start) * progress
            )
        elif self.pose_residual_schedule == "switch":
            switch_epoch = self.pose_residual_switch_epoch
            if switch_epoch < 0:
                switch_epoch = int(getattr(self.opt, "num_epochs", 1)) // 2
            if (not self._pose_residual_switched) and self.epoch >= switch_epoch:
                mode = self.pose_residual_switch_mode
                if mode not in ("rt", "t"):
                    mode = "rt"
                self.pose_residual_mode = mode
                self.pose_residual_reg_weight = float(getattr(self.opt, "pose_residual_reg_weight", 1e-3))
                self._pose_residual_switched = True
                print(f"[pose] switch to residual mode={mode} at epoch {self.epoch}")

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()
        self.training = True

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()
        self.training = False

    def _disp_to_depth_for_method(self, disp: torch.Tensor) -> torch.Tensor:
        """Convert disp tensor to depth according to current method semantics."""
        if str(getattr(self.opt, "methods", "")) == "SPIDepth":
            # SPIdepth decoder stores metric depth directly in ('disp', scale).
            return disp
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        return depth

    def train(self):
        """Run the entire training pipeline
        """
        # 训练阶段从 step=0 开始；所有日志/调度器均基于该计数。
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self._update_pose_residual_schedule()
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        #self.model_lr_scheduler.step()

        print("Training")
        self.set_train()
        if self.collect_debug_metrics:
            self._reset_metrics("train")
        self.model_optimizer.zero_grad()
        #self.val()
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            # 前向 + 损失计算统一由 process_batch 执行，内部会处理中间缓存/多尺度输出。
            # GasMono 固定对比方法使用 batch 内多次迭代优化，并维护 best pseudo-depth。
            if self.use_gasmono_fixed:
                iiter = self.gasmono_iiters if self.epoch >= self.gasmono_iiter_start_epoch else 1
                if hasattr(self, "loss"):
                    setattr(self.loss, "bestpp", {})
                for _ in range(iiter):
                    # GasMono performs multiple optimization iters on the same mini-batch.
                    # Build a detached view of tensor inputs each iter to prevent any
                    # accidental autograd graph carry-over across inner-loop steps.
                    iter_inputs = {}
                    for key, val in inputs.items():
                        if isinstance(val, torch.Tensor):
                            iter_inputs[key] = val.detach()
                        else:
                            iter_inputs[key] = val
                    outputs, losses = self.process_batch(iter_inputs)
                    losses["loss"].backward()
                    self.model_optimizer.step()
                    self.model_optimizer.zero_grad()
            else:
                outputs, losses = self.process_batch(inputs)
                losses["loss"].backward()
                self.model_optimizer.step()
                self.model_optimizer.zero_grad()

            if self.collect_debug_metrics:
                self._accumulate_metrics("train", losses)

            duration = time.time() - before_op_time

            # 每隔 100 步记录一次日志
            #if self.step % self.opt.log_frequency == 0:
            if self.step > 0 and (self.step % self.opt.log_frequency == 0):

                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)

            self.step += 1

        if self.collect_debug_metrics:
            self._log_epoch_metrics("train")
        self.val()

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """

        # DataLoader 产出的 batch 是 CPU tensor/Numpy，需在此统一搬运到目标设备。
        for key, val in list(inputs.items()):
            if isinstance(val, torch.Tensor):
                inputs[key] = val.to(self.device, non_blocking=True)
            elif isinstance(val, np.ndarray):
                t = torch.from_numpy(val)
                inputs[key] = t.to(self.device, non_blocking=True)
            else:
                # 非张量/数组的元数据（list/str/int/None 等）保持不动
                inputs[key] = val

            # 同步 step 给 loss（可选，但建议；用于 freeze_gamma_until）
        if hasattr(self, "loss") and hasattr(self.loss, "global_step"):
            self.loss.global_step = self.step
        if hasattr(self, "loss") and hasattr(self.loss, "global_epoch"):
            self.loss.global_epoch = self.epoch

        # ========= 将原先的大型 if/elif 分支替换为外部 handler =========
        # 根据 self.opt.methods 选择合适的 forward 函数，保持 Trainer 逻辑紧凑。
        handler = get_forward_handler(self.opt.methods)
        outputs, losses = handler(self, inputs)

        if self.collect_debug_metrics:
            pose_metrics = self._update_pose_metrics(outputs)
            if pose_metrics:
                losses.update(pose_metrics)

        return outputs, losses

    def _reset_metrics(self, mode):
        return trainer_reset_metrics(self, mode)

    def _accumulate_metrics(self, mode, losses):
        return trainer_accumulate_metrics(self, mode, losses)

    def _log_epoch_metrics(self, mode):
        return trainer_log_epoch_metrics(self, mode)

    def _update_pose_metrics(self, outputs):
        return trainer_update_pose_metrics(self, outputs)

    def _get_aux_source_image_for_pose(self, outputs, frame_id):
        # Keep PoseNet source inputs in the color_aug domain, matching
        # original Monodepth2 pose/depth encoder semantics.
        if bool(getattr(self, "use_posegt", False)):
            aux_img = outputs.get(("irw_img_aug", frame_id, 0), None)
            if torch.is_tensor(aux_img):
                return aux_img
            aux_img = outputs.get(("irw_img", frame_id, 0), None)
            if torch.is_tensor(aux_img):
                return aux_img
        if bool(getattr(self, "use_vggt_prewarp", False)):
            aux_img = outputs.get(("pre_warp_img_aug", frame_id, 0), None)
            if torch.is_tensor(aux_img):
                return aux_img
            aux_img = outputs.get(("pre_warp_img", frame_id, 0), None)
            if torch.is_tensor(aux_img):
                return aux_img
        return None

    def _get_aux_source_image_for_photo(self, outputs, frame_id):
        # Photometric reprojection/automask stay in the raw-color domain.
        if bool(getattr(self, "use_posegt", False)):
            aux_img = outputs.get(("irw_img", frame_id, 0), None)
            if torch.is_tensor(aux_img):
                return aux_img
        if bool(getattr(self, "use_vggt_prewarp", False)):
            aux_img = outputs.get(("pre_warp_img", frame_id, 0), None)
            if torch.is_tensor(aux_img):
                return aux_img
        return None

    def _get_aux_source_transform(self, outputs, frame_id):
        if bool(getattr(self, "use_posegt", False)):
            aux_T = outputs.get(("posegt_cam_T_cam", 0, frame_id), None)
            if torch.is_tensor(aux_T):
                return aux_T
        if bool(getattr(self, "use_vggt_prewarp", False)):
            aux_T = outputs.get(("pre_warp_cam_T_cam", 0, frame_id), None)
            if torch.is_tensor(aux_T):
                return aux_T
        return None

    @staticmethod
    def _rotation_only_transform(T):
        if not torch.is_tensor(T):
            return T
        T_rot = T.clone()
        T_rot[:, :3, 3] = 0.0
        return T_rot

    def _rotation_only_warp_grid(self, K, inv_K, T_rot, height, width):
        """Build a depth-free sampling grid using rotation homography K R K^-1."""
        batch = K.shape[0]
        device = K.device
        dtype = K.dtype
        target_px = self._target_pixel_grid(batch, height, width, device=device, dtype=dtype)
        ones = torch.ones(batch, height, width, 1, device=device, dtype=dtype)
        target_h = torch.cat([target_px, ones], dim=-1)
        target_h = target_h.view(batch, -1, 3).transpose(1, 2)

        K3 = K[:, :3, :3]
        inv_K3 = inv_K[:, :3, :3]
        R = T_rot[:, :3, :3]
        H = torch.matmul(K3, torch.matmul(R, inv_K3))
        source_h = torch.matmul(H, target_h)

        eps = 1e-7
        x = source_h[:, 0, :] / (source_h[:, 2, :] + eps)
        y = source_h[:, 1, :] / (source_h[:, 2, :] + eps)

        pix_coords = torch.stack([x, y], dim=1).view(batch, 2, height, width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= float(max(width - 1, 1))
        pix_coords[..., 1] /= float(max(height - 1, 1))
        pix_coords = (pix_coords - 0.5) * 2.0
        return pix_coords

    def _identity_warp_grid(self, batch, height, width, device, dtype):
        target_px = self._target_pixel_grid(batch, height, width, device=device, dtype=dtype)
        x = target_px[..., 0] / float(max(width - 1, 1))
        y = target_px[..., 1] / float(max(height - 1, 1))
        grid = torch.stack([x, y], dim=-1)
        return (grid - 0.5) * 2.0

    def _rotation_only_flow(self, K, inv_K, T_rot, height, width):
        """Return normalized target-to-source rotation displacement flow as [B, 2, H, W]."""
        rot_grid = self._rotation_only_warp_grid(K, inv_K, T_rot, height=height, width=width)
        identity_grid = self._identity_warp_grid(
            batch=K.shape[0],
            height=height,
            width=width,
            device=K.device,
            dtype=K.dtype,
        )
        flow = rot_grid - identity_grid
        return flow.permute(0, 3, 1, 2).contiguous()

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        use_vggt_tdir_posenet_mag = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_TDir_PoseMag"
        use_vggt_tprior_alpha = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_TPrior_Alpha"
        use_vggt_tprior_align = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_TPrior_AlignRes"
        use_vggt_rprior_tpose = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_RPrior_TPose"
        use_monovit_rprior_resr_tpose = (
            str(getattr(self.opt, "methods", "")) == "MonoViT_VGGT_RPrior_ResR_TPose"
        )
        use_monovit_rflow_pose = (
            str(getattr(self.opt, "methods", "")) == "MonoViT_VGGT_RFlow_Pose"
        )
        use_monovit_rflow_resr_tpose = (
            str(getattr(self.opt, "methods", "")) == "MonoViT_VGGT_RFlow_ResR_TPose"
        )
        use_monovit_rflow_resr_tpose_singlehead = (
            str(getattr(self.opt, "methods", "")) == "MonoViT_VGGT_RFlow_ResR_TPose_SingleHead"
        )
        use_monovit_rflow_tinj = (
            str(getattr(self.opt, "methods", "")) == "MonoViT_VGGT_RFlow_TInj"
        )
        use_spidepth_posecnn = bool(getattr(self, "spidepth_use_posecnn", False))
        use_monovit_rflow = bool(
            use_monovit_rflow_pose
            or use_monovit_rflow_resr_tpose
            or use_monovit_rflow_resr_tpose_singlehead
            or use_monovit_rflow_tinj
        )
        alpha_mode = str(getattr(self.opt, "pose_alpha_mode", "tanh")).lower()
        alpha_tanh_scale = float(getattr(self.opt, "pose_alpha_tanh_scale", 0.1))
        alpha_exp_scale = float(getattr(self.opt, "pose_alpha_exp_scale", 0.01))
        pose_alpha_reg_weight = float(getattr(self.opt, "pose_alpha_reg_weight", 0.0))
        align_scale_mode = str(getattr(self.opt, "pose_align_scale_mode", "tanh")).lower()
        align_scale_tanh_scale = float(getattr(self.opt, "pose_align_scale_tanh_scale", 0.1))
        align_scale_exp_scale = float(getattr(self.opt, "pose_align_scale_exp_scale", 0.01))
        align_res_tanh_scale = float(getattr(self.opt, "pose_align_res_tanh_scale", 0.01))
        align_res_scale_by_prior = bool(getattr(self.opt, "pose_align_res_scale_by_prior_norm", False))
        align_scale_reg_weight = float(getattr(self.opt, "pose_align_scale_reg_weight", 0.0))
        align_res_reg_weight = float(getattr(self.opt, "pose_align_res_reg_weight", 0.0))
        if not self.use_pose_net:
            missing = []
            pose_scale = self.models.get("pose_scale", None)
            for f_i in self.opt.frame_ids[1:]:
                if f_i == "s":
                    continue
                key = ("external_cam_T_cam", 0, f_i)
                T = inputs.get(key, None)
                if T is None:
                    missing.append(f_i)
                    continue
                if pose_scale is not None:
                    scale = pose_scale(T.shape[0], device=T.device, dtype=T.dtype)
                    t_scaled = T[:, :3, 3:4] * scale
                    T = torch.cat([torch.cat([T[:, :3, :3], t_scaled], dim=2), T[:, 3:4, :]], dim=1)
                outputs[("cam_T_cam", 0, f_i)] = T
            if missing:
                raise KeyError(f"Missing external poses for frame ids: {missing}")
            return outputs

        residual_mode = getattr(self, "pose_residual_mode", None)
        pose_reg_terms = []
        alpha_reg_terms = []
        align_reg_terms = []

        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_img = pose_feats[f_i]
                        aux_pose_img = self._get_aux_source_image_for_pose(outputs, f_i)
                        if torch.is_tensor(aux_pose_img):
                            pose_img = aux_pose_img
                        pose_inputs = [pose_img, pose_feats[0]]
                    else:
                        pose_img = pose_feats[f_i]
                        aux_pose_img = self._get_aux_source_image_for_pose(outputs, f_i)
                        if torch.is_tensor(aux_pose_img):
                            pose_img = aux_pose_img
                        pose_inputs = [pose_feats[0], pose_img]

                    pose_pair = torch.cat(pose_inputs, 1)
                    if use_spidepth_posecnn:
                        axisangle, translation = self.models["pose"](pose_pair)
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
                        )
                        continue
                    if use_monovit_rflow:
                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for rotation-flow method, frame id={f_i}")

                        pose_dtype = pose_pair.dtype
                        pose_device = pose_pair.device
                        K = inputs[("K", 0)].to(device=pose_device, dtype=pose_dtype)
                        inv_K = inputs[("inv_K", 0)].to(device=pose_device, dtype=pose_dtype)
                        T_rot = self._rotation_only_transform(
                            base_T.to(device=pose_device, dtype=pose_dtype)
                        )
                        rotation_flow = self._rotation_only_flow(
                            K,
                            inv_K,
                            T_rot,
                            height=pose_pair.shape[-2],
                            width=pose_pair.shape[-1],
                        )
                        pose_inputs = [self.models["pose_encoder"](pose_pair, rotation_flow)]
                        if self.collect_debug_metrics:
                            outputs[("pose_rotation_flow", f_i, 0)] = rotation_flow.detach()
                    else:
                        pose_inputs = [self.models["pose_encoder"](pose_pair)]

                    if use_monovit_rflow_tinj:
                        t_prior = base_T.to(device=pose_pair.device, dtype=pose_pair.dtype)[:, :3, 3]
                        pose_out = self.models["pose"](pose_inputs, t_prior)
                        if not isinstance(pose_out, (tuple, list)) or len(pose_out) != 3:
                            raise RuntimeError(
                                "MonoViT_VGGT_RFlow_TInj expects PoseTPriorDecoder output tuple."
                            )
                        axisangle, trans_scale, trans_res = pose_out
                        outputs[("axisangle", 0, f_i)] = axisangle

                        scale = 1.0 + trans_scale[:, 0, 0]
                        t_res = trans_res[:, 0, 0]
                        t_final = scale * t_prior + t_res

                        translation = trans_res.clone()
                        translation[:, 0, 0, :] = t_final
                        outputs[("translation", 0, f_i)] = translation

                        T_prior = base_T.to(device=axisangle.device, dtype=axisangle.dtype)
                        R_delta = rot_from_axisangle(axisangle[:, 0])
                        if f_i < 0:
                            R_delta = R_delta.transpose(1, 2)
                        R_final = torch.matmul(R_delta[:, :3, :3], T_prior[:, :3, :3])

                        T = T_prior.clone()
                        T[:, :3, :3] = R_final
                        T[:, :3, 3] = t_final
                        outputs[("cam_T_cam", 0, f_i)] = T
                        continue

                    if use_monovit_rflow_resr_tpose_singlehead:
                        pose_out = self.models["pose"](pose_inputs)
                        if not isinstance(pose_out, (tuple, list)) or len(pose_out) != 2:
                            raise RuntimeError(
                                "MonoViT_VGGT_RFlow_ResR_TPose_SingleHead expects PoseDecoder output tuple."
                            )
                        axisangle, translation = pose_out
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation

                        delta_T = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
                        )

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(
                                f"Missing external pose for rotation-flow single-head method, frame id={f_i}"
                            )

                        T_rprior = base_T.to(device=delta_T.device, dtype=delta_T.dtype).clone()
                        T_rprior[:, :3, 3] = 0.0
                        outputs[("cam_T_cam", 0, f_i)] = torch.matmul(delta_T, T_rprior)
                        continue

                    if use_monovit_rprior_resr_tpose or use_monovit_rflow_resr_tpose:
                        axisangle = self.models["pose_r"](pose_inputs)
                        translation = self.models["pose_t"](pose_inputs)
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation

                        delta_T = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0)
                        )

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for rotation-prior method, frame id={f_i}")

                        T_rprior = base_T.to(device=delta_T.device, dtype=delta_T.dtype).clone()
                        T_rprior[:, :3, 3] = 0.0
                        outputs[("cam_T_cam", 0, f_i)] = torch.matmul(delta_T, T_rprior)
                        continue

                    pose_out = self.models["pose"](pose_inputs)
                    if use_vggt_rprior_tpose:
                        if not isinstance(pose_out, (tuple, list)):
                            raise RuntimeError("MD2_VGGT_RPrior_TPose expects PoseDecoder output tuple.")
                        axisangle, translation = pose_out
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation

                        T_pose = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for r-prior method, frame id={f_i}")

                        T_prior = base_T.to(T_pose.device)
                        T = T_prior.clone()
                        T[:, :3, 3] = T_pose[:, :3, 3]
                        outputs[("cam_T_cam", 0, f_i)] = T
                        continue
                    if use_vggt_tdir_posenet_mag:
                        if isinstance(pose_out, (tuple, list)) and len(pose_out) == 2:
                            axisangle, translation = pose_out
                            t_mag = torch.linalg.norm(translation[:, 0], dim=2, keepdim=True)
                        else:
                            axisangle, translation = None, None
                            t_mag = pose_out
                            if t_mag.ndim == 4:
                                t_mag = t_mag[:, 0]
                            elif t_mag.ndim == 3:
                                t_mag = t_mag[:, 0:1]
                            elif t_mag.ndim == 2:
                                t_mag = t_mag[:, None, None]

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for t-dir method, frame id={f_i}")

                        T_prior = base_T.to(t_mag.device)
                        t_prior = T_prior[:, :3, 3]
                        t_norm = torch.linalg.norm(t_prior, dim=1, keepdim=True).clamp_min(1e-6)
                        t_dir = t_prior / t_norm
                        t_mag = t_mag.to(device=t_dir.device, dtype=t_dir.dtype)
                        t_combined = t_dir.unsqueeze(1) * t_mag

                        T = T_prior.clone()
                        T[:, :3, 3] = t_combined[:, 0]
                        outputs[("cam_T_cam", 0, f_i)] = T

                        if self.collect_debug_metrics:
                            outputs[("pose_mag", 0, f_i)] = t_mag.detach()
                            outputs[("pose_prior_norm", 0, f_i)] = t_norm.unsqueeze(1).detach()
                        continue
                    if use_vggt_tprior_alpha:
                        if isinstance(pose_out, (tuple, list)):
                            raise RuntimeError(
                                "MD2_VGGT_TPrior_Alpha expects PoseAlphaDecoder output (tensor), got tuple."
                            )
                        alpha_raw = pose_out
                        if not torch.is_tensor(alpha_raw):
                            raise TypeError("PoseAlphaDecoder output must be a tensor.")
                        if alpha_raw.ndim == 4:
                            alpha_raw = alpha_raw[:, 0]
                        elif alpha_raw.ndim == 3:
                            alpha_raw = alpha_raw[:, 0:1]
                        elif alpha_raw.ndim == 2:
                            alpha_raw = alpha_raw[:, None, None]

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for t-prior method, frame id={f_i}")

                        T_prior = base_T.to(alpha_raw.device)
                        t_prior = T_prior[:, :3, 3]
                        alpha_raw = alpha_raw.to(device=t_prior.device, dtype=t_prior.dtype)
                        if alpha_mode == "exp":
                            alpha = torch.exp(alpha_raw * alpha_exp_scale)
                        else:
                            alpha = 1.0 + alpha_tanh_scale * torch.tanh(alpha_raw)

                        alpha_scalar = alpha.view(alpha.shape[0], -1)
                        if alpha_scalar.shape[1] != 1:
                            alpha_scalar = alpha_scalar[:, 0:1]
                        # Scale full prior translation (no direction normalization).
                        t_scaled = t_prior * alpha_scalar

                        T = T_prior.clone()
                        T[:, :3, 3] = t_scaled
                        outputs[("cam_T_cam", 0, f_i)] = T

                        if pose_alpha_reg_weight > 0:
                            log_alpha = torch.log(alpha.clamp_min(1e-6))
                            alpha_reg_terms.append((log_alpha ** 2).mean())

                        if self.collect_debug_metrics:
                            outputs[("pose_alpha", 0, f_i)] = alpha.detach()
                        continue
                    if use_vggt_tprior_align:
                        align_out = self.models.get("pose_align", None)
                        if align_out is None:
                            raise RuntimeError("MD2_VGGT_TPrior_AlignRes expects pose_align decoder.")
                        align_raw = align_out(pose_inputs)
                        if not torch.is_tensor(align_raw):
                            raise TypeError("PoseAlignDecoder output must be a tensor.")
                        if align_raw.ndim == 4:
                            align_raw = align_raw[:, 0]
                        elif align_raw.ndim == 3:
                            align_raw = align_raw[:, 0:1]
                        elif align_raw.ndim == 2:
                            align_raw = align_raw[:, None, :]
                        if align_raw.shape[-1] != 4:
                            raise RuntimeError("PoseAlignDecoder output last dim must be 4.")

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for align method, frame id={f_i}")

                        T_prior = base_T.to(align_raw.device)
                        t_prior = T_prior[:, :3, 3]
                        t_norm = torch.linalg.norm(t_prior, dim=1, keepdim=True).clamp_min(1e-6)

                        align_raw = align_raw.to(device=t_prior.device, dtype=t_prior.dtype)
                        scale_raw = align_raw[..., :1]
                        res_raw = align_raw[..., 1:]

                        if align_scale_mode == "exp":
                            scale = torch.exp(scale_raw * align_scale_exp_scale)
                        else:
                            scale = 1.0 + align_scale_tanh_scale * torch.tanh(scale_raw)

                        scale_scalar = scale.view(scale.shape[0], -1)
                        if scale_scalar.shape[1] != 1:
                            scale_scalar = scale_scalar[:, 0:1]
                        t_scaled = t_prior * scale_scalar

                        res = torch.tanh(res_raw) * align_res_tanh_scale
                        if align_res_scale_by_prior:
                            res = res * t_norm.view(-1, 1, 1)
                        if res.ndim == 3:
                            t_res = res[:, 0]
                        else:
                            t_res = res.view(res.shape[0], 3)

                        t_corr = t_scaled + t_res

                        if not isinstance(pose_out, (tuple, list)):
                            raise RuntimeError("MD2_VGGT_TPrior_AlignRes expects PoseDecoder output tuple.")
                        axisangle = pose_out[0]
                        if axisangle.ndim == 4:
                            axisangle = axisangle[:, 0]
                        elif axisangle.ndim == 3:
                            axisangle = axisangle[:, 0:1]
                        elif axisangle.ndim == 2:
                            axisangle = axisangle[:, None, :]

                        R_delta = rot_from_axisangle(axisangle)
                        if f_i < 0:
                            R_delta = R_delta.transpose(1, 2)
                        R_prior = T_prior[:, :3, :3]
                        R_final = torch.matmul(R_delta[:, :3, :3], R_prior)

                        T = T_prior.clone()
                        T[:, :3, :3] = R_final
                        T[:, :3, 3] = t_corr
                        outputs[("cam_T_cam", 0, f_i)] = T

                        if align_scale_reg_weight > 0:
                            log_scale = torch.log(scale.clamp_min(1e-6))
                            align_reg_terms.append((log_scale ** 2).mean() * align_scale_reg_weight)
                        if align_res_reg_weight > 0:
                            align_reg_terms.append((t_res ** 2).mean() * align_res_reg_weight)

                        if self.collect_debug_metrics:
                            outputs[("pose_align_scale", 0, f_i)] = scale.detach()
                            outputs[("pose_align_res", 0, f_i)] = t_res.detach()
                            outputs[("pose_prior_norm", 0, f_i)] = t_norm.unsqueeze(1).detach()
                        continue
                    axisangle, translation = pose_out
                    if residual_mode is not None:
                        if residual_mode == "t":
                            axisangle = torch.zeros_like(axisangle)
                        scale = getattr(self, "pose_residual_scale", 1.0)
                        if scale != 1.0:
                            axisangle = axisangle * scale
                            translation = translation * scale

                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation

                        delta_T = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

                        key = ("external_cam_T_cam", 0, f_i)
                        base_T = inputs.get(key, None)
                        if base_T is None:
                            raise KeyError(f"Missing external pose for residual method, frame id={f_i}")
                        T_prior = base_T.to(delta_T.device)
                        order = str(getattr(self, "pose_residual_order", "left")).lower()
                        if order == "right":
                            outputs[("cam_T_cam", 0, f_i)] = torch.matmul(T_prior, delta_T)
                        else:
                            outputs[("cam_T_cam", 0, f_i)] = torch.matmul(delta_T, T_prior)

                        if getattr(self, "pose_residual_reg_weight", 0.0) > 0:
                            reg = (translation[:, 0] ** 2).mean()
                            if residual_mode != "t":
                                reg = reg + (axisangle[:, 0] ** 2).mean()
                            pose_reg_terms.append(reg)
                    else:
                        outputs[("axisangle", 0, f_i)] = axisangle
                        outputs[("translation", 0, f_i)] = translation

                        # Invert the matrix if the frame id is negative
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        if pose_reg_terms and getattr(self, "pose_residual_reg_weight", 0.0) > 0:
            reg_tensor = torch.stack(pose_reg_terms).sum() * self.pose_residual_reg_weight
            outputs["pose_residual_reg"] = reg_tensor
        if alpha_reg_terms and pose_alpha_reg_weight > 0:
            outputs["pose_alpha_reg"] = torch.stack(alpha_reg_terms).sum() * pose_alpha_reg_weight
        if align_reg_terms and (align_scale_reg_weight > 0 or align_res_reg_weight > 0):
            outputs["pose_align_reg"] = torch.stack(align_reg_terms).sum()

        return outputs

    def val(self):
        """Validate the model over the entire val_loader (once per epoch)."""
        self.set_eval()
        if self.collect_debug_metrics:
            self._reset_metrics("val")

        total_loss = 0.0
        count = 0

        # 为了沿用你现有的可视化，只缓存第一批样本用于出图
        vis_cache = None

        # ====== 新增：进入验证的文字提示 ======
        try:
            num_val_batches = len(self.val_loader)
        except TypeError:
            num_val_batches = -1  # 不可用时给个占位
        print(f"[Validation] start | step={self.step} | num_batches={num_val_batches}")

        with torch.no_grad():
            for inputs in self.val_loader:
                outputs, losses = self.process_batch(inputs)
                if self.collect_debug_metrics:
                    self._accumulate_metrics("val", losses)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                # 累积标量损失
                val = losses["loss"]
                val_float = float(val.detach().cpu().item() if isinstance(val, torch.Tensor) else val)
                total_loss += val_float
                count += 1

                # 只缓存第一批用于可视化，避免大量 I/O
                if vis_cache is None:
                    # 深拷或保留引用均可，这里保留引用即可；后面会及时 del
                    vis_cache = (inputs, outputs, losses)

            # 计算平均验证损失
            avg_val_loss = total_loss / max(count, 1)
            if self.collect_debug_metrics:
                self._log_epoch_metrics("val")

            # —— 记录到 W&B（或 TensorBoard）——
            if self.using_wandb:
                # 1) 标量：使用训练时累积的 step 对齐横轴
                #wandb.log({"val/loss": avg_val_loss}, step=self.step)

                # 2) 可视化：复用你现有的 log()，并将 losses["loss"] 覆盖为 epoch 平均值，避免数值不一致
                if vis_cache is not None:
                    v_inputs, v_outputs, v_losses = vis_cache
                    if isinstance(v_losses.get("loss", None), torch.Tensor):
                        device = v_losses["loss"].device
                        v_losses["loss"] = torch.tensor(avg_val_loss, device=device, dtype=torch.float32)
                    else:
                        v_losses["loss"] = avg_val_loss

                    # 仅记录一批的图像，避免大流量
                    self.log("val", v_inputs, v_outputs, v_losses)

            # 释放缓存，避免占显存/内存
            if vis_cache is not None:
                del v_inputs, v_outputs, v_losses
            del inputs, outputs, losses
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        self.set_train()

    def generate_images_posegt(self, inputs, outputs):
        """Generate posegt pre-warped images using external poses."""
        disp = outputs[("disp", 0)]
        disp = F.interpolate(
            disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        source_scale = 0

        depth = self._disp_to_depth_for_method(disp)

        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                continue

            raw_img = inputs[("color_aug", frame_id, 0)]
            tgt_img = inputs[("color_aug", 0, 0)]
            if frame_id < 0:
                pose_inputs = [raw_img, tgt_img]
            else:
                pose_inputs = [tgt_img, raw_img]

            pose_inputs = [self.models["trans_encoder"](torch.cat(pose_inputs, 1))]
            translation, scale = self.models["trans"](pose_inputs)

            key = ("external_cam_T_cam", 0, frame_id)
            T_prior = inputs.get(key, None)
            if T_prior is None:
                raise KeyError(f"Missing external pose for posegt, frame id={frame_id}")

            T = transformation_from_gtmsrtpose(
                T_prior.to(translation.device), scale[:, 0], translation[:, 0], invert=(frame_id < 0)
            )
            outputs[("posegt_cam_T_cam", 0, frame_id)] = T.detach()

            cam_points = self.backproject_depth[source_scale](
                depth, inputs[("inv_K", source_scale)])
            pix_coords = self.project_3d[source_scale](
                cam_points, inputs[("K", source_scale)], T)

            source_img = inputs[("color", frame_id, 0)]
            source_img_aug = inputs[("color_aug", frame_id, 0)]
            outputs[("irw_img", frame_id, 0)] = F.grid_sample(
                source_img, pix_coords, padding_mode="border", align_corners=True)
            outputs[("irw_img_aug", frame_id, 0)] = F.grid_sample(
                source_img_aug, pix_coords, padding_mode="border", align_corners=True)

    def generate_images_vggt_prewarp(self, inputs, outputs):
        """Generate raw/aug source images pre-warped only by the external VGGT rotation."""
        source_scale = 0
        K = inputs[("K", source_scale)]
        inv_K = inputs[("inv_K", source_scale)]
        height = int(self.opt.height)
        width = int(self.opt.width)

        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                continue

            key = ("external_cam_T_cam", 0, frame_id)
            T_prior = inputs.get(key, None)
            if T_prior is None:
                raise KeyError(f"Missing external pose for VGGT pre-warp, frame id={frame_id}")

            T_prior = T_prior.to(device=K.device, dtype=K.dtype)
            T_prewarp = self._rotation_only_transform(T_prior)
            outputs[("pre_warp_cam_T_cam", 0, frame_id)] = T_prewarp.detach()

            pix_coords = self._rotation_only_warp_grid(K, inv_K, T_prewarp, height=height, width=width)
            outputs[("pre_warp_sample", frame_id, 0)] = pix_coords.detach()

            source_img = inputs[("color", frame_id, 0)]
            source_img_aug = inputs[("color_aug", frame_id, 0)]
            outputs[("pre_warp_img", frame_id, 0)] = F.grid_sample(
                source_img, pix_coords, padding_mode="border", align_corners=True)
            outputs[("pre_warp_img_aug", frame_id, 0)] = F.grid_sample(
                source_img_aug, pix_coords, padding_mode="border", align_corners=True)

            if self.opt.distorted_mask:
                outputs[("pre_warp_valid", frame_id, 0)] = self.get_valid_warp_mask(pix_coords)

    def get_valid_warp_mask(self, pixel_coords):
        x = pixel_coords[..., 0]
        y = pixel_coords[..., 1]

        valid_x = (x >= -1.0) & (x <= 1.0)
        valid_y = (y >= -1.0) & (y <= 1.0)
        valid = valid_x & valid_y  # [B, H, W]

        valid_mask = valid.unsqueeze(1).float()  # [B, 1, H, W]
        return valid_mask

    @staticmethod
    def _grid_to_pixel_coords(grid):
        """Convert normalized grid [-1, 1] to pixel coordinates."""
        h = grid.shape[1]
        w = grid.shape[2]
        x = (grid[..., 0] * 0.5 + 0.5) * float(max(w - 1, 1))
        y = (grid[..., 1] * 0.5 + 0.5) * float(max(h - 1, 1))
        return torch.stack([x, y], dim=-1)

    @staticmethod
    def _target_pixel_grid(batch, h, w, device, dtype):
        """Build target pixel coordinates [B, H, W, 2] as (x, y)."""
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack([xs, ys], dim=-1).unsqueeze(0)
        return grid.expand(batch, -1, -1, -1)

    def _compute_derot_parallax(self, cam_points, K, T_full, source_scale):
        """Compute pixel-wise de-rotation translational parallax magnitude."""
        obs_pack = self._compute_source_observability(cam_points, K, T_full, source_scale)
        return obs_pack["mag"]

    def _compute_source_observability(self, cam_points, K, T_full, source_scale):
        """Compute target-space observability from full-vs-rotation-only sampling displacement."""
        pix_full = self.project_3d[source_scale](cam_points, K, T_full)
        T_rot = T_full.clone()
        T_rot[:, :3, 3] = 0.0
        pix_rot = self.project_3d[source_scale](cam_points, K, T_rot)

        pix_full_px = self._grid_to_pixel_coords(pix_full)
        pix_rot_px = self._grid_to_pixel_coords(pix_rot)
        delta = pix_full_px - pix_rot_px
        mag = torch.linalg.norm(delta, dim=-1, keepdim=False).unsqueeze(1)
        valid_full = self.get_valid_warp_mask(pix_full) > 0.5
        valid_rot = self.get_valid_warp_mask(pix_rot) > 0.5
        valid = valid_full & valid_rot & torch.isfinite(mag)
        return {
            "full_grid": pix_full,
            "rot_grid": pix_rot,
            "mag": mag,
            "valid": valid.float(),
        }

    def _predict_disp_outputs_for_viz(self, image):
        """Run depth net for visualization only (no grad, no BN stat update)."""
        encoder = self.models.get("encoder", None)
        depth_decoder = self.models.get("depth", None)
        if depth_decoder is None:
            depth_decoder = self.models.get("decoder", None)
        if encoder is None or depth_decoder is None:
            return None

        enc_mode = encoder.training
        dec_mode = depth_decoder.training
        try:
            encoder.eval()
            depth_decoder.eval()
            with torch.no_grad():
                features = encoder(image)
                disp_outputs = depth_decoder(features)
        finally:
            encoder.train(enc_mode)
            depth_decoder.train(dec_mode)
        return disp_outputs

    def _compute_target_space_depth_cycle(
        self,
        grid_t2s,
        T_t2s,
        target_depth,
        source_depth_sampled,
        inv_K,
    ):
        """Compute target-space depth consistency maps from sampled source depth."""
        eps = 1e-6
        h = grid_t2s.shape[1]
        w = grid_t2s.shape[2]

        x = (grid_t2s[..., 0] * 0.5 + 0.5) * float(max(w - 1, 1))
        y = (grid_t2s[..., 1] * 0.5 + 0.5) * float(max(h - 1, 1))
        ones = torch.ones_like(x)
        pix_s = torch.stack([x, y, ones], dim=-1)  # [B, H, W, 3]

        inv_K_33 = inv_K[:, :3, :3]
        rays_s = torch.einsum("bij,bhwj->bhwi", inv_K_33, pix_s)
        points_s = rays_s * source_depth_sampled.squeeze(1).unsqueeze(-1)

        T_s2t = torch.linalg.inv(T_t2s)
        R_s2t = T_s2t[:, :3, :3]
        t_s2t = T_s2t[:, :3, 3]
        points_t = torch.einsum("bij,bhwj->bhwi", R_s2t, points_s) + t_s2t.unsqueeze(1).unsqueeze(1)
        depth_t_from_s = points_t[..., 2].unsqueeze(1)

        valid_warp = self.get_valid_warp_mask(grid_t2s) > 0.5
        finite = (
            torch.isfinite(target_depth)
            & torch.isfinite(source_depth_sampled)
            & torch.isfinite(depth_t_from_s)
        )
        positive = (
            (target_depth > eps)
            & (source_depth_sampled > eps)
            & (depth_t_from_s > eps)
        )
        valid = valid_warp & finite & positive
        valid_f = valid.to(dtype=target_depth.dtype)

        abs_diff = torch.abs(depth_t_from_s - target_depth)
        log_diff = torch.abs(
            torch.log(depth_t_from_s.clamp_min(eps)) - torch.log(target_depth.clamp_min(eps))
        )

        return {
            "target_from_source_depth": depth_t_from_s,
            "abs_diff": abs_diff,
            "log_diff": log_diff,
            "abs_diff_masked": abs_diff * valid_f,
            "log_diff_masked": log_diff * valid_f,
            "valid_mask": valid_f,
        }

    @staticmethod
    def _compose_rotation_prior_pose(T_pose, T_prior):
        """Replace predicted rotation with external rotation while keeping predicted translation."""
        if not (torch.is_tensor(T_pose) and torch.is_tensor(T_prior)):
            return None
        if T_pose.ndim != 3 or T_prior.ndim != 3:
            return None
        if T_pose.shape[-2:] != (4, 4) or T_prior.shape[-2:] != (4, 4):
            return None

        T_mix = T_pose.clone()
        T_prior = T_prior.to(device=T_pose.device, dtype=T_pose.dtype)
        T_mix[:, :3, :3] = T_prior[:, :3, :3]
        T_mix[:, 3, :] = T_pose[:, 3, :]
        return T_mix

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        mrfe_align_corners = str(getattr(self.opt, "methods", "")) != "MRFEDepth"
        scale_align_factor = None
        scale_align_ref = getattr(self.scale_aligner, "reference_scale", None) if self.scale_aligner else None
        align_mode = "train" if self.training else "val"
        align_attempted = False
        enable_pose_gating = str(getattr(self.opt, "methods", "")) == "MD2_VGGT_Gated"
        enable_rdistill = bool(getattr(self, "use_rdistill", False))
        enable_rmask_switch = bool(getattr(self, "use_rmask_switch", False))
        enable_badscore_weight = str(getattr(self.opt, "methods", "")) in {
            "MD2_VGGT_PoseGT_BadScoreWeight",
            "MD2_VGGT_PoseGT_BadScoreLocalWeight",
            "MonoViT_VGGT_PoseGT_BadScoreWeight",
        }
        enable_derot_hardmask = bool(getattr(self, "use_derot_hardmask", False))
        enable_derot_sigmoid_weight = bool(getattr(self, "use_derot_sigmoid_weight", False))
        enable_derot_method = bool(enable_derot_hardmask or enable_derot_sigmoid_weight)
        derot_start_epoch = int(getattr(self.opt, "derot_start_epoch", 15))
        derot_thresh_px = float(getattr(self.opt, "derot_thresh_px", 1.0))
        derot_sigmoid_tau = max(float(getattr(self.opt, "derot_sigmoid_tau", 1.0)), 1e-6)
        derot_enabled = bool(enable_derot_method and self.epoch >= derot_start_epoch)
        outputs["derot_enabled"] = float(1.0 if derot_enabled else 0.0)
        outputs["derot_thresh_px"] = float(derot_thresh_px)
        outputs["derot_sigmoid_tau"] = float(derot_sigmoid_tau)
        outputs["r_distill_enabled"] = float(1.0 if enable_rdistill else 0.0)
        outputs["r_mask_switch_enabled"] = float(1.0 if enable_rmask_switch else 0.0)
        teacher_photo_weight = float(getattr(self.opt, "teacher_photo_weight", 0.0))
        enable_teacher_photo = (
            str(getattr(self.opt, "methods", "")) == "MD2_VGGT_Teacher_Photo" and teacher_photo_weight > 0.0
        )
        enable_depth_cycle_viz = bool(getattr(self, "enable_depth_cycle_viz", False))
        depth_cycle_scale = int(getattr(self, "depth_cycle_viz_scale", 0))
        depth_cycle_source_depths = {}
        enable_depth_sensitivity_viz = bool(getattr(self, "enable_depth_sensitivity_viz", False))
        enable_depth_pixshift_viz = bool(getattr(self, "enable_depth_pixshift_viz", False))
        depth_sensitivity_scale = int(getattr(self, "depth_sensitivity_viz_scale", 0))
        depth_sensitivity_factor = float(getattr(self, "depth_sensitivity_factor", 1.1))
        if depth_sensitivity_factor <= 0.0:
            if enable_depth_sensitivity_viz and not self._depth_sensitivity_viz_warned:
                print(
                    f"[depth-sensitivity-viz] depth_sensitivity_factor={depth_sensitivity_factor} 非法，"
                    f"将改为 1.1"
                )
                self._depth_sensitivity_viz_warned = True
            depth_sensitivity_factor = 1.1
            self.depth_sensitivity_factor = depth_sensitivity_factor
        if enable_depth_cycle_viz:
            if depth_cycle_scale not in self.opt.scales:
                if not self._depth_cycle_viz_warned:
                    print(
                        f"[depth-cycle-viz] depth_cycle_viz_scale={depth_cycle_scale} "
                        f"不在 scales={self.opt.scales} 中，已跳过可视化"
                    )
                    self._depth_cycle_viz_warned = True
                enable_depth_cycle_viz = False
            else:
                for frame_id in self.opt.frame_ids[1:]:
                    if frame_id == "s":
                        continue
                    src_aug = inputs.get(("color_aug", frame_id, 0), None)
                    if not torch.is_tensor(src_aug):
                        continue
                    src_disp_outputs = self._predict_disp_outputs_for_viz(src_aug)
                    if not isinstance(src_disp_outputs, dict):
                        if not self._depth_cycle_viz_warned:
                            print("[depth-cycle-viz] 当前方法未检测到 depth/decoder，已跳过可视化")
                            self._depth_cycle_viz_warned = True
                        enable_depth_cycle_viz = False
                        depth_cycle_source_depths = {}
                        break
                    src_disp = src_disp_outputs.get(("disp", depth_cycle_scale), None)
                    if not torch.is_tensor(src_disp):
                        continue
                    src_disp = F.interpolate(
                        src_disp,
                        [self.opt.height, self.opt.width],
                        mode="bilinear",
                        align_corners=False,
                    )
                    src_depth = self._disp_to_depth_for_method(src_disp)
                    depth_cycle_source_depths[frame_id] = src_depth.detach()
        if enable_depth_sensitivity_viz and depth_sensitivity_scale not in self.opt.scales:
            if not self._depth_sensitivity_viz_warned:
                print(
                    f"[depth-sensitivity-viz] depth_sensitivity_viz_scale={depth_sensitivity_scale} "
                    f"不在 scales={self.opt.scales} 中，已跳过可视化"
                )
                self._depth_sensitivity_viz_warned = True
            enable_depth_sensitivity_viz = False
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            depth = self._disp_to_depth_for_method(disp)

            outputs[("depth", 0, scale)] = depth

            depth_for_warp = depth
            if self.scale_aligner is not None:
                align_attempted = True
                allow_compute = (scale_align_factor is not None) or (scale_align_ref is None) or (scale == scale_align_ref)
                if allow_compute:
                    depth_for_warp, candidate_factor = self.scale_aligner.apply(depth, inputs, cached_factor=scale_align_factor)
                    if candidate_factor is not None:
                        scale_align_factor = candidate_factor
                        outputs["scale_align_factor"] = scale_align_factor.view(scale_align_factor.shape[0])
                # 若尚未到参考尺度，则保持原深度参与 warp
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175

                cam_points = self.backproject_depth[source_scale](
                    depth_for_warp, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                if enable_badscore_weight:
                    T_posegt_obs = outputs.get(("posegt_cam_T_cam", 0, frame_id), None)
                    if torch.is_tensor(T_posegt_obs):
                        T_full_obs = torch.matmul(
                            T,
                            T_posegt_obs.to(device=T.device, dtype=T.dtype),
                        )
                    else:
                        T_full_obs = T
                    obs_pack = self._compute_source_observability(
                        cam_points,
                        inputs[("K", source_scale)],
                        T_full_obs,
                        source_scale,
                    )
                    outputs[("badscore_obs", frame_id, scale)] = obs_pack["mag"].detach()
                    outputs[("badscore_obs_valid", frame_id, scale)] = obs_pack["valid"].detach()

                if enable_derot_method and self.use_posegt:
                    T_posegt = outputs.get(("posegt_cam_T_cam", 0, frame_id), None)
                    if torch.is_tensor(T_posegt):
                        T_full = torch.matmul(
                            T,
                            T_posegt.to(device=T.device, dtype=T.dtype),
                        )
                    else:
                        T_full = T
                    derot_map = self._compute_derot_parallax(
                        cam_points, inputs[("K", source_scale)], T_full, source_scale
                    ).detach()
                    derot_mask_raw = (derot_map > derot_thresh_px).float()
                    outputs[("derot_parallax", frame_id, scale)] = derot_map
                    outputs[("derot_mask_raw", frame_id, scale)] = derot_mask_raw
                    if enable_derot_hardmask:
                        if derot_enabled:
                            derot_mask = derot_mask_raw
                        else:
                            derot_mask = torch.ones_like(derot_mask_raw)
                        outputs[("derot_mask", frame_id, scale)] = derot_mask
                    if enable_derot_sigmoid_weight:
                        derot_weight_raw = torch.sigmoid((derot_map - derot_thresh_px) / derot_sigmoid_tau)
                        if derot_enabled:
                            derot_weight = derot_weight_raw
                        else:
                            derot_weight = torch.ones_like(derot_weight_raw)
                        outputs[("derot_weight_raw", frame_id, scale)] = derot_weight_raw
                        outputs[("derot_weight", frame_id, scale)] = derot_weight

                photo_T = T
                photo_uses_composed_transform = False
                if bool(getattr(self, "use_vggt_prewarp", False)):
                    # Keep PoseNet on pre-warped aug images, but supervise photometric loss
                    # with a single raw-source warp under the composed total pose.
                    aux_source_T = self._get_aux_source_transform(outputs, frame_id)
                    if torch.is_tensor(aux_source_T):
                        photo_T = torch.matmul(
                            T,
                            aux_source_T.to(device=T.device, dtype=T.dtype),
                        )
                        pix_coords = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], photo_T
                        )
                        photo_uses_composed_transform = True

                outputs[("sample", frame_id, scale)] = pix_coords

                grid = outputs[("sample", frame_id, scale)]
                if enable_rdistill and scale == 0:
                    key = ("external_cam_T_cam", 0, frame_id)
                    T_prior_rot = inputs.get(key, None)
                    T_rdistill = self._compose_rotation_prior_pose(T, T_prior_rot)
                    if T_rdistill is not None:
                        T_rdistill = T_rdistill.detach()
                        pix_coords_rdistill = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T_rdistill
                        )
                        outputs[("cam_T_cam_rdistill", 0, frame_id)] = T_rdistill
                        outputs[("sample_rdistill", frame_id, scale)] = pix_coords_rdistill
                        outputs[("color_rdistill", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords_rdistill,
                            padding_mode="border",
                            align_corners=True,
                        )
                        if self.opt.distorted_mask:
                            outputs[("distorted_mask_rdistill", frame_id, scale)] = self.get_valid_warp_mask(
                                pix_coords_rdistill
                            )
                if enable_rmask_switch:
                    key = ("external_cam_T_cam", 0, frame_id)
                    T_prior_rot = inputs.get(key, None)
                    T_rswitch = self._compose_rotation_prior_pose(T.detach(), T_prior_rot)
                    if T_rswitch is not None:
                        pix_coords_rswitch = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T_rswitch
                        )
                        outputs[("cam_T_cam_rswitch", 0, frame_id)] = T_rswitch
                        outputs[("sample_rswitch", frame_id, scale)] = pix_coords_rswitch
                        outputs[("color_rswitch", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords_rswitch,
                            padding_mode="border",
                            align_corners=True,
                        )
                        if self.opt.distorted_mask:
                            outputs[("distorted_mask_rswitch", frame_id, scale)] = self.get_valid_warp_mask(
                                pix_coords_rswitch
                            )
                if enable_depth_cycle_viz and scale == depth_cycle_scale:
                    T_cycle = T
                    grid_cycle = grid
                    aux_source_T = self._get_aux_source_transform(outputs, frame_id)
                    if torch.is_tensor(aux_source_T):
                        T_cycle = torch.matmul(
                            T,
                            aux_source_T.to(device=T.device, dtype=T.dtype),
                        )
                        grid_cycle = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T_cycle
                        )
                    src_depth = depth_cycle_source_depths.get(frame_id, None)
                    if torch.is_tensor(src_depth):
                        with torch.no_grad():
                            src_depth_sampled = F.grid_sample(
                                src_depth,
                                grid_cycle,
                                padding_mode="border",
                                align_corners=True,
                            )
                            depth_cycle = self._compute_target_space_depth_cycle(
                                grid_t2s=grid_cycle,
                                T_t2s=T_cycle,
                                target_depth=depth_for_warp,
                                source_depth_sampled=src_depth_sampled,
                                inv_K=inputs[("inv_K", source_scale)],
                            )
                        outputs[("depth_cycle_target_depth", frame_id, scale)] = depth_for_warp.detach()
                        outputs[("depth_cycle_source_depth_sampled", frame_id, scale)] = src_depth_sampled.detach()
                        outputs[("depth_cycle_target_from_source", frame_id, scale)] = (
                            depth_cycle["target_from_source_depth"].detach()
                        )
                        outputs[("depth_cycle_absdiff", frame_id, scale)] = depth_cycle["abs_diff"].detach()
                        outputs[("depth_cycle_logdiff", frame_id, scale)] = depth_cycle["log_diff"].detach()
                        outputs[("depth_cycle_absdiff_masked", frame_id, scale)] = (
                            depth_cycle["abs_diff_masked"].detach()
                        )
                        outputs[("depth_cycle_logdiff_masked", frame_id, scale)] = (
                            depth_cycle["log_diff_masked"].detach()
                        )
                        outputs[("depth_cycle_valid", frame_id, scale)] = depth_cycle["valid_mask"].detach()

                source_img = inputs[("color", frame_id, source_scale)]
                if not photo_uses_composed_transform:
                    aux_source_img = self._get_aux_source_image_for_photo(outputs, frame_id)
                    if torch.is_tensor(aux_source_img):
                        source_img = aux_source_img

                # warped image
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    source_img,
                    grid,
                    padding_mode="border",
                    align_corners=mrfe_align_corners,
                )
                if enable_depth_sensitivity_viz and scale == depth_sensitivity_scale:
                    with torch.no_grad():
                        # 预扭正 source 模式下，敏感性可视化改用“总位姿”：
                        # raw source --(T_aux + T_residual)--> target
                        depth_sens_source_img = source_img
                        depth_sens_T = photo_T
                        depth_sens_grid_base = grid
                        aux_source_T = self._get_aux_source_transform(outputs, frame_id)
                        if (not photo_uses_composed_transform) and torch.is_tensor(aux_source_T):
                            depth_sens_T = torch.matmul(
                                T,
                                aux_source_T.to(device=T.device, dtype=T.dtype),
                            )
                            depth_sens_source_img = inputs[("color", frame_id, source_scale)]
                            depth_sens_grid_base = self.project_3d[source_scale](
                                cam_points,
                                inputs[("K", source_scale)],
                                depth_sens_T,
                            )

                        depth_perturbed = depth_for_warp * depth_sensitivity_factor
                        cam_points_perturbed = self.backproject_depth[source_scale](
                            depth_perturbed,
                            inputs[("inv_K", source_scale)],
                        )
                        grid_perturbed = self.project_3d[source_scale](
                            cam_points_perturbed,
                            inputs[("K", source_scale)],
                            depth_sens_T,
                        )
                        color_base = F.grid_sample(
                            depth_sens_source_img,
                            depth_sens_grid_base,
                            padding_mode="border",
                            align_corners=True,
                        )
                        color_perturbed = F.grid_sample(
                            depth_sens_source_img,
                            grid_perturbed,
                            padding_mode="border",
                            align_corners=True,
                        )
                        target_img = inputs[("color", 0, source_scale)]
                        color_base = color_base.detach()
                        loss_base = self.loss.compute_reprojection_loss(color_base, target_img)
                        loss_perturbed = self.loss.compute_reprojection_loss(color_perturbed, target_img)
                        loss_delta = torch.abs(loss_perturbed - loss_base)
                        loss_rel_delta = loss_delta / (loss_base.abs() + 1e-6)

                        valid_base = self.get_valid_warp_mask(depth_sens_grid_base) > 0.5
                        valid_perturbed = self.get_valid_warp_mask(grid_perturbed) > 0.5
                        valid_joint = valid_base & valid_perturbed
                        valid_joint_f = valid_joint.float()
                        denom = valid_joint_f.sum(dim=(1, 2, 3)).clamp_min(1.0)
                        loss_delta_mean = (
                            (loss_delta * valid_joint_f).sum(dim=(1, 2, 3)) / denom
                        )
                        pix_mag_base = None
                        pix_mag_perturbed = None
                        pix_mag_delta = None
                        pix_mag_rel_delta = None
                        if enable_depth_pixshift_viz:
                            h = depth_sens_grid_base.shape[1]
                            w = depth_sens_grid_base.shape[2]
                            batch = depth_sens_grid_base.shape[0]
                            base_px = self._grid_to_pixel_coords(depth_sens_grid_base)
                            perturbed_px = self._grid_to_pixel_coords(grid_perturbed)
                            target_px = self._target_pixel_grid(
                                batch=batch,
                                h=h,
                                w=w,
                                device=base_px.device,
                                dtype=base_px.dtype,
                            )
                            flow_base = base_px - target_px
                            flow_perturbed = perturbed_px - target_px
                            pix_mag_base = torch.linalg.norm(flow_base, dim=-1).unsqueeze(1)
                            pix_mag_perturbed = torch.linalg.norm(flow_perturbed, dim=-1).unsqueeze(1)
                            # Geometric sensitivity should measure actual sampling-coordinate displacement
                            # caused by depth perturbation, including direction changes.
                            flow_delta = flow_perturbed - flow_base
                            pix_mag_delta = torch.linalg.norm(flow_delta, dim=-1).unsqueeze(1)
                            # Suppress low-motion pixels to avoid unstable relative ratios.
                            pix_base_motion_mask = pix_mag_base >= 1.0
                            pix_mag_rel_delta = torch.where(
                                pix_base_motion_mask,
                                pix_mag_delta / (pix_mag_base + 1e-6),
                                torch.zeros_like(pix_mag_delta),
                            )
                            pix_delta_mean = (
                                (pix_mag_delta * valid_joint_f).sum(dim=(1, 2, 3)) / denom
                            )
                            outputs[("depth_sens_pixmag_delta_mean", frame_id, scale)] = pix_delta_mean.detach()

                        outputs[("depth_sens_sample", frame_id, scale)] = grid_perturbed.detach()
                        outputs[("depth_sens_sample_base", frame_id, scale)] = depth_sens_grid_base.detach()
                        outputs[("depth_sens_color_base", frame_id, scale)] = color_base
                        outputs[("depth_sens_color_perturbed", frame_id, scale)] = color_perturbed.detach()
                        outputs[("depth_sens_loss_base", frame_id, scale)] = (loss_base * valid_joint_f).detach()
                        outputs[("depth_sens_loss_perturbed", frame_id, scale)] = (
                            loss_perturbed * valid_joint_f
                        ).detach()
                        outputs[("depth_sens_loss_delta", frame_id, scale)] = (
                            loss_delta * valid_joint_f
                        ).detach()
                        outputs[("depth_sens_loss_rel_delta", frame_id, scale)] = (
                            loss_rel_delta * valid_joint_f
                        ).detach()
                        if (
                            pix_mag_base is not None
                            and pix_mag_perturbed is not None
                            and pix_mag_delta is not None
                            and pix_mag_rel_delta is not None
                        ):
                            outputs[("depth_sens_pixmag_base", frame_id, scale)] = (
                                pix_mag_base * valid_joint_f
                            ).detach()
                            outputs[("depth_sens_pixmag_perturbed", frame_id, scale)] = (
                                pix_mag_perturbed * valid_joint_f
                            ).detach()
                            outputs[("depth_sens_pixmag_delta", frame_id, scale)] = (
                                pix_mag_delta * valid_joint_f
                            ).detach()
                            outputs[("depth_sens_pixmag_rel_delta", frame_id, scale)] = (
                                pix_mag_rel_delta * valid_joint_f
                            ).detach()
                        outputs[("depth_sens_valid", frame_id, scale)] = valid_joint_f.detach()
                        outputs[("depth_sens_delta_mean", frame_id, scale)] = loss_delta_mean.detach()

                warp_valid = self.get_valid_warp_mask(grid)
                outputs[("warp_valid", frame_id, scale)] = warp_valid.detach()
                # ✅ 加入 distorted mask 控制
                if self.opt.distorted_mask:
                    outputs[("distorted_mask", frame_id, scale)] = warp_valid

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

                if enable_pose_gating:
                    key = ("external_cam_T_cam", 0, frame_id)
                    T_prior = inputs.get(key, None)
                    if torch.is_tensor(T_prior):
                        T_prior = T_prior.to(cam_points.device)
                        pix_coords_prior = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T_prior)
                        outputs[("color_vggt", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords_prior,
                            padding_mode="border",
                            align_corners=True,
                        )
                        if self.opt.distorted_mask:
                            outputs[("distorted_mask_vggt", frame_id, scale)] = self.get_valid_warp_mask(
                                pix_coords_prior)

                if enable_teacher_photo:
                    key = ("external_cam_T_cam", 0, frame_id)
                    T_prior = inputs.get(key, None)
                    T_pose = outputs.get(("cam_T_cam", 0, frame_id), None)
                    if torch.is_tensor(T_prior) and torch.is_tensor(T_pose):
                        T_prior = T_prior.to(cam_points.device)
                        t_prior = T_prior[:, :3, 3]
                        t_norm = torch.linalg.norm(t_prior, dim=1, keepdim=True).clamp_min(1e-6)
                        t_dir = t_prior / t_norm

                        t_mag = torch.linalg.norm(T_pose[:, :3, 3].detach(), dim=1, keepdim=True)
                        t_teacher = t_dir * t_mag

                        T_teacher = T_prior.clone()
                        T_teacher[:, :3, 3] = t_teacher

                        outputs[("cam_T_cam_teacher", 0, frame_id)] = T_teacher
                        pix_coords_teacher = self.project_3d[source_scale](
                            cam_points, inputs[("K", source_scale)], T_teacher)
                        outputs[("color_teacher", frame_id, scale)] = F.grid_sample(
                            inputs[("color", frame_id, source_scale)],
                            pix_coords_teacher,
                            padding_mode="border",
                            align_corners=True,
                        )
                        if self.opt.distorted_mask:
                            outputs[("distorted_mask_teacher", frame_id, scale)] = self.get_valid_warp_mask(
                                pix_coords_teacher)

                        if self.collect_debug_metrics and scale == 0:
                            t_mag_log = t_mag.detach()
                            if t_mag_log.ndim == 2:
                                t_mag_log = t_mag_log.unsqueeze(1)
                            outputs[("pose_mag_teacher", 0, frame_id)] = t_mag_log
                            t_norm_log = t_norm.detach()
                            if t_norm_log.ndim == 2:
                                t_norm_log = t_norm_log.unsqueeze(1)
                            outputs[("pose_prior_norm_teacher", 0, frame_id)] = t_norm_log

        if align_attempted:
            monitor = self._scale_align_monitor.get(align_mode)
            if monitor is not None:
                monitor["attempts"] += 1
                if scale_align_factor is not None:
                    monitor["success"] += 1
                else:
                    if not self._scale_align_warned.get(align_mode, False):
                        print(f"[scale-align][{align_mode}] 未能找到有效的锚点深度用于尺度估计，请检查训练数据的 depth_gt/depth_conf 配置")
                        self._scale_align_warned[align_mode] = True

    def generate_features_pred(self, inputs, outputs):
        mrfe_align_corners = str(getattr(self.opt, "methods", "")) != "MRFEDepth"
        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, [int(self.opt.height / 2), int(self.opt.width / 2)], mode="bilinear",
                             align_corners=False)
        depth = self._disp_to_depth_for_method(disp)
        for i, frame_id in enumerate(self.opt.frame_ids[1:]):  # -1 1
            T = outputs[("cam_T_cam", 0, frame_id)]  # T = (-1，0)  (1，0)

            cam_points = self.backproject_depth[1](depth, inputs[("inv_K", 1)])
            pix_coords = self.project_3d[1](cam_points, inputs[("K", 1)],
                                            T)  # [batch,height,width,2] ==> torch.Size([2, 96, 128 , 2])

            src_f = self.models['FeatureEncoder'](inputs[("color", frame_id, 0)])[0]  # torch.Size([2, 64, 96, 128])

            outputs[("feature", frame_id, 0)] = F.grid_sample(
                src_f, pix_coords, padding_mode="border", align_corners=mrfe_align_corners
            )

        # print('OK_generate_features_pred !! ')

    def compute_depth_losses(self, inputs, outputs, losses):
        return trainer_compute_depth_losses(self, inputs, outputs, losses)

    def _compute_kitti_depth_metrics(self, inputs, outputs, losses):
        return trainer_compute_kitti_depth_metrics(self, inputs, outputs, losses)

    def _compute_uavid_depth_metrics(self, inputs, outputs, losses):
        return trainer_compute_uavid_depth_metrics(self, inputs, outputs, losses)

    def log_time(self, batch_idx, duration, loss):
        return trainer_log_time(self, batch_idx, duration, loss)

    def log(self, mode, inputs, outputs, losses):
        return trainer_log(self, mode, inputs, outputs, losses)

    def save_opts(self):
        return trainer_save_opts(self)

    def save_model(self):
        return trainer_save_model(self)

    def load_model(self):
        return trainer_load_model(self)

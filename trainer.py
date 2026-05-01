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

from trainer_init.model_init import init_models, get_forward_handler
from trainer_init.loss_init import init_losses
from trainer_init.data_init import init_dataloaders
from trainer_init.logging_init import init_logging
from trainer_init.optim_init import init_optimizers
from trainer_init.geometry_init import init_geometry


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
        self.use_pose_net = True

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

        if not self.opt.no_ssim:
            self.ssim = SSIM().to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
        self.collect_debug_metrics = bool(getattr(self.opt, "enable_debug_metrics", False))
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
        method_name = str(getattr(self.opt, "methods", ""))
        use_tinj = method_name == "MonoViT_VGGT_RFlow_TInj"

        if self.num_pose_frames != 2:
            raise NotImplementedError("Only pose_model_input='pairs' is supported.")

        pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

        for f_i in self.opt.frame_ids[1:]:
            if f_i == "s":
                continue

            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_pair = torch.cat(pose_inputs, 1)

            if use_tinj:
                key = ("external_cam_T_cam", 0, f_i)
                base_T = inputs.get(key, None)
                if base_T is None:
                    raise KeyError(f"Missing external pose for rotation-flow method, frame id={f_i}")

                pose_dtype = pose_pair.dtype
                pose_device = pose_pair.device
                K = inputs[("K", 0)].to(device=pose_device, dtype=pose_dtype)
                inv_K = inputs[("inv_K", 0)].to(device=pose_device, dtype=pose_dtype)
                T_prior = base_T.to(device=pose_device, dtype=pose_dtype)
                T_rot = self._rotation_only_transform(T_prior)
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

                t_prior = T_prior[:, :3, 3]
                pose_out = self.models["pose"](pose_inputs, t_prior)
                if not isinstance(pose_out, (tuple, list)) or len(pose_out) != 3:
                    raise RuntimeError("MonoViT_VGGT_RFlow_TInj expects PoseTPriorDecoder output tuple.")
                axisangle, trans_scale, trans_res = pose_out
                outputs[("axisangle", 0, f_i)] = axisangle

                scale = 1.0 + trans_scale[:, 0, 0]
                t_res = trans_res[:, 0, 0]
                t_final = scale * t_prior + t_res

                translation = trans_res.clone()
                translation[:, 0, 0, :] = t_final
                outputs[("translation", 0, f_i)] = translation

                T_prior = T_prior.to(device=axisangle.device, dtype=axisangle.dtype)
                R_delta = rot_from_axisangle(axisangle[:, 0])
                if f_i < 0:
                    R_delta = R_delta.transpose(1, 2)
                R_final = torch.matmul(R_delta[:, :3, :3], T_prior[:, :3, :3])

                T = T_prior.clone()
                T[:, :3, :3] = R_final
                T[:, :3, 3] = t_final
                outputs[("cam_T_cam", 0, f_i)] = T
                continue

            pose_inputs = [self.models["pose_encoder"](pose_pair)]
            axisangle, translation = self.models["pose"](pose_inputs)
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

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

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

            depth = self._disp_to_depth_for_method(disp)

            outputs[("depth", 0, scale)] = depth

            depth_for_warp = depth
            
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175

                cam_points = self.backproject_depth[source_scale](
                    depth_for_warp, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                grid = outputs[("sample", frame_id, scale)]
                source_img = inputs[("color", frame_id, source_scale)]

                # warped image
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    source_img,
                    grid,
                    padding_mode="border",
                    align_corners=True,
                )

                if not self.opt.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

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

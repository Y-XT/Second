# 训练参数与方法说明

本文分为三部分：
1) 参数与指标速查（含默认值/含义）
2) 方法列表（做什么）
3) 各方法的关键参数与实现细节（对照代码实现，避免混淆）

## 1) 参数与指标速查

### 1.1 训练/数据/输入
- `--dataset`：训练数据集类型。
- `--split`：训练/验证划分文件目录（`methods/splits/<split>`）。
- `--data_path`：数据根目录。
- `--triplet_root`：triplet jsonl 根目录（UAVula/UAVid triplet 数据集用）。
- `--height`, `--width`：输入分辨率。
- `--frame_ids`：帧索引（默认 `[0, -1, 1]`）。
- `--scales`：loss 计算的尺度列表（默认 `[0,1,2,3]`）。
- `--min_depth`, `--max_depth`：预测深度范围。
- `--batch_size`, `--learning_rate`, `--num_epochs`：训练超参。
- `--weights_init`：encoder 初始化方式（`pretrained`/`scratch`）。
- `--png`：KITTI png 训练开关。
- `--enable_flip`：训练时随机水平翻转（triplet 会同步外部位姿翻转）。

### 1.2 光度/几何损失相关
- `--disparity_smoothness`：视差平滑损失权重。
- `--no_ssim`：关闭 SSIM，只用 L1。
- `--avg_reprojection`：多帧重投影误差取平均（否则使用逐像素最小）。
- `--disable_automasking`：关闭 automask。
- `--distorted_mask`：使用 warp 边界 mask 过滤无效区域。

### 1.3 VGGT/位姿相关（含 teacher）
- `--pose_prior_scale_init`：`MD2_VGGT_NoPose_TScale` 的全局平移尺度初值。
- `--pose_mag_scale_init`/`--pose_mag_scale_learnable`：`MD2_VGGT_TDir_PoseMag` 的幅值尺度设置。
- `--pose_alpha_mode`/`--pose_alpha_tanh_scale`/`--pose_alpha_exp_scale`：`MD2_VGGT_TPrior_Alpha` 的 alpha 参数化。
- `--pose_alpha_reg_weight`：`MD2_VGGT_TPrior_Alpha` 的 `log(alpha)^2` 正则权重。
- `--pose_align_scale_mode`/`--pose_align_scale_tanh_scale`/`--pose_align_scale_exp_scale`：`MD2_VGGT_TPrior_AlignRes` 的尺度参数化。
- `--pose_align_res_tanh_scale`/`--pose_align_res_scale_by_prior_norm`：`MD2_VGGT_TPrior_AlignRes` 的残差幅值设置。
- `--pose_align_scale_reg_weight`/`--pose_align_res_reg_weight`：`MD2_VGGT_TPrior_AlignRes` 的正则权重。
- `--pose_gating_mode`/`--pose_gating_tau`：`MD2_VGGT_Gated` 的 hard min / softmin 以及温度。

**残差/切换策略**
- `--pose_residual_reg_weight`：ResPose 系列残差正则权重。
- `--pose_residual_scale_start`/`--pose_residual_scale_end`/`--pose_residual_decay_epochs`：ResPose_Decay 的线性衰减配置。
- `--pose_residual_switch_epoch`/`--pose_residual_switch_mode`：PoseToRes 切换 epoch 与残差模式（`rt`/`t`）。

**Pose Teacher（位姿蒸馏）**
- `--pose_teacher_rot_weight`/`--pose_teacher_trans_weight`：旋转/平移方向蒸馏项权重。
- `--pose_teacher_schedule`/`--pose_teacher_schedule_by`/`--pose_teacher_phase0_end`/`--pose_teacher_phase1_end`/`--pose_teacher_w0/w1/w2`：蒸馏阶段调度。
- `--pose_teacher_conf_key`/`--pose_teacher_conf_floor`/`--pose_teacher_conf_thresh`：teacher 置信度权重与硬阈值。
- `--pose_teacher_min_prior_t_norm`/`--pose_teacher_min_pred_t_norm`/`--pose_teacher_max_rot_deg`：跳过异常样本的阈值。

**Teacher photometric**
- `--teacher_photo_weight`：`MD2_VGGT_Teacher_Photo` 的 teacher photometric 权重（默认 `0.1`）。
- `--r_mask_switch_keep_thresh`：`MonoViT_VGGT_RMaskSwitch` 的样本级切换阈值；当 external-R 分支的 automask keep ratio 比 PoseNet 分支高出该值时，整条监督链路切到 external-R 分支（默认 `0.03`）。
- `--www`：PoseGT 系列方法（如 `MD2_VGGT_PoseGT*` / `MonoViT_PoseGT*`）的 posegt photometric 权重（默认 `0.2`）。
- `--posegt_reprojection_mode`：posegt 重投影损失模式（`gasmono`=相对 L1，`md2`=绝对 L1，默认 `md2`）。
- `--derot_start_epoch`：`MD2_VGGT_PoseGT_DeRotHardMask` / `MD2_VGGT_PoseGT_DeRotSigmoidWeight` 的启用 epoch（默认 `15`）。
- `--derot_thresh_px`：de-rotation 像素阈值（hardmask: `P_derot > thresh` 保留；sigmoid: 作为中心阈值，默认 `1.0`）。
- `--derot_sigmoid_tau`：`MD2_VGGT_PoseGT_DeRotSigmoidWeight` 的 sigmoid 温度（默认 `1.0`）。

**DepthSensWeight（`MD2_VGGT_PoseGT_DepthSensWeight` 专属）**
- `--depth_sens_weight_start_epoch`：warmup 截止 epoch（默认 `10`）；小于该 epoch 时不启用敏感性加权。
- `--depth_sens_weight_q_low`/`--depth_sens_weight_q_high`：`Sg/Sp` 的分位归一化区间（默认 `5/95`）。
- `Sg/Sp` 取值来源（当前实现）：权重分支使用 `depth_sens_loss_delta` 与 `depth_sens_pixmag_delta`（不再使用 `*_rel_delta`）。
- `--depth_sens_wpix_scale`：像素内压制强度 `λ_pix`，`Wpix = clip(1 - λ_pix * mismatch, weight_min, 1)`（默认 `0.8`）。
- `--depth_sens_wimg_scale`：目标图像级压制强度 `λ_img`，`Wimg(target) = clip(1 - λ_img * norm_bs(score_target), weight_min, 1)`（默认 `0.6`）。
- `--depth_sens_weight_min`：权重下限 `weight_min`，用于限制像素/帧间压制后的最小权重（默认 `0.2`）。

**BadScoreWeight（`MD2_VGGT_PoseGT_BadScoreWeight` 专属）**
- `--badscore_start_epoch`：warmup 截止 epoch（默认 `10`）；小于该 epoch 时不启用图像级坏监督抑制。
- `--badscore_alpha_r`：残差分数 `r_map` 的 sigmoid 斜率 `alpha_r`（默认 `1.0`）。
- `--badscore_alpha_o`：低可观测性分数 `o_map` 的 sigmoid 斜率 `alpha_o`（默认 `1.0`）。
- `--badscore_wimg_scale`：图像级压制强度 `λ_img`，`W_img = clip(1 - λ_img * sigmoid(B_hat), w_min, 1)`（默认 `0.6`）。
- `--badscore_weight_min`：图像级权重下限 `w_min`（默认 `0.2`）。
- `--badscore_norm_clip`：`r_map/o_map/B_hat` 标准化后的截断范围（默认 `3.0`）。
- 核心流程：先按 source-wise 计算 `E_s` 与 `O_s = ||p_full - p_rot||_2`，再按 winner 拼到 target，得到 `R_map/O_map/K_map`，随后构造 `r_map/o_map/b_map/B_img/W_img`。

**BadScoreLocalWeight（`MD2_VGGT_PoseGT_BadScoreLocalWeight` 专属）**
- 复用 `BadScoreWeight` 的前半段 `R/O/K -> r/o -> q`，其中图像级 `q_map` 仍然等于现有 `b_map`。
- 上一版问题：
  - 图像级已经有 `W_img` 负责“整张图是否可信”，像素级如果再走一套阈值式 `z_loc -> l_map -> l_tilde`，语义会和真正的 per-pixel fragile score 脱节。
  - 阈值式 local 分支依赖 `z_thresh/gamma`，会把像素压制变成 hard-ish 开关，调参成本更高，也更容易出现“同一类坏像素在不同图里忽然压/忽然不压”的不连续现象。
  - 当前实现里的 `r_map/o_map` 已经是标准化后的 batch-shared 分数，同时 automask margin `M` 也天然提供了“这个 keep 到底稳不稳”的信号；因此更合适的做法是直接把 `M_tilde` 乘进无阈值 fragile score，而不是再额外做一套 local 阈值链路。
- 本次修改：
  - 图像级 `W_img` 保持不变，继续基于 `B_img/B_hat` 做 batch 内整图质量压制。
  - 先把 automask margin 标准化成 `M_tilde = (M - mu_M) / (sigma_M + eps)`；其中 `M` 越大表示 keep 越稳，`M` 越小表示 keep 越脆弱。
  - 像素级 fragile score 改成 `fragile_score = sigmoid(alpha_r * r_map) * sigmoid(alpha_o * o_map) * sigmoid(-beta_m * M_tilde)`。
  - 像素权重改成 `W_pix = clip(1 - lambda_pix * fragile_score, w_pix_min, 1)`，最终 `W = W_img * W_pix`。
  - 旧版 `badscore_local_gamma` / `badscore_local_z_thresh` 参数仅为兼容旧命令行保留，当前 no-threshold 公式不再使用。
- `--badscore_start_epoch`：warmup 截止 epoch（默认 `10`）；小于该 epoch 时不启用局部坏监督抑制。
- `--badscore_alpha_r` / `--badscore_alpha_o`：像素级 `fragile_score` 中 `R/O` 两项的 sigmoid 斜率（默认 `1.0/1.0`）。
- `--badscore_beta_m`：margin 项的 sigmoid 斜率 `beta_m`，用于 `sigmoid(-beta_m * M_tilde)`（默认 `1.0`）。
- `--badscore_norm_clip`：`r/o/B_hat` 的稳定性裁剪上限（默认 `3.0`）。
- `--badscore_local_scale`：像素级压制强度 `lambda_pix`（默认 `0.2`）。
- `--badscore_local_weight_min`：像素权重下限 `w_pix_min`（默认 `0.2`）。
- `--badscore_local_gamma` / `--badscore_local_z_thresh`：兼容旧版 CLI 的保留参数，当前 no-threshold `W_pix` 公式忽略。
- 核心流程：先由 `BadScoreWeight` 得到图像级 `W_img` 与标准化后的 `r_map/o_map`，再用 automask margin 构造 `M_tilde`，最后得到 `fragile_score`、`W_pix` 和 `W = W_img * W_pix`。
- 可视化命名约定：W&B 主面板里的 `w_pix` 现在就是实际的 `W_pix`，不再只是旧版 `W_loc` 的别名。

### 1.4 尺度对齐（Scale Align）
- `--scale_align_mode`：`off`/`depth`。
- `--scale_align_anchor_key`/`--scale_align_conf_key`：对齐锚点深度与置信度输入键。
- `--scale_align_min_valid_ratio`/`--scale_align_min_valid_pixels`：最少有效像素阈值。
- `--scale_align_conf_floor`/`--scale_align_eps`：数值稳定相关。
- `--scale_align_scale_min`/`--scale_align_scale_max`：尺度解的范围。
- `--scale_align_reference_scale`：用于估计尺度的 decoder scale 索引。

### 1.5 日志与指标（W&B）
**相关参数**
- `--project_name`：W&B 项目名；会作为 `wandb.init(project=...)` 的值，用于面板归类。
- `--log_dir`：日志根目录；模型权重、可视化输出、（可选）W&B 本地落盘目录都会放在该路径下。
- `--model_name`：实验名；会同时用于保存目录名与 W&B run 名称。设置为 `auto` 会根据配置自动生成；空/未设时也会走自动命名逻辑。
- `--log_frequency`：训练阶段每隔多少 step 记录一次日志；影响 `train/loss`、深度评估与可视化图像的刷新频率。
- `--save_frequency`：每隔多少 epoch 保存一次模型权重；与 W&B 记录频率无关，仅影响 checkpoint 产出频率。
- `--enable_debug_metrics`：开启后按 epoch 汇总 debug 指标（`metrics/*` 前缀）；会增加一定计算与日志开销。

**记录时机**
- 训练：每 `log_frequency` step 记录一次
- 验证：每个 epoch 记录一次（只用首个 batch 出图）
- Debug 指标：仅在 `--enable_debug_metrics` 开启时按 epoch 汇总

**基础指标**
- `train/loss`, `val/loss` — 总损失（多尺度 photometric + disparity smooth + 其他启用分支，如 pose_teacher / teacher_photo；先按尺度平均，再与附加项相加）

**深度评估（需要 depth GT）**
- `train/de/abs_rel`, `val/de/abs_rel` — 平均相对误差 `|pred-gt|/gt`。
- `train/de/sq_rel`, `val/de/sq_rel` — 平均平方相对误差 `(pred-gt)^2/gt`。
- `train/de/rms`, `val/de/rms` — RMSE，`sqrt(mean((pred-gt)^2))`。
- `train/de/log_rms`, `val/de/log_rms` — log RMSE，`sqrt(mean((log pred - log gt)^2))`。
- `train/da/a1`, `val/da/a1` — 准确率：`max(gt/pred, pred/gt) < 1.25` 的比例。
- `train/da/a2`, `val/da/a2` — 同上阈值 `1.25^2`。
- `train/da/a3`, `val/da/a3` — 同上阈值 `1.25^3`。
- `train/metrics/uavid2020/scale_ratio_median`, `val/metrics/uavid2020/scale_ratio_median` — 每张图的中值尺度比（`median(gt)/median(pred)`）的中位数。

**尺度对齐监控（启用 scale align 才有）**
- `train/scale_align_factor_mean`, `val/scale_align_factor_mean` — 当前 batch 内求解到的尺度因子均值。
- `train/scale_align_success_ratio`, `val/scale_align_success_ratio` — 成功求解次数 / 尝试次数。

**Pose Teacher（位姿蒸馏）**
- `train/loss/pose_teacher`, `val/loss/pose_teacher` — 旋转/平移方向蒸馏的加权总和（已乘 rot/trans 权重与阶段权重）。
- `train/loss/pose_teacher_rot`, `val/loss/pose_teacher_rot` — 旋转蒸馏项（SO(3) 角距离，单位弧度，未乘权重）。
- `train/loss/pose_teacher_trans`, `val/loss/pose_teacher_trans` — 平移方向蒸馏项（`1 - cos_sim`，未乘权重）。
- `train/loss/pose_teacher_weight`, `val/loss/pose_teacher_weight` — 蒸馏阶段权重（调度系数）。

**Teacher photometric（MD2_VGGT_Teacher_Photo）**
- `train/loss/teacher_photo`, `val/loss/teacher_photo` — teacher photometric 损失（`raw * weight`）。
- `train/loss/teacher_photo_raw`, `val/loss/teacher_photo_raw` — teacher photometric 原始均值（未乘权重）。
- `train/loss/teacher_photo_weight`, `val/loss/teacher_photo_weight` — teacher photometric 权重（常数）。

**Debug 位姿指标（`--enable_debug_metrics`）**
- `train/pose/t_norm_*`, `val/pose/t_norm_*` — `cam_T_cam` 平移向量范数的统计量（`mean/std/p50/p90/roll_var`）。
- `train/pose/t_over_z_*`, `val/pose/t_over_z_*` — `t_norm / z_mean` 的统计量（`z_mean` 为预测深度 5~95 分位内均值）。
  - 注意（PoseGT 语义）：`MD2_VGGT_PoseGT` 下该指标使用**组合后的全量位姿**统计，`T_full = delta_T @ T_posegt`（`delta_T` 为 irw_img 上 PoseNet 残差，`T_posegt` 为外部 pose + trans 预对齐）。因此反映的是**全量位姿尺度**，而非残差尺度。
- 适用性提醒（`t_norm/t_over_z` 的语义是否“可与基线对比”取决于 `cam_T_cam` 的来源）：
  - 语义基本不变（仍为原始帧间 PoseNet 位姿幅值）：`Monodepth2`、`MD2_VGGT`、`MD2_VGGT_Gated`、`MD2_VGGT_Teacher`、`MD2_VGGT_Teacher_Distill`、`MD2_VGGT_Teacher_Photo`。
  - 语义改变（外部位姿或混合位姿）：`MD2_VGGT_NoPose`、`MD2_VGGT_NoPose_UniformT`、`MD2_VGGT_NoPose_TScale`（外部位姿尺度）、`MD2_VGGT_NoPose_SAlign`（外部位姿 + 对齐深度）、`MD2_VGGT_TDir_PoseMag`（外部方向 + 预测幅值）、`MD2_VGGT_TPrior_Alpha`（外部位姿缩放）、`MD2_VGGT_TPrior_AlignRes`（外部缩放 + 残差）、`MD2_VGGT_ResPose_*` / `MD2_VGGT_ResPose_Decay` / `MD2_VGGT_PoseToRes`（外部位姿与残差组合/切换）、`MD2_VGGT_PoseGT`（外部位姿 + 残差组合后的全量位姿）。
- `train/pose/t_mag_mean`, `val/pose/t_mag_mean` — PoseNet 预测的平移幅值均值（仅 `MD2_VGGT_TDir_PoseMag`）。
- `train/pose/prior_t_norm_mean`, `val/pose/prior_t_norm_mean` — 外部位姿平移向量范数均值。
- `train/pose/t_mag_ratio_mean`, `val/pose/t_mag_ratio_mean` — `t_mag_mean / prior_t_norm_mean` 的比例。
- `train/pose/alpha_mean`, `val/pose/alpha_mean` — `MD2_VGGT_TPrior_Alpha` 的 alpha 均值。
- `train/pose/alpha_std`, `val/pose/alpha_std` — `MD2_VGGT_TPrior_Alpha` 的 alpha 标准差。
- `train/pose/align_scale_mean`, `val/pose/align_scale_mean` — `MD2_VGGT_TPrior_AlignRes` 的尺度因子均值。
- `train/pose/align_scale_std`, `val/pose/align_scale_std` — `MD2_VGGT_TPrior_AlignRes` 的尺度因子标准差。
- `train/pose/align_res_norm_mean`, `val/pose/align_res_norm_mean` — `MD2_VGGT_TPrior_AlignRes` 的残差平移范数均值。
- `train/pose/align_res_norm_std`, `val/pose/align_res_norm_std` — `MD2_VGGT_TPrior_AlignRes` 的残差平移范数标准差。
- `train/pose/align_res_ratio_mean`, `val/pose/align_res_ratio_mean` — 残差范数均值 / `prior_t_norm_mean` 的比例。
**Teacher photometric 的幅值统计（`--enable_debug_metrics`）**
- `train/pose_teacher/t_mag_mean`, `val/pose_teacher/t_mag_mean` — teacher pose 使用的平移幅值均值（来自 PoseNet 幅值，已 detach）。
- `train/pose_teacher/prior_t_norm_mean`, `val/pose_teacher/prior_t_norm_mean` — teacher pose 使用的外部平移向量范数均值。
- `train/pose_teacher/t_mag_ratio_mean`, `val/pose_teacher/t_mag_ratio_mean` — teacher 分支的 `t_mag_mean / prior_t_norm_mean` 比例。

**Debug 光度指标（`--enable_debug_metrics`，仅 scale0）**
- 主 photometric：`train/photometric/*`, `val/photometric/*`；teacher 分支同名字段前缀为 `teacher_photometric/*`。
- 下面这些字段在主/teacher 分支含义一致（按 scale0 统计）：
  - `*_prev_frame_id` / `*_next_frame_id`：被选为最优重投影的前/后帧实际 offset。
  - `*_valid_ratio_prev` / `*_valid_ratio_next`：前/后帧在“所有像素”上的有效比例。
  - `*_min_from_prev_ratio` / `*_min_from_next_ratio`：在“被监督像素”中前/后帧成为最小误差来源的比例。
  - `*_prev_p50/p90`、`*_next_p50/p90`：前/后帧重投影误差的分位数（仅有效像素）。
  - `*_valid_ratio_both`：前后两帧都有效的像素比例。
  - `*_min_p50/p90`：最终用于优化的最小误差分位数。

**Debug automask 指标（`--enable_debug_metrics`，仅 scale0）**
- `train/automask/scale0_keep_ratio`, `val/automask/scale0_keep_ratio` — 通过 automask 最终保留下来的像素比例（kept / valid），越低表示屏蔽越多。
- `train/automask/scale0_high_res_ratio_in_mask`, `val/automask/scale0_high_res_ratio_in_mask` — 被屏蔽的像素里有多少属于“高残差”（high-residual & masked / masked）。
- `train/automask/scale0_bad_keep_ratio`, `val/automask/scale0_bad_keep_ratio` — 高残差像素中仍被保留的比例（high-residual & kept / high-residual）。
- 相关参数：`--automask_hr_percentile`（阈值分位数），`--automask_hr_scope`（阈值统计范围，all 或 mask）。

**Debug final_loss_mask 指标（最终进入 loss 的掩码，scale0）**
- `train/final_loss_mask/scale0_keep_ratio`, `val/final_loss_mask/scale0_keep_ratio` — 最终监督掩码保留比例。
- `train/final_loss_mask/scale0_high_res_ratio_in_mask`, `val/final_loss_mask/scale0_high_res_ratio_in_mask` — 最终屏蔽像素中的高残差比例。
- `train/final_loss_mask/scale0_bad_keep_ratio`, `val/final_loss_mask/scale0_bad_keep_ratio` — 最终高残差仍保留比例。
- 说明：这些就是训练实际进入 loss 的“最终掩码”统计（可视化面板中命名为 `final_loss_mask`）。

**Depth Sensitivity / Weight 指标（可视化相关）**
- `train/depth_sensitivity/delta_mean[_f{fid}]`, `val/*` — depth 扰动前后 photometric 绝对变化均值。
- `train/depth_sensitivity/rel_delta_mean[_f{fid}]`, `val/*` — photometric 相对变化均值。
- `train/depth_sensitivity/rel_delta_mean_after_automask[_f{fid}]`, `val/*` — 仅在 automask 保留区域内的相对变化均值。
- `train/depth_pixshift/delta_mean[_f{fid}]`, `val/*` — depth 扰动导致的 target->source 采样坐标位移变化均值，定义为 `||flow_pert-flow_base||_2`（单位：pixel）。
- `train/depth_pixshift/rel_delta_mean[_f{fid}]`, `val/*` — 上述位移变化相对基准流幅值 `||flow_base||_2` 的均值。
- `train/depth_pixshift/rel_delta_mean_after_automask[_f{fid}]`, `val/*` — automask 保留区域内的位移变化相对均值。
- `train/depth_sens_weight/mean`, `train/depth_sens_weight/p90`（验证同名）— `MD2_VGGT_PoseGT_DepthSensWeight` 的最终像素权重统计。
- `train/depth_sens_weight/scale0_weight_mean`, `train/depth_sens_weight/scale0_weight_p90`（验证同名）— debug 聚合后的 scale0 权重统计（来源 `metrics/depth_sens_weight/*`）。
- `train/depth_sens_weight/scale0_img_weight_mean`, `train/depth_sens_weight/scale0_img_weight_min`（验证同名）— target 级帧间权重（`Wimg(target)`）统计；用于观察是否存在整张 target 被下调现象。
- 可视化面板（W&B）：`depth_sens_weighted_loss_*`，用于对比 `loss_after_mask` 与 `loss_after_mask_weighted`，并附带 `depth_sens_weight` 热力图。

**BadScoreWeight 指标（`MD2_VGGT_PoseGT_BadScoreWeight`，scale0）**
- `train/badscore/scale0_w_img_mean`, `val/badscore/scale0_w_img_mean` — 当前 batch 内 `W_img` 的均值。
- `train/badscore/scale0_w_img_min`, `val/badscore/scale0_w_img_min` — 当前 batch 内最小 `W_img`，用于观察是否有 target 被明显压低。
- `train/badscore/scale0_B_img_mean`, `val/badscore/scale0_B_img_mean` — 图像级坏监督分数 `B_img` 均值。
- `train/badscore/scale0_B_img_max`, `val/badscore/scale0_B_img_max` — 图像级坏监督分数 `B_img` 最大值。
- `train/badscore/scale0_b_map_mean`, `val/badscore/scale0_b_map_mean` — 像素级坏监督分数 `b_map` 在有效区域内的均值。
- 可视化面板（W&B）：`badscore_R_map_*`、`badscore_O_map_*`、`badscore_norm_maps_*`、`badscore_weighted_loss_*`、`badscore_weight_map_*`。

**BadScoreLocalWeight 指标（`MD2_VGGT_PoseGT_BadScoreLocalWeight`，scale0）**
- `metrics/badscorelocal/scale0_weight_mean`：有效像素上的最终权重 `W` 均值。
- `metrics/badscorelocal/scale0_weight_min`：有效像素上的最终权重 `W` 最小值。
- `metrics/badscorelocal/scale0_margin_tilde_mean`：有效像素上的标准化 margin `M_tilde` 均值。
- `metrics/badscorelocal/scale0_margin_gate_mean`：有效像素上的 margin gate `sigmoid(-beta_m * M_tilde)` 均值。
- `metrics/badscorelocal/scale0_fragile_mean`：有效像素上的无阈值 fragile score 均值。
- `metrics/badscorelocal/scale0_w_pix_mean`：有效像素上的像素权重 `W_pix` 均值。
- 可视化面板（W&B）：`badscorelocal_maps_*`、`badscorelocal_score_maps_*`、`badscorelocal_weighted_loss_*`、`badscorelocal_weight_map_*`。
- `badscorelocal_maps_*`：主四联图，固定显示 `w_img / w_pix / w_img*w_pix / loss*img*pix`；其中 `w_img` 会直接在标题里写成 `w_img=0.xxx`。
- `badscorelocal_score_maps_*`：公式拆解图，显示 `M_tilde / margin_gate / fragile_score / W_pix / W`。
- `loss_split_0_0` 最后一张图标题会优先显示 `W_mean=...`；图像级分支仍可通过 `w_img=...` 单独观察。

**DeRot HardMask 指标（`MD2_VGGT_PoseGT_DeRotHardMask`，scale0）**
- 定义：`P_derot(u,v)=||p_full(u,v)-p_rot(u,v)||_2`，其中 `p_full` 为完整位姿投影，`p_rot` 为仅旋转投影。
- `train/derot/scale0_keep_ratio`, `val/derot/scale0_keep_ratio` — `W_derot = 1(P_derot > derot_thresh_px)` 的保留比例。
- `train/derot/scale0_p50`, `val/derot/scale0_p50` — `P_derot` 的 50% 分位数（像素）。
- `train/derot/scale0_p90`, `val/derot/scale0_p90` — `P_derot` 的 90% 分位数（像素）。
- `train/derot/scale0_p95`, `val/derot/scale0_p95` — `P_derot` 的 95% 分位数（像素）。
- 可视化面板（W&B）：`P_derot` 热力图、`W_derot` 二值图、`final_loss_mask` 最终监督二值图（`fid=-1/+1`，默认每次记录第一个样本）。

**DeRot SigmoidWeight 指标（`MD2_VGGT_PoseGT_DeRotSigmoidWeight`，scale0）**
- `train/derot/scale0_keep_ratio`, `val/derot/scale0_keep_ratio` — 与 hardmask 同源的 keep 区域比例参考值。
- `train/derot/scale0_weight_mean`, `val/derot/scale0_weight_mean` — `W_derot` 平均权重（sigmoid 输出）。
- `train/derot/scale0_final_weight_mean`, `val/derot/scale0_final_weight_mean` — 最终 loss 权重均值（已合并 automask/其他权重）。
- `train/derot/scale0_p50/p90/p95`, `val/*` — `P_derot` 分位统计（像素）。

**Debug HRMask 指标（仅 `MD2_VGGT_PoseGT_HRMask` + `--enable_debug_metrics`，scale0）**
- `train/hrmask/scale0_final_keep_ratio`, `val/hrmask/scale0_final_keep_ratio` — 最终参与监督的像素比例（automask 后再去掉高残差）。
- `train/hrmask/scale0_final_bad_keep_ratio`, `val/hrmask/scale0_final_bad_keep_ratio` — 高残差像素中仍被保留的比例（用于确认 HRMask 是否生效）。

**RMaskSwitch 指标（仅 `MonoViT_VGGT_RMaskSwitch`，scale0）**
- `train/r_mask_switch/scale0_route_ratio`, `val/*` — 切到 external-R 整链路监督的样本比例。
- `train/r_mask_switch/scale0_route_valid_ratio`, `val/*` — keep-rate 比较有效的样本比例。
- `train/r_mask_switch/scale0_pose_keep_mean`, `val/*` — PoseNet 分支的 automask keep ratio 样本均值。
- `train/r_mask_switch/scale0_external_keep_mean`, `val/*` — external-R 分支的 automask keep ratio 样本均值。
- `train/r_mask_switch/scale0_delta_keep_mean`, `val/*` — `keep_ext - keep_pose` 的有效样本均值。
- `train/r_mask_switch/scale0_pose_loss_mean`, `val/*` — PoseNet 分支的样本级 photometric loss 均值。
- `train/r_mask_switch/scale0_external_loss_mean`, `val/*` — external-R 分支的样本级 photometric loss 均值。
- 可视化面板（W&B）：`r_mask_switch_panel_*`。
- 面板内容：`target / pose_auto_keep / ext_auto_keep / pose_final_keep / ext_final_keep / selected_keep / route`。
- `pose_auto_keep` 与 `ext_auto_keep` 显示两条并行分支各自的 automask 保留图，当前实现里 `1=保留并进入 photometric`、`0=被 automask 屏蔽`，不是“被屏蔽区域 mask=1”的反向定义。
- `pose_final_keep` / `ext_final_keep` 是在 automask keep 的基础上，再与外部 mask、de-rotation mask、warp valid 等额外 keep 条件求交后的最终保留图；`selected_keep` 是最终真正进入 loss 的 keep 图。
- `route` 颜色约定：绿色表示该样本切到 external-R，蓝色表示继续使用 PoseNet，灰色表示 keep-rate 比较无效；标题里的 `dk` 是 `keep_ext - keep_pose`。

## 2) 方法列表（做什么）

### Monodepth2 系列
- `Monodepth2` — 标准自监督 Monodepth2。
- `MD2_Mask` — Monodepth2 + 外部 mask 与 automask 联合监督。
- `Monodepth2_DINO` — DINOv3 ConvNeXt encoder + UPerDispHead 解码头。
- `LiteMono` — LiteMono 编码器/解码器。
- `MonoViT` — MPViT 编码器/解码器。
- `MonoViT_VGGT_RMaskSwitch` — MonoViT + PoseNet/external-R 双分支监督，按 automask keep rate 做样本级整链路切换。
- `MonoViT_PoseGT` — MonoViT + PoseGT 预对齐闭环。
- `MonoViT_PoseGT_Mask` — MonoViT_PoseGT + 外部 mask。
- `MonoViT_PoseGT_HRMask` — MonoViT_PoseGT + 高残差像素抑制。
- `Madhuanand` — 使用 DepthDecoder_3d + Madhuanand loss。
- `MRFEDepth` — HRNet18 编码器 + MRFEDepthDecoder。

### MD2_VGGT 系列（需要 triplet 数据时会读取外部位姿）
- `MD2_VGGT` — 使用 PoseNet 位姿，外部位姿不参与训练。
- `MD2_VGGT_DepthCycleViz` — `MD2_VGGT` + target-space depth cycle 可视化（仅调试）。
- `MD2_VGGT_DepthSensitivityViz` — `MD2_VGGT` + depth 扰动 photometric 敏感性可视化（仅调试）。
- `MD2_VGGT_NoPose` — 关闭 PoseNet，直接用外部位姿。
- `MD2_VGGT_NoPose_UniformT` — 同 NoPose，但使用均匀 t 采样的 triplets。
- `MD2_VGGT_NoPose_SAlign` — NoPose + 预测深度对齐 vggt_depth 尺度。
- `MD2_VGGT_NoPose_TScale` — NoPose + 全局平移尺度 `pose_scale`。
- `MD2_VGGT_TDir_PoseMag` — 旋转/方向来自外部位姿，平移幅值由 PoseNet 预测。
- `MD2_VGGT_TPrior_Alpha` — 外部平移向量整体乘 alpha（PoseNet 预测）。
- `MD2_VGGT_TPrior_AlignRes` — 外部平移向量先缩放再加残差，旋转做残差修正。
- `MD2_VGGT_RPrior_TPose` — 旋转来自外部位姿，平移来自 PoseNet。
- `MD2_VGGT_ResPose_RT` — 残差位姿（旋转+平移）左乘外部位姿。
- `MD2_VGGT_ResPose_RT_RMul` — 残差位姿右乘外部位姿。
- `MD2_VGGT_ResPose_RT_Reg` — ResPose_RT + 残差正则。
- `MD2_VGGT_ResPose_T` — 仅平移残差。
- `MD2_VGGT_ResPose_T_Reg` — ResPose_T + 残差正则。
- `MD2_VGGT_ResPose_Decay` — ResPose_RT 但残差逐步衰减。
- `MD2_VGGT_PoseToRes` — 前期纯 PoseNet，后期切换为残差。
- `MD2_VGGT_Gated` — PoseNet/VGGT photometric gating。
- `MD2_VGGT_Teacher` — PoseNet 位姿 + 位姿方向蒸馏（非 photometric）。
- `MD2_VGGT_Teacher_Distill` — Teacher 蒸馏带调度和置信度门控。
- `MD2_VGGT_Teacher_Photo` — 主闭环保持 PoseNet，再加 teacher photometric（只训 Depth）。
- `MD2_VGGT_PoseGT` — 外部位姿 + posegt 预先 warp + posegt photometric（`www` 权重）。
- `MD2_VGGT_PoseGT_DepthCycleViz` — `PoseGT` + target-space depth cycle 可视化（仅调试）。
- `MD2_VGGT_PoseGT_DepthSensitivityViz` — `PoseGT` + photometric 敏感性可视化（仅调试）。
- `MD2_VGGT_PoseGT_DepthSensViz` — `PoseGT` + photometric/pixshift 双敏感性可视化（仅调试）。
- `MD2_VGGT_PoseGT_DepthSensWeight` — PoseGT + 深度敏感性加权训练（两张敏感性图参与 loss，支持 warmup）。
- `MD2_VGGT_PoseGT_BadScoreWeight` — PoseGT + target-space residual/observability bad-score 图像级加权。
- `MD2_VGGT_PoseGT_BadScoreLocalWeight` — PoseGT + bad-score 图像级加权与图内局部异常加权联合训练。
- `MD2_VGGT_PoseGT_Mask` — PoseGT + 外部 mask（与 automask 组合）监督。
- `MD2_VGGT_PoseGT_HRMask` — PoseGT 基础上，对“高残差 top 10%”像素做额外 mask（默认阈值 `--posegt_hr_percentile=90`）。
- `MD2_VGGT_PoseGT_DeRotHardMask` — PoseGT 基础上，引入去旋转平移幅值 `P_derot` 的硬门控（`P_derot > --derot_thresh_px`），并与 automask 合并。
- `MD2_VGGT_PoseGT_DeRotSigmoidWeight` — PoseGT 基础上，引入去旋转平移幅值的 sigmoid 连续权重并与最终 loss 权重相乘。

## 3) 方法关键参数与详细说明（对照实现）

### 3.1 Monodepth2 系列

**Monodepth2**
- 实现：`handle_monodepth2` + `Monodepth2Loss`。
- 损失：photometric（SSIM+L1）+ automask + min-reproj + 多尺度 + 平滑项。
- 关键参数：`--disparity_smoothness`, `--no_ssim`, `--avg_reprojection`, `--disable_automasking`。

**Monodepth2_DINO**
- 实现：`handle_monodepth2_dino`。
- 变更点：encoder 替换为 DINOv3 ConvNeXt，多尺度特征喂给 `UPerDispHead`。
- 输入：若存在 `color_norm` 则用归一化输入，否则回退 `color_aug`。
- 其余流程与 Monodepth2 相同。

**LiteMono / MonoViT / Madhuanand / MRFEDepth**
- 均使用各自 encoder/decoder，但光度/平滑损失逻辑与 Monodepth2 基本一致（具体以对应 loss 实现为准）。

**MD2_Mask / MonoViT_VGGT_RMaskSwitch / MonoViT_PoseGT / MonoViT_PoseGT_Mask / MonoViT_PoseGT_HRMask**
- `MD2_Mask`：Monodepth2 基础上叠加外部 mask 与 automask 的联合监督。
- `MonoViT_VGGT_RMaskSwitch`：同时计算 PoseNet warp 与 `external-R + detached pose-t` warp 两条监督链路；先各自跑完整的 `warp -> reprojection -> min rule -> automask`，再按 scale0 的 keep-rate 增益做样本级切换。被切到 external-R 分支的样本只训练 DepthNet，不训练 PoseNet。
- `MonoViT_PoseGT*`：MonoViT 版本的 PoseGT 系列，核心思想与 `MD2_VGGT_PoseGT`/`PoseGT_Mask`/`PoseGT_HRMask` 对齐。

### 3.2 MD2_VGGT 系列

**MD2_VGGT**
- 位姿：完全用 PoseNet 预测（`T = T_pose`）。
- 外部位姿：不参与训练（`use_triplet_pose=False`）。
- 损失：与 Monodepth2 相同。

**MD2_VGGT_DepthCycleViz / MD2_VGGT_DepthSensitivityViz**
- 训练主流程与 `MD2_VGGT` 相同，仅额外开启可视化，不直接改变 loss。
- 常用参数：`--enable_depth_cycle_viz`、`--enable_depth_sensitivity_viz`、`--depth_sensitivity_factor`。

**MD2_VGGT_NoPose**
- 位姿：禁用 PoseNet，直接用 `external_cam_T_cam`。
- 训练：photometric + smooth 不变。
- 依赖：triplet 数据必须提供外部位姿。

**MD2_VGGT_NoPose_UniformT**
- 与 NoPose 相同，但数据集使用 `triplets_uniform_t.jsonl`。

**MD2_VGGT_NoPose_SAlign**
- 与 NoPose 相同，但在重投影前对预测深度做尺度对齐。
- 实现：`scale_align_mode=depth`，anchor 设为 `vggt_depth`，conf 设为 `vggt_conf`。

**MD2_VGGT_NoPose_TScale**
- 与 NoPose 相同，但 `t_prior` 乘全局可学习尺度 `pose_scale`。
- 关键参数：`--pose_prior_scale_init`。

**MD2_VGGT_TDir_PoseMag**
- 位姿：`R = R_prior`，`t = t_dir * t_mag`。
- `t_dir` 来自外部位姿，`t_mag` 由 `PoseMagDecoder` 预测（softplus * 0.01 * scale）。

**MD2_VGGT_PoseGT**
- 位姿：外部位姿 `T_prior` 的旋转保持不变，平移 `t = scale * t_prior + t_res`。
- 训练：先用 posegt 生成 `irw_img`（预对齐图），PoseNet 输入与重投影均优先用 `irw_img`。
- 损失：主 photometric + posegt photometric（`--www`）；posegt 的 L1 模式由 `--posegt_reprojection_mode` 控制。
- 关键参数：`--pose_mag_scale_init`, `--pose_mag_scale_learnable`。

**MD2_VGGT_PoseGT_DepthCycleViz / MD2_VGGT_PoseGT_DepthSensitivityViz / MD2_VGGT_PoseGT_DepthSensViz**
- 训练主流程与 `MD2_VGGT_PoseGT` 相同，仅额外开启可视化。
- 区别：
  - `DepthCycleViz`：target-space depth cycle 可视化。
  - `DepthSensitivityViz`：photometric depth sensitivity 可视化。
  - `DepthSensViz`：photometric + pixshift 双敏感性可视化。
- 常用参数：`--depth_sensitivity_factor`、`--depth_sensitivity_viz_scale`、`--enable_depth_pixshift_viz`。

**MD2_VGGT_PoseGT_DepthSensWeight**
- 基线：继承 `MD2_VGGT_PoseGT_DepthSensViz` 的两张敏感性图（光度 `Sg` 与像素位移 `Sp`）。
- 几何敏感性 `Sp` 变更记录（2026-03-17）：
  - 原问题：旧定义使用 `| ||flow_pert||_2 - ||flow_base||_2 |`，当位移方向变化明显但幅值接近时会低估几何扰动。
  - 修改内容：改为 `||flow_pert - flow_base||_2`（等价于 `||pert_px-base_px||_2`），直接衡量扰动前后采样坐标的实际位移差。
  - 修改原因：几何敏感性应反映采样点位置变化强度，而非仅比较两次流幅值；新定义能同时覆盖方向与幅值变化。
- `Sg/Sp` 定义（当前权重分支）：`Sg <- depth_sens_loss_delta`，`Sp <- depth_sens_pixmag_delta`（两者均为绝对变化量）。
- 核心变更（2026-03-19，target-first 版本）：
  - 旧问题：先按 `fid=±1` 分别计算后再按 winner 选权重，`+1/-1` 覆盖区域差异会造成帧间统计偏置（补充帧容易“少而精”虚高）。
  - 新方案：先按最小重投影 winner 把敏感性拼到 target，再在 target 域分别计算帧内/帧间权重。
  - 目标：帧内只管“target 哪些区域该压”，帧间只管“这张 target 整体是否应下调”，并避免两条归一化链互相污染。
- 权重定义（当前实现）：
  - 步骤1（target拼接）：按 `reproj winner + keep mask` 得到 target 单图 `Sg_tgt/Sp_tgt`。
  - 步骤2（帧内像素权重）：在每张 target 内做分位归一化（q5~q95），`mismatch_intra = Sg_norm_img * (1 - Sp_norm_img)`，`Wpix = clip(1 - λ_pix * mismatch_intra, weight_min, 1)`。
  - 步骤3（帧间target权重）：不使用步骤2的归一化结果；改用 target 原始量经 batch 归一化得到 `mismatch_batch`，再计算每张 target 的 `score_target`，在 batch 内归一化后得到 `Wimg(target) = clip(1 - λ_img * norm_bs(score_target), weight_min, 1)`。
  - 步骤4（最终权重）：`W = Wpix * Wimg(target)`，再乘到 `to_optimise`。
- 本次变更解决的问题：
  - 避免把“帧内归一化后数值”误用于帧间统计，保留跨样本相对量级关系。
  - 把帧间抑制统一收敛到 target 级标量，减少 `+1/-1` 覆盖不均导致的反直觉权重。
- warmup：`epoch < --depth_sens_weight_start_epoch` 时不启用该权重（保持原始 loss）。
- 可视化：除双敏感性图外，额外记录 `depth_sens_weighted_loss_*` 面板，对比 `loss_after_mask` 与 `loss_after_mask_weighted`。
- 关键参数：
  - `--depth_sens_weight_start_epoch`：warmup 结束 epoch（默认 `10`）。
  - `--depth_sens_weight_q_low`/`--depth_sens_weight_q_high`：分位归一化区间（默认 `5/95`）。
  - `--depth_sens_wpix_scale`：像素内压制强度 `λ_pix`（默认 `0.8`）。
  - `--depth_sens_wimg_scale`：target级帧间压制强度 `λ_img`（默认 `0.6`）。
  - `--depth_sens_weight_min`：权重下限 `weight_min`（默认 `0.2`）。
- 潜在未完成项（需后续确认）：
  - `depth_sensitivity_factor=1.5` 的倍率影响尚未在当前若干敏感性统计中做显式消除；跨实验对比时可能引入缩放偏差（建议后续补充按倍率归一化版本指标）。
  - 新权重会改变 photometric 分支的有效 loss 尺度，可能影响与其他 loss 项（如 posegt/teacher/smooth 等）的相对平衡；建议在首轮实验后复查并必要时重调相关权重超参。

**MD2_VGGT_PoseGT_BadScoreWeight**
- 基线：继承 `MD2_VGGT_PoseGT` 的 photometric 主链，不使用 `DepthSensWeight` 的 `Sg/Sp` 双图压制，而是改用 target-space `R/O/K` 流程。
- source-wise 量：
  - `E_s`：每个 source 的 reprojection residual map。
  - `O_s`：每个 source 的 observability，定义为 `||p_full - p_rot||_2`，单位为 pixel。
- target-first winner：
  - `winner = argmin_s E_s`
  - `R_map = min_s E_s`
  - `O_map = gather(O_s, winner)`
  - `K_map = auto-mask keep`
- 归一化分数：
  - `r_map`：对 `R_map` 在 `K_map=1` 的有效像素上做 `log + batch 标准化 + clip`。
  - `o_map`：先对 `O_map` 做同样的 `log + batch 标准化`，再取反，表示“低 observability 程度”。
  - 符号约定：raw `O_map` 是 observability，数值越大表示越可观测、通常越不该压；而 `o_map` 是翻号后的 `low-observability score`，数值越大表示越不可观测、越可疑。
- 坏监督分数：
  - `b_map = K_map * sigmoid(alpha_r * r_map) * sigmoid(alpha_o * o_map)`
  - `B_img = sum(b_map) / (sum(K_map) + eps)`
  - `B_hat = clip((B_img - mu_B) / (sigma_B + eps))`
  - `W_img = clip(1 - lambda * sigmoid(B_hat), w_min, 1)`
- 计算层次：
  - 像素级：`b_map` 表示当前 target 内，每个有效像素“高 residual 且低 observability”的联合坏分数。
  - 图像级：`B_img` 是 `b_map` 在有效像素上的平均值，表示“这张图整体有多坏”。
  - 图像级归一化：`B_hat` 不是单张图自己归一化，而是当前 batch 内所有有效样本的 `B_img` 再做一次标准化，用于帧间比较。
- 跨图像比较：
  - `r_map/o_map` 的标准化统计量 `mu/sigma` 是在当前 batch 的所有有效像素上共享计算，因此它们不是“单张图内自归一化”，而是 batch-shared 分数。
  - 真正用于“这张图比别的图更坏多少”的量是 `B_hat`；`B_hat > 0` 表示该图的 `B_img` 高于当前 batch 平均，`B_hat < 0` 表示低于当前 batch 平均。
  - 当前实现只做 batch 内比较，不做 EMA 或跨 step 的全局运行统计，所以它不是全训练历史上的全局分数。
  - 当 `batch_size=1` 时，`B_hat` 基本会退化到接近 `0`，此时 `W_img` 会接近固定值，图像级比较能力会明显减弱。
- 最终用法：`W_img` 会乘进 `final_loss_weight`，因此 `loss_split` 最后一张图会显示 `w_img=...`。
- 可视化：
  - `badscore_R_map_*`：`R_map` 热力图。
  - `badscore_O_map_*`：`O_map` 热力图。
  - `badscore_norm_maps_*`：归一化后的 `r_map / o_map(low_obs)`；这里的 `o_map` 已经不是 raw observability，而是翻号后的“低可观测性分数”。
  - `badscore_weighted_loss_*`：对比 `loss(mask)` 与 `loss(mask*w_img)`。
  - `badscore_weight_map_*`：当前 target 的 `W_img`。
- 关键参数：
  - `--badscore_start_epoch`：warmup 结束 epoch（默认 `10`）。
  - `--badscore_alpha_r` / `--badscore_alpha_o`：`r_map/o_map` 的 sigmoid 斜率（默认 `1.0/1.0`）。
  - `--badscore_wimg_scale`：图像级压制强度 `lambda`（默认 `0.6`）。
  - `--badscore_weight_min`：权重下限 `w_min`（默认 `0.2`）。
  - `--badscore_norm_clip`：标准化分数裁剪范围（默认 `3.0`）。

**MD2_VGGT_PoseGT_BadScoreLocalWeight**
- 基线：复用 `MD2_VGGT_PoseGT_BadScoreWeight` 的 `R_map/O_map/K_map/r_map/o_map`，保留图像级 `W_img`。
- 上一版问题：
  - 旧版 local 分支仍然走 `z_loc -> l_map -> l_tilde -> W_loc` 的阈值链路，等于把 pixel suppression 建立在“图内相对异常”上，而不是直接建立在当前像素的 fragile 程度上。
  - 这条阈值链路引入了 `z_thresh/gamma` 两个额外超参，使 `W_pix` 的变化更像分段开关，不够平滑，也更难稳定迁移到别的数据分布。
  - 当前实现里的 `r_map/o_map` 已经是对 `log(R)` / `log(O)` 的标准化结果，而 automask margin `M = identity_comp - reproj_comp` 又正好提供了 keep 稳定性；继续走 local 阈值化，会把这个更直接的脆弱性信号浪费掉。
- 本次修改：
  - 保留图像级 `W_img`，继续让它负责 batch 内的整图质量压制。
  - 引入 automask margin 项：先把 `M` 标准化成 `M_tilde = (M - mu_M) / (sigma_M + eps)`。
  - 像素级改成无阈值 fragile score，再由 fragile score 直接映射到 `W_pix`。
  - 旧版 `badscore_local_gamma` / `badscore_local_z_thresh` 仅为兼容旧脚本而保留，当前 no-threshold 实现不再使用。
- 无阈值 fragile score：
  - `M = automask_margin = identity_comp - reproj_comp`
  - `M_tilde = (M - mu_M) / (sigma_M + eps)`
  - `fragile_score = K_map * sigmoid(alpha_r * r_map) * sigmoid(alpha_o * o_map) * sigmoid(-beta_m * M_tilde)`
  - 其中 `M_tilde` 大表示 keep 更稳，因此通过负号 `sigmoid(-beta_m * M_tilde)` 把“小 margin / 脆弱 keep”映射成更高的 fragile score。
- 像素权重：
  - `W_pix = clip(1 - lambda_pix * fragile_score, w_pix_min, 1)`
- 最终权重：
  - `W = W_img * W_pix`
  - 最终乘到 photometric loss 的是 `keep_mask * W`。
- 语义：
  - `W_img` 负责压整张坏图。
  - `fragile_score / W_pix` 负责平滑压低当前图里“高 residual、低 observability、且 automask margin 小”的可疑像素；不再依赖图内 hard threshold。
- 可视化：
  - 共享 `badscore_R_map_*` / `badscore_O_map_*` / `badscore_norm_maps_*` 观察 `R/O/r/o`。
  - `badscorelocal_maps_*`：主四联图，显示 `w_img / w_pix / w_img*w_pix / loss*img*pix`；这里的 `w_pix` 现在就是实际的 `W_pix`。
  - `badscorelocal_score_maps_*`：公式拆解图，显示 `M_tilde / margin_gate / fragile_score / W_pix / W`。
  - `badscorelocal_weighted_loss_*`：对比 `loss(mask)` 与 `loss(mask*w_img*w_pix)`。
  - `badscorelocal_weight_map_*`：最终 `w_img*w_pix` 热力图。
- 关键参数：
  - `--badscore_start_epoch`：warmup 结束 epoch（默认 `10`）。
  - `--badscore_alpha_r` / `--badscore_alpha_o`：`fragile_score` 中 `R/O` 两项的 sigmoid 斜率（默认 `1.0/1.0`）。
  - `--badscore_beta_m`：margin 项的 sigmoid 斜率（默认 `1.0`）。
  - `--badscore_local_scale`：像素级系数 `lambda_pix`（默认 `0.2`）。
  - `--badscore_local_weight_min`：像素权重下限 `w_pix_min`（默认 `0.2`）。
  - `--badscore_local_gamma` / `--badscore_local_z_thresh`：兼容旧版 CLI 的保留参数；当前实现忽略。
  - `--badscore_norm_clip`：`r/o/B_hat` 的裁剪范围（默认 `3.0`）。

**MD2_VGGT_PoseGT_Mask**
- 在 `MD2_VGGT_PoseGT` 基础上引入外部 mask，和 automask 联合决定最终监督区域。
- 常用参数：`--use_external_mask`, `--external_mask_dir`, `--external_mask_ext`, `--external_mask_thresh`。

**MD2_VGGT_PoseGT_HRMask**
- 在 `MD2_VGGT_PoseGT` 基础上抑制高残差像素（按分位阈值过滤），再做 photometric 监督。
- 常用参数：`--posegt_hr_percentile`、`--posegt_hr_scope`。

**MD2_VGGT_PoseGT_DeRotHardMask**
- 基线：继承 `MD2_VGGT_PoseGT` 的两次修正流程（`T_full = delta_T @ T_posegt`）。
- 去旋转平移幅值：在每个像素上计算
  - 完整投影：`p_full = π(K(RX+t))`
  - 纯旋转投影：`p_rot = π(K(RX))`
  - `P_derot = ||p_full - p_rot||_2`（单位：pixel）
- 硬门控：`W_derot = 1(P_derot > --derot_thresh_px)`。
- 启用时机：`epoch < --derot_start_epoch` 时 `W_derot` 置 1（不启用）；达到阈值后启用硬门控。
- 最终监督掩码（`final_loss_mask`）：`W_auto ∩ W_derot ∩ W_ext(optional) ∩ W_distorted(optional)`。
- 常用关注指标：
  - `final_loss_mask/scale0_keep_ratio`（最终保留比例）
  - `derot/scale0_keep_ratio`, `derot/scale0_p50/p90/p95`（门控强度与分布）

**MD2_VGGT_PoseGT_DeRotSigmoidWeight**
- 与 `DeRotHardMask` 共用 `P_derot`，但改用连续权重：
  - `W_derot = sigmoid((P_derot - derot_thresh_px) / derot_sigmoid_tau)`。
- 启用时机：由 `--derot_start_epoch` 控制；未到启用 epoch 时该分支权重退化为 1。
- 最终 loss 权重：`W_final = W_auto * W_derot`。
- 关键参数：`--derot_start_epoch`、`--derot_thresh_px`、`--derot_sigmoid_tau`。

**MD2_VGGT_TPrior_Alpha**
- 位姿：`t = alpha * t_prior`，`R = R_prior`。
- `alpha` 由 `PoseAlphaDecoder` 预测，可 `tanh` 或 `exp` 参数化。
- 关键参数：`--pose_alpha_mode`, `--pose_alpha_tanh_scale`, `--pose_alpha_exp_scale`, `--pose_alpha_reg_weight`。

**MD2_VGGT_TPrior_AlignRes**
- 位姿：`t = s * t_prior + delta_t`，`R = delta_R * R_prior`（平移来自 AlignNet，旋转为残差修正）。
- `s`/`delta_t` 由 `PoseAlignDecoder` 预测，`s` 可 `tanh/exp` 参数化，`delta_t` 走 `tanh` 并可按 `||t_prior||` 缩放。
- 关键参数：`--pose_align_scale_mode`, `--pose_align_scale_tanh_scale`, `--pose_align_scale_exp_scale`, `--pose_align_res_tanh_scale`, `--pose_align_res_scale_by_prior_norm`, `--pose_align_scale_reg_weight`, `--pose_align_res_reg_weight`。

**MD2_VGGT_RPrior_TPose**
- 位姿：`R = R_prior`，平移直接使用 PoseNet 预测（`t = t_pose`）。
- 适用场景：保留外部旋转先验，同时让网络自主学习平移。

**MD2_VGGT_ResPose_RT / RT_RMul / RT_Reg**
- 位姿：PoseNet 预测残差 `delta_T`，与 `T_prior` 组合。
- `RT` 为左乘：`T = delta_T @ T_prior`；`RT_RMul` 为右乘：`T = T_prior @ delta_T`。
- `RT_Reg` 增加残差正则：`--pose_residual_reg_weight`。

**MD2_VGGT_ResPose_T / T_Reg**
- 位姿：仅使用平移残差（旋转残差置零）。
- `T_Reg` 增加残差正则：`--pose_residual_reg_weight`。

**MD2_VGGT_ResPose_Decay**
- 与 ResPose_RT 相同，但残差整体乘线性衰减系数。
- 关键参数：`--pose_residual_scale_start`, `--pose_residual_scale_end`, `--pose_residual_decay_epochs`。

**MD2_VGGT_PoseToRes**
- 前期纯 PoseNet 位姿；到 `switch_epoch` 切换为残差模式。
- 关键参数：`--pose_residual_switch_epoch`, `--pose_residual_switch_mode`。

**MD2_VGGT_Gated**
- 使用 `T_pose` 与 `T_prior` 各自产生 photometric 误差。
- 逐像素取 `min` 或 `softmin` 作为监督。
- 关键参数：`--pose_gating_mode`, `--pose_gating_tau`。

**MD2_VGGT_Teacher**
- 主闭环仍是 `T_pose` photometric。
- 额外加位姿蒸馏：对齐 `R` 与 `t_dir`，不约束平移幅值。
- 该蒸馏只依赖 PoseNet 输出（不涉及 Depth）。
- 关键参数：`--pose_teacher_rot_weight`, `--pose_teacher_trans_weight`。

**MD2_VGGT_Teacher_Distill**
- 与 `MD2_VGGT_Teacher` 相同，但增加调度与置信度过滤。
- 关键参数：`--pose_teacher_schedule*`, `--pose_teacher_conf_*`, `--pose_teacher_min_*`, `--pose_teacher_max_rot_deg`。

**MD2_VGGT_Teacher_Photo**
- 主闭环：仍使用 `T_pose` 做 photometric（更新 Depth+Pose）。
- Teacher photometric：
  - 旋转与平移方向来自外部位姿 `T_prior`。
  - 平移幅值来自 PoseNet 预测 `t_mag`（**detach**，不反传给 PoseNet）。
  - 计算流程与主 photometric 保持一致（automask + min-reproj + 多尺度）。
  - 只更新 DepthNet。
- 关键参数：`--teacher_photo_weight`（默认 `0.1`）。

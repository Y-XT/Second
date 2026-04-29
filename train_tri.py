# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
from argparse import ArgumentParser
from utils import set_global_seed
import sys
import os
# 固定 Qt / Matplotlib 的渲染平台，避免训练节点缺少显示支持导致崩溃。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

# `log_dir` and `model_name` together define where and how the model is saved.
#(1024,576),(512,288)
# remember to change k in dataset
# Madhuanand, Monodepth2, "MRFEDepth","MonoViT","LiteMono"
# madhuanand (lower-case alias) is also supported
#         "UAVid2020": UAVid2020_Dataset,
#         "UAVula_Dataset": UAVula_Dataset,
#         "UAVula_TriDataset": UAVTripletJsonDataset,
# 通过向 sys.argv 追加参数的方式提供常用训练配置，便于快速切换数据集/三元组设置。
# 如果要改用其他配置，直接调整或注释对应块即可。
"""
# UAVid Germany 三元组配置，帧间隔/数据根均与此数据集对应。


sys.argv += [
    "--project_name","UAVid_China",
    "--methods", "MonoViT_PoseGT",  # ✅ W_img + W_pix 联合加权
    "--log_dir", "/mnt/data_nvme3n1p1/mono_weights/weights/UAVid_China",
    "--model_name", "",
    "--split", "UAVid2020_China",
    "--dataset", 'UAVid_TriDataset',
    "--data_path", "/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China",
    "--triplet_root","/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China_tri/tri_win20_disp0.05_lap0_rot10d_U0.5_S0",
    "--batch_size", "8",
    "--learning_rate", "1e-4",
    "--num_workers", "16",
    "--frame_ids", "0", "-1", "1",
    "--height", "288",
    "--width", "512",
    "--num_epochs", "40",
    "--scheduler_step_size", "20",
    "--max_depth", "150.0",
    "--log_frequency","300",
    # LiteMono 仅支持 3 个输出尺度，对应 scales=0/1/2；MD2/MonoViT 等方法通常用 0/1/2/3
    "--scales", "0", "1", "2",

    # posegt 重投影损失模式：gasmono=相对 L1，md2=绝对 L1
    #"--posegt_reprojection_mode", "gasmono",
    # 如果要切回 md2，取消下面注释：
    #"--posegt_reprojection_mode", "md2",

    # DeRot 配置（MD2_VGGT_PoseGT_DeRotHardMask / MD2_VGGT_PoseGT_DeRotSigmoidWeight）
    #"--derot_start_epoch", "10",
    # 用于控制hardmask和sigmoid位置
    #"--derot_thresh_px", "1.0",
    # MD2_VGGT_PoseGT_DeRotSigmoidWeight
    #"--derot_sigmoid_tau", "1.0",

    #DepthSensitivityViz
    #"--depth_sensitivity_factor","1.5",
    #"--depth_sens_weight_start_epoch", "5",

    # BadScore 共享配置（BadScoreWeight / BadScoreLocalWeight 共用）
    # badscore_start_epoch: warmup 截止 epoch；在此之前不启用 BadScore 图像级/局部级压制
    # alpha_r / alpha_o 不是 Local 专属：
    # 1) 图像级 W_img 用它们构造 q_map = sigmoid(alpha_r * r_map) * sigmoid(alpha_o * o_map)
    # 2) 若使用 Local 版本，像素级 fragile_score 也会复用 alpha_r / alpha_o
    # badscore_wimg_scale: 图像级压制强度，控制 W_img = clip(1 - lambda_img * sigmoid(B_hat), w_min, 1)
    # badscore_weight_min: 图像级权重 W_img 的下限
    # badscore_norm_clip: 对 r_map / o_map / B_hat 的标准化结果做裁剪，防止极端值过大
    #"--badscore_start_epoch", "10",
    #"--badscore_alpha_r", "1.0",
    #"--badscore_alpha_o", "1.0",
    #"--badscore_wimg_scale", "0.6",
    #"--badscore_weight_min", "0.2",
    #"--badscore_norm_clip", "3.0",

    # BadScoreLocalWeight 专属配置（在共享配置基础上额外启用）
    # M_tilde = (automask_margin - mu_M) / (sigma_M + eps), 其中 automask_margin 越大 keep 越稳
    # badscore_beta_m: margin 项斜率，控制 sigmoid(-beta_m * M_tilde) 的陡峭程度
    # fragile_score = sigmoid(alpha_r * r_map) * sigmoid(alpha_o * o_map) * sigmoid(-beta_m * M_tilde)
    # badscore_local_scale: 像素级压制强度，控制 W_pix = clip(1 - lambda_pix * fragile_score, w_pix_min, 1)
    # badscore_local_weight_min: 像素权重 W_pix 的下限
    # badscore_local_gamma / badscore_local_z_thresh: 旧版阈值式 local 分支的兼容参数；
    # 当前无阈值 fragile_score -> W_pix 公式不再使用它们
    #"--badscore_beta_m", "1.0",
    #"--badscore_local_scale", "0.6",
    #"--badscore_local_weight_min", "0.2",
    #"--badscore_local_gamma", "1.0",
    #"--badscore_local_z_thresh", "1.5",  # legacy no-op, kept only for backward-compatible CLI

    # automask 可视化：训练时直接生成 identity/reproj/margin 对比图
    "--enable_automask_margin_viz",
    #"--save_automask_margin_viz_local",
    "--automask_margin_viz_samples", "1",
    "--r_distill_margin_thresh", "0.1",
    
    # ✅ Enable this store_true flag:
    "--enable_debug_metrics",
    "--enable_flip",
    #"--distorted_mask"
]


sys.argv += [
    "--project_name","UAVid_Germany",
    "--methods", "MD2_VGGT_PoseGT_Mask",  # ✅ 添加该参数
    "--log_dir", "/mnt/data_nvme3n1p1/mono_weights/weights/UAVid_Germany",
    "--model_name", "",
    "--split", "UAVid2020_Germany",
    "--dataset", 'UAVid_TriDataset',
    "--data_path", "/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany",
    "--triplet_root","/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany_tri/tri_win10_disp0.05_lap0_rot10d_U0.5_S0",
    "--batch_size", "8",
    "--learning_rate", "1e-4",
    "--num_workers", "16",
    "--frame_ids", "0", "-1", "1",
    "--height", "288",
    "--width", "512",
    "--num_epochs", "40",
    "--scheduler_step_size", "20",
    "--max_depth", "150.0",
    "--log_frequency","300",
    "--scales", "0", "1", "2",

    # posegt 重投影损失模式：gasmono=相对 L1，md2=绝对 L1
    #"--posegt_reprojection_mode", "gasmono",
    # 如果要切回 md2，取消下面注释：
    #"--posegt_reprojection_mode", "md2",

    "--enable_debug_metrics",
    "--enable_flip",
    # ✅ Enable this store_true flag:
    #"--distorted_mask"
]

sys.argv += [
    "--project_name","UAVula_R1",
    #"--methods", "MonoViT_VGGT_RFlow_Pose",  # final R,t from RGB pair + VGGT rotation flow
    # 若要切回不注入 t_prior 的单头版本，改成 MonoViT_VGGT_RFlow_ResR_TPose_SingleHead
    "--methods", "MonoViT_VGGT_RFlow_TInj",
    "--log_dir", "/mnt/data_nvme3n1p1/mono_weights/weights/UAVula_R1",
    "--model_name", "",
    "--split", "UAVula_R1",
    "--dataset", 'UAVula_TriDataset',
    "--data_path", "/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_dataset",
    "--triplet_root","/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_tri/tri_win5_disp0.05_lap0_rot10d_U0.5_S0",
    "--batch_size", "8",
    "--learning_rate", "1e-4",
    "--num_workers", "16",
    "--frame_ids", "0", "-1", "1",
    "--height", "288",
    "--width", "512",
    "--num_epochs", "40",
    "--scheduler_step_size", "20",
    "--max_depth", "150.0",
    "--log_frequency","300",
    "--scales", "0", "1", "2", "3",

    # posegt 重投影损失模式：gasmono=相对 L1，md2=绝对 L1
    #"--posegt_reprojection_mode", "gasmono",
    # 如果要切回 md2，取消下面注释：
    #"--posegt_reprojection_mode", "md2",

    # DeRot 配置（MD2_VGGT_PoseGT_DeRotHardMask / MD2_VGGT_PoseGT_DeRotSigmoidWeight）
    #"--derot_start_epoch", "10",
    # 用于控制hardmask和sigmoid位置
    #"--derot_thresh_px", "1.0",
    # MD2_VGGT_PoseGT_DeRotSigmoidWeight
    #"--derot_sigmoid_tau", "1.0",

    #DepthSensitivityViz
    #"--depth_sensitivity_factor","1.5",
    #"--depth_sens_weight_start_epoch", "5",

    # BadScore 共享配置（BadScoreWeight / BadScoreLocalWeight 共用）
    # badscore_start_epoch: warmup 截止 epoch；在此之前不启用 BadScore 图像级/局部级压制
    # alpha_r / alpha_o 不是 Local 专属：
    # 1) 图像级 W_img 用它们构造 q_map = sigmoid(alpha_r * r_map) * sigmoid(alpha_o * o_map)
    # 2) 若使用 Local 版本，像素级 fragile_score 也会复用 alpha_r / alpha_o
    # badscore_wimg_scale: 图像级压制强度，控制 W_img = clip(1 - lambda_img * sigmoid(B_hat), w_min, 1)
    # badscore_weight_min: 图像级权重 W_img 的下限
    # badscore_norm_clip: 对 r_map / o_map / B_hat 的标准化结果做裁剪，防止极端值过大
    #"--badscore_start_epoch", "10",
    #"--badscore_alpha_r", "1.0",
    #"--badscore_alpha_o", "1.0",
    #"--badscore_wimg_scale", "0.6",
    #"--badscore_weight_min", "0.2",
    #"--badscore_norm_clip", "3.0",

    # BadScoreLocalWeight 专属配置（在共享配置基础上额外启用）
    # M_tilde = (automask_margin - mu_M) / (sigma_M + eps), 其中 automask_margin 越大 keep 越稳
    # badscore_beta_m: margin 项斜率，控制 sigmoid(-beta_m * M_tilde) 的陡峭程度
    # fragile_score = sigmoid(alpha_r * r_map) * sigmoid(alpha_o * o_map) * sigmoid(-beta_m * M_tilde)
    # badscore_local_scale: 像素级压制强度，控制 W_pix = clip(1 - lambda_pix * fragile_score, w_pix_min, 1)
    # badscore_local_weight_min: 像素权重 W_pix 的下限
    # badscore_local_gamma / badscore_local_z_thresh: 旧版阈值式 local 分支的兼容参数；
    # 当前无阈值 fragile_score -> W_pix 公式不再使用它们
    #"--badscore_beta_m", "1.0",
    #"--badscore_local_scale", "0.6",
    #"--badscore_local_weight_min", "0.2",
    #"--badscore_local_gamma", "1.0",
    #"--badscore_local_z_thresh", "1.5",  # legacy no-op, kept only for backward-compatible CLI

    # automask 可视化：训练时直接生成 identity/reproj/margin 对比图
    "--enable_automask_margin_viz",
    #"--save_automask_margin_viz_local",
    "--automask_margin_viz_samples", "1",
    #"--r_distill_margin_thresh", "0.1",
    
    # ✅ Enable this store_true flag:
    "--enable_debug_metrics",
    "--enable_flip",
    #"--distorted_mask"
]

"""
#MD2_VGGT_ResPose_RT
#MD2_VGGT_ResPose_RT_Reg
#MD2_VGGT_ResPose_RT_RMul
#MD2_VGGT_ResPose_T
#MD2_VGGT_ResPose_T_Reg
#MD2_VGGT_ResPose_Decay
#MD2_VGGT_TPrior_AlignRes

#MD2_VGGT
#MD2_VGGT_DepthCycleViz
#MD2_VGGT_DepthSensitivityViz
#MD2_VGGT_NoPose
#MD2_VGGT_NoPose_UniformT
#MD2_VGGT_NoPose_SAlign
#MD2_VGGT_NoPose_TScale
#MD2_VGGT_TDir_PoseMag
#MD2_VGGT_TPrior_Alpha
#MD2_VGGT_RPrior_TPose

#MD2_VGGT_Gated
#MD2_VGGT_Teacher
#MD2_VGGT_Teacher_Photo
#MD2_VGGT_Teacher_Distill

#MD2_VGGT_PoseToRes
#MD2_VGGT_PoseGT
#MD2_VGGT_PoseGT_DepthCycleViz
#MD2_VGGT_PoseGT_DepthSensitivityViz
#MD2_VGGT_PoseGT_DepthSensViz
#MD2_VGGT_PoseGT_DepthSensWeight
#MD2_VGGT_PoseGT_BadScoreWeight
#MD2_VGGT_PoseGT_BadScoreLocalWeight
#MD2_VGGT_PoseGT_HRMask
#MD2_VGGT_PoseGT_Mask
#MD2_VGGT_PoseGT_DeRotHardMask
#MD2_VGGT_PoseGT_DeRotSigmoidWeight
#MD2_Mask

#MonoViT
#MonoViT_ResNet50_Pose
#MonoViT_ConvNeXt_Pose
#MonoViT_ConvNeXtSmall_Pose
#MonoViT_ConvNeXtBase_Pose
#MonoViT_VGGT_RDistill
#MonoViT_VGGT_RMaskSwitch
#MonoViT_VGGT_PreWarp
#MonoViT_VGGT_RPrior_ResR_TPose
#MonoViT_VGGT_RFlow_Pose
#MonoViT_VGGT_RFlow_ResR_TPose
#MonoViT_VGGT_RFlow_ResR_TPose_SingleHead
#MonoViT_VGGT_RFlow_TInj
#MonoViT_PoseGT
#MonoViT_PoseGT_Mask
#MonoViT_PoseGT_HRMask
#MonoViT_VGGT_PoseGT_BadScoreWeight

################################################Compare#####################################################
# MD2_VGGT
# - scales: 0 1 2 3
# - status: finish
#
# MonoViT
# - scales: 0 1 2 3
# - status: finish
# - PoseNet backbone: ResNet18 (default)
#
# MonoViT_ResNet50_Pose
# - scales: 0 1 2 3
# - PoseNet backbone: ResNet50
#
# MonoViT_ConvNeXt_Pose
# - scales: 0 1 2 3
# - PoseNet backbone: ConvNeXt-Tiny (timm: convnext_tiny)
#
# MonoViT_ConvNeXtSmall_Pose
# - scales: 0 1 2 3
# - PoseNet backbone: ConvNeXt-Small (timm: convnext_small)
#
# MonoViT_ConvNeXtBase_Pose
# - scales: 0 1 2 3
# - PoseNet backbone: ConvNeXt-Base (timm: convnext_base)
#
# GasMono
# - scales: 0 1 2 3
# - backbone: MPViT
# - decoder: official GasMono decoder
# - default: --iiters 2 --wpp 0.1 --www 0.2
#
# LiteMono
# - scales: 0 1 2
# - variant: --litemono_variant lite-mono-8m
# - pretrained: --litemono_pretrained (encoder.pth or weights folder)
# - status: test is ok, waiting for run
#
# SPIDepth
# - scales: 0
# - note: disp is treated as depth in current framework
#
# madhuanand
# - scales: "0","1","2","3"
#
# MRFEDepth
# - scales: "0","1","2","3"

sys.argv += [
    "--project_name","UAVula_R1",
    "--methods", "MonoViT_VGGT_RFlow_TInj",
    "--log_dir", "/mnt/data_nvme3n1p1/mono_weights/weights/UAVula_R1",
    "--model_name", "",
    "--split", "UAVula_R1",
    "--dataset", 'UAVula_TriDataset',
    "--data_path", "/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_dataset",
    "--triplet_root","/mnt/data_nvme3n1p1/dataset/UAV_ula/R1_tri/tri_win5_disp0.05_lap0_rot10d_U0.5_S0",
    "--batch_size", "8",
    "--learning_rate", "1e-4",
    "--num_workers", "16",
    "--frame_ids", "0", "-1", "1",
    "--height", "288",
    "--width", "512",
    "--num_epochs", "40",
    "--scheduler_step_size", "20",
    "--max_depth", "150.0",
    "--log_frequency","300",
    "--scales", "0","1","2","3",

    # automask 可视化：训练时直接生成 identity/reproj/margin 对比图
    #"--enable_automask_margin_viz",
    #"--save_automask_margin_viz_local",
    #"--automask_margin_viz_samples", "1",
    #"--r_distill_margin_thresh", "0.1",
    
    # ✅ Enable this store_true flag:
    "--enable_debug_metrics",
    "--enable_flip",
    #"--distorted_mask"
]







def main():
    options = MonodepthOptions()
    # 解析命令行参数并生成最终的 model_name。
    opts = options.parse()  # ★ 自动命名在 options.parse() 内完成
    if getattr(opts, "seed", None) is not None and int(opts.seed) >= 0:
        set_global_seed(int(opts.seed), deterministic=True)
        print(f"[INFO] Seed:        {int(opts.seed)} (cudnn deterministic)")
    else:
        print("[INFO] Seed:        disabled")

    # 友好输出与保存目录准备
    #save_dir = os.path.join(opts.log_dir, opts.model_name)
    print(f"[INFO] Project:      {opts.project_name}")
    print(f"[INFO] Method:       {opts.methods}")
    print(f"[INFO] Model name:   {opts.model_name}")
    #print(f"[INFO] Save dir:     {save_dir}")

    # 启动标准训练循环；Trainer 内部已支持三元组数据。
    trainer = Trainer(opts)
    trainer.train()

if __name__ == "__main__":
    main()

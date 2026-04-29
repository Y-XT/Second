# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse


METHOD_DESCRIPTIONS = {
    "Monodepth2": "Baseline self-supervised Monodepth2 pipeline",
    "MD2_Mask": "Monodepth2 + external motion mask (mask+automask keep only)",
    "Madhuanand": "Custom variant with Madhuanand loss",
    "madhuanand": "Alias of Madhuanand (lower-case method name)",
    "MRFEDepth": "HRNet-based encoder-decoder for depth",
    "MonoViT": "Transformer-based MonoViT depth model",
    "MonoViT_ResNet50_Pose": "MonoViT depth model with a ResNet-50 pose encoder",
    "MonoViT_ConvNeXt_Pose": "MonoViT depth model with a ConvNeXt-Tiny pose encoder",
    "MonoViT_ConvNeXtSmall_Pose": "MonoViT depth model with a ConvNeXt-Small pose encoder",
    "MonoViT_ConvNeXtBase_Pose": "MonoViT depth model with a ConvNeXt-Base pose encoder",
    "MonoViT_VGGT_RDistill": "MonoViT + full-image reprojection verification + external-R rotation distillation",
    "MonoViT_VGGT_RMaskSwitch": "MonoViT + sample-wise supervision switch between PoseNet warp and external-R warp (external branch trains depth only)",
    "MonoViT_VGGT_PreWarp": "MonoViT + VGGT-rotation-only pre-warped source image",
    "MonoViT_VGGT_RPrior_ResR_TPose": "MonoViT + VGGT rotation prior + PoseNet residual rotation head + PoseNet translation head",
    "MonoViT_VGGT_RFlow_Pose": "MonoViT + rotation-flow-conditioned PoseNet predicting final pose",
    "MonoViT_VGGT_RFlow_ResR_TPose": "MonoViT + rotation-flow-conditioned PoseNet residual rotation head + final translation head",
    "MonoViT_VGGT_RFlow_ResR_TPose_SingleHead": "MonoViT + rotation-flow-conditioned single-head PoseDecoder, residual rotation on prior + final translation",
    "MonoViT_VGGT_RFlow_TInj": "MonoViT + rotation-flow-conditioned single-head pose decoder with t-prior injection",
    "MonoViT_PoseGT": "MonoViT + PoseGT pre-warp",
    "MonoViT_PoseGT_Mask": "MonoViT + PoseGT pre-warp + external motion mask",
    "MonoViT_PoseGT_HRMask": "MonoViT + PoseGT pre-warp + high-residual masking",
    "MonoViT_VGGT_PoseGT_BadScoreWeight": "MonoViT + PoseGT pre-warp + target-space residual/observability bad-score image weighting",
    "LiteMono": "Lightweight CNN depth network",
    "SPIDepth": "SPIdepth feature encoder + query-transformer depth head under the current training framework",
    "MD2_VGGT": "Monodepth2 + VGGT triplet + PoseNet pose",
    "MD2_VGGT_DepthCycleViz": "MD2_VGGT + target-space depth cycle visualization (debug only)",
    "MD2_VGGT_DepthSensitivityViz": "MD2_VGGT + depth perturbation sensitivity visualization (debug only)",
    "MD2_VGGT_NoPose": "Use pure VGGT pose (no PoseNet) + MD2 depth",
    "MD2_VGGT_NoPose_UniformT": "Same as NoPose but uniform triplet sampling",
    "MD2_VGGT_NoPose_SAlign": "NoPose variant with per-sequence VGGT depth scale alignment",
    "MD2_VGGT_NoPose_TScale": "VGGT pose prior with global translation scale (fixed rotation/direction)",
    "MD2_VGGT_TDir_PoseMag": "VGGT rotation + translation direction with PoseNet magnitude",
    "MD2_VGGT_TPrior_Alpha": "VGGT translation scaled by PoseNet alpha (around 1)",
    "MD2_VGGT_TPrior_AlignRes": "VGGT translation scale+residual (AlignNet) + PoseNet residual rotation",
    "MD2_VGGT_RPrior_TPose": "VGGT rotation prior + PoseNet translation (cam_T_cam)",
    "MD2_VGGT_ResPose_RT": "VGGT pose prior + PoseNet residual (rot+trans)",
    "MD2_VGGT_ResPose_RT_RMul": "ResPose_RT but right-multiply prior (T_prior @ delta_T)",
    "MD2_VGGT_ResPose_RT_Reg": "Residual RT + L2 regularization",
    "MD2_VGGT_ResPose_T": "VGGT pose prior with translation-only residual",
    "MD2_VGGT_ResPose_T_Reg": "Translation residual + L2 regularization",
    "MD2_VGGT_Gated": "PoseNet/VGGT gated photometric supervision (per-pixel best)",
    "MD2_VGGT_Teacher": "PoseNet photometric + pose prior regularization",
    "MD2_VGGT_Teacher_Distill": "PoseNet photometric + scheduled VGGT pose distillation",
    "MD2_VGGT_Teacher_Photo": "PoseNet photometric + teacher photometric (VGGT R/t_dir + PoseNet t_mag)",
    "MD2_VGGT_ResPose_Decay": "VGGT pose prior + PoseNet residual with linear decay",
    "MD2_VGGT_PoseToRes": "PoseNet pose first, then switch to VGGT residual",
    "MD2_VGGT_PoseGT": "VGGT pose prior + posegt pre-warp (translation residual + scale)",
    "MD2_VGGT_PoseGT_DepthCycleViz": "PoseGT + target-space depth cycle visualization (debug only)",
    "MD2_VGGT_PoseGT_DepthSensitivityViz": "PoseGT + depth perturbation photometric sensitivity visualization (debug only)",
    "MD2_VGGT_PoseGT_DepthSensViz": "PoseGT + depth perturbation sensitivity visualization (photometric + pixel-shift, debug only)",
    "MD2_VGGT_PoseGT_DepthSensWeight": "PoseGT + depth sensitivity driven photometric reweighting (warmup + pixel/frame weight)",
    "MD2_VGGT_PoseGT_BadScoreWeight": "PoseGT + target-space residual/observability bad-score image weighting",
    "MD2_VGGT_PoseGT_BadScoreLocalWeight": "PoseGT + bad-score image weighting with within-image local anomaly refinement",
    "MD2_VGGT_PoseGT_HRMask": "PoseGT + high-residual masking (drop top percentile pixels)",
    "MD2_VGGT_PoseGT_Mask": "PoseGT + external motion mask (mask+automask keep only)",
    "MD2_VGGT_PoseGT_DeRotHardMask": "PoseGT + de-rotation parallax hard mask (pixel threshold)",
    "MD2_VGGT_PoseGT_DeRotSigmoidWeight": "PoseGT + de-rotation sigmoid pixel weighting",
    "GasMono": "GasMono fixed baseline: MPViT depth backbone + PoseGT pre-warp + iterative self-distillation (selfpp)",
    "Monodepth2_DINO": "Monodepth2 decoder fed by DINOv3 encoder"
}

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

DEFAULT_MODELS_TO_LOAD = [
    "encoder", "depth", "decoder",
    "pose_encoder", "pose", "pose_r", "pose_t",
    "pose_align", "pose_scale",
    "trans_encoder", "trans",
]

MRFE_MODELS_TO_LOAD = [
    "DepthEncoder", "DepthDecoder",
    "FeatureEncoder", "FeatureDecoder",
    "pose_encoder", "pose",
]


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        #METHODS
        self.parser.add_argument("--methods",
                                 type=str,
                                 help="methods for training",
                                 choices=[
                                     "Monodepth2", "MD2_Mask", "Madhuanand", "MRFEDepth",
                                     "madhuanand",
                                     "MonoViT", "MonoViT_ResNet50_Pose", "MonoViT_ConvNeXt_Pose",
                                     "MonoViT_ConvNeXtSmall_Pose", "MonoViT_ConvNeXtBase_Pose", "MonoViT_VGGT_RDistill",
                                     "MonoViT_VGGT_RMaskSwitch",
                                     "MonoViT_VGGT_PreWarp",
                                     "MonoViT_VGGT_RPrior_ResR_TPose",
                                     "MonoViT_VGGT_RFlow_Pose",
                                     "MonoViT_VGGT_RFlow_ResR_TPose",
                                     "MonoViT_VGGT_RFlow_ResR_TPose_SingleHead",
                                     "MonoViT_VGGT_RFlow_TInj",
                                     "MonoViT_PoseGT", "MonoViT_PoseGT_Mask",
                                     "MonoViT_PoseGT_HRMask", "MonoViT_VGGT_PoseGT_BadScoreWeight",
                                     "LiteMono", "SPIDepth",
                                     "MD2_VGGT", "MD2_VGGT_DepthCycleViz",
                                     "MD2_VGGT_DepthSensitivityViz",
                                     "MD2_VGGT_NoPose", "MD2_VGGT_NoPose_UniformT",
                                     "MD2_VGGT_NoPose_SAlign", "MD2_VGGT_NoPose_TScale",
                                     "MD2_VGGT_TDir_PoseMag", "MD2_VGGT_TPrior_Alpha",
                                     "MD2_VGGT_TPrior_AlignRes",
                                     "MD2_VGGT_RPrior_TPose",
                                     "MD2_VGGT_ResPose_RT", "MD2_VGGT_ResPose_RT_RMul", "MD2_VGGT_ResPose_RT_Reg",
                                     "MD2_VGGT_ResPose_T", "MD2_VGGT_ResPose_T_Reg",
                                     "MD2_VGGT_Gated", "MD2_VGGT_Teacher", "MD2_VGGT_Teacher_Distill",
                                     "MD2_VGGT_Teacher_Photo",
                                     "MD2_VGGT_ResPose_Decay", "MD2_VGGT_PoseToRes",
                                     "MD2_VGGT_PoseGT", "MD2_VGGT_PoseGT_DepthCycleViz",
                                     "MD2_VGGT_PoseGT_DepthSensitivityViz",
                                     "MD2_VGGT_PoseGT_DepthSensViz",
                                     "MD2_VGGT_PoseGT_DepthSensWeight",
                                     "MD2_VGGT_PoseGT_BadScoreWeight",
                                     "MD2_VGGT_PoseGT_BadScoreLocalWeight",
                                     "MD2_VGGT_PoseGT_HRMask", "MD2_VGGT_PoseGT_Mask",
                                     "MD2_VGGT_PoseGT_DeRotHardMask",
                                     "MD2_VGGT_PoseGT_DeRotSigmoidWeight",
                                     "GasMono", "gasmono",
                                     "Monodepth2_DINO"
                                 ],
                                 default="Monodepth2")


        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument(
            "--describe_methods",
            action="store_true",
            help="print available methods and descriptions, then exit"
        )
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--project_name",
                                 type=str,
                                 help="the name of the project",
                                 default="UAVula")
        self.parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="random seed for reproducibility; set <0 to disable seeding"
        )

        # 1) __init__ 里：定义/覆盖 --model_name（默认 None）
        self.parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="模型保存目录名；留空或设为 'auto' 则按约定字段自动生成"
        )

        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["UAVid2020_China","UAVid2020_Germany","UAVula","UAVula_R1","wilduav"],
                                 default="UAVula")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="UAVid2020",
                                 choices=["UAVid2020","UAVula_Dataset","UAVula_TriDataset","UAVid_TriDataset","WildUAV"]) 
        # UAVula TriDataset 额外参数
        self.parser.add_argument(
            "--triplet_root",
            type=str,
            default="/mnt/data_nvme3n1p1/dataset/UAV_ula/uav_triplets",
            help="阶段二输出根目录（包含 <seq>/triplets.jsonl）"
        )
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=512)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--use_exponential_lr",
                                 action='store_true',
                                 help="Use ExponentialLR instead of StepLR")
        self.parser.add_argument("--drop_path",
                                 type=float,
                                 help="drop path rate",
                                 default=0.2)
        # ABLATION options
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument(
            "--litemono_variant",
            type=str,
            default="lite-mono-8m",
            choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"],
            help="LiteMono encoder variant"
        )
        self.parser.add_argument(
            "--litemono_pretrained",
            type=str,
            default="/mnt/data_nvme3n1p1/PycharmProjects/monodepth2/ckpt",
            help="optional LiteMono pretrained path; accepts encoder checkpoint file or a folder containing encoder.pth/depth.pth"
        )
        self.parser.add_argument(
            "--spidepth_backbone",
            type=str,
            default="convnext_large",
            choices=["resnet", "resnet_lite", "resnet18_lite", "convnextv2_huge", "convnext_large"],
            help="SPIdepth depth feature backbone variant"
        )
        self.parser.add_argument(
            "--spidepth_num_layers",
            type=int,
            default=50,
            choices=[18, 34, 50, 101, 152],
            help="SPIdepth ResNet depth encoder layers for resnet/resnet_lite variants"
        )
        self.parser.add_argument(
            "--spidepth_num_features",
            type=int,
            default=512,
            help="SPIdepth decoder bottleneck width for the depth feature encoder"
        )
        self.parser.add_argument(
            "--spidepth_model_dim",
            type=int,
            default=32,
            help="SPIdepth dense feature channel count fed into the query-transformer head"
        )
        self.parser.add_argument(
            "--spidepth_patch_size",
            type=int,
            default=20,
            help="SPIdepth patch size used before the transformer encoder"
        )
        self.parser.add_argument(
            "--spidepth_query_nums",
            type=int,
            default=32,
            help="SPIdepth number of learned queries"
        )
        self.parser.add_argument(
            "--spidepth_dim_out",
            type=int,
            default=64,
            help="SPIdepth adaptive bin count"
        )
        self.parser.add_argument(
            "--spidepth_dec_channels",
            nargs="+",
            type=int,
            default=[1024, 512, 256, 128],
            help="SPIdepth Unet decoder channels for convnext-style backbones"
        )
        self.parser.add_argument(
            "--spidepth_pose_backbone",
            type=str,
            default="resnet18",
            help="SPIdepth PoseCNN timm backbone name"
        )
        self.parser.add_argument(
            "--spidepth_pretrained",
            type=str,
            default="",
            help="optional SPIdepth pretrained source; accepts a weights folder containing encoder.pth/depth.pth/pose.pth or a single checkpoint file"
        )
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        self.parser.add_argument("--distorted_mask",
                                 help="Use warp boundary mask to ignore padded regions",
                                 action="store_true")
        self.parser.add_argument(
            "--pose_residual_reg_weight",
            type=float,
            default=1e-3,
            help="L2 regularization weight for Pose residual methods (ignored otherwise)"
        )
        self.parser.add_argument(
            "--pose_residual_scale_start",
            type=float,
            default=1.0,
            help="start scale for Pose residual decay (1.0 keeps full residual)"
        )
        self.parser.add_argument(
            "--pose_residual_scale_end",
            type=float,
            default=0.0,
            help="end scale for Pose residual decay (0.0 becomes pure prior)"
        )
        self.parser.add_argument(
            "--pose_residual_decay_epochs",
            type=int,
            default=0,
            help="epochs to decay residual scale (0 uses num_epochs)"
        )
        self.parser.add_argument(
            "--pose_residual_switch_epoch",
            type=int,
            default=-1,
            help="epoch to switch from PoseNet to residual mode (-1 uses num_epochs//2)"
        )
        self.parser.add_argument(
            "--pose_residual_switch_mode",
            type=str,
            default="rt",
            choices=["rt", "t"],
            help="residual mode after switching (rt or t)"
        )
        self.parser.add_argument(
            "--pose_prior_scale_init",
            type=float,
            default=1.0,
            help="initial global scale for prior translation (MD2_VGGT_NoPose_TScale)"
        )
        self.parser.add_argument(
            "--pose_mag_scale_init",
            type=float,
            default=1.0,
            help="initial scale for PoseMagDecoder output (MD2_VGGT_TDir_PoseMag)"
        )
        self.parser.add_argument(
            "--pose_mag_scale_learnable",
            action="store_true",
            help="learn a global scale for PoseMagDecoder output (MD2_VGGT_TDir_PoseMag)"
        )
        self.parser.add_argument(
            "--pose_alpha_mode",
            type=str,
            default="tanh",
            choices=["tanh", "exp"],
            help="alpha parameterization (MD2_VGGT_TPrior_Alpha)"
        )
        self.parser.add_argument(
            "--pose_alpha_tanh_scale",
            type=float,
            default=0.1,
            help="alpha scale for tanh mode (MD2_VGGT_TPrior_Alpha)"
        )
        self.parser.add_argument(
            "--pose_alpha_exp_scale",
            type=float,
            default=0.01,
            help="alpha scale for exp mode (MD2_VGGT_TPrior_Alpha)"
        )
        self.parser.add_argument(
            "--pose_alpha_reg_weight",
            type=float,
            default=1e-3,
            help="regularization weight for log(alpha)^2 (MD2_VGGT_TPrior_Alpha)"
        )
        self.parser.add_argument(
            "--pose_align_scale_mode",
            type=str,
            default="tanh",
            choices=["tanh", "exp"],
            help="scale parameterization (MD2_VGGT_TPrior_AlignRes)"
        )
        self.parser.add_argument(
            "--pose_align_scale_tanh_scale",
            type=float,
            default=0.1,
            help="scale tanh factor (MD2_VGGT_TPrior_AlignRes)"
        )
        self.parser.add_argument(
            "--pose_align_scale_exp_scale",
            type=float,
            default=0.01,
            help="scale exp factor (MD2_VGGT_TPrior_AlignRes)"
        )
        self.parser.add_argument(
            "--pose_align_res_tanh_scale",
            type=float,
            default=0.01,
            help="residual tanh factor (MD2_VGGT_TPrior_AlignRes)"
        )
        self.parser.add_argument(
            "--pose_align_res_scale_by_prior_norm",
            action="store_true",
            help="scale residual by ||t_prior|| (MD2_VGGT_TPrior_AlignRes)"
        )
        self.parser.add_argument(
            "--pose_align_scale_reg_weight",
            type=float,
            default=0.0,
            help="regularization weight for log(scale)^2 (MD2_VGGT_TPrior_AlignRes)"
        )
        self.parser.add_argument(
            "--pose_align_res_reg_weight",
            type=float,
            default=0.0,
            help="regularization weight for ||residual||^2 (MD2_VGGT_TPrior_AlignRes)"
        )
        self.parser.add_argument(
            "--pose_gating_mode",
            type=str,
            default="min",
            choices=["min", "softmin"],
            help="gating mode for PoseNet/VGGT photometric supervision"
        )
        self.parser.add_argument(
            "--pose_gating_tau",
            type=float,
            default=0.1,
            help="softmin temperature for pose gating (smaller = sharper)"
        )
        self.parser.add_argument(
            "--pose_teacher_rot_weight",
            type=float,
            default=1.0,
            help="rotation prior weight for pose teacher regularization"
        )
        self.parser.add_argument(
            "--pose_teacher_trans_weight",
            type=float,
            default=1.0,
            help="translation direction prior weight for pose teacher regularization"
        )
        self.parser.add_argument(
            "--pose_teacher_schedule",
            type=str,
            default="off",
            choices=["off", "linear", "cosine"],
            help="pose teacher distillation schedule (off/linear/cosine for Phase1)"
        )
        self.parser.add_argument(
            "--pose_teacher_schedule_by",
            type=str,
            default="epoch",
            choices=["epoch", "step"],
            help="schedule progress unit for pose teacher distillation"
        )
        self.parser.add_argument(
            "--pose_teacher_phase0_end",
            type=int,
            default=-1,
            help="phase0 end (epoch or step; <0 disables schedule)"
        )
        self.parser.add_argument(
            "--pose_teacher_phase1_end",
            type=int,
            default=-1,
            help="phase1 end (epoch or step; <=0 disables schedule)"
        )
        self.parser.add_argument(
            "--pose_teacher_w0",
            type=float,
            default=1.0,
            help="distillation weight for phase0"
        )
        self.parser.add_argument(
            "--pose_teacher_w1",
            type=float,
            default=1.0,
            help="distillation weight at the start of phase1"
        )
        self.parser.add_argument(
            "--pose_teacher_w2",
            type=float,
            default=0.0,
            help="distillation weight after phase1"
        )
        self.parser.add_argument(
            "--pose_teacher_conf_key",
            type=str,
            default="vggt_conf",
            help="inputs key for teacher confidence map"
        )
        self.parser.add_argument(
            "--pose_teacher_conf_floor",
            type=float,
            default=0.0,
            help="confidence floor for soft distillation weighting"
        )
        self.parser.add_argument(
            "--pose_teacher_conf_thresh",
            type=float,
            default=0.0,
            help="hard reject threshold for teacher confidence"
        )
        self.parser.add_argument(
            "--pose_teacher_min_prior_t_norm",
            type=float,
            default=1e-6,
            help="skip distillation when ||t_prior|| is too small"
        )
        self.parser.add_argument(
            "--pose_teacher_min_pred_t_norm",
            type=float,
            default=1e-6,
            help="skip translation direction distillation when ||t_pred|| is too small"
        )
        self.parser.add_argument(
            "--pose_teacher_max_rot_deg",
            type=float,
            default=0.0,
            help="skip distillation when rotation error exceeds this threshold (degrees; 0 disables)"
        )
        self.parser.add_argument(
            "--teacher_photo_weight",
            type=float,
            default=0.1,
            help="weight for teacher photometric loss (MD2_VGGT_Teacher_Photo)"
        )
        self.parser.add_argument(
            "--r_distill_loss_weight",
            type=float,
            default=0.02,
            help="weight for reprojection-gated rotation distillation (MonoViT_VGGT_RDistill)"
        )
        self.parser.add_argument(
            "--r_distill_warmup_epochs",
            type=int,
            default=5,
            help="start epoch for image-gated rotation distillation"
        )
        self.parser.add_argument(
            "--r_distill_margin_thresh",
            type=float,
            default=0.10,
            help="reference threshold for full-image margin mean logging in R-distill"
        )
        self.parser.add_argument(
            "--r_distill_delta_rel_min",
            type=float,
            default=0.02,
            help="minimum relative photometric improvement to activate R-distill gate"
        )
        self.parser.add_argument(
            "--r_distill_delta_rel_max",
            type=float,
            default=0.10,
            help="relative photometric improvement mapped to full R-distill gate"
        )
        self.parser.add_argument(
            "--r_mask_switch_keep_thresh",
            type=float,
            default=0.03,
            help="minimum automask keep-ratio gain required to switch a sample to external-R warp supervision"
        )
        self.parser.add_argument(
            "--www",
            type=float,
            default=0.2,
            help="weight for posegt photometric loss (PoseGT-based methods)"
        )
        self.parser.add_argument(
            "--iiters",
            type=int,
            default=2,
            help="number of inner optimization iterations per batch (GasMono-style)"
        )
        self.parser.add_argument(
            "--wpp",
            type=float,
            default=0.1,
            help="self-distillation (pseudo label) loss weight (GasMono-style)"
        )
        self.parser.add_argument(
            "--gasmono_selfpp_weight",
            type=float,
            default=0.1,
            help="legacy alias of --wpp for GasMono iterative self-distillation loss"
        )
        self.parser.add_argument(
            "--gasmono_iiters",
            type=int,
            default=2,
            help="legacy alias of --iiters for GasMono inner optimization iterations"
        )
        self.parser.add_argument(
            "--gasmono_iiter_start_epoch",
            type=int,
            default=10,
            help="epoch to start multi-iteration per batch in GasMono"
        )
        self.parser.add_argument(
            "--gasmono_selfpp_warmup_epochs",
            type=int,
            default=20,
            help="linear warmup epochs for GasMono selfpp weight"
        )
        self.parser.add_argument(
            "--posegt_reprojection_mode",
            type=str,
            default="md2",
            choices=["gasmono", "md2"],
            help="posegt reprojection loss mode (gasmono=relative L1, md2=absolute L1)"
        )
        self.parser.add_argument(
            "--posegt_hr_percentile",
            type=float,
            default=90.0,
            help="PoseGT_HRMask 中高残差阈值分位数（如 90 表示 top 10%）"
        )
        self.parser.add_argument(
            "--posegt_hr_scope",
            type=str,
            default="mask",
            choices=["all", "mask"],
            help="PoseGT_HRMask 的阈值统计范围：all=全像素，mask=仅 keep 像素"
        )
        self.parser.add_argument(
            "--derot_start_epoch",
            type=int,
            default=15,
            help="epoch >= this value to enable de-rotation masking/weighting"
        )
        self.parser.add_argument(
            "--derot_thresh_px",
            type=float,
            default=1.0,
            help="de-rotation center threshold in pixel (used by hardmask/sigmoid)"
        )
        self.parser.add_argument(
            "--derot_sigmoid_tau",
            type=float,
            default=1.0,
            help="temperature tau for DeRot sigmoid weighting: sigmoid((P_derot-thresh)/tau)"
        )
        self.parser.add_argument(
            "--enable_debug_metrics",
            action="store_true",
            help="开启额外的 photometric/pose 统计指标（默认关闭，节省开销）"
        )
        self.parser.add_argument(
            "--enable_depth_cycle_viz",
            action="store_true",
            help="开启 target-space depth cycle 可视化（仅调试，不参与 loss）"
        )
        self.parser.add_argument(
            "--depth_cycle_viz_scale",
            type=int,
            default=0,
            help="depth cycle 可视化所使用的尺度（默认 0）"
        )
        self.parser.add_argument(
            "--enable_depth_sensitivity_viz",
            action="store_true",
            help="开启 depth 扰动敏感性可视化（仅调试，不参与 loss）"
        )
        self.parser.add_argument(
            "--depth_sensitivity_viz_scale",
            type=int,
            default=0,
            help="depth 敏感性可视化所使用的尺度（默认 0）"
        )
        self.parser.add_argument(
            "--depth_sensitivity_factor",
            type=float,
            default=1.1,
            help="depth 扰动倍数（例如 1.1 表示 +10%）"
        )
        self.parser.add_argument(
            "--enable_depth_pixshift_viz",
            action="store_true",
            help="开启 depth 扰动下 target->source 像素位移强度敏感性可视化（仅调试，不参与 loss）"
        )
        self.parser.add_argument(
            "--depth_sens_weight_start_epoch",
            type=int,
            default=10,
            help="DepthSensWeight 开启 epoch（前若干 epoch warmup 不启用加权）"
        )
        self.parser.add_argument(
            "--depth_sens_weight_q_low",
            type=float,
            default=5.0,
            help="DepthSensWeight 归一化下分位（百分位，默认 5）"
        )
        self.parser.add_argument(
            "--depth_sens_weight_q_high",
            type=float,
            default=95.0,
            help="DepthSensWeight 归一化上分位（百分位，默认 95）"
        )
        self.parser.add_argument(
            "--depth_sens_wpix_scale",
            type=float,
            default=0.8,
            help="DepthSensWeight 像素压制强度（针对高光度低几何区域）：Wpix = 1 - scale * (Sg*(1-Sp))"
        )
        self.parser.add_argument(
            "--depth_sens_wimg_scale",
            type=float,
            default=0.6,
            help="DepthSensWeight target级帧间压制强度（按batch归一化后的target score压制）"
        )
        self.parser.add_argument(
            "--depth_sens_weight_min",
            type=float,
            default=0.2,
            help="DepthSensWeight 权重下限（像素与帧间压制后的最小值）"
        )
        self.parser.add_argument(
            "--badscore_start_epoch",
            type=int,
            default=10,
            help="BadScoreWeight 开启 epoch（前若干 epoch warmup 不启用图像级加权）"
        )
        self.parser.add_argument(
            "--badscore_alpha_r",
            type=float,
            default=1.0,
            help="BadScoreWeight 中 residual 分数 sigmoid 的斜率 alpha_r；BadScoreLocalWeight 中也参与 fragile_score = sigmoid(alpha_r*r_map) * sigmoid(alpha_o*o_map) * sigmoid(-beta_m*M_tilde)"
        )
        self.parser.add_argument(
            "--badscore_alpha_o",
            type=float,
            default=1.0,
            help="BadScoreWeight 中 low-observability 分数 sigmoid 的斜率 alpha_o；BadScoreLocalWeight 中也参与 fragile_score = sigmoid(alpha_r*r_map) * sigmoid(alpha_o*o_map) * sigmoid(-beta_m*M_tilde)"
        )
        self.parser.add_argument(
            "--badscore_beta_m",
            type=float,
            default=1.0,
            help="BadScoreLocalWeight 中 margin 项的斜率 beta_m，用于 margin_gate = sigmoid(-beta_m * M_tilde)"
        )
        self.parser.add_argument(
            "--badscore_wimg_scale",
            type=float,
            default=0.6,
            help="BadScoreWeight 图像级压制强度 lambda，用于 W_img = clip(1 - lambda * sigmoid(B_hat), w_min, 1)"
        )
        self.parser.add_argument(
            "--badscore_weight_min",
            type=float,
            default=0.2,
            help="BadScoreWeight 图像级权重下限 w_min"
        )
        self.parser.add_argument(
            "--badscore_norm_clip",
            type=float,
            default=3.0,
            help="BadScoreWeight 中 r/o 标准化分数的截断范围"
        )
        self.parser.add_argument(
            "--badscore_local_scale",
            type=float,
            default=0.2,
            help="BadScoreLocalWeight 中像素级压制强度 lambda_pix，用于 W_pix = clip(1 - lambda_pix * fragile_score, w_min, 1)"
        )
        self.parser.add_argument(
            "--badscore_local_weight_min",
            type=float,
            default=0.2,
            help="BadScoreLocalWeight 中像素权重下限 w_pix_min"
        )
        self.parser.add_argument(
            "--badscore_local_gamma",
            type=float,
            default=1.0,
            help="保留旧版阈值式 local 分支的兼容参数；当前无阈值 W_pix 公式不再使用"
        )
        self.parser.add_argument(
            "--badscore_local_z_thresh",
            type=float,
            default=1.0,
            help="保留旧版阈值式 local 分支的兼容参数；当前无阈值 W_pix 公式不再使用"
        )
        self.parser.add_argument(
            "--automask_hr_percentile",
            type=float,
            default=90.0,
            help="automask 统计中高残差阈值的分位数（如 90 表示 top 10%）"
        )
        self.parser.add_argument(
            "--automask_hr_scope",
            type=str,
            default="all",
            choices=["all", "mask"],
            help="高残差阈值的像素范围：all=全像素，mask=仅在 keep 像素中估计阈值"
        )
        self.parser.add_argument(
            "--enable_automask_margin_viz",
            action="store_true",
            help="训练 log() 时可视化 automask 比较图：identity_min / reproj_min / margin"
        )
        self.parser.add_argument(
            "--save_automask_margin_viz_local",
            action="store_true",
            help="除 W&B 外，也将 automask 比较图保存到 <log_path>/automask_margin_viz/"
        )
        self.parser.add_argument(
            "--automask_margin_viz_samples",
            type=int,
            default=1,
            help="每次 train/val 日志时保存多少个 automask 比较样本（默认 1）"
        )
        self.parser.add_argument(
            "--enable_flip",
            action="store_true",
            help="启用训练阶段的随机水平翻转增强（默认关闭）"
        )
        self.parser.add_argument(
            "--use_external_mask",
            action="store_true",
            help="Enable external mask for photometric losses (white=masked, black=keep)"
        )
        self.parser.add_argument(
            "--disable_external_mask",
            action="store_true",
            help="Force disable external mask even if method enables it"
        )
        self.parser.add_argument(
            "--external_mask_dir",
            type=str,
            default="mask",
            help="Subfolder name under each seq dir for external masks (default: mask)"
        )
        self.parser.add_argument(
            "--external_mask_ext",
            type=str,
            default=".png",
            help="Mask file extension (default: .png)"
        )
        self.parser.add_argument(
            "--external_mask_thresh",
            type=float,
            default=0.5,
            help="Threshold on [0,1] mask tensor to treat as masked (default: 0.5)"
        )

        # SCALE ALIGNMENT options
        self.parser.add_argument(
            "--scale_align_mode",
            type=str,
            default="off",
            choices=["off", "depth"],
            help="per-mini-sequence scale alignment mode"
        )
        self.parser.add_argument(
            "--scale_align_anchor_key",
            type=str,
            default="depth_gt",
            help="inputs key containing anchor depth (default depth_gt)"
        )
        self.parser.add_argument(
            "--scale_align_conf_key",
            type=str,
            default="depth_conf",
            help="inputs key containing anchor confidence (optional)"
        )
        self.parser.add_argument(
            "--scale_align_min_valid_ratio",
            type=float,
            default=0.01,
            help="minimum ratio of valid anchor pixels required to solve scale"
        )
        self.parser.add_argument(
            "--scale_align_min_valid_pixels",
            type=int,
            default=2048,
            help="minimum number of valid anchor pixels required to solve scale"
        )
        self.parser.add_argument(
            "--scale_align_conf_floor",
            type=float,
            default=1e-3,
            help="lower bound applied to anchor confidences"
        )
        self.parser.add_argument(
            "--scale_align_eps",
            type=float,
            default=1e-6,
            help="numerical epsilon for the weighted LS solver"
        )
        self.parser.add_argument(
            "--scale_align_scale_min",
            type=float,
            default=0.05,
            help="minimum clamp for the solved scale factor"
        )
        self.parser.add_argument(
            "--scale_align_scale_max",
            type=float,
            default=40.0,
            help="maximum clamp for the solved scale factor"
        )
        self.parser.add_argument(
            "--scale_align_reference_scale",
            type=int,
            default=0,
            help="decoder scale index used to estimate alignment"
        )

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=DEFAULT_MODELS_TO_LOAD)




        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each logging step",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)


    def parse(self, args=None, namespace=None):
        import re

        def _sci(x: float) -> str:
            # 1e-04 -> 1e-4
            s = f"{float(x):.0e}"
            base, exp = s.split("e")
            exp = exp.lstrip("+0") or "0"
            return f"{base}e{exp}"

        def _slug(s: str) -> str:
            s = str(s).lower()
            return re.sub(r"[^a-z0-9._-]+", "-", s).strip("-")

        opts = self.parser.parse_args(args=args, namespace=namespace)

        method_aliases = {
            "madhuanand": "Madhuanand",
            "gasmono": "GasMono",
        }
        opts.methods = method_aliases.get(str(opts.methods), opts.methods)

        if getattr(opts, "describe_methods", False):
            print("Available methods:")
            for name in sorted(METHOD_DESCRIPTIONS):
                desc = METHOD_DESCRIPTIONS[name]
                print(f"  {name:>25}: {desc}")
            raise SystemExit(0)

        method_name = str(getattr(opts, "methods", ""))
        if method_name == "SPIDepth":
            fixed_height = 288
            fixed_width = 512
            fixed_backbone = "convnext_large"
            fixed_model_dim = 32
            fixed_patch_size = 32
            fixed_query_nums = 32
            fixed_dim_out = 64
            fixed_dec_channels = [1024, 512, 256, 128]
            fixed_pose_backbone = "resnet18"
            fixed_pretrained_dir = "/mnt/data_nvme3n1p1/PycharmProjects/monodepth2/ckpt/SPIDepth"

            if (
                int(getattr(opts, "height", fixed_height)) != fixed_height
                or int(getattr(opts, "width", fixed_width)) != fixed_width
                or str(getattr(opts, "spidepth_backbone", fixed_backbone)) != fixed_backbone
                or int(getattr(opts, "spidepth_model_dim", fixed_model_dim)) != fixed_model_dim
                or int(getattr(opts, "spidepth_patch_size", fixed_patch_size)) != fixed_patch_size
                or int(getattr(opts, "spidepth_query_nums", fixed_query_nums)) != fixed_query_nums
                or int(getattr(opts, "spidepth_dim_out", fixed_dim_out)) != fixed_dim_out
                or list(getattr(opts, "spidepth_dec_channels", fixed_dec_channels)) != fixed_dec_channels
                or str(getattr(opts, "spidepth_pose_backbone", fixed_pose_backbone)) != fixed_pose_backbone
            ):
                print(
                    "[options] SPIDepth is pinned to fixed config: "
                    f"height={fixed_height}, width={fixed_width}, "
                    f"backbone={fixed_backbone}, model_dim={fixed_model_dim}, "
                    f"patch={fixed_patch_size}, query_nums={fixed_query_nums}, "
                    f"dim_out={fixed_dim_out}, dec_channels={fixed_dec_channels}, "
                    f"pose_backbone={fixed_pose_backbone}"
                )

            opts.height = fixed_height
            opts.width = fixed_width
            opts.spidepth_backbone = fixed_backbone
            opts.spidepth_model_dim = fixed_model_dim
            opts.spidepth_patch_size = fixed_patch_size
            opts.spidepth_query_nums = fixed_query_nums
            opts.spidepth_dim_out = fixed_dim_out
            opts.spidepth_dec_channels = fixed_dec_channels
            opts.spidepth_pose_backbone = fixed_pose_backbone
            if not str(getattr(opts, "spidepth_pretrained", "") or "").strip() and os.path.isdir(fixed_pretrained_dir):
                opts.spidepth_pretrained = fixed_pretrained_dir
                print(f"[options] SPIDepth default pretrained folder -> {fixed_pretrained_dir}")
        if method_name == "GasMono":
            fixed_posegt_reprojection_mode = "gasmono"
            fixed_posegt_weight = 0.2
            fixed_selfpp_weight = 0.1
            fixed_iiters = 2
            fixed_iiter_start_epoch = 10
            fixed_selfpp_warmup_epochs = 20
            if (
                str(getattr(opts, "posegt_reprojection_mode", fixed_posegt_reprojection_mode)).lower()
                != fixed_posegt_reprojection_mode
                or float(getattr(opts, "www", fixed_posegt_weight)) != fixed_posegt_weight
                or float(getattr(opts, "wpp", fixed_selfpp_weight)) != fixed_selfpp_weight
                or int(getattr(opts, "iiters", fixed_iiters)) != fixed_iiters
                or int(getattr(opts, "gasmono_iiter_start_epoch", fixed_iiter_start_epoch)) != fixed_iiter_start_epoch
                or int(getattr(opts, "gasmono_selfpp_warmup_epochs", fixed_selfpp_warmup_epochs))
                != fixed_selfpp_warmup_epochs
            ):
                print(
                    "[options] GasMono is pinned to fixed config: "
                    f"posegt_reprojection_mode={fixed_posegt_reprojection_mode}, "
                    f"www={fixed_posegt_weight}, wpp={fixed_selfpp_weight}, "
                    f"iiters={fixed_iiters}, gasmono_iiter_start_epoch={fixed_iiter_start_epoch}, "
                    f"gasmono_selfpp_warmup_epochs={fixed_selfpp_warmup_epochs}"
                )
            opts.posegt_reprojection_mode = fixed_posegt_reprojection_mode
            opts.www = fixed_posegt_weight
            opts.wpp = fixed_selfpp_weight
            opts.iiters = fixed_iiters
            opts.gasmono_selfpp_weight = fixed_selfpp_weight
            opts.gasmono_iiters = fixed_iiters
            opts.gasmono_iiter_start_epoch = fixed_iiter_start_epoch
            opts.gasmono_selfpp_warmup_epochs = fixed_selfpp_warmup_epochs

        # 只有未显式给出 model_name 或给了 'auto'/空串时才自动生成
        want_auto = (
                opts.model_name is None
                or (isinstance(opts.model_name, str) and opts.model_name.strip().lower() in {"", "auto"})
        )
        if want_auto:
            methods = str(opts.methods).lower()
            dataset = str(opts.dataset).lower()
            res_token = f"{int(opts.width)}x{int(opts.height)}"  # ★ 把分辨率纳入命名
            bs_token = f"bs{int(opts.batch_size)}"
            lr_token = f"lr{_sci(opts.learning_rate)}"
            e_token = f"e{int(opts.num_epochs)}"
            step_token = f"step{int(opts.scheduler_step_size)}"

            name = "_".join([
                methods, dataset, res_token,  # 模型/数据集/输入尺寸
                bs_token, lr_token, e_token, step_token  # 优化超参
            ])
            opts.model_name = _slug(name)

        # ---- external mask defaults ----
        if method_name == "MRFEDepth" and list(getattr(opts, "models_to_load", [])) == DEFAULT_MODELS_TO_LOAD:
            opts.models_to_load = MRFE_MODELS_TO_LOAD.copy()
            print(f"[options] MRFEDepth defaults models_to_load -> {opts.models_to_load}")
        if method_name in {"MD2_VGGT_PoseGT_Mask", "MonoViT_PoseGT_Mask", "MD2_Mask"} and not getattr(opts, "use_external_mask", False):
            opts.use_external_mask = True
        if method_name in {"MD2_VGGT_PoseGT_DepthCycleViz", "MD2_VGGT_DepthCycleViz"}:
            opts.enable_depth_cycle_viz = True
        if method_name == "SPIDepth" and list(getattr(opts, "scales", [])) != [0]:
            print(f"[options] SPIDepth only supports scale [0], overriding {opts.scales} -> [0]")
            opts.scales = [0]
        if method_name in {
            "MD2_VGGT_PoseGT_DepthSensitivityViz",
            "MD2_VGGT_DepthSensitivityViz",
            "MD2_VGGT_PoseGT_DepthSensViz",
            "MD2_VGGT_PoseGT_DepthSensWeight",
        }:
            opts.enable_depth_sensitivity_viz = True
        if method_name in {"MD2_VGGT_PoseGT_DepthSensViz", "MD2_VGGT_PoseGT_DepthSensWeight"}:
            opts.enable_depth_pixshift_viz = True
        if getattr(opts, "disable_external_mask", False):
            opts.use_external_mask = False
        if getattr(opts, "external_mask_ext", None):
            ext = str(opts.external_mask_ext)
            if ext and not ext.startswith("."):
                opts.external_mask_ext = "." + ext

        return opts

# methods/init_and_forward.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from methods import networks
from layers import disp_to_depth, get_smooth_loss

# ========== 常量 ==========
MD2_VGGT_METHODS = {
    "MD2_VGGT", "MD2_VGGT_DepthCycleViz", "MD2_VGGT_DepthSensitivityViz",
    "MD2_VGGT_NoPose", "MD2_VGGT_NoPose_UniformT", "MD2_VGGT_NoPose_SAlign",
    "MD2_VGGT_NoPose_TScale", "MD2_VGGT_TDir_PoseMag", "MD2_VGGT_TPrior_Alpha",
    "MD2_VGGT_TPrior_AlignRes", "MD2_VGGT_RPrior_TPose",
    "MD2_VGGT_ResPose_RT", "MD2_VGGT_ResPose_RT_Reg", "MD2_VGGT_ResPose_RT_RMul",
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
}
EXTERNAL_POSE_METHODS = {
    "MD2_VGGT_NoPose", "MD2_VGGT_NoPose_UniformT", "MD2_VGGT_NoPose_SAlign",
    "MD2_VGGT_NoPose_TScale",
}
PRIOR_SCALE_METHODS = {"MD2_VGGT_NoPose_TScale"}
SCALE_ALIGN_METHODS = {"MD2_VGGT_NoPose_SAlign"}
POSEGT_METHODS = {
    "MD2_VGGT_PoseGT", "MD2_VGGT_PoseGT_DepthCycleViz",
    "MD2_VGGT_PoseGT_DepthSensitivityViz",
    "MD2_VGGT_PoseGT_DepthSensViz",
    "MD2_VGGT_PoseGT_DepthSensWeight",
    "MD2_VGGT_PoseGT_BadScoreWeight",
    "MD2_VGGT_PoseGT_BadScoreLocalWeight",
    "MD2_VGGT_PoseGT_HRMask", "MD2_VGGT_PoseGT_Mask",
    "MD2_VGGT_PoseGT_DeRotHardMask",
    "MD2_VGGT_PoseGT_DeRotSigmoidWeight",
}
MONOVIT_POSEGT_METHODS = {
    "MonoViT_PoseGT",
    "MonoViT_PoseGT_Mask",
    "MonoViT_PoseGT_HRMask",
    "MonoViT_VGGT_PoseGT_BadScoreWeight",
}
MONOVIT_RPRIOR_RESR_TPOSE_METHODS = {
    "MonoViT_VGGT_RPrior_ResR_TPose",
}
MONOVIT_RFLOW_METHODS = {
    "MonoViT_VGGT_RFlow_Pose",
    "MonoViT_VGGT_RFlow_ResR_TPose",
    "MonoViT_VGGT_RFlow_ResR_TPose_SingleHead",
    "MonoViT_VGGT_RFlow_TInj",
}
MONOVIT_RFLOW_RESR_TPOSE_METHODS = {
    "MonoViT_VGGT_RFlow_ResR_TPose",
}
MADHUANAND_METHODS = {"Madhuanand", "madhuanand"}
POSE_RESIDUAL_METHOD_CONFIG = {
    "MD2_VGGT_ResPose_RT": {"mode": "rt", "use_reg": False},
    "MD2_VGGT_ResPose_RT_Reg": {"mode": "rt", "use_reg": True},
    "MD2_VGGT_ResPose_RT_RMul": {"mode": "rt", "use_reg": False, "order": "right"},
    "MD2_VGGT_ResPose_T": {"mode": "t", "use_reg": False},
    "MD2_VGGT_ResPose_T_Reg": {"mode": "t", "use_reg": True},
    "MD2_VGGT_ResPose_Decay": {"mode": "rt", "use_reg": False},
}


class PoseScale(nn.Module):
    """Learnable global scale for prior translation."""

    def __init__(self, init_scale: float = 1.0):
        super().__init__()
        init = max(float(init_scale), 1e-6)
        self.log_scale = nn.Parameter(torch.log(torch.tensor(init, dtype=torch.float32)))

    def forward(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        scale = torch.exp(self.log_scale)
        if device is not None or dtype is not None:
            scale = scale.to(device=device, dtype=dtype)
        return scale.view(1, 1, 1).repeat(batch_size, 1, 1)

# ========== 公共工具 ==========
def _count_params(m):
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train

def _print_init(tag: str, **kwargs):
    kv = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    print(f"[init_models] {tag}, {kv}")

def _print_pretrained_status(tag: str, enabled: bool, source: str, module=None):
    # 统一预训练加载提示：确保所有方法都能看到“已加载/未加载”的明确日志。
    status = "loaded" if enabled else "disabled"
    extra = ""
    if module is not None:
        matched = getattr(module, "pretrained_num_loaded", None)
        total = getattr(module, "pretrained_num_total", None)
        if matched is None and hasattr(module, "encoder"):
            matched = getattr(module.encoder, "pretrained_num_loaded", None)
            total = getattr(module.encoder, "pretrained_num_total", None)
        if matched is not None and total is not None:
            extra = f", matched={matched}/{total}"
    print(f"[init_models][pretrained] {tag}: {status} ({source}{extra})")

def _get_module_pretrained_match(module):
    matched = getattr(module, "pretrained_num_loaded", None)
    total = getattr(module, "pretrained_num_total", None)
    if matched is None and hasattr(module, "encoder"):
        matched = getattr(module.encoder, "pretrained_num_loaded", None)
        total = getattr(module.encoder, "pretrained_num_total", None)
    return matched, total

def _print_local_ckpt_status(tag: str, path: str, stats: dict):
    if not stats:
        print(f"[init_models][pretrained] {tag}: source=local_ckpt ({path}), applied=no")
        return
    applied = "yes" if bool(stats.get("applied", False)) else "no"
    print(
        f"[init_models][pretrained] {tag}: source=local_ckpt ({path}), applied={applied}, "
        f"matched={stats.get('matched', 0)}/{stats.get('total_model', 0)}, "
        f"shape_mismatch={stats.get('shape_mismatch', 0)}, "
        f"missing_in_ckpt={stats.get('missing_in_ckpt', 0)}"
    )

def _load_partial_state_dict(module, path: str, tag: str, min_match_ratio: float = None, log_loaded: bool = True):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model_dict = module.state_dict()
    matched = {}
    shape_mismatch = 0
    model_tensor_keys = [k for k, v in model_dict.items() if hasattr(v, "shape")]
    for k, v in state_dict.items():
        if k not in model_dict or not hasattr(v, "shape"):
            continue
        if model_dict[k].shape == v.shape:
            matched[k] = v
        else:
            shape_mismatch += 1
    total_model = len(model_tensor_keys)
    match_ratio = (len(matched) / total_model) if total_model > 0 else 0.0
    if min_match_ratio is not None and match_ratio < float(min_match_ratio):
        print(
            f"[init_models][warning] skip loading {tag} from {path}: "
            f"low match ratio {len(matched)}/{total_model} ({match_ratio:.3f} < {float(min_match_ratio):.3f}), "
            f"shape_mismatch={shape_mismatch}, missing_in_ckpt={sum(1 for k in model_tensor_keys if k not in state_dict)}"
        )
        return {
            "matched": len(matched),
            "total_model": total_model,
            "shape_mismatch": shape_mismatch,
            "missing_in_ckpt": sum(1 for k in model_tensor_keys if k not in state_dict),
            "applied": False,
        }
    model_dict.update(matched)
    module.load_state_dict(model_dict, strict=False)
    missing_in_ckpt = 0
    for k in model_tensor_keys:
        if k not in state_dict:
            missing_in_ckpt += 1
    if log_loaded:
        print(
            f"[init_models] loaded {len(matched)}/{total_model} tensors from {tag}: {path} "
            f"(shape_mismatch={shape_mismatch}, missing_in_ckpt={missing_in_ckpt})"
        )
    return {
        "matched": len(matched),
        "total_model": total_model,
        "shape_mismatch": shape_mismatch,
        "missing_in_ckpt": missing_in_ckpt,
        "applied": True,
    }

def _apply_pose_regularizers(outputs, losses):
    for key in ("pose_residual_reg", "pose_alpha_reg", "pose_align_reg"):
        reg = outputs.pop(key, None)
        if reg is not None:
            losses[key] = reg
            losses["loss"] = losses["loss"] + reg

# ========== models 初始化：仅构建模型，不涉及 loss ==========
def init_models(self):
    """
    按 self.opt.methods 构建 encoder/decoder，并在初始化时打印单行摘要。
    兼容 PoseNet 的启用/禁用策略，打印关键信息（与现有风格一致）。
    """
    self.use_pose_net = self.opt.methods not in EXTERNAL_POSE_METHODS
    self.pose_residual_cfg = POSE_RESIDUAL_METHOD_CONFIG.get(self.opt.methods)
    self.pose_residual_order = "left"
    if self.pose_residual_cfg is not None:
        self.use_pose_net = True
        self.pose_residual_mode = self.pose_residual_cfg.get("mode", "rt")
        self.pose_residual_order = self.pose_residual_cfg.get("order", "left")
        if self.pose_residual_cfg.get("use_reg"):
            self.pose_residual_reg_weight = float(getattr(self.opt, "pose_residual_reg_weight", 1e-3))
        else:
            self.pose_residual_reg_weight = 0.0
    else:
        self.pose_residual_mode = None
        self.pose_residual_reg_weight = 0.0

    if self.opt.methods in SCALE_ALIGN_METHODS:
        if getattr(self.opt, "scale_align_mode", "off") == "off":
            self.opt.scale_align_mode = "depth"
        if getattr(self.opt, "scale_align_anchor_key", "depth_gt") == "depth_gt":
            self.opt.scale_align_anchor_key = "vggt_depth"
        if getattr(self.opt, "scale_align_conf_key", "depth_conf") == "depth_conf":
            self.opt.scale_align_conf_key = "vggt_conf"
    print(f"[init_models] dataset={self.opt.dataset}, use_pose_net={self.use_pose_net}")

    self.models = {}
    self.parameters_to_train = []
    self.spidepth_use_posecnn = False
    self.spidepth_pose_ckpt = None

    def _init_posegt_translation_modules():
        self.models["trans_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=2 if self.opt.pose_model_input == "pairs" else len(self.opt.frame_ids)
        )
        self.models["trans_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["trans_encoder"].parameters())

        self.models["trans"] = networks.TransDecoder(
            self.models["trans_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2,
        )
        self.models["trans"].to(self.device)
        self.parameters_to_train += list(self.models["trans"].parameters())

        te, tre = _count_params(self.models["trans_encoder"])
        td, trd = _count_params(self.models["trans"])
        _print_init("trans_encoder=ResnetEncoder",
                    num_layers=self.opt.num_layers,
                    params=f"{te/1e6:.2f}M/{tre/1e6:.2f}M")
        _print_pretrained_status(
            "trans_encoder",
            self.opt.weights_init == "pretrained",
            f"weights_init={self.opt.weights_init}",
            module=self.models["trans_encoder"],
        )
        _print_init("trans_decoder=TransDecoder",
                    frames_to_predict_for=2,
                    params=f"{td/1e6:.2f}M/{trd/1e6:.2f}M")

    # ---------- Monodepth2 ----------
    if self.opt.methods in {"Monodepth2", "MD2_Mask"}:
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        t, tr = _count_params(self.models["encoder"])
        _print_init("encoder=ResnetEncoder",
                    num_layers=self.opt.num_layers,
                    weights_init=self.opt.weights_init,
                    params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M")
        _print_pretrained_status(
            "encoder=ResnetEncoder",
            self.opt.weights_init == "pretrained",
            f"weights_init={self.opt.weights_init}",
            module=self.models["encoder"],
        )

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        t, tr = _count_params(self.models["depth"])
        _print_init("decoder=DepthDecoder",
                    num_ch_enc=tuple(self.models["encoder"].num_ch_enc),
                    scales=list(self.opt.scales),
                    params=f'{t/1e6:.2f}M/{tr/1e6:.2f}M')

    # ---------- Monodepth2_DINO ----------
    elif self.opt.methods == "Monodepth2_DINO":
        #dino_arch = getattr(self.opt, "dino_arch", "dinov3_convnext_tiny")  # << 用 arch 名
        dino_arch = getattr(self.opt, "dino_arch", "dinov3_convnext_small")  # << 用 arch 名
        self.models["encoder"] = networks.DinoConvNeXtMultiScale(arch=dino_arch)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        enc_ch = tuple(getattr(self.models["encoder"], "num_ch_enc", []))
        t, tr = _count_params(self.models["encoder"])
        _print_init("encoder=DinoConvNeXtMultiScale",
                    arch=dino_arch,
                    out_indices=getattr(self.models["encoder"], "out_indices", None),
                    num_ch_enc=enc_ch,
                    params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M")
        _print_pretrained_status(
            "encoder=DinoConvNeXtMultiScale",
            True,
            f"weights={getattr(self.models['encoder'], 'pretrained_weights_path', 'auto')}",
            module=self.models["encoder"],
        )

        assert len(self.opt.scales) == 4 and list(self.opt.scales) == [0, 1, 2, 3], \
            f"opt.scales 必须为 [0,1,2,3]，当前为 {self.opt.scales}"

        # 三选一：若切换其他头，替换下一行实例化即可
        # self.models["depth"] = networks.DepthDecoderDINO(enc_ch, self.opt.scales)
        # self.models["depth"] = networks.MSegDPTMonoDepth2Head(
        #     in_channels=tuple(enc.num_ch_enc),  # (96,192,384,768)
        #     channels=256,
        #     out_channel=1,
        #     align_corners=False
        # )
        self.models["depth"] = networks.UPerDispHead(in_channels=[96, 192, 384, 768], compat="monodepth2")
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        t, tr = _count_params(self.models["depth"])
        head_name = self.models["depth"].__class__.__name__
        _print_init(f"decoder={head_name}",
                    in_channels=getattr(self.models["depth"], "in_channels", [96, 192, 384, 768]),
                    scales=list(self.opt.scales),
                    params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M")

    # ---------- GasMono (official-style MPViT depth backbone) ----------
    elif self.opt.methods == "GasMono":
        self.models["encoder"] = networks.mpvit_small()
        self.models["encoder"].num_ch_enc = [64, 128, 216, 288, 288]
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        t, tr = _count_params(self.models["encoder"])
        _print_init(
            "encoder=MPViT(GasMono)",
            num_ch_enc=tuple(self.models["encoder"].num_ch_enc),
            params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M",
        )
        _print_pretrained_status(
            "encoder=MPViT(GasMono)",
            True,
            "mpvit_small checkpoint",
            module=self.models["encoder"],
        )

        self.models["depth"] = networks.GasMonoFSDepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales
        )
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        t, tr = _count_params(self.models["depth"])
        _print_init(
            "decoder=GasMonoFSDepthDecoder",
            num_ch_enc=tuple(self.models["encoder"].num_ch_enc),
            scales=list(self.opt.scales),
            params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M",
        )

        _init_posegt_translation_modules()

    # ---------- MD2_VGGT / 派生 ----------
    elif self.opt.methods in MD2_VGGT_METHODS:
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        t, tr = _count_params(self.models["encoder"])
        suffix = "VGGT_NoPose" if self.opt.methods in EXTERNAL_POSE_METHODS else "VGGT"
        _print_init(f"encoder=ResnetEncoder({suffix})",
                    num_layers=self.opt.num_layers,
                    weights_init=self.opt.weights_init,
                    params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M")
        _print_pretrained_status(
            f"encoder=ResnetEncoder({suffix})",
            self.opt.weights_init == "pretrained",
            f"weights_init={self.opt.weights_init}",
            module=self.models["encoder"],
        )

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        t, tr = _count_params(self.models["depth"])
        _print_init(f"decoder=DepthDecoder({suffix})",
                    num_ch_enc=tuple(self.models["encoder"].num_ch_enc),
                    scales=list(self.opt.scales),
                    params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M")

        if self.opt.methods in PRIOR_SCALE_METHODS:
            init_scale = float(getattr(self.opt, "pose_prior_scale_init", 1.0))
            self.models["pose_scale"] = PoseScale(init_scale=init_scale).to(self.device)
            self.parameters_to_train += list(self.models["pose_scale"].parameters())
            _print_init("pose_scale=PoseScale", init_scale=init_scale)

        if self.opt.methods in POSEGT_METHODS:
            _init_posegt_translation_modules()

    # ---------- Madhuanand ----------
    elif self.opt.methods in MADHUANAND_METHODS:
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        t, tr = _count_params(self.models["encoder"])
        _print_init("encoder=ResnetEncoder(Madhuanand)",
                    num_layers=self.opt.num_layers,
                    weights_init=self.opt.weights_init,
                    params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M")
        _print_pretrained_status(
            "encoder=ResnetEncoder(Madhuanand)",
            self.opt.weights_init == "pretrained",
            f"weights_init={self.opt.weights_init}",
            module=self.models["encoder"],
        )

        self.models["depth"] = networks.DepthDecoder_3d(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        t, tr = _count_params(self.models["depth"])
        _print_init("decoder=DepthDecoder_3d",
                    num_ch_enc=tuple(self.models["encoder"].num_ch_enc),
                    scales=list(self.opt.scales),
                    params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M")

    # ---------- MRFEDepth ----------
    elif self.opt.methods == "MRFEDepth":
        mrfe_pretrained = self.opt.weights_init == "pretrained"
        # Depth Encoder (HRNet18)
        self.models["DepthEncoder"] = networks.MRFE_depth_encoder.hrnet18(mrfe_pretrained)
        self.models["DepthEncoder"].num_ch_enc = [64, 18, 36, 72, 144]
        self.models["DepthEncoder"].to(self.device)
        self.parameters_to_train += list(self.models["DepthEncoder"].parameters())
        print("DepthEncoder params:", sum(p.numel() for p in self.models["DepthEncoder"].parameters()))
        _print_init("encoder=HRNet18(MRFE)",
                    num_ch_enc=tuple(self.models["DepthEncoder"].num_ch_enc))
        _print_pretrained_status(
            "encoder=HRNet18(MRFE)",
            mrfe_pretrained,
            f"weights_init={self.opt.weights_init}",
            module=self.models["DepthEncoder"],
        )

        # Depth Decoder
        self.models["DepthDecoder"] = networks.MRFEDepthDecoder(self.models["DepthEncoder"].num_ch_enc, self.opt.scales)
        self.models["DepthDecoder"].to(self.device)
        self.parameters_to_train += list(self.models["DepthDecoder"].parameters())
        print("DepthDecoder params:", sum(p.numel() for p in self.models["DepthDecoder"].parameters()))
        _print_init("decoder=MRFEDepthDecoder",
                    num_ch_enc=tuple(self.models["DepthEncoder"].num_ch_enc),
                    scales=list(self.opt.scales))

        # Feature Encoder
        self.models["FeatureEncoder"] = networks.FeatureEncoder(self.opt.num_layers, pretrained=mrfe_pretrained)
        self.models["FeatureEncoder"].to(self.device)
        self.parameters_to_train += list(self.models["FeatureEncoder"].parameters())
        print("FeatureEncoder params:", sum(p.numel() for p in self.models["FeatureEncoder"].parameters()))
        _print_init("feat_encoder=FeatureEncoder", pretrained=mrfe_pretrained)
        _print_pretrained_status(
            "feat_encoder=FeatureEncoder",
            mrfe_pretrained,
            f"weights_init={self.opt.weights_init}",
            module=self.models["FeatureEncoder"],
        )

        # Feature Decoder
        self.models["FeatureDecoder"] = networks.FeatureDecoder(self.models["FeatureEncoder"].num_ch_enc,
                                                                num_output_channels=3)
        self.models["FeatureDecoder"].to(self.device)
        self.parameters_to_train += list(self.models["FeatureDecoder"].parameters())
        print("FeatureDecoder params:", sum(p.numel() for p in self.models["FeatureDecoder"].parameters()))
        _print_init("feat_decoder=FeatureDecoder")

    # ---------- MonoViT ----------
    elif self.opt.methods in {
        "MonoViT",
        "MonoViT_ResNet50_Pose",
        "MonoViT_ConvNeXt_Pose",
        "MonoViT_ConvNeXtSmall_Pose",
        "MonoViT_ConvNeXtBase_Pose",
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
    }:
        self.model = networks.DeepNet(type='mpvitnet')
        self.model.to(self.device)
        self.models["encoder"] = self.model.encoder
        self.models["decoder"] = self.model.decoder
        self.parameters_to_train += list(self.model.encoder.parameters())
        self.parameters_to_train += list(self.model.decoder.parameters())
        te, tre = _count_params(self.models["encoder"])
        td, trd = _count_params(self.models["decoder"])
        _print_init("encoder=MonoViT.encoder", params=f"{te/1e6:.2f}M/{tre/1e6:.2f}M")
        _print_pretrained_status(
            "encoder=MonoViT.encoder",
            True,
            "mpvit_small checkpoint",
            module=self.models["encoder"],
        )
        _print_init("decoder=MonoViT.decoder", params=f"{td/1e6:.2f}M/{trd/1e6:.2f}M")
        if self.opt.methods in MONOVIT_POSEGT_METHODS:
            _init_posegt_translation_modules()

    # ---------- LiteMono ----------
    elif self.opt.methods == "LiteMono":
        litemono_variant = str(getattr(self.opt, "litemono_variant", "lite-mono"))
        if list(self.opt.scales) != [0, 1, 2]:
            print(f"[init_models] LiteMono variants support scales [0, 1, 2], overriding {self.opt.scales} -> [0, 1, 2]")
            self.opt.scales = [0, 1, 2]

        self.models["encoder"] = networks.LiteMono(model=litemono_variant,
                                                   drop_path_rate=self.opt.drop_path,
                                                   width=self.opt.width, height=self.opt.height)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        t, tr = _count_params(self.models["encoder"])
        _print_init("encoder=LiteMono",
                    variant=litemono_variant,
                    width=self.opt.width, height=self.opt.height,
                    drop_path=self.opt.drop_path,
                    params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M")

        self.models["depth"] = networks.DepthDecoder_litemono(self.models["encoder"].num_ch_enc,
                                                              self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        t, tr = _count_params(self.models["depth"])
        _print_init("decoder=DepthDecoder_litemono",
                    num_ch_enc=tuple(self.models["encoder"].num_ch_enc),
                    scales=list(self.opt.scales),
                    params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M")

        litemono_pretrained = str(getattr(self.opt, "litemono_pretrained", "") or "").strip()
        if litemono_pretrained:
            encoder_ckpt = None
            depth_ckpt = None
            if os.path.isdir(litemono_pretrained):
                encoder_candidate = os.path.join(litemono_pretrained, "encoder.pth")
                depth_candidate = os.path.join(litemono_pretrained, "depth.pth")
                variant_name = litemono_variant.strip()
                variant_candidates = [
                    os.path.join(litemono_pretrained, f"{variant_name}-pretrain.pth"),
                    os.path.join(litemono_pretrained, f"{variant_name}.pth"),
                ]
                if os.path.isfile(encoder_candidate):
                    encoder_ckpt = encoder_candidate
                else:
                    for candidate in variant_candidates:
                        if os.path.isfile(candidate):
                            encoder_ckpt = candidate
                            break
                if os.path.isfile(depth_candidate):
                    depth_ckpt = depth_candidate
            elif os.path.isfile(litemono_pretrained):
                encoder_ckpt = litemono_pretrained
            else:
                print(f"[init_models] LiteMono pretrained path not found, skip loading: {litemono_pretrained}")

            if encoder_ckpt:
                _load_partial_state_dict(self.models["encoder"], encoder_ckpt, "LiteMono.encoder")
            if depth_ckpt:
                _load_partial_state_dict(self.models["depth"], depth_ckpt, "LiteMono.depth")
        else:
            _print_pretrained_status(
                "LiteMono.partial_ckpt",
                False,
                "litemono_pretrained is empty",
            )

    # ---------- SPIDepth ----------
    elif self.opt.methods == "SPIDepth":
        if list(self.opt.scales) != [0]:
            print(f"[init_models] SPIDepth only supports scale [0], overriding {self.opt.scales} -> [0]")
            self.opt.scales = [0]

        spidepth_pretrained = str(getattr(self.opt, "spidepth_pretrained", "") or "").strip()
        encoder_ckpt = None
        depth_ckpt = None
        pose_ckpt = None
        if spidepth_pretrained:
            if os.path.isdir(spidepth_pretrained):
                encoder_candidate = os.path.join(spidepth_pretrained, "encoder.pth")
                depth_candidate = os.path.join(spidepth_pretrained, "depth.pth")
                pose_candidate = os.path.join(spidepth_pretrained, "pose.pth")
                if os.path.isfile(encoder_candidate):
                    encoder_ckpt = encoder_candidate
                if os.path.isfile(depth_candidate):
                    depth_ckpt = depth_candidate
                if os.path.isfile(pose_candidate):
                    pose_ckpt = pose_candidate
            elif os.path.isfile(spidepth_pretrained):
                base_name = os.path.basename(spidepth_pretrained).lower()
                if "depth" in base_name:
                    depth_ckpt = spidepth_pretrained
                elif "pose" in base_name:
                    pose_ckpt = spidepth_pretrained
                else:
                    encoder_ckpt = spidepth_pretrained
            else:
                print(f"[init_models] SPIDepth pretrained path not found, skip loading: {spidepth_pretrained}")

        spidepth_backbone = str(getattr(self.opt, "spidepth_backbone", "resnet"))
        spidepth_num_layers = int(getattr(self.opt, "spidepth_num_layers", 50))
        spidepth_num_features = int(getattr(self.opt, "spidepth_num_features", 512))
        spidepth_model_dim = int(getattr(self.opt, "spidepth_model_dim", 32))
        spidepth_patch_size = int(getattr(self.opt, "spidepth_patch_size", 20))
        spidepth_query_nums = int(getattr(self.opt, "spidepth_query_nums", 128))
        spidepth_dim_out = int(getattr(self.opt, "spidepth_dim_out", 128))
        spidepth_dec_channels = list(getattr(self.opt, "spidepth_dec_channels", [1024, 512, 256, 128]))
        max_query_tokens = max(
            1,
            (int(self.opt.height) // spidepth_patch_size) * (int(self.opt.width) // spidepth_patch_size),
        )
        if spidepth_query_nums > max_query_tokens:
            print(
                f"[init_models] SPIDepth query_nums={spidepth_query_nums} exceeds token count="
                f"{max_query_tokens} for input {self.opt.height}x{self.opt.width} and patch={spidepth_patch_size}; "
                f"overriding query_nums -> {max_query_tokens}"
            )
            spidepth_query_nums = max_query_tokens
            self.opt.spidepth_query_nums = spidepth_query_nums

        # Encoder policy for SPIDepth: prefer timm pretrained first.
        encoder_use_timm_pretrained = (self.opt.weights_init == "pretrained")

        if spidepth_backbone == "resnet18_lite":
            self.models["encoder"] = networks.SPILiteResnetEncoderDecoder(
                model_dim=spidepth_model_dim,
                pretrained=encoder_use_timm_pretrained,
            )
            transformer_ff_dim = 512
        elif spidepth_backbone in {"resnet", "resnet_lite"}:
            self.models["encoder"] = networks.SPIResnetEncoderDecoder(
                num_layers=spidepth_num_layers,
                num_features=spidepth_num_features,
                model_dim=spidepth_model_dim,
                pretrained=encoder_use_timm_pretrained,
            )
            transformer_ff_dim = 1024
        else:
            self.models["encoder"] = networks.SPIUnet(
                backbone=spidepth_backbone,
                pretrained=encoder_use_timm_pretrained,
                in_channels=3,
                num_classes=spidepth_model_dim,
                decoder_channels=tuple(spidepth_dec_channels),
            )
            transformer_ff_dim = 1024

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        t, tr = _count_params(self.models["encoder"])
        _print_init(
            "encoder=SPIDepthFeatureEncoder",
            backbone=spidepth_backbone,
            depth_num_layers=18 if spidepth_backbone == "resnet18_lite" else spidepth_num_layers,
            model_dim=spidepth_model_dim,
            dec_channels=spidepth_dec_channels if spidepth_backbone not in {"resnet", "resnet_lite", "resnet18_lite"} else "n/a",
            params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M",
        )
        enc_pre_m, enc_pre_t = _get_module_pretrained_match(self.models["encoder"])
        if encoder_use_timm_pretrained:
            extra = f", matched={enc_pre_m}/{enc_pre_t}" if enc_pre_m is not None and enc_pre_t is not None else ""
            print(
                "[init_models][pretrained] encoder=SPIDepthFeatureEncoder: "
                f"source=timm_pretrained, enabled=yes, policy=timm-first{extra}"
            )
        else:
            print(
                "[init_models][pretrained] encoder=SPIDepthFeatureEncoder: "
                "source=timm_pretrained, enabled=no, policy=timm-first"
            )
        if encoder_use_timm_pretrained:
            print("[init_models][note] SPIDepth.encoder uses timm pretrained first (large backbone compatible mode).")
        if encoder_ckpt:
            if encoder_use_timm_pretrained:
                print(
                    f"[init_models][note] skip SPIDepth.encoder local ckpt (timm-first policy): {encoder_ckpt}"
                )
            else:
                enc_stats = _load_partial_state_dict(
                    self.models["encoder"],
                    encoder_ckpt,
                    "SPIDepth.encoder",
                    min_match_ratio=0.8,
                    log_loaded=False,
                )
                _print_local_ckpt_status("encoder=SPIDepthFeatureEncoder", encoder_ckpt, enc_stats)
        elif not encoder_use_timm_pretrained:
            print("[init_models][warning] SPIDepth.encoder has no pretrained weights loaded (timm disabled, local ckpt missing).")

        self.models["depth"] = networks.SPIDepthDecoderQueryTr(
            in_channels=spidepth_model_dim,
            embedding_dim=spidepth_model_dim,
            patch_size=spidepth_patch_size,
            query_nums=spidepth_query_nums,
            dim_out=spidepth_dim_out,
            transformer_ff_dim=transformer_ff_dim,
            min_val=self.opt.min_depth,
            max_val=self.opt.max_depth,
        )
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        t, tr = _count_params(self.models["depth"])
        _print_init(
            "decoder=SPIDepthDecoderQueryTr",
            patch_size=spidepth_patch_size,
            query_nums=spidepth_query_nums,
            dim_out=spidepth_dim_out,
            min_depth=self.opt.min_depth,
            max_depth=self.opt.max_depth,
            params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M",
        )

        if depth_ckpt:
            depth_stats = _load_partial_state_dict(
                self.models["depth"],
                depth_ckpt,
                "SPIDepth.depth",
                log_loaded=False,
            )
            _print_local_ckpt_status("depth=SPIDepthDecoderQueryTr", depth_ckpt, depth_stats)
            if (not depth_stats.get("applied", False)) or depth_stats["matched"] == 0:
                print("[init_models][warning] SPIDepth.depth local pretrained was not applied effectively.")
        else:
            print("[init_models][warning] SPIDepth.depth has no local pretrained checkpoint (depth.pth missing).")
        if pose_ckpt:
            self.spidepth_pose_ckpt = pose_ckpt
        else:
            self.spidepth_pose_ckpt = None

    # ---------- Pose 网络 ----------
    if self.use_pose_net:
        if self.opt.methods == "SPIDepth":
            self.spidepth_use_posecnn = True
            pose_num_input_frames = 2 if self.opt.pose_model_input == "pairs" else len(self.opt.frame_ids)
            pose_backbone = str(getattr(self.opt, "spidepth_pose_backbone", "resnet18"))
            pose_use_timm_pretrained = (self.opt.weights_init == "pretrained") and (self.spidepth_pose_ckpt is None)
            self.models["pose"] = networks.PoseCNN(
                num_input_frames=pose_num_input_frames,
                enc_name=pose_backbone,
                pretrained=pose_use_timm_pretrained,
            )
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())
            if self.spidepth_pose_ckpt:
                print("[init_models][pretrained] pose=PoseCNN: source=timm_pretrained, enabled=no")
            elif pose_use_timm_pretrained:
                pose_pre_m, pose_pre_t = _get_module_pretrained_match(self.models["pose"])
                extra = f", matched={pose_pre_m}/{pose_pre_t}" if pose_pre_m is not None and pose_pre_t is not None else ""
                print(
                    "[init_models][pretrained] pose=PoseCNN: "
                    f"source=timm_pretrained, enabled=yes{extra}"
                )
            else:
                print("[init_models][warning] pose=PoseCNN has no pretrained weights loaded (timm disabled, local pose ckpt missing).")
            if self.spidepth_pose_ckpt:
                pose_stats = _load_partial_state_dict(
                    self.models["pose"],
                    self.spidepth_pose_ckpt,
                    "SPIDepth.pose",
                    log_loaded=False,
                )
                _print_local_ckpt_status("pose=PoseCNN", self.spidepth_pose_ckpt, pose_stats)
                if (not pose_stats.get("applied", False)) or pose_stats["matched"] == 0:
                    print("[init_models][warning] SPIDepth.pose local pretrained was not applied effectively.")
            t, tr = _count_params(self.models["pose"])
            _print_init(
                "pose=PoseCNN",
                num_input_frames=pose_num_input_frames,
                backbone=pose_backbone,
                params=f"{t/1e6:.2f}M/{tr/1e6:.2f}M",
            )
            return

        pose_num_input_images = 2 if self.opt.pose_model_input == "pairs" else len(self.opt.frame_ids)
        pose_backbone = "resnet18"
        if self.opt.methods == "MonoViT_ResNet50_Pose":
            pose_backbone = "resnet50"
        elif self.opt.methods in {"MonoViT_ConvNeXt_Pose"}:
            pose_backbone = "convnext_tiny"
        elif self.opt.methods == "MonoViT_ConvNeXtSmall_Pose":
            pose_backbone = "convnext_small"
        elif self.opt.methods == "MonoViT_ConvNeXtBase_Pose":
            pose_backbone = "convnext_base"

        pose_num_layers = 50 if pose_backbone == "resnet50" else self.opt.num_layers
        pose_encoder_init_meta = {}
        if self.opt.methods in MONOVIT_RFLOW_METHODS:
            self.models["pose_encoder"] = networks.PoseFlowResnetEncoder(
                pose_num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=pose_num_input_images,
                num_flow_channels=2,
                flow_init_scale=1.0,
            )
            pose_encoder_init_meta["num_layers"] = pose_num_layers
            pose_encoder_init_meta["flow_channels"] = 2
            pose_encoder_init_meta["fusion_mode"] = "concat_1x1"
            pose_encoder_init_meta["flow_init_scale"] = 1.0
        elif self.opt.methods == "MRFEDepth":
            self.models["pose_encoder"] = networks.PoseEncoder(
                pose_num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=pose_num_input_images,
            )
            pose_encoder_init_meta["num_layers"] = pose_num_layers
        else:
            if pose_backbone.startswith("convnext_"):
                self.models["pose_encoder"] = networks.PoseTimmEncoder(
                    enc_name=pose_backbone,
                    pretrained=self.opt.weights_init == "pretrained",
                    num_input_images=pose_num_input_images,
                )
                pose_encoder_init_meta["backbone"] = pose_backbone
            else:
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    pose_num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=pose_num_input_images
                )
                pose_encoder_init_meta["num_layers"] = pose_num_layers
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        pose_encoder_init_tag = f"pose_encoder={self.models['pose_encoder'].__class__.__name__}"
        _print_pretrained_status(
            pose_encoder_init_tag,
            self.opt.weights_init == "pretrained",
            f"weights_init={self.opt.weights_init}",
            module=self.models["pose_encoder"],
        )

        if self.opt.methods == "MD2_VGGT_TDir_PoseMag":
            scale_init = float(getattr(self.opt, "pose_mag_scale_init", 1.0))
            scale_learnable = bool(getattr(self.opt, "pose_mag_scale_learnable", False))
            self.models["pose"] = networks.PoseMagDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2,
                scale_init=scale_init,
                learnable_scale=scale_learnable,
            )
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

            te, tre = _count_params(self.models["pose_encoder"])
            td, trd = _count_params(self.models["pose"])
            encoder_kwargs = dict(pose_encoder_init_meta)
            encoder_kwargs["params"] = f"{te/1e6:.2f}M/{tre/1e6:.2f}M"
            _print_init(pose_encoder_init_tag, **encoder_kwargs)
            _print_init("pose_decoder=PoseMagDecoder",
                        frames_to_predict_for=2,
                        scale_init=scale_init,
                        scale_learnable=scale_learnable,
                        params=f"{td/1e6:.2f}M/{trd/1e6:.2f}M")
        elif self.opt.methods == "MD2_VGGT_TPrior_Alpha":
            alpha_mode = str(getattr(self.opt, "pose_alpha_mode", "tanh")).lower()
            self.models["pose"] = networks.PoseAlphaDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2,
            )
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

            te, tre = _count_params(self.models["pose_encoder"])
            td, trd = _count_params(self.models["pose"])
            encoder_kwargs = dict(pose_encoder_init_meta)
            encoder_kwargs["params"] = f"{te/1e6:.2f}M/{tre/1e6:.2f}M"
            _print_init(pose_encoder_init_tag, **encoder_kwargs)
            _print_init("pose_decoder=PoseAlphaDecoder",
                        frames_to_predict_for=2,
                        alpha_mode=alpha_mode,
                        params=f"{td/1e6:.2f}M/{trd/1e6:.2f}M")
        elif self.opt.methods == "MD2_VGGT_TPrior_AlignRes":
            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2
            )
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

            self.models["pose_align"] = networks.PoseAlignDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2,
            )
            self.models["pose_align"].to(self.device)
            self.parameters_to_train += list(self.models["pose_align"].parameters())

            te, tre = _count_params(self.models["pose_encoder"])
            td, trd = _count_params(self.models["pose"])
            ta, tra = _count_params(self.models["pose_align"])
            encoder_kwargs = dict(pose_encoder_init_meta)
            encoder_kwargs["params"] = f"{te/1e6:.2f}M/{tre/1e6:.2f}M"
            _print_init(pose_encoder_init_tag, **encoder_kwargs)
            _print_init("pose_decoder=PoseDecoder",
                        frames_to_predict_for=2,
                        params=f"{td/1e6:.2f}M/{trd/1e6:.2f}M")
            _print_init("pose_align=PoseAlignDecoder",
                        frames_to_predict_for=2,
                        params=f"{ta/1e6:.2f}M/{tra/1e6:.2f}M")
        elif self.opt.methods in (MONOVIT_RPRIOR_RESR_TPOSE_METHODS | MONOVIT_RFLOW_RESR_TPOSE_METHODS):
            self.models["pose_r"] = networks.PoseVectorDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2,
                num_output_channels=3,
                zero_init=True,
            )
            self.models["pose_r"].to(self.device)
            self.parameters_to_train += list(self.models["pose_r"].parameters())

            self.models["pose_t"] = networks.PoseVectorDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2,
                num_output_channels=3,
            )
            self.models["pose_t"].to(self.device)
            self.parameters_to_train += list(self.models["pose_t"].parameters())

            te, tre = _count_params(self.models["pose_encoder"])
            trt, trtr = _count_params(self.models["pose_r"])
            ttt, tttr = _count_params(self.models["pose_t"])
            encoder_kwargs = dict(pose_encoder_init_meta)
            encoder_kwargs["params"] = f"{te/1e6:.2f}M/{tre/1e6:.2f}M"
            _print_init(pose_encoder_init_tag, **encoder_kwargs)
            _print_init("pose_r=PoseVectorDecoder",
                        frames_to_predict_for=2,
                        num_output_channels=3,
                        zero_init=True,
                        params=f"{trt/1e6:.2f}M/{trtr/1e6:.2f}M")
            _print_init("pose_t=PoseVectorDecoder",
                        frames_to_predict_for=2,
                        num_output_channels=3,
                        zero_init=False,
                        params=f"{ttt/1e6:.2f}M/{tttr/1e6:.2f}M")
        elif self.opt.methods == "MonoViT_VGGT_RFlow_TInj":
            self.models["pose"] = networks.PoseTPriorDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2,
            )
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

            te, tre = _count_params(self.models["pose_encoder"])
            td, trd = _count_params(self.models["pose"])
            encoder_kwargs = dict(pose_encoder_init_meta)
            encoder_kwargs["params"] = f"{te/1e6:.2f}M/{tre/1e6:.2f}M"
            _print_init(pose_encoder_init_tag, **encoder_kwargs)
            _print_init("pose_decoder=PoseTPriorDecoder",
                        frames_to_predict_for=2,
                        prior_dim=4,
                        prior_mlp="4-64-128",
                        params=f"{td/1e6:.2f}M/{trd/1e6:.2f}M")
        else:
            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2
            )
            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

            te, tre = _count_params(self.models["pose_encoder"])
            td, trd = _count_params(self.models["pose"])
            encoder_kwargs = dict(pose_encoder_init_meta)
            encoder_kwargs["params"] = f"{te/1e6:.2f}M/{tre/1e6:.2f}M"
            _print_init(pose_encoder_init_tag, **encoder_kwargs)
            _print_init("pose_decoder=PoseDecoder",
                        frames_to_predict_for=2,
                        params=f"{td/1e6:.2f}M/{trd/1e6:.2f}M")
    else:
        print("[init_models] PoseNet disabled (external pose source)")

# ================================== forward handlers ==================================

# ---- Monodepth2 / Monodepth2App / Monodepth2_DINO 等价路径 ----
def handle_monodepth2(runner, inputs):
    # 只使用帧 0 的图像
    features = runner.models["encoder"](inputs[("color_aug", 0, 0)])
    outputs = runner.models["depth"](features)

    outputs.update(runner.predict_poses(inputs, features))

    runner.generate_images_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs)
    _apply_pose_regularizers(outputs, losses)
    return outputs, losses

def handle_monodepth2_dino(runner, inputs):
    # 只使用帧 0 的图像
    use_norm = bool(getattr(runner, "normalize", {}).get("enabled", False))

    # 2) 选择编码器输入：优先 color_norm，若不存在则回退 color_aug
    if use_norm and ("color_norm", 0, 0) in inputs:
        enc_in = inputs[("color_norm", 0, 0)]
    else:
        enc_in = inputs[("color_aug", 0, 0)]
    # 3) 前向
    features = runner.models["encoder"](enc_in)
    outputs = runner.models["depth"](features)

    outputs.update(runner.predict_poses(inputs, features))

    runner.generate_images_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs)
    return outputs, losses

# ---- MD2_VGGT（Monodepth2 路径 + 外部位姿注入） ----
def handle_md2_vggt(runner, inputs):
    features = runner.models["encoder"](inputs[("color_aug", 0, 0)])
    outputs = runner.models["depth"](features)

    if getattr(runner, "use_posegt", False):
        runner.generate_images_posegt(inputs, outputs)

    outputs.update(runner.predict_poses(inputs, features))

    runner.generate_images_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs)
    _apply_pose_regularizers(outputs, losses)
    return outputs, losses


# ---- GasMono（独立 handler，避免与 MD2_VGGT 共用入口） ----
def handle_gasmono(runner, inputs):
    features = runner.models["encoder"](inputs[("color_aug", 0, 0)])
    outputs = runner.models["depth"](features)

    if getattr(runner, "use_posegt", False):
        runner.generate_images_posegt(inputs, outputs)

    outputs.update(runner.predict_poses(inputs, features))

    runner.generate_images_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs)
    _apply_pose_regularizers(outputs, losses)
    return outputs, losses


def handle_madhuanand(runner, inputs):
    features0 = runner.models["encoder"](inputs[("color_aug", 0, 0)])

    available_source_ids = [
        fid for fid in runner.opt.frame_ids[1:]
        if fid != "s" and ("color_aug", fid, 0) in inputs
    ]
    if not available_source_ids:
        raise KeyError("Madhuanand requires at least one source frame in frame_ids[1:]")

    # Paper setting: depth branch uses reference Ir and successive frame Ir+1.
    if 1 in available_source_ids:
        depth_source_id = 1
    elif len(available_source_ids) == 1:
        depth_source_id = available_source_ids[0]
    else:
        positive_ids = [fid for fid in available_source_ids if isinstance(fid, int) and fid > 0]
        depth_source_id = min(positive_ids) if positive_ids else available_source_ids[0]

    features_i = runner.models["encoder"](inputs[("color_aug", depth_source_id, 0)])
    fused_features = [
        torch.stack([feat0, feat_i], dim=2) for feat0, feat_i in zip(features0, features_i)
    ]
    outputs = runner.models["depth"](fused_features)

    outputs.update(runner.predict_poses(inputs, features0))

    runner.generate_images_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs)
    _apply_pose_regularizers(outputs, losses)
    return outputs, losses


# ---- MRFEDepth（保持原分支逻辑） ----
def handle_mrfedepth(runner, inputs):
    Depthfeatures = runner.models["DepthEncoder"](inputs["color_aug", 0, 0])
    outputs = runner.models["DepthDecoder"](Depthfeatures)

    outputs.update(runner.predict_poses(inputs, Depthfeatures))

    features = runner.models['FeatureEncoder'](inputs['color', 0, 0])
    outputs.update(runner.models['FeatureDecoder'](features, 0))

    runner.generate_images_pred(inputs, outputs)
    runner.generate_features_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs, features)
    return outputs, losses

# ---- MonoViT（encoder + decoder） ----
def handle_monovit(runner, inputs):
    input_image = inputs[("color_aug", 0, 0)]
    features = runner.models["encoder"](input_image)
    outputs = runner.models["decoder"](features)

    if getattr(runner, "use_posegt", False):
        runner.generate_images_posegt(inputs, outputs)
    if getattr(runner, "use_vggt_prewarp", False):
        runner.generate_images_vggt_prewarp(inputs, outputs)

    outputs.update(runner.predict_poses(inputs, features))

    runner.generate_images_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs)
    return outputs, losses

# ---- LiteMono（与 Monodepth2 等价路径） ----
def handle_litemono(runner, inputs):
    features = runner.models["encoder"](inputs[("color_aug", 0, 0)])
    outputs = runner.models["depth"](features)

    outputs.update(runner.predict_poses(inputs, features))

    runner.generate_images_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs)
    return outputs, losses


def handle_spidepth(runner, inputs):
    features = runner.models["encoder"](inputs[("color_aug", 0, 0)])
    outputs = runner.models["depth"](features)
    target_h, target_w = inputs[("color", 0, 0)].shape[-2:]
    disp0 = outputs.get(("disp", 0), None)
    if torch.is_tensor(disp0) and disp0.shape[-2:] != (target_h, target_w):
        disp0 = F.interpolate(
            disp0,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        outputs[("disp", 0)] = disp0
        if ("spi_depth", 0) in outputs:
            outputs[("spi_depth", 0)] = disp0

    outputs.update(runner.predict_poses(inputs, features))

    runner.generate_images_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs)
    return outputs, losses


# ---- 路由入口 ----
def get_forward_handler(method: str):
    """
    返回指定 method 的 handler，可直接调用：handler(runner, inputs) -> (outputs, losses)
    未命中时回落到 Monodepth2。
    """
    table = {
        "Monodepth2": handle_monodepth2,
        "MD2_Mask": handle_monodepth2,
        "Monodepth2_DINO": handle_monodepth2_dino,
        "Monodepth2App": handle_monodepth2,
        "MD2_VGGT": handle_md2_vggt,
        "MD2_VGGT_DepthCycleViz": handle_md2_vggt,
        "MD2_VGGT_DepthSensitivityViz": handle_md2_vggt,
        "MD2_VGGT_NoPose": handle_md2_vggt,
        "MD2_VGGT_NoPose_UniformT": handle_md2_vggt,
        "MD2_VGGT_NoPose_SAlign": handle_md2_vggt,
        "MD2_VGGT_NoPose_TScale": handle_md2_vggt,
        "MD2_VGGT_TDir_PoseMag": handle_md2_vggt,
        "MD2_VGGT_TPrior_Alpha": handle_md2_vggt,
        "MD2_VGGT_TPrior_AlignRes": handle_md2_vggt,
        "MD2_VGGT_RPrior_TPose": handle_md2_vggt,
        "MD2_VGGT_Teacher_Photo": handle_md2_vggt,
        "MD2_VGGT_PoseGT": handle_md2_vggt,
        "MD2_VGGT_PoseGT_DepthCycleViz": handle_md2_vggt,
        "MD2_VGGT_PoseGT_DepthSensitivityViz": handle_md2_vggt,
        "MD2_VGGT_PoseGT_DepthSensViz": handle_md2_vggt,
        "MD2_VGGT_PoseGT_DepthSensWeight": handle_md2_vggt,
        "MD2_VGGT_PoseGT_BadScoreWeight": handle_md2_vggt,
        "MD2_VGGT_PoseGT_BadScoreLocalWeight": handle_md2_vggt,
        "MD2_VGGT_PoseGT_HRMask": handle_md2_vggt,
        "MD2_VGGT_PoseGT_Mask": handle_md2_vggt,
        "MD2_VGGT_PoseGT_DeRotHardMask": handle_md2_vggt,
        "MD2_VGGT_PoseGT_DeRotSigmoidWeight": handle_md2_vggt,
        "GasMono": handle_gasmono,
        "Madhuanand": handle_madhuanand,
        "madhuanand": handle_madhuanand,
        "MRFEDepth": handle_mrfedepth,
        "MonoViT": handle_monovit,
        "MonoViT_ResNet50_Pose": handle_monovit,
        "MonoViT_ConvNeXt_Pose": handle_monovit,
        "MonoViT_ConvNeXtSmall_Pose": handle_monovit,
        "MonoViT_ConvNeXtBase_Pose": handle_monovit,
        "MonoViT_VGGT_RDistill": handle_monovit,
        "MonoViT_VGGT_RMaskSwitch": handle_monovit,
        "MonoViT_VGGT_PreWarp": handle_monovit,
        "MonoViT_VGGT_RPrior_ResR_TPose": handle_monovit,
        "MonoViT_VGGT_RFlow_Pose": handle_monovit,
        "MonoViT_VGGT_RFlow_ResR_TPose": handle_monovit,
        "MonoViT_VGGT_RFlow_ResR_TPose_SingleHead": handle_monovit,
        "MonoViT_VGGT_RFlow_TInj": handle_monovit,
        "MonoViT_PoseGT": handle_monovit,
        "MonoViT_PoseGT_Mask": handle_monovit,
        "MonoViT_PoseGT_HRMask": handle_monovit,
        "MonoViT_VGGT_PoseGT_BadScoreWeight": handle_monovit,
        "LiteMono": handle_litemono,
        "SPIDepth": handle_spidepth,
    }
    return table.get(str(method), handle_monodepth2)

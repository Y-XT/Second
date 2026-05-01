# methods/init_and_forward.py
# -*- coding: utf-8 -*-

from methods import networks

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

# ========== models 初始化：仅构建模型，不涉及 loss ==========
def init_models(self):
    """
    按 self.opt.methods 构建 encoder/decoder，并在初始化时打印单行摘要。
    兼容 PoseNet 的启用/禁用策略，打印关键信息（与现有风格一致）。
    """
    self.use_pose_net = True
    print(f"[init_models] dataset={self.opt.dataset}, use_pose_net={self.use_pose_net}")

    self.models = {}
    self.parameters_to_train = []

    # ---------- Monodepth2 ----------
    if self.opt.methods in {"Monodepth2"}:
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

    # ---------- MonoViT ----------
    elif self.opt.methods in {
        "MonoViT",
        "MonoViT_VGGT_RFlow_TInj",
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
    else:
        raise ValueError(f"Unsupported method: {self.opt.methods}")

    # ---------- Pose 网络 ----------
    pose_num_input_images = 2 if self.opt.pose_model_input == "pairs" else len(self.opt.frame_ids)
    pose_num_layers = self.opt.num_layers
    pose_encoder_init_meta = {}
    if self.opt.methods == "MonoViT_VGGT_RFlow_TInj":
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

    if self.opt.methods == "MonoViT_VGGT_RFlow_TInj":
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

# ================================== forward handlers ==================================

# ---- Monodepth2 ----
def handle_monodepth2(runner, inputs):
    # 只使用帧 0 的图像
    features = runner.models["encoder"](inputs[("color_aug", 0, 0)])
    outputs = runner.models["depth"](features)

    outputs.update(runner.predict_poses(inputs, features))

    runner.generate_images_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs)
    return outputs, losses

# ---- MonoViT（encoder + decoder） ----
def handle_monovit(runner, inputs):
    input_image = inputs[("color_aug", 0, 0)]
    features = runner.models["encoder"](input_image)
    outputs = runner.models["decoder"](features)

    outputs.update(runner.predict_poses(inputs, features))

    runner.generate_images_pred(inputs, outputs)
    losses = runner.loss.compute_losses(inputs, outputs)
    return outputs, losses

# ---- 路由入口 ----
def get_forward_handler(method: str):
    """
    返回指定 method 的 handler，可直接调用：handler(runner, inputs) -> (outputs, losses)
    """
    table = {
        "Monodepth2": handle_monodepth2,
        "MonoViT": handle_monovit,
        "MonoViT_VGGT_RFlow_TInj": handle_monovit,
    }
    method = str(method)
    if method not in table:
        raise ValueError(f"Unsupported method: {method}")
    return table[method]

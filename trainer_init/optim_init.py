import math
import torch.optim as optim


def init_optimizers(self):
    # Legacy Monovit optimizer branch intentionally disabled.
    # if self.opt.methods == "Monovit":
    #     self.params = [
    #         {"params": self.parameters_to_train, "lr": 1e-4},
    #         {"params": list(self.models["encoder"].parameters()), "lr": self.opt.learning_rate},
    #     ]
    #     self.model_optimizer = optim.AdamW(self.params)
    #     self.model_lr_scheduler = optim.lr_scheduler.StepLR(
    #         self.model_optimizer,
    #         step_size=self.opt.scheduler_step_size,
    #         gamma=0.1
    #     )
    #     if self.opt.use_exponential_lr:
    #         self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
    #             self.model_optimizer,
    #             gamma=0.9
    #         )

    if self.opt.methods == "Monodepth2_DINO":
        # ===== 新增：DINOv3 ConvNeXt Tiny 的优化器/调度配置 =====

        # 以 options 的 --learning_rate 作为“基准 lr”
        base_lr = float(getattr(self.opt, "learning_rate", 1e-4))

        # 比例系数（若没在 opt 里设，就用默认）
        head_lr_ratio = float(getattr(self.opt, "head_lr_ratio", 1.0))  # depth/head 相对基准 lr
        bb_top_lr_ratio = float(getattr(self.opt, "bb_top_lr_ratio", 0.5))  # backbone 顶层相对基准 lr
        pose_lr_ratio = float(getattr(self.opt, "pose_lr_ratio", head_lr_ratio))

        # 允许显式覆盖（若没提供则用基准×倍率）
        lr_head = getattr(self.opt, "lr_head", None)
        if lr_head is None:
            lr_head = base_lr * head_lr_ratio

        lr_backbone_top = getattr(self.opt, "lr_backbone_top", None)
        if lr_backbone_top is None:
            lr_backbone_top = base_lr * bb_top_lr_ratio

        lr_pose = base_lr * pose_lr_ratio  # 给 pose 子网（若启用）

        # 其它常用超参
        llrd = float(getattr(self.opt, "llrd", 0.8))  # Layer-wise LR Decay
        weight_decay = float(getattr(self.opt, "weight_decay", 0.05))
        betas = tuple(getattr(self.opt, "betas", (0.9, 0.999)))
        use_cosine = bool(getattr(self.opt, "use_cosine_for_dino", False))
        warmup_steps = int(getattr(self.opt, "warmup_steps", 3000))
        min_lr_scale = float(getattr(self.opt, "min_lr_scale", 0.05))

        # 估计 total_steps（若外部未给出）
        total_steps = getattr(self.opt, "total_steps", None)
        if not total_steps and hasattr(self, "num_total_steps"):
            total_steps = self.num_total_steps
        if not total_steps and hasattr(self, "num_epochs") and hasattr(self, "train_loader"):
            total_steps = self.num_epochs * max(1, len(self.train_loader))
        if not total_steps:
            total_steps = 100000

        # no-decay 规则：Norm 或 bias 不做 weight decay
        def is_no_decay(n, p):
            return (p.ndim == 1) or n.endswith(".bias")

        param_groups = []
        seen = set()  # 防止重复加入

        def add_named_params(named_params, lr, wd):
            dec, nde = [], []
            for n, p in named_params:
                if (not p.requires_grad) or (id(p) in seen):
                    continue
                seen.add(id(p))
                (nde if is_no_decay(n, p) else dec).append(p)
            if dec:
                param_groups.append({"params": dec, "lr": lr, "weight_decay": wd})
            if nde:
                param_groups.append({"params": nde, "lr": lr, "weight_decay": 0.0})

        # ===== 1) encoder（DinoConvNeXtMultiScale）按 stage 做 LLRD =====
        enc = self.models.get("encoder", None)
        if enc is not None:
            if hasattr(enc, "stages"):
                stages = list(enc.stages)  # 常见 ConvNeXt 命名：stages[0..3]
                for si, stage in enumerate(stages[::-1]):  # 深 -> 浅
                    layer_lr = lr_backbone_top * (llrd ** si)
                    add_named_params(list(stage.named_parameters()), lr=layer_lr, wd=weight_decay)
                if hasattr(enc, "stem"):
                    stem_lr = lr_backbone_top * (llrd ** len(stages))
                    add_named_params(list(enc.stem.named_parameters()), lr=stem_lr, wd=weight_decay)

            elif hasattr(enc, "backbone") and hasattr(enc.backbone, "stages"):
                stages = list(enc.backbone.stages)
                for si, stage in enumerate(stages[::-1]):
                    layer_lr = lr_backbone_top * (llrd ** si)
                    add_named_params(list(stage.named_parameters()), lr=layer_lr, wd=weight_decay)
                if hasattr(enc.backbone, "stem"):
                    stem_lr = lr_backbone_top * (llrd ** len(stages))
                    add_named_params(list(enc.backbone.stem.named_parameters()), lr=stem_lr, wd=weight_decay)

            elif hasattr(enc, "downsample_layers") and hasattr(enc, "stages"):
                stages = list(enc.stages)
                for si, stage in enumerate(stages[::-1]):
                    layer_lr = lr_backbone_top * (llrd ** si)
                    add_named_params(list(stage.named_parameters()), lr=layer_lr, wd=weight_decay)
                small_lr = lr_backbone_top * (llrd ** len(stages))
                for mod in list(enc.downsample_layers):
                    add_named_params(list(mod.named_parameters()), lr=small_lr, wd=weight_decay)

            else:
                # 兜底：整体 encoder 一个较小 lr
                add_named_params(list(enc.named_parameters()), lr=lr_backbone_top, wd=weight_decay)

        # ===== 2) depth 头（UPerDispHead / 其它）用较大学习率 =====
        if "depth" in self.models:
            add_named_params(list(self.models["depth"].named_parameters()), lr=lr_head, wd=weight_decay)

        # ===== 3) pose 分支（若启用）也一起训练，lr 默认与 head 相同 =====
        if "pose_encoder" in self.models:
            add_named_params(list(self.models["pose_encoder"].named_parameters()), lr=lr_pose, wd=weight_decay)
        if "pose" in self.models:
            add_named_params(list(self.models["pose"].named_parameters()), lr=lr_pose, wd=weight_decay)

        # 注意：这里不再把 self.parameters_to_train 整包加入，避免与上面重复

        # ===== 优化器：AdamW =====
        self.model_optimizer = optim.AdamW(param_groups, betas=betas)

        # ===== 调度器 =====
        if use_cosine:
            # Warmup + Cosine
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                progress = max(0.0, min(1.0, progress))
                return min_lr_scale + (1.0 - min_lr_scale) * 0.5 * (1.0 + math.cos(math.pi * progress))

            self.model_lr_scheduler = optim.lr_scheduler.LambdaLR(self.model_optimizer, lr_lambda)
        else:
            # 默认仍用 StepLR（和你其余分支一致）
            self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer,
                step_size=self.opt.scheduler_step_size,
                gamma=0.1
            )

        # 如果开启了 use_exponential_lr，则覆盖
        if getattr(self.opt, "use_exponential_lr", False):
            self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.model_optimizer,
                gamma=0.9
            )

        # （可选调试）打印各组 lr / wd / 参数量，确认 LLRD/分组是否生效
        try:
            tot = 0
            for i, g in enumerate(self.model_optimizer.param_groups):
                n_params = sum(p.numel() for p in g["params"])
                tot += n_params
                print(f"[DINO OPT] group {i}: lr={g['lr']:.2e}, wd={g.get('weight_decay', 0)}, n={n_params / 1e6:.2f}M")
            print(f"[DINO OPT] total params in optimizer: {tot / 1e6:.2f}M")
        except Exception:
            pass

    else:
        # ===== 原样保留（未改动） =====
        # 默认的优化器设置
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        # 学习率调度器设置（默认使用 StepLR）
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer,
            step_size=self.opt.scheduler_step_size,
            gamma=0.1
        )

        # LiteMono 特有的调度器：ChainedScheduler（需已安装相关库）
        # 如果开启了 use_exponential_lr，则覆盖默认的 StepLR 或 LiteMono 的配置
        if self.opt.use_exponential_lr:
            self.model_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                self.model_optimizer,
                gamma=0.9
            )
    """
    elif self.opt.methods == "LiteMono":
        self.model_lr_scheduler = ChainedScheduler(
            self.model_optimizer,
            T_0=int(self.opt.lr[2]),
            T_mul=1,
            eta_min=self.opt.lr[1],
            last_epoch=-1,
            max_lr=self.opt.lr[0],
            warmup_steps=0,
            gamma=0.9
        )

        # 如果使用独立的 pose 优化器和调度器，也一并配置
        self.model_pose_lr_scheduler = ChainedScheduler(
            self.model_pose_optimizer,
            T_0=int(self.opt.lr[5]),
            T_mul=1,
            eta_min=self.opt.lr[4],
            last_epoch=-1,
            max_lr=self.opt.lr[3],
            warmup_steps=0,
            gamma=0.9
        )    

    """

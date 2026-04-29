# trainer_init/logging_init.py
import wandb
import os
def init_logging(trainer):
    """
    初始化 W&B 日志（可兼容本地单跑与 sweep）。
    约定：Trainer.__init__ 会在 init_logging(self) 之后调用
          _rebuild_model_name_and_paths() 来用“最终参数”生成 model_name。
    """
    opt = trainer.opt
    trainer.using_wandb = True

    # === 0) 将 W&B 本地目录固定到 opt.log_dir 下 ===
    # 例如：logs/exp_001/wandb
    #base_log_dir = getattr(opt, "log_dir", None)
    #wandb_dir = None
    #if base_log_dir:
    #    wandb_dir = os.path.join(base_log_dir, "wandb")
    #    os.makedirs(wandb_dir, exist_ok=True)
        # 可选：用环境变量影响内部默认（Artifacts/缓存亦可定向）
    #    os.environ.setdefault("WANDB_DIR", wandb_dir)
        # os.environ.setdefault("WANDB_ARTIFACT_DIR", os.path.join(base_log_dir, "artifacts"))
        # os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(base_log_dir, "cache"))
    wandb_dir= "/home/yxt/文档/mono_result/wandb"
    # === 1) 准备“安全配置”：排除会引发冲突/无意义写入的键 ===
    EXCLUDE_KEYS = {"model_name", "log_dir", "log_path"}
    safe_cfg = {k: v for k, v in vars(opt).items() if k not in EXCLUDE_KEYS}

    # === 2) 初始化或复用现有 run ===
    # 说明：
    # - 不在这里设置 name（run 名称）；等 Trainer 重建完 model_name 后再设置：
    #   你已在 Trainer.save_opts() 内通过 wandb.run.name = self.opt.model_name 完成该步骤。
    if wandb.run is None:
        init_kwargs = {
            "project": getattr(opt, "project_name", None) or "monodepth",
            "config": safe_cfg
        }
        entity = getattr(opt, "wandb_entity", None)
        if entity:
            init_kwargs["entity"] = entity
        if wandb_dir:
            init_kwargs["dir"] = wandb_dir  # 关键：落盘到 opts.log_dir/wandb
        try:
            wandb.init(**init_kwargs)
        except Exception as e:
            # 不阻断训练，降级为非 W&B 模式
            trainer.using_wandb = False
            print(f"[WARN] W&B 初始化失败：{e}")
            return
    else:
        # 已存在 run：仅同步配置（允许值变化，避免 sweep/代码二次覆盖冲突）
        try:
            wandb.config.update(safe_cfg, allow_val_change=True)
        except Exception as e:
            print(f"[WARN] W&B 配置更新失败：{e}")

    # === 3) 将（可能来自 sweep 的）最终配置回灌到 opt ===
    # 这样 Trainer 后续使用的 opt 是“最终值”，保证命名/保存路径一致
    try:
        cfg = dict(wandb.config)
        for k, v in cfg.items():
            if hasattr(opt, k):
                setattr(opt, k, v)
    except Exception as e:
        print(f"[WARN] 回灌 W&B 配置到 opt 失败：{e}")

    # === 4) 定义聚合指标（便于在 W&B 面板聚合/早停） ===
    try:
        wandb.define_metric("val/loss", summary="min")
        # 可选：如需在面板中追踪最后一步的 train/loss
        # wandb.define_metric("train/loss", summary="last")
    except Exception:
        pass

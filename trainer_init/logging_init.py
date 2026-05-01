import os

import wandb


def init_logging(trainer):
    """Initialize W&B logging without making it required for training."""
    opt = trainer.opt
    trainer.using_wandb = True

    base_log_dir = getattr(opt, "log_dir", None)
    wandb_dir = os.path.join(base_log_dir, "wandb") if base_log_dir else None
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)

    exclude_keys = {"model_name", "log_dir", "log_path"}
    safe_cfg = {k: v for k, v in vars(opt).items() if k not in exclude_keys}

    if wandb.run is None:
        init_kwargs = {
            "project": getattr(opt, "project_name", None) or "monodepth",
            "config": safe_cfg,
        }
        entity = getattr(opt, "wandb_entity", None)
        if entity:
            init_kwargs["entity"] = entity
        if wandb_dir:
            init_kwargs["dir"] = wandb_dir

        try:
            wandb.init(**init_kwargs)
        except Exception as exc:
            trainer.using_wandb = False
            print(f"[WARN] W&B initialization failed, logging disabled: {exc}")
            return
    else:
        try:
            wandb.config.update(safe_cfg, allow_val_change=True)
        except Exception as exc:
            print(f"[WARN] W&B config update failed: {exc}")

    try:
        wandb.define_metric("val/loss", summary="min")
    except Exception:
        pass

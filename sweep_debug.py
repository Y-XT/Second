# sweep_debug.py
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import yaml
import wandb
from trainer import Trainer
from options import MonodepthOptions

# ==== 修改成你的 sweep 配置 / 项目信息 ====
SWEEP_YAML = "./sweep/Monodepth2_DINO.yaml"
ENTITY = "Yxt_Ikyrie"
PROJECT = "UAVula_new_DINODPT"
# =========================================

# 这是你现在 train.py 里 sys.argv 注入的“UAVula_new”配置，原样搬过来
BASE_ARGS = [
    "--project_name","UAVula_new",
    "--methods","Monodepth2_DINO",
    "--log_dir","/home/yxt/文档/mono_result/weights/UAVula_new",
    "--model_name","",
    "--split","UAVula",
    "--dataset","UAVula_Dataset",
    "--data_path","/mnt/data_nvme3n1p1/dataset/UAV_ula/dataset",
    "--batch_size","8",
    # 【注意】learning_rate / scheduler_step_size 由 wandb.config 覆盖，不在这里写死
    "--num_workers","16",
    "--frame_ids","0","-1","1",
    "--height","288",
    "--width","512",
    "--num_epochs","1",
    "--max_depth","150.0",
    "--log_frequency","300",  # 调试建议先降低，确保 val 会触发
    "--scales","0","1","2","3",
]

def train_fn():
    run = wandb.init(project=PROJECT, entity=ENTITY)

    # 1) 解析你的既有 CLI 参数（不从命令行读，直接用 BASE_ARGS）
    opt = MonodepthOptions().parse(BASE_ARGS)

    # 2) 用 sweep 的超参覆盖（与 sweep.yaml 的键一致）
    cfg = wandb.config
    if "learning_rate" in cfg:
        opt.learning_rate = float(cfg.learning_rate)
    if "scheduler_step_size" in cfg:
        opt.scheduler_step_size = int(cfg.scheduler_step_size)

    # 3) 开训（在 trainer.py 的 val()/log() 里打断点）
    Trainer(opt).train()

    run.finish()

if __name__ == "__main__":
    with open(SWEEP_YAML, "r", encoding="utf-8") as f:
        sweep_cfg = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep_cfg, entity=ENTITY, project=PROJECT)

    # 调试先跑 1 次；需要多 trial 把 count 调大即可
    wandb.agent(sweep_id, function=train_fn, count=10)

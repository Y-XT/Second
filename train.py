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
# 设置图形相关的环境变量，确保在无显示服务器的训练机器上也能正常导入可视化库。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
# `log_dir` and `model_name` together define where and how the model is saved.
#(1024,576),(512,288)
# remember to change k in dataset
# Madhuanand, Monodepth2

#wandb login
#wandb sweep ./sweep/Monodepth2_DINO.yaml
#CUDA_VISIBLE_DEVICES=0 wandb agent --count 10 Yxt_Ikyrie/UAVula_new_DINODPT/w0ps02ma



"""

"""
# 下面的预设参数通过直接修改 sys.argv 注入默认 CLI，用于快速在不同数据集上启动训练。
# 根据需要取消注释对应块即可；保持字符串列表顺序与 MonodepthOptions 定义一致。
"""

sys.argv += [
    "--project_name","UAVid_germany",
    "--methods", "MonoViT",  # ✅ 添加该参数
    "--log_dir", "/mnt/data_nvme3n1p1/mono_weights/weights/UAVid_germany",
    "--model_name", "",
    "--split", "UAVid2020_Germany",
    "--dataset", "UAVid2020",
    "--data_path", "/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/Germany",
    "--batch_size", "8",
    "--learning_rate", "1e-4",
    "--num_workers", "16",
    "--frame_ids", "0", "-10", "10",
    "--height", "288",
    "--width", "512",
    "--num_epochs", "40",
    "--scheduler_step_size", "20",
    "--max_depth", "150.0",
    "--log_frequency","300",
    "--scales", "0", "1", "2", "3",
    "--enable_debug_metrics",
    # ✅ Enable this store_true flag:
    #"--distorted_mask"
]

sys.argv += [
    "--project_name","UAVula",
    "--methods", "MonoViT",  # ✅ 添加该参数
    "--log_dir", "/mnt/data_nvme3n1p1/mono_weights/weights/UAVula",
    "--model_name", "",
    "--split", "UAVula",
    "--dataset", "UAVula_Dataset",
    "--data_path", "/mnt/data_nvme3n1p1/dataset/UAV_ula/dataset",
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
    "--enable_debug_metrics",
    "--enable_flip",
    # ✅ Enable this store_true flag:
    #"--distorted_mask"
]
# UAVid Germany 配置：与上面相同，只是数据源与帧间距不同。 frame+-10
# UAVid China 配置：与上面相同，只是数据源与帧间距不同。 frame+-5
# UAVula 配置：与上面相同，只是数据源与帧间距不同。 frame+-1

"""
sys.argv += [
    "--project_name","UAVid_China",
    "--methods", "MonoViT",  # ✅ 添加该参数
    "--log_dir", "/mnt/data_nvme3n1p1/mono_weights/weights/UAVid_China",
    "--model_name", "",
    "--split", "UAVid2020_China",
    "--dataset", "UAVid2020",
    "--data_path", "/mnt/data_nvme3n1p1/dataset/UAVid2020/uavid_v1.5_official_release/China",
    "--batch_size", "8",
    "--learning_rate", "1e-4",
    "--num_workers", "16",
    "--frame_ids", "0", "-5", "5",
    "--height", "288",
    "--width", "512",
    "--num_epochs", "40",
    "--scheduler_step_size", "20",
    "--max_depth", "150.0",
    "--log_frequency","300",
    "--scales", "0", "1", "2", "3",
    "--enable_debug_metrics",
    "--enable_flip",

    # ✅ Enable this store_true flag:
    #"--distorted_mask"
        # automask 可视化：训练时直接生成 identity/reproj/margin 对比图
    "--enable_automask_margin_viz",
    #"--save_automask_margin_viz_local",
    "--automask_margin_viz_samples", "1",
    #"--r_distill_margin_thresh", "0.1",
]

def main():
    options = MonodepthOptions()
    # 解析命令行参数（含上方注入的默认参数），内部会自动补全 run 名称。
    opts = options.parse()  # ★ 自动命名在 options.parse() 内完成
    if getattr(opts, "seed", None) is not None and int(opts.seed) >= 0:
        set_global_seed(int(opts.seed), deterministic=True)
        print(f"[INFO] Seed:        {int(opts.seed)} (cudnn deterministic)")
    else:
        print("[INFO] Seed:        disabled")

    # 友好输出与保存目录准备
    #save_dir = os.path.join(opts.log_dir, opts.model_name)
    print(f"[INFO] Project:      {opts.project_name}")
    print(f"[INFO] Model name:   {opts.model_name}")

    # 构建 Trainer 并启动完整的训练流程。
    trainer = Trainer(opts)
    trainer.train()

if __name__ == "__main__":
    main()

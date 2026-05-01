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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

#China
"""
sys.argv += [
    "--project_name","UAVid_China",
    "--methods", "MonoViT_VGGT_RFlow_TInj",  # ✅ W_img + W_pix 联合加权
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
    "--scales", "0","1","2","3",

    # automask 可视化：训练时直接生成 identity/reproj/margin 对比图
    "--enable_automask_margin_viz",
    #"--save_automask_margin_viz_local",
    "--automask_margin_viz_samples", "1",
    
    # ✅ Enable this store_true flag:
    "--enable_debug_metrics",
    "--enable_flip",
]
"""

#Germany
"""
sys.argv += [
"--project_name","UAVid_Germany",
    "--methods", "MonoViT_VGGT_RFlow_TInj",  # ✅ 添加该参数
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
    "--scales", "0","1","2","3",

    # automask 可视化：训练时直接生成 identity/reproj/margin 对比图
    "--enable_automask_margin_viz",
    #"--save_automask_margin_viz_local",
    "--automask_margin_viz_samples", "1",
    
    # ✅ Enable this store_true flag:
    "--enable_debug_metrics",
    "--enable_flip",
]
"""
#ULA
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
    "--enable_automask_margin_viz",
    #"--save_automask_margin_viz_local",
    "--automask_margin_viz_samples", "1",
    
    # ✅ Enable this store_true flag:
    "--enable_debug_metrics",
    "--enable_flip",
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

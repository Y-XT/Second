# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

DEFAULT_MODELS_TO_LOAD = {
    "Monodepth2": ["encoder", "depth", "pose_encoder", "pose"],
    "MonoViT": ["encoder", "decoder", "pose_encoder", "pose"],
    "MonoViT_VGGT_RFlow_TInj": ["encoder", "decoder", "pose_encoder", "pose"],
}

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        #METHODS
        self.parser.add_argument("--methods",
                                 type=str,
                                 help="methods for training",
                                 choices=[
                                     "Monodepth2",
                                     "MonoViT",
                                     "MonoViT_VGGT_RFlow_TInj",
                                 ],
                                 default="Monodepth2")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))

        # TRAINING options
        self.parser.add_argument("--project_name",
                                 type=str,
                                 help="the name of the project",
                                 default="UAVula")
        self.parser.add_argument("--seed",
                                type=int,
                                default=42,
                                help="random seed for reproducibility; set <0 to disable seeding"
                                )

        # 1) __init__ 里：定义/覆盖 --model_name（默认 None）
        self.parser.add_argument("--model_name",
                                type=str,
                                default=None,
                                help="模型保存目录名；留空或设为 'auto' 则按约定字段自动生成"
        )
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["UAVid2020_China","UAVid2020_Germany","UAVula","UAVula_R1"],
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
                                 choices=["UAVid2020","UAVula_Dataset","UAVula_TriDataset","UAVid_TriDataset"]) 
        
        # UAVula TriDataset 额外参数
        self.parser.add_argument("--triplet_root",
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
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])

        self.parser.add_argument("--enable_debug_metrics",
                                action="store_true",
                                help="开启额外的 photometric/pose 统计指标（默认关闭，节省开销）"
                                )
        self.parser.add_argument("--automask_hr_percentile",
                                type=float,
                                default=90.0,
                                help="automask 统计中高残差阈值的分位数（如 90 表示 top 10%）"
                                )
        self.parser.add_argument("--automask_hr_scope",
                                type=str,
                                default="all",
                                choices=["all", "mask"],
                                help="高残差阈值的像素范围：all=全像素，mask=仅在 keep 像素中估计阈值"
                                )
        self.parser.add_argument("--enable_automask_margin_viz",
                                action="store_true",
                                help="训练 log() 时可视化 automask 比较图：identity_min / reproj_min / margin"
                                )
        self.parser.add_argument("--automask_margin_viz_samples",
                                type=int,
                                default=1,
                                help="每次 train/val 日志时保存多少个 automask 比较样本（默认 1）"
                                )
        self.parser.add_argument("--enable_flip",
                                action="store_true",
                                help="启用训练阶段的随机水平翻转增强（默认关闭）"
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
                                 default=None)
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

        method_name = str(getattr(opts, "methods", ""))
        if opts.models_to_load is None:
            opts.models_to_load = list(DEFAULT_MODELS_TO_LOAD.get(method_name, []))

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

        return opts

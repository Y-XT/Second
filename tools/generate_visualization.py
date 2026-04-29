# generate_visualization.py
# ---------------------------------------------
# 用途：
#   - 从预测结果文件夹中读取视差图（.npy），形状为 [1,1,H,W]
#   - 从 GT 文件夹读取对应的深度图（.tif）
#   - 将预测结果 resize 到 GT 大小，并用中值缩放统一尺度
#   - 生成统一色图范围下的彩色视差图（colormap）
#   - 生成 GT 与预测之间的差值图（colormap）
#   - 支持控制是否只生成彩色图或只生成差值图，默认都生成
#
# 使用示例：
#   python generate_visualization.py --pred_dir ./pred --gt_dir ./gt
#   python generate_visualization.py --pred_dir ./pred --gt_dir ./gt --gen_pred_color
#   python generate_visualization.py --pred_dir ./pred --gt_dir ./gt --gen_diff_color

import os
import numpy as np
from PIL import Image
import argparse
import matplotlib as mpl
import matplotlib.cm as cm
from glob import glob
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate colorized prediction and error maps")
    parser.add_argument('--pred_dir', type=str, required=True, help="Path to directory with *_depth.npy files")
    parser.add_argument('--gt_dir', type=str, required=True, help="Path to directory with ground truth .tif files")
    parser.add_argument('--output_dir', type=str, default="./output_vis", help="Directory to save color images")
    parser.add_argument('--gen_pred_color', action='store_true', help="Generate prediction colormap image")
    parser.add_argument('--gen_gt_color', action='store_true', help="Generate GT colormap image")
    parser.add_argument('--gen_diff_color', action='store_true', help="Generate difference colormap image")
    return parser.parse_args()

def create_colormap_with_fixed_range(img_np, vmin, vmax):
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap='magma')
    return (mapper.to_rgba(img_np)[:, :, :3] * 255).astype(np.uint8)

def main():
    args = parse_args()

    # 默认三个都生成
    gen_pred = args.gen_pred_color or not (args.gen_pred_color or args.gen_gt_color or args.gen_diff_color)
    gen_gt = args.gen_gt_color or not (args.gen_pred_color or args.gen_gt_color or args.gen_diff_color)
    gen_diff = args.gen_diff_color or not (args.gen_pred_color or args.gen_gt_color or args.gen_diff_color)

    if gen_pred:
        os.makedirs(os.path.join(args.output_dir, "pred_color"), exist_ok=True)
    if gen_gt:
        os.makedirs(os.path.join(args.output_dir, "gt_color"), exist_ok=True)
    if gen_diff:
        os.makedirs(os.path.join(args.output_dir, "diff_color"), exist_ok=True)

    gt_files = sorted(glob(os.path.join(args.gt_dir, "*.tif")))
    matched = 0

    for gt_path in tqdm(gt_files):
        base_name = os.path.splitext(os.path.basename(gt_path))[0]
        pred_path = os.path.join(args.pred_dir, base_name + "_depth.npy")
        if not os.path.exists(pred_path):
            continue  # 如果找不到对应的预测，跳过

        matched += 1

        gt = np.array(Image.open(gt_path), dtype=np.float32)
        pred_raw = np.load(pred_path)
        pred = pred_raw.squeeze()

        print(f"[{base_name}] 原始pred shape: {pred_raw.shape}")
        print(f"[{base_name}] Squeeze 后pred shape: {pred.shape}")
        print(f"[{base_name}] GT shape: {gt.shape}")


        # GT 范围过滤：只用 [0.01, 150] 范围参与计算
        valid_gt_mask = (gt > 0.01) & (gt < 150)
        valid_gt = gt[valid_gt_mask]
        if len(valid_gt) == 0:
            print(f"[{base_name}] ⚠️ GT 中无有效像素，跳过")
            continue

        gt_median = np.median(valid_gt)
        pred_median = np.median(pred)
        print(f"[{base_name}] pred min={pred.min():.4f}, max={pred.max():.4f}, median={pred_median:.4f}")
        print(f"[{base_name}] GT     min={gt.min():.4f}, max={gt.max():.4f}, median={gt_median:.4f}")

        if pred_median == 0:
            print(f"[{base_name}] ⚠️ 预测中值为 0，跳过")
            continue

        scale = gt_median / pred_median
        pred_scaled = pred * scale

        # Resize 预测图到 GT 尺寸
        pred_resized = np.array(Image.fromarray(pred_scaled).resize(gt.shape[::-1], Image.BILINEAR))

        # 差值图（误差图）
        # 使用与 GT 中值一样的有效像素条件
        valid_mask = (gt > 0.01) & (gt < 150)

        # 安全：GT 形状为 (H, W)，pred_resized 也是 (H, W)
        diff = np.zeros_like(gt)
        diff[valid_mask] = np.abs(pred_resized[valid_mask] - gt[valid_mask])

        # 联合色图范围（GT 和 预测）
        combined_min = min(pred_resized.min(), gt.min())
        combined_max = max(pred_resized.max(), gt.max())

        print(f"[{base_name}] pred_min={pred_resized.min():.2f}, max={pred_resized.max():.2f}, "
              f"gt_min={gt.min():.2f}, max={gt.max():.2f}")

        if gen_gt:
            gt_color = create_colormap_with_fixed_range(gt, vmin=combined_min, vmax=combined_max)
            gt_save_path = os.path.join(args.output_dir, "gt_color", base_name + "_gt.jpeg")
            Image.fromarray(gt_color).save(gt_save_path)

        if gen_pred:
            pred_color = create_colormap_with_fixed_range(pred_resized, vmin=combined_min, vmax=combined_max)
            pred_save_path = os.path.join(args.output_dir, "pred_color", base_name + "_pred.jpeg")
            Image.fromarray(pred_color).save(pred_save_path)

        if gen_diff:
            diff_color = create_colormap_with_fixed_range(diff, vmin=0, vmax=np.percentile(diff, 95))
            diff_save_path = os.path.join(args.output_dir, "diff_color", base_name + "_diff.jpeg")
            Image.fromarray(diff_color).save(diff_save_path)

    print(f"\n✅ 共处理成功 {matched} 对 GT + 预测深度图")

if __name__ == "__main__":
    main()
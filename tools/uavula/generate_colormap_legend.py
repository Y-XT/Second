import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Generate standalone color legend image")
    parser.add_argument('--vmin', type=float, default=0.0, help='Minimum value of the color scale (default: 0)')
    parser.add_argument('--vmax', type=float, default=100.0, help='Maximum value of the color scale (default: 100)')
    parser.add_argument('--output', type=str, default='legend.png', help='Output path of the legend image')
    parser.add_argument('--label_min', type=str, default='0', help='Label for the minimum value')
    parser.add_argument('--label_max', type=str, default='max', help='Label for the maximum value')
    parser.add_argument('--cmap', type=str, default='magma', help='Colormap to use (default: magma)')
    return parser.parse_args()


def main():
    args = parse_args()

    fig, ax = plt.subplots(figsize=(1, 4))  # 长条状竖直图例
    fig.subplots_adjust(left=0.5, right=0.8)

    norm = mpl.colors.Normalize(vmin=args.vmin, vmax=args.vmax)
    cbar = mpl.colorbar.ColorbarBase(
        ax, cmap=plt.get_cmap(args.cmap), norm=norm, orientation='vertical'
    )
    cbar.set_ticks([args.vmin, args.vmax])
    cbar.set_ticklabels([args.label_min, args.label_max])
    cbar.ax.tick_params(labelsize=12)

    plt.savefig(args.output, dpi=300, bbox_inches='tight', transparent=False)
    print(f"✅ 图例已保存至: {args.output}")

if __name__ == '__main__':
    main()

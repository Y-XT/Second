#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UAVula 深度统计可视化脚本。

脚本读取 `analyze_val_depth_distribution.py` 生成的 JSON，输出三类图像：
  1. 数据集总览（分位曲线 + 裁剪区间摘要）
  2. 按场景的 5-95% 区间对比
  3. 按场景的分位曲线

默认输入输出均在 tools/uavula/ 目录，可直接执行：

    python tools/uavula/plot_depth_scene.py

如需自定义路径，可使用以下参数：

    python tools/uavula/plot_depth_scene.py \\
        --stats-json tools/uavula/uavula_depth_stats.json \\
        --overview tools/uavula/depth_distribution_overview.png \\
        --scene-bars tools/uavula/depth_distribution_by_scene.png \\
        --scene-quantiles tools/uavula/depth_scene_quantiles.png
"""

import argparse
import json
import math
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATS_JSON = os.path.join(CURRENT_DIR, "uavula_depth_stats.json")
DEFAULT_OVERVIEW = os.path.join(CURRENT_DIR, "depth_distribution_overview.png")
DEFAULT_SCENE_BARS = os.path.join(CURRENT_DIR, "depth_distribution_by_scene.png")
DEFAULT_SCENE_QUANT = os.path.join(CURRENT_DIR, "depth_scene_quantiles.png")

COLOR_TRIMMED = "#FFA94D"  # 亮橙色
COLOR_TRIM_BAND = "#748FFC"
COLOR_TRIM_FILL = "#D0EBFF"
COLOR_Q25 = "#5C7CFA"
COLOR_Q50 = "#2B8A3E"
COLOR_Q75 = "#FAB005"
VALUE_MODES = ("raw", "trimmed", "scaled")
DEFAULT_VALUE_MODE = "trimmed"


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def _configure_fonts() -> None:
    """尝试自动选择支持中文的无衬线字体，避免图像中文乱码。"""
    try:
        from matplotlib import font_manager as fm
    except ImportError:
        return

    candidates = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "PingFang SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    for name in candidates:
        try:
            fm.findfont(name, fallback_to_default=False)
        except (ValueError, RuntimeError):
            continue
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [name]
        break
    matplotlib.rcParams["axes.unicode_minus"] = False


_configure_fonts()


def _parse_quantiles(quantiles: Dict[str, float]) -> Tuple[List[float], List[float]]:
    """将 {"q05": value, ...} 排序转换为百分位和取值列表。"""
    if not quantiles:
        return [], []
    items = []
    for key, value in quantiles.items():
        try:
            percent = float(key.lstrip("q"))
        except ValueError:
            continue
        items.append((percent, value))
    items.sort(key=lambda x: x[0])
    if not items:
        return [], []
    percents, values = zip(*items)
    return list(percents), list(values)


def _resolve_scale_window(trim_stats: Dict) -> Tuple[float, float]:
    """若截尾统计可用，则返回 (low, high)。"""
    if not trim_stats:
        return math.nan, math.nan
    low = trim_stats.get("low")
    high = trim_stats.get("high")
    if low is None or high is None:
        return math.nan, math.nan
    if not math.isfinite(low) or not math.isfinite(high):
        return math.nan, math.nan
    if high <= low:
        return math.nan, math.nan
    return float(low), float(high)


def _scale_series(values: List[float], trim_stats: Dict, mode: str) -> List[float]:
    """根据截尾区间将序列缩放到 [0, 1]（仅 scaled 模式下执行）。"""
    if mode != "scaled":
        return values
    low, high = _resolve_scale_window(trim_stats)
    if math.isnan(low) or math.isnan(high):
        return values
    span = high - low
    if span <= 0:
        return values
    arr = np.asarray(values, dtype=np.float64)
    scaled = (arr - low) / span
    return list(np.clip(scaled, 0.0, 1.0))


def _scale_value(val: float, trim_stats: Dict, mode: str) -> float:
    """在 scaled 模式下按截尾区间缩放单个数值。"""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return val
    if mode != "scaled":
        return val
    low, high = _resolve_scale_window(trim_stats)
    if math.isnan(low) or math.isnan(high):
        return val
    span = high - low
    if span <= 0:
        return val
    return float(np.clip((val - low) / span, 0.0, 1.0))


def _get_quantile(summary: Dict, percent: int) -> float:
    quantiles = summary.get("quantiles", {})
    key = f"q{percent:02d}"
    val = quantiles.get(key)
    return None if val is None else float(val)


def _strip_scene_label(name: str) -> str:
    """裁剪场景名称，仅保留末尾片段，避免左侧标签过长。"""
    if not name:
        return name
    if "/" in name:
        _, tail = name.rsplit("/", 1)
        return tail
    return name


def _apply_trimmed_limits(ax, trim_stats: Dict, mode: str, axis: str = "y") -> None:
    """依据显示模式调整坐标轴范围。"""
    if mode == "scaled":
        if axis == "y":
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_xlim(0.0, 1.0)
        return

    if mode != "trimmed":
        return

    low, high = _resolve_scale_window(trim_stats)
    if math.isnan(low) or math.isnan(high):
        return
    margin = (high - low) * 0.05 if high > low else 1.0
    lower = low - margin
    upper = high + margin
    if axis == "y":
        ax.set_ylim(lower, upper)
    else:
        ax.set_xlim(lower, upper)


def _get_value_label(mode: str) -> str:
    if mode == "scaled":
        return "Scaled depth (q05->0, q95->1)"
    if mode == "trimmed":
        return "Depth value (focused on q05~q95)"
    return "Depth value"


def _add_dataset_info(ax, summary: Dict, dataset_name: str, mode: str) -> None:
    """在角标展示样本数量与有效像素占比。"""
    files = summary.get("files", {})
    pixels = summary.get("pixels", {})
    trim_stats = summary.get("trim_05_95", {})
    samples = files.get("processed", 0)
    missing = files.get("missing_depth", 0)
    ratio = pixels.get("valid_ratio", None)
    text_lines = [
        f"Samples: {samples} (missing {missing})",
        f"Valid ratio: {ratio:.2%}" if ratio not in (None, 0) else "Valid ratio: N/A",
    ]
    if mode in ("scaled", "trimmed"):
        low, high = _resolve_scale_window(trim_stats)
        if not math.isnan(low) and not math.isnan(high):
            label = "Scale window" if mode == "scaled" else "Trimmed window"
            text_lines.append(f"{label}: q05={low:.2f}, q95={high:.2f}")

    ax.text(
        0.98,
        0.95,
        "\n".join(text_lines),
        ha="right",
        va="top",
        fontsize=9,
        color="dimgray",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6, edgecolor="lightgray"),
    )
    ax.set_title(dataset_name, fontsize=12, fontweight="bold")


# --------------------------------------------------------------------------- #
# 数据集总览绘图
# --------------------------------------------------------------------------- #
def _plot_dataset_quantiles(ax, summary: Dict, dataset_label: str, mode: str) -> None:
    percents, values = _parse_quantiles(summary.get("quantiles", {}))
    if not percents:
        ax.text(0.5, 0.5, "No quantile data", ha="center", va="center", fontsize=11, color="firebrick")
        ax.axis("off")
        return

    trim_stats = summary.get("trim_05_95", {})
    perc_arr = np.asarray(percents, dtype=np.float64)
    val_arr = np.asarray(values, dtype=np.float64)

    if mode == "trimmed":
        mask = (perc_arr >= 5.0) & (perc_arr <= 95.0)
        if np.any(mask):
            perc_arr = perc_arr[mask]
            val_arr = val_arr[mask]

    if mode == "scaled":
        val_arr = np.asarray(_scale_series(list(val_arr), trim_stats, mode), dtype=np.float64)

    ax.plot(perc_arr, val_arr, marker="o", color=COLOR_TRIM_BAND, linewidth=2)
    if perc_arr.size >= 2:
        ax.fill_between(perc_arr, val_arr, color=COLOR_TRIM_FILL, alpha=0.35)

    if mode == "trimmed":
        ax.set_xlim(5, 95)
    ax.set_xticks(perc_arr)
    ax.set_xticklabels([f"{p:.0f}" for p in perc_arr])
    ax.set_xlabel("Quantile (%)")
    ax.set_ylabel(_get_value_label(mode))
    _apply_trimmed_limits(ax, trim_stats, mode)

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title(f"{dataset_label}: Quantile Trend (5-95%)", fontsize=11)


def _plot_dataset_trimmed_summary(ax, summary: Dict, dataset_label: str, mode: str) -> None:
    trim_stats = summary.get("trim_05_95", {})
    low_val = trim_stats.get("low")
    high_val = trim_stats.get("high")
    if low_val is None or high_val is None:
        ax.text(0.5, 0.5, "No trimmed statistics", ha="center", va="center", fontsize=11, color="firebrick")
        ax.axis("off")
        return

    low_disp = _scale_value(low_val, trim_stats, mode)
    high_disp = _scale_value(high_val, trim_stats, mode)
    if low_disp is None or high_disp is None:
        ax.text(0.5, 0.5, "Trimmed window unavailable", ha="center", va="center", fontsize=11, color="firebrick")
        ax.axis("off")
        return

    base_y = 0.0
    ax.hlines(base_y, low_disp, high_disp, color=COLOR_TRIM_BAND, linewidth=8, alpha=0.4)

    legend_entries: List[Tuple[object, str]] = []
    seen = set()
    occupied: Dict[float, int] = {}
    span = high_disp - low_disp
    if not math.isfinite(span) or span <= 0:
        span = 1.0
    shift_unit = max(span * 0.18, 0.2)
    if mode == "scaled":
        x_lower, x_upper = 0.0, 1.0
    else:
        margin_ratio = 0.35 if mode == "trimmed" else 0.25
        margin = span * margin_ratio if span > 0 else 1.0
        x_lower = min(low_disp, high_disp) - margin
        x_upper = max(low_disp, high_disp) + margin
        if not math.isfinite(x_lower) or not math.isfinite(x_upper) or x_upper <= x_lower:
            x_lower, x_upper = low_disp - 1.0, high_disp + 1.0
    clamp_pad = max((x_upper - x_lower) * 0.04, 0.05)

    def add_marker(
        value: float,
        marker: str,
        color: str,
        label: str,
        text_offset: float,
        *,
        text_prefix: str,
        x_offset: float = 0.0,
        ha: str = "center",
    ) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return
        display_val = _scale_value(value, trim_stats, mode)
        if display_val is None or (isinstance(display_val, float) and math.isnan(display_val)):
            return
        key = round(display_val, 3)
        count = occupied.get(key, 0)
        shift = 0.0
        if count > 0:
            direction = -1 if count % 2 else 1
            magnitude = shift_unit * ((count + 1) // 2)
            shift = direction * magnitude
        occupied[key] = count + 1
        text_x = display_val + x_offset + shift
        text_x = min(max(text_x, x_lower + clamp_pad), x_upper - clamp_pad)
        artist = ax.scatter(
            display_val,
            base_y,
            marker=marker,
            s=90,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label if label not in seen else None,
            zorder=5,
        )
        if label not in seen:
            legend_entries.append((artist, label))
            seen.add(label)
        text = f"{text_prefix}{value:.2f}"
        ax.text(
            text_x,
            base_y + text_offset,
            text,
            ha=ha,
            va="bottom" if text_offset >= 0 else "top",
            fontsize=8,
            color=color,
        )

    q05 = _get_quantile(summary, 5)
    q50 = _get_quantile(summary, 50)
    q95 = _get_quantile(summary, 95)
    trim_mean = trim_stats.get("mean")

    add_marker(
        q05,
        marker="|",
        color=COLOR_TRIM_BAND,
        label="5% quantile",
        text_offset=-0.12,
        text_prefix="5%: ",
        x_offset=-shift_unit * 0.4,
        ha="right",
    )
    add_marker(
        q50,
        marker="o",
        color="#2B8A3E",
        label="Median",
        text_offset=0.18,
        text_prefix="Median: ",
    )
    add_marker(
        q95,
        marker="|",
        color=COLOR_TRIM_BAND,
        label="95% quantile",
        text_offset=-0.12,
        text_prefix="95%: ",
        x_offset=shift_unit * 0.4,
        ha="left",
    )
    add_marker(
        trim_mean,
        marker="D",
        color=COLOR_TRIMMED,
        label="Trimmed mean (5-95%)",
        text_offset=0.30,
        text_prefix="Trim mean: ",
        ha="left",
        x_offset=shift_unit * 0.25,
    )

    ax.set_ylim(-0.4, 0.6)
    ax.set_yticks([])
    ax.set_xlabel(_get_value_label(mode))
    ax.set_title(f"{dataset_label}: 5-95% range summary", fontsize=11)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlim(x_lower, x_upper)

    if legend_entries:
        handles, labels = zip(*legend_entries)
        ax.legend(
            handles,
            labels,
            fontsize=8,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.95),
            ncol=1,
            framealpha=0.6,
            borderaxespad=0.1,
        )


def plot_overview(stats: Dict, output_path: str, mode: str, dpi: int = 150) -> None:
    datasets = list(stats.keys())
    if not datasets:
        raise ValueError("statistics JSON does not contain any datasets.")

    fig_rows = len(datasets)
    fig_height = max(5.2, 5.0 * fig_rows)
    fig, axes = plt.subplots(fig_rows, 2, figsize=(12.5, fig_height), constrained_layout=False)
    if fig_rows == 1:
        axes = np.array([axes])

    for row_index, dataset_name in enumerate(datasets):
        summary = stats[dataset_name].get("overall", {})
        ax_quant = axes[row_index, 0]
        ax_mean = axes[row_index, 1]
        label = dataset_name.upper()
        _plot_dataset_quantiles(ax_quant, summary, label, mode)
        _add_dataset_info(ax_quant, summary, label, mode)
        _plot_dataset_trimmed_summary(ax_mean, summary, label, mode)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.08, hspace=0.65, wspace=0.32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 场景截尾区间对比
# --------------------------------------------------------------------------- #
def _plot_scene_summary(ax, scenes: Dict[str, Dict], dataset_label: str, mode: str) -> None:
    if not scenes:
        ax.text(0.5, 0.5, "No scene statistics", ha="center", va="center", fontsize=11, color="firebrick")
        ax.axis("off")
        return

    rows = []
    for scene in sorted(scenes.keys()):
        summary = scenes[scene]
        trim_stats = summary.get("trim_05_95", {})
        low_val = trim_stats.get("low")
        high_val = trim_stats.get("high")
        if low_val is None or high_val is None:
            continue
        files = summary.get("files", {})
        q05 = _get_quantile(summary, 5)
        q50 = _get_quantile(summary, 50)
        q95 = _get_quantile(summary, 95)
        trim_mean = trim_stats.get("mean")
        rows.append(
            {
                "scene": scene,
                "scene_short": _strip_scene_label(scene),
                "trim_stats": trim_stats,
                "low": float(low_val),
                "high": float(high_val),
                "q05": q05,
                "q50": q50,
                "q95": q95,
                "trim_mean": trim_mean,
                "count": files.get("processed", 0),
            }
        )

    if not rows:
        ax.text(0.5, 0.5, "No scene statistics", ha="center", va="center", fontsize=11, color="firebrick")
        ax.axis("off")
        return

    y_positions = np.arange(len(rows))
    legend_entries = []
    seen_labels = set()

    def register(label: str, artist) -> None:
        if artist is None:
            return
        if label in seen_labels:
            return
        legend_entries.append((artist, label))
        seen_labels.add(label)
    display_values: List[float] = []

    for idx, row in enumerate(rows):
        trim_stats = row["trim_stats"]
        base_y = y_positions[idx]
        low_disp = _scale_value(row["low"], trim_stats, mode)
        high_disp = _scale_value(row["high"], trim_stats, mode)

        ax.hlines(base_y, low_disp, high_disp, color=COLOR_TRIM_BAND, linewidth=4, alpha=0.75, zorder=1)
        occupied: Dict[float, int] = {}
        if low_disp is None or high_disp is None:
            continue
        span = high_disp - low_disp
        if not math.isfinite(span) or span <= 0:
            raw_span = row["high"] - row["low"] if row["high"] is not None and row["low"] is not None else None
            span = raw_span if raw_span and math.isfinite(raw_span) and raw_span > 0 else 1.0
        shift_unit = max(span * 0.18, 0.2)
        if mode == "scaled":
            row_lower, row_upper = 0.0, 1.0
        else:
            margin_ratio = 0.35 if mode == "trimmed" else 0.25
            margin = span * margin_ratio if span > 0 else 1.0
            base_low = min(low_disp, high_disp)
            base_high = max(low_disp, high_disp)
            row_lower = base_low - margin
            row_upper = base_high + margin
            if not math.isfinite(row_lower) or not math.isfinite(row_upper) or row_upper <= row_lower:
                row_lower, row_upper = base_low - 1.0, base_high + 1.0
        clamp_pad = max((row_upper - row_lower) * 0.04, 0.05)
        display_values.extend([row_lower, row_upper])

        def annotate_value(
            value: float,
            marker: str,
            color: str,
            label: str,
            *,
            text_prefix: str,
            text_offset: float,
            x_offset: float = 0.0,
            ha: str = "center",
            size: float = 60.0,
            edgecolor: str = "white",
            linewidths: float = 0.8,
            zorder: float = 4.0,
        ) -> None:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return
            display_val = _scale_value(value, trim_stats, mode)
            if display_val is None or (isinstance(display_val, float) and math.isnan(display_val)):
                return
            key = round(display_val, 3)
            count = occupied.get(key, 0)
            shift = 0.0
            if count > 0:
                direction = -1 if count % 2 else 1
                magnitude = shift_unit * ((count + 1) // 2)
                shift = direction * magnitude
            occupied[key] = count + 1
            scatter = ax.scatter(
                display_val,
                base_y,
                marker=marker,
                s=size,
                color=color,
                edgecolor=edgecolor,
                linewidths=linewidths,
                zorder=zorder,
            )
            register(label, scatter)
            display_values.append(display_val)
            text_x = display_val + x_offset + shift
            text_x = min(max(text_x, row_lower + clamp_pad), row_upper - clamp_pad)
            ax.text(
                text_x,
                base_y + text_offset,
                f"{text_prefix}{value:.2f}",
                ha=ha,
                va="bottom" if text_offset >= 0 else "top",
                fontsize=7,
                color=color,
            )

        annotate_value(
            row["q05"],
            marker="|",
            color=COLOR_TRIM_BAND,
            label="5% / 95% quantile",
            text_prefix="5%: ",
            text_offset=-0.12,
            x_offset=-shift_unit * 0.4,
            ha="right",
            size=140.0,
            edgecolor=COLOR_TRIM_BAND,
            linewidths=0.0,
            zorder=3.0,
        )
        annotate_value(
            row["q95"],
            marker="|",
            color=COLOR_TRIM_BAND,
            label="5% / 95% quantile",
            text_prefix="95%: ",
            text_offset=-0.12,
            x_offset=shift_unit * 0.4,
            ha="left",
            size=140.0,
            edgecolor=COLOR_TRIM_BAND,
            linewidths=0.0,
            zorder=3.0,
        )
        annotate_value(
            row["q50"],
            marker="o",
            color="#2B8A3E",
            label="Median",
            text_prefix="Median: ",
            text_offset=0.18,
        )
        annotate_value(
            row["trim_mean"],
            marker="D",
            color=COLOR_TRIMMED,
            label="Trimmed mean (5-95%)",
            text_prefix="Trim mean: ",
            text_offset=0.30,
            size=70.0,
            x_offset=shift_unit * 0.2,
            ha="left",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row['scene_short']} (n={row['count']})" for row in rows])
    ax.set_ylim(len(rows) - 0.45, -0.45)
    ax.invert_yaxis()
    ax.set_xlabel(_get_value_label(mode))
    ax.set_title(f"{dataset_label}: Scenes 5-95% range", fontsize=11)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)

    if mode == "scaled":
        ax.set_xlim(0.0, 1.0)
    else:
        valid_vals = [val for val in display_values if val is not None and math.isfinite(val)]
        if valid_vals:
            vmin = min(valid_vals)
            vmax = max(valid_vals)
            if math.isfinite(vmin) and math.isfinite(vmax):
                span = vmax - vmin
                margin = span * 0.08 if span > 0 else 1.0
                ax.set_xlim(vmin - margin, vmax + margin)

    if legend_entries:
        handles, labels = zip(*legend_entries)
        ax.legend(handles, labels, fontsize=8, loc="upper right", bbox_to_anchor=(0.98, 0.95), framealpha=0.6, borderaxespad=0.1)


def plot_scene_bars(stats: Dict, output_path: str, mode: str, dpi: int = 150) -> None:
    datasets = list(stats.keys())
    if not datasets:
        raise ValueError("statistics JSON does not contain any datasets.")

    fig_rows = len(datasets)
    max_scene_count = max(max(len(stats[name].get("scenes", {})), 1) for name in datasets)
    fig_height = max(5.2 * fig_rows, 1.9 * max_scene_count)
    fig, axes = plt.subplots(fig_rows, 1, figsize=(12.5, fig_height), squeeze=False)

    for row_index, dataset_name in enumerate(datasets):
        ax = axes[row_index, 0]
        scenes = stats[dataset_name].get("scenes", {})
        label = dataset_name.upper()
        _plot_scene_summary(ax, scenes, label, mode)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.07, hspace=0.75)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 场景分位曲线
# --------------------------------------------------------------------------- #
def plot_scene_quantiles(stats: Dict, output_path: str, mode: str, dpi: int = 150) -> None:
    datasets = list(stats.keys())
    if not datasets:
        raise ValueError("statistics JSON does not contain datasets.")

    rows = len(datasets)
    cols = max(len(stats[name].get("scenes", {})) for name in datasets)
    if cols == 0:
        raise ValueError("statistics JSON does not contain per-scene data.")

    fig_width = max(4.8 * cols, 6.5)
    fig_height = max(3.8 * rows, 3.4)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)

    for r, dataset_name in enumerate(datasets):
        scenes = stats[dataset_name].get("scenes", {})
        scene_names = sorted(scenes.keys())
        for c in range(cols):
            ax = axes[r, c]
            if c >= len(scene_names):
                ax.axis("off")
                continue

            scene_name = scene_names[c]
            summary = scenes[scene_name]
            scene_short = _strip_scene_label(scene_name)
            percents, values = _parse_quantiles(summary.get("quantiles", {}))
            if percents:
                trim_stats = summary.get("trim_05_95", {})
                perc_arr = np.asarray(percents, dtype=np.float64)
                val_arr = np.asarray(values, dtype=np.float64)
                if mode == "trimmed":
                    mask = (perc_arr >= 5.0) & (perc_arr <= 95.0)
                    if np.any(mask):
                        perc_arr = perc_arr[mask]
                        val_arr = val_arr[mask]
                if perc_arr.size == 0 or val_arr.size == 0:
                    ax.text(0.5, 0.5, "No trimmed quantiles", ha="center", va="center", color="firebrick")
                    ax.axis("off")
                    continue
                if mode == "scaled":
                    val_arr = np.asarray(_scale_series(list(val_arr), trim_stats, mode), dtype=np.float64)

                # 场景分位曲线，高亮截尾范围的分布走势
                ax.plot(perc_arr, val_arr, marker="o", color=COLOR_TRIM_BAND, linewidth=1.8)
                if perc_arr.size >= 2:
                    ax.fill_between(perc_arr, val_arr, color=COLOR_TRIM_FILL, alpha=0.25)
                ax.set_xticks(perc_arr)
                ax.set_xticklabels([f"{p:.0f}" for p in perc_arr])
                ax.set_xlabel("Quantile (%)")
                ax.set_ylabel(_get_value_label(mode))
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
                if mode == "trimmed":
                    ax.set_xlim(5, 95)
                _apply_trimmed_limits(ax, trim_stats, mode)
            else:
                ax.text(0.5, 0.5, "No quantile data", ha="center", va="center", color="firebrick")
                ax.axis("off")
                continue

            files = summary.get("files", {})
            pixels = summary.get("pixels", {})
            processed = files.get("processed", 0)
            missing = files.get("missing_depth", 0)
            ratio = pixels.get("valid_ratio", None)
            info = f"n={processed} (missing {missing})"
            if ratio not in (None, 0):
                info += f"\nvalid {ratio:.2%}"
            ax.set_title(f"{dataset_name.upper()} - {scene_short}", fontsize=10)
            ax.text(
                0.98,
                0.02,
                info,
                ha="right",
                va="bottom",
                fontsize=7,
                transform=ax.transAxes,
                color="dimgray",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.5, edgecolor="lightgray"),
            )

    fig.subplots_adjust(hspace=0.55, wspace=0.28, top=0.93, bottom=0.08, left=0.08, right=0.98)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Plot UAVula depth statistics.")
    parser.add_argument(
        "--stats-json",
        default=DEFAULT_STATS_JSON,
        help=f"Statistics JSON path (default: {DEFAULT_STATS_JSON}).",
    )
    parser.add_argument(
        "--overview",
        default=DEFAULT_OVERVIEW,
        help=f"Dataset overview plot output (default: {DEFAULT_OVERVIEW}).",
    )
    parser.add_argument(
        "--scene-bars",
        default=DEFAULT_SCENE_BARS,
        help=f"Scene mean comparison plot output (default: {DEFAULT_SCENE_BARS}).",
    )
    parser.add_argument(
        "--scene-quantiles",
        default=DEFAULT_SCENE_QUANT,
        help=f"Scene quantile curves plot output (default: {DEFAULT_SCENE_QUANT}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default 150).",
    )
    parser.add_argument(
        "--value-mode",
        choices=VALUE_MODES,
        default=DEFAULT_VALUE_MODE,
        help="Value display mode: 'raw' for full range, 'trimmed' to focus axis on q05~q95, 'scaled' to normalize by that window.",
    )
    args = parser.parse_args()

    stats_path = os.path.abspath(args.stats_json)
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"Statistics JSON not found: {stats_path}")

    with open(stats_path, "r", encoding="utf-8") as fp:
        stats = json.load(fp)

    overview_path = os.path.abspath(args.overview)
    plot_overview(stats, overview_path, mode=args.value_mode, dpi=args.dpi)
    print(f"[INFO] Overview plot saved to: {overview_path}")

    scene_bar_path = os.path.abspath(args.scene_bars)
    plot_scene_bars(stats, scene_bar_path, mode=args.value_mode, dpi=args.dpi)
    print(f"[INFO] Scene bar plot saved to: {scene_bar_path}")

    scene_quant_path = os.path.abspath(args.scene_quantiles)
    plot_scene_quantiles(stats, scene_quant_path, mode=args.value_mode, dpi=args.dpi)
    print(f"[INFO] Scene quantile plot saved to: {scene_quant_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select representative image IDs (names only) from two (or more) model metrics files.
Now with optional *grouped statistics* by parsing a key from image_id (e.g., "DJI_0166_image_..." -> group "0166").

Features
- Metrics: abs_rel, sq_rel, rmse, rmse_log (auto-detects 'rems_log')
- Score: rank-based average of per-metric percentiles (lower = better)
- Modes:
    1) consensus: choose the same image_ids across models via averaged error_score
    2) per_model: choose top/bottom independently for each model
- Outputs (names only, no images):
    - CSV summary(ies) with selected IDs and scores/metrics
    - Plain-text ID lists for quick copy/paste
    - (Optional) Group-level statistics and per-group selections

Author: ChatGPT
"""

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

# --------------------------- Defaults ---------------------------
DEFAULT_MODELS: List[str] = [
    # "M1|/data/model1/metrics.txt",
    # "M2|/data/model2/metrics.txt",
]

DEFAULT_GROUP_REGEX = r"DJI_([^_]+)_image"  # captures "0166" from "DJI_0166_image_..." and "0502b" from "DJI_0502b_image_..."


# ------------------------ Arg Parsing ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select representative image IDs (best/worst) and optional grouped stats.")
    p.add_argument("--models", nargs="+", required=False,
                   help="每个模型形如 '模型名|指标文件路径'（示例：\"M1|/data/m1.txt\" \"M2|/data/m2.txt\"）")
    p.add_argument("--models-file", type=str,
                   help="包含多行 '模型名|指标文件路径' 的文本文件；支持以 # 开头的注释行与空行。")
    p.add_argument("--use-default", action="store_true",
                   help="使用脚本内置的 DEFAULT_MODELS（可在脚本顶部编辑）。")

    p.add_argument("--mode", choices=["consensus", "per_model"], default="consensus",
                   help="consensus：对齐同一批图像；per_model：各模型独立选择。默认 consensus。")
    p.add_argument("--k", type=int, default=10, help="总体（好/坏）选取数量，默认 10。")

    p.add_argument("--metrics-columns", default="abs_rel,sq_rel,rmse,rmse_log",
                   help="用于评分的指标列名，逗号分隔。默认 'abs_rel,sq_rel,rmse,rmse_log'。")
    p.add_argument("--output-dir", default="./selection_output", help="输出目录，默认 ./selection_output")

    # Grouping options
    p.add_argument("--grouped", action="store_true",
                   help="启用分组统计与分组内选取（按 --group-regex 从 image_id 提取分组键）。")
    p.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX,
                   help=f"用于从 image_id 提取分组键的正则（默认：{DEFAULT_GROUP_REGEX}，捕获组 #1 作为分组键）。")
    p.add_argument("--per-group-k", type=int, default=10, help="每个分组内（好/坏）选取数量，默认 10。")
    p.add_argument("--group-selection", choices=["best","worst","both"], default="best", help="分组内选取方向：best=只选误差小，worst=只选误差大，both=两者都选。默认 best。")
    # Overall K with per-group proportional allocation
    p.add_argument("--overall-k", type=int, default=None, help="按各大类占比，从所有分组中总计选取 K 个（而非每组各取）。")
    p.add_argument("--allocation", choices=["proportional", "equal"], default="proportional", help="overall-k 模式下的配额分配策略：proportional=按各组样本占比；equal=各组尽量均分。")

    return p.parse_args()


# ----------------------- Utilities ------------------------------
def infer_separator(path: str) -> Optional[str]:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    if "," in header and "\t" not in header and ";" not in header:
        return ","
    if "\t" in header and "," not in header:
        return "\t"
    return None  # let pandas auto-detect


def read_metrics_df(path: str, wanted_cols: List[str]) -> pd.DataFrame:
    sep = infer_separator(path)
    try:
        df = pd.read_csv(
            path,
            sep=sep,
            engine="python",
            comment="#",
            on_bad_lines="skip",
        )
    except Exception:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            comment="#",
            on_bad_lines="skip",
        )
    # normalize columns
    df.columns = [c.strip() for c in df.columns]
    # normalize image_id-like column
    image_id_col = None
    for cand in ["image_id", "img_id", "id", "filename", "name"]:
        if cand in df.columns:
            image_id_col = cand
            break
    if image_id_col is None:
        raise ValueError(f"指标文件缺少 image_id 列（支持：image_id/img_id/id/filename/name）。文件：{path}")

    # handle rmse_log typo and presence check
    resolved_cols: List[Optional[str]] = []
    cols_lower = {c.lower(): c for c in df.columns}
    for w in wanted_cols:
        lw = w.lower()
        if lw in cols_lower:
            resolved_cols.append(cols_lower[lw])
        elif lw == "rmse_log" and "rems_log" in cols_lower:
            resolved_cols.append(cols_lower["rems_log"])
        else:
            resolved_cols.append(None)
    missing = [w for w, r in zip(wanted_cols, resolved_cols) if r is None]
    if missing:
        raise ValueError(f"指标文件缺少以下列：{missing}（文件：{path}）")

    use_cols = [image_id_col] + [c for c in resolved_cols if c is not None]
    out = df[use_cols].copy()
    out.rename(columns={image_id_col: "image_id"}, inplace=True)
    out["image_id"] = out["image_id"].astype(str).str.strip()
    # coerce metrics to numeric & drop invalid rows (filters summary lines etc.)
    metric_cols_resolved = [c for c in use_cols if c != "image_id"]
    for mc in metric_cols_resolved:
        out[mc] = pd.to_numeric(out[mc], errors="coerce")
    out = out.dropna(subset=metric_cols_resolved)
    return out


def compute_error_score(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    work = df.copy()
    pct_cols = []
    for col in metric_cols:
        if col not in work.columns:
            if col == "rmse_log" and "rems_log" in work.columns:
                col = "rems_log"
            else:
                raise ValueError(f"缺少指标列：{col}")
        pct = work[col].rank(pct=True, ascending=True)
        work[f"pct_{col}"] = pct
        pct_cols.append(f"pct_{col}")
    work["error_score"] = work[pct_cols].mean(axis=1)
    return work



def allocate_quota(group_sizes: Dict[str, int], K: int, mode: str = "proportional") -> Dict[str, int]:
    """
    使用 Hamilton (largest remainder) 方法按占比分配整数配额，使总和为 K。
    mode="proportional": 按 n_g / sum(n) * K 分配；
    mode="equal": 尽量均分，等价于将每组权重视为 1。
    返回 dict: {group -> quota}，可能出现某些小组为 0（若 K < 组数）。
    """
    if K <= 0 or not group_sizes:
        return {g: 0 for g in group_sizes}
    if mode not in ("proportional", "equal"):
        mode = "proportional"
    sizes = group_sizes.copy()
    total = sum(sizes.values()) if mode == "proportional" else len(sizes)
    if total == 0:
        return {g: 0 for g in sizes}
    quotas_float = {g: (sizes[g] / total) * K for g in sizes}
    quotas_floor = {g: int(q // 1) for g, q in quotas_float.items()}
    remainder = {g: quotas_float[g] - quotas_floor[g] for g in sizes}
    assigned = sum(quotas_floor.values())
    remaining = max(0, K - assigned)
    # 分配剩余名额给余数最大的组
    order = sorted(remainder.items(), key=lambda x: x[1], reverse=True)
    quotas = quotas_floor.copy()
    i = 0
    while remaining > 0 and i < len(order):
        g = order[i][0]
        quotas[g] += 1
        remaining -= 1
        i += 1
    return quotas
def extract_group_key(image_id: str, pattern: str) -> str:
    m = re.search(pattern, image_id)
    return m.group(1) if m else "UNGROUPED"


def load_models_from_file(path: str) -> List[str]:
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items


def resolve_models(args) -> List[str]:
    if args.models:
        return args.models
    if args.models_file:
        filespecs = load_models_from_file(args.models_file)
        if filespecs:
            return filespecs
        else:
            print(f"警告：{args.models_file} 为空或仅有注释。", file=sys.stderr)
    if args.use_default:
        if not DEFAULT_MODELS:
            raise ValueError("DEFAULT_MODELS 为空。请编辑脚本顶部的 DEFAULT_MODELS，或改用 --models/--models-file。")
        return DEFAULT_MODELS
    raise ValueError("未提供模型配置。请使用 --models 或 --models-file，或加 --use-default 并编辑 DEFAULT_MODELS。")


# --------------------------- Main -------------------------------
def main():
    args = parse_args()

    models_raw: List[str] = resolve_models(args)
    if len(models_raw) < 2:
        print("警告：建议至少提供 2 个模型。", file=sys.stderr)

    specs: List[Tuple[str, str]] = []
    for s in models_raw:
        parts = s.split("|", 1)
        if len(parts) != 2:
            raise ValueError(f"--models 参数格式错误：'{s}'（应为 'name|metrics'）")
        specs.append((parts[0].strip(), parts[1].strip()))

    metrics_cols = [c.strip() for c in args.metrics_columns.split(",") if c.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and score each model
    per_model_df: List[pd.DataFrame] = []
    for name, mpath in specs:
        df = read_metrics_df(mpath, metrics_cols)
        df_scored = compute_error_score(df, metrics_cols)
        df_scored["model"] = name
        per_model_df.append(df_scored)

    # ------------ consensus mode ------------
    if args.mode == "consensus":
        # intersection of image_ids across all models
        id_sets = [set(df["image_id"].astype(str)) for df in per_model_df]
        common_ids = set.intersection(*id_sets) if id_sets else set()
        if not common_ids:
            print("错误：各模型之间没有共同的 image_id，无法执行 consensus。请改用 --mode per_model。", file=sys.stderr)
            sys.exit(1)

        # Build consensus rows
        # maps for quick lookup
        err_maps: List[Dict[str, float]] = [{r["image_id"]: float(r["error_score"]) for _, r in df.iterrows()} for df in per_model_df]
        met_maps: List[Dict[str, Dict[str, float]]] = []
        for df in per_model_df:
            mm = {}
            for _, r in df.iterrows():
                mm[r["image_id"]] = {mc: float(r[mc]) if mc in r and pd.notna(r[mc]) else np.nan for mc in metrics_cols}
            met_maps.append(mm)

        rows = []
        for iid in common_ids:
            errs = [m.get(iid, np.nan) for m in err_maps]
            errs = [e for e in errs if np.isfinite(e)]
            if len(errs) != len(err_maps):
                continue
            cons_err = float(np.mean(errs))
            rec = {"image_id": iid, "consensus_error_score": cons_err}
            # average metrics across models row-wise
            for mc in metrics_cols:
                vals = [m.get(iid, {}).get(mc, np.nan) for m in met_maps]
                vals = [v for v in vals if np.isfinite(v)]
                rec[mc] = float(np.mean(vals)) if vals else np.nan
            # group key
            rec["group"] = extract_group_key(iid, args.group_regex) if args.grouped else None
            rows.append(rec)

        if not rows:
            print("错误：无法计算共识结果。", file=sys.stderr)
            sys.exit(1)

        df_cons = pd.DataFrame(rows)
        # overall selections (best/worst)
        df_sorted_asc = df_cons.sort_values("consensus_error_score", ascending=True)
        best_ids = df_sorted_asc["image_id"].head(min(args.k, len(df_sorted_asc))).tolist()
        df_sorted_desc = df_cons.sort_values("consensus_error_score", ascending=False)
        worst_ids = df_sorted_desc["image_id"].head(min(args.k, len(df_sorted_desc))).tolist()

        # CSV summary (overall selections) + txt lists
        # Also attach per-model error_score if needed
        overall_rows = []
        for label, id_list in [("best", best_ids), ("worst", worst_ids)]:
            for iid in id_list:
                row = {"mode": "consensus", "selection": label, "image_id": iid}
                row["consensus_error_score"] = float(df_cons.loc[df_cons["image_id"] == iid, "consensus_error_score"].values[0])
                for mc in metrics_cols:
                    if mc in df_cons.columns:
                        row[mc] = float(df_cons.loc[df_cons["image_id"] == iid, mc].values[0])
                if args.grouped:
                    row["group"] = df_cons.loc[df_cons["image_id"] == iid, "group"].values[0]
                overall_rows.append(row)
        pd.DataFrame(overall_rows).to_csv(os.path.join(args.output_dir, "selection_summary_consensus.csv"),
                                          index=False, encoding="utf-8")
        with open(os.path.join(args.output_dir, "best_ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(best_ids))
        with open(os.path.join(args.output_dir, "worst_ids.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(worst_ids))

        # Grouped statistics & selections
        if args.grouped:
            # group stats table
            agg = {
                "image_id": "count",
                "consensus_error_score": ["mean", "std", "median", "min", "max"],
            }
            for mc in metrics_cols:
                agg[mc] = ["mean", "std", "median"]
            gs = df_cons.groupby("group").agg(agg)
            # flatten columns
            gs.columns = ["_".join([c for c in col if c]).replace("__", "_") for col in gs.columns.values]
            gs = gs.rename(columns={"image_id_count": "n"}).reset_index()
            gs.to_csv(os.path.join(args.output_dir, "group_stats_consensus.csv"), index=False, encoding="utf-8")

            # overall-k proportional/equal allocation across groups (consensus)
            if args.overall_k is not None and args.overall_k > 0:
                # compute per-group sizes in df_cons and allocate quotas
                group_sizes = df_cons.groupby("group")["image_id"].count().to_dict()
                quotas = allocate_quota(group_sizes, args.overall_k, args.allocation)
                sel_rows_overall = []
                selected_ids = []
                for g, q in quotas.items():
                    if q <= 0:
                        continue
                    gdf = df_cons[df_cons["group"] == g]
                    # choose direction as per --group-selection (best/worst/both but here typically best/worst)
                    if args.group_selection in ("best", "both"):
                        g_sorted = gdf.sort_values("consensus_error_score", ascending=True)
                    else:
                        g_sorted = gdf.sort_values("consensus_error_score", ascending=False)
                    picks = g_sorted.head(min(q, len(g_sorted)))
                    for rank, (_, r) in enumerate(picks.iterrows(), start=1):
                        rec = {"group": g, "image_id": r["image_id"],
                               "consensus_error_score": float(r["consensus_error_score"]),
                               "allocation_quota": int(q), "allocation_mode": args.allocation,
                               "selection": ("best" if args.group_selection in ("best","both") else "worst"),
                               "rank_within_group": rank}
                        for mc in metrics_cols:
                            rec[mc] = float(r[mc]) if pd.notna(r[mc]) else np.nan
                        sel_rows_overall.append(rec)
                        selected_ids.append(str(r["image_id"]))
                pd.DataFrame(sel_rows_overall).to_csv(
                    os.path.join(args.output_dir, "group_quota_selection_consensus.csv"), index=False, encoding="utf-8")
                with open(os.path.join(args.output_dir, "selected_ids_overall.txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(selected_ids))
                print(f"[已生成] group_quota_selection_consensus.csv / selected_ids_overall.txt (K={args.overall_k}, {args.allocation})")


            # per-group selections
            sel_rows = []
            for g, gdf in df_cons.groupby("group"):
                # Determine which selections to produce based on --group-selection
                if args.group_selection in ("best", "both"):
                    g_best = gdf.sort_values("consensus_error_score", ascending=True).head(min(args.per_group_k, len(gdf)))
                    for rank, (_, r) in enumerate(g_best.iterrows(), start=1):
                        rec = {"group": g, "selection": "best", "rank": rank, "image_id": r["image_id"],
                               "consensus_error_score": float(r["consensus_error_score"])}
                        for mc in metrics_cols:
                            rec[mc] = float(r[mc]) if pd.notna(r[mc]) else np.nan
                        sel_rows.append(rec)
                if args.group_selection in ("worst", "both"):
                    g_worst = gdf.sort_values("consensus_error_score", ascending=False).head(min(args.per_group_k, len(gdf)))
                    for rank, (_, r) in enumerate(g_worst.iterrows(), start=1):
                        rec = {"group": g, "selection": "worst", "rank": rank, "image_id": r["image_id"],
                               "consensus_error_score": float(r["consensus_error_score"])}
                        for mc in metrics_cols:
                            rec[mc] = float(r[mc]) if pd.notna(r[mc]) else np.nan
                        sel_rows.append(rec)
            pd.DataFrame(sel_rows).to_csv(os.path.join(args.output_dir, "group_selections_consensus.csv"),
                                          index=False, encoding="utf-8")

            # Aggregate per-group selections into final union (no further top-10 filtering)
            if sel_rows:
                df_sel = pd.DataFrame(sel_rows)
                # BEST aggregate
                if (args.group_selection in ("best", "both")) and not df_sel[df_sel['selection']=="best"].empty:
                    best_df = df_sel[df_sel['selection']=="best"].copy()
                    best_df.to_csv(os.path.join(args.output_dir, "group_aggregate_best_consensus.csv"), index=False, encoding="utf-8")
                    best_ids = best_df['image_id'].astype(str).drop_duplicates().tolist()
                    with open(os.path.join(args.output_dir, "aggregate_best_ids.txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(best_ids))
                # WORST aggregate
                if (args.group_selection in ("worst", "both")) and not df_sel[df_sel['selection']=="worst"].empty:
                    worst_df = df_sel[df_sel['selection']=="worst"].copy()
                    worst_df.to_csv(os.path.join(args.output_dir, "group_aggregate_worst_consensus.csv"), index=False, encoding="utf-8")
                    worst_ids = worst_df['image_id'].astype(str).drop_duplicates().tolist()
                    with open(os.path.join(args.output_dir, "aggregate_worst_ids.txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(worst_ids))
            print("[已生成] group_aggregate_*_consensus.csv 与 aggregate_*_ids.txt（来自每个大类各取 per-group-k 的并集）")

        # Console print
        print("Best IDs (consensus):")
        print("\n".join(best_ids))
        print("\nWorst IDs (consensus):")
        print("\n".join(worst_ids))
        if args.grouped:
            print("\n[已生成] group_stats_consensus.csv / group_selections_consensus.csv")
        print("[已生成] selection_summary_consensus.csv, best_ids.txt, worst_ids.txt")

    # ------------ per_model mode ------------
    else:
        all_rows = []
        best_txt_paths = []
        worst_txt_paths = []
        # grouped outputs
        grouped_all_sel = []  # for combined CSV
        for (name, _), df in zip(specs, per_model_df):
            # overall best/worst
            df_sorted = df.sort_values("error_score", ascending=True)
            k_best = min(args.k, len(df_sorted))
            best_ids = df_sorted["image_id"].head(k_best).tolist()

            df_sorted = df.sort_values("error_score", ascending=False)
            k_worst = min(args.k, len(df_sorted))
            worst_ids = df_sorted["image_id"].head(k_worst).tolist()

            # accumulate CSV rows (overall selections)
            for label, id_list in [("best", best_ids), ("worst", worst_ids)]:
                for iid in id_list:
                    r = df.loc[df["image_id"] == iid].head(1)
                    row = {"mode": "per_model", "model": name, "selection": label, "image_id": iid}
                    row["error_score"] = float(r["error_score"].values[0])
                    for mc in metrics_cols:
                        mc_resolved = mc if mc in r.columns else ("rems_log" if mc == "rmse_log" and "rems_log" in r.columns else None)
                        if mc_resolved:
                            row[mc] = float(r[mc_resolved].values[0])
                    all_rows.append(row)

            # write plain-text lists per model
            best_path = os.path.join(args.output_dir, f"{name}_best_ids.txt")
            worst_path = os.path.join(args.output_dir, f"{name}_worst_ids.txt")
            with open(best_path, "w", encoding="utf-8") as f:
                for iid in best_ids:
                    f.write(str(iid) + "\n")
            with open(worst_path, "w", encoding="utf-8") as f:
                for iid in worst_ids:
                    f.write(str(iid) + "\n")
            best_txt_paths.append(best_path)
            worst_txt_paths.append(worst_path)

            # grouped stats & selections per model
            if args.grouped:
                df_g = df.copy()
                df_g["group"] = df_g["image_id"].astype(str).apply(lambda s: extract_group_key(s, args.group_regex))
                agg = {
                    "image_id": "count",
                    "error_score": ["mean", "std", "median", "min", "max"],
                }
                for mc in metrics_cols:
                    agg[mc] = ["mean", "std", "median"]
                gs = df_g.groupby("group").agg(agg)
                gs.columns = ["_".join([c for c in col if c]).replace("__", "_") for col in gs.columns.values]
                gs = gs.rename(columns={"image_id_count": "n"}).reset_index()
                gs.insert(0, "model", name)
                gs.to_csv(os.path.join(args.output_dir, f"group_stats_{name}.csv"), index=False, encoding="utf-8")                # per-group selections
                for g, gdf in df_g.groupby("group"):
                    if args.group_selection in ("best", "both"):
                        g_best = gdf.sort_values("error_score", ascending=True).head(min(args.per_group_k, len(gdf)))
                        for rank, (_, r) in enumerate(g_best.iterrows(), start=1):
                            rec = {"model": name, "group": g, "selection": "best", "rank": rank,
                                   "image_id": r["image_id"], "error_score": float(r["error_score"])}
                            for mc in metrics_cols:
                                rec[mc] = float(r[mc]) if mc in r and pd.notna(r[mc]) else np.nan
                            grouped_all_sel.append(rec)
                    if args.group_selection in ("worst", "both"):
                        g_worst = gdf.sort_values("error_score", ascending=False).head(min(args.per_group_k, len(gdf)))
                        for rank, (_, r) in enumerate(g_worst.iterrows(), start=1):
                            rec = {"model": name, "group": g, "selection": "worst", "rank": rank,
                                   "image_id": r["image_id"], "error_score": float(r["error_score"])}
                            for mc in metrics_cols:
                                rec[mc] = float(r[mc]) if mc in r and pd.notna(r[mc]) else np.nan
                            grouped_all_sel.append(rec)

        # write overall selection CSV
        pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, "selection_summary_per_model.csv"),
                                      index=False, encoding="utf-8")
        print(f"[已生成] selection_summary_per_model.csv")
        print(f"[已生成] {', '.join(os.path.basename(p) for p in best_txt_paths + worst_txt_paths)}")

        if args.grouped and grouped_all_sel:
            pd.DataFrame(grouped_all_sel).to_csv(os.path.join(args.output_dir, "group_selections_per_model.csv"),
                                                 index=False, encoding="utf-8")
            print("[已生成] group_stats_{model}.csv（每模型一个）与 group_selections_per_model.csv")

            # Aggregate per-group selections into final union per model (no further top-10 filtering)
            if grouped_all_sel:
                df_gsel = pd.DataFrame(grouped_all_sel)
                models = sorted(df_gsel['model'].unique())
                for m in models:
                    mdf = df_gsel[df_gsel['model']==m]
                    # BEST aggregate per model
                    if (args.group_selection in ("best", "both")) and not mdf[mdf['selection']=="best"].empty:
                        mbest = mdf[mdf['selection']=="best"].copy()
                        mbest.to_csv(os.path.join(args.output_dir, f"group_aggregate_best_{m}.csv"), index=False, encoding="utf-8")
                        ids = mbest['image_id'].astype(str).drop_duplicates().tolist()
                        with open(os.path.join(args.output_dir, f"{m}_aggregate_best_ids.txt"), "w", encoding="utf-8") as f:
                            f.write("\n".join(ids))
                    # WORST aggregate per model
                    if (args.group_selection in ("worst", "both")) and not mdf[mdf['selection']=="worst"].empty:
                        mworst = mdf[mdf['selection']=="worst"].copy()
                        mworst.to_csv(os.path.join(args.output_dir, f"group_aggregate_worst_{m}.csv"), index=False, encoding="utf-8")
                        ids = mworst['image_id'].astype(str).drop_duplicates().tolist()
                        with open(os.path.join(args.output_dir, f"{m}_aggregate_worst_ids.txt"), "w", encoding="utf-8") as f:
                            f.write("\n".join(ids))
            print("[已生成] group_aggregate_*_{MODEL}.csv 与 {MODEL}_aggregate_*_ids.txt（来自每个大类各取 per-group-k 的并集）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[失败] {e}", file=sys.stderr)
        sys.exit(2)
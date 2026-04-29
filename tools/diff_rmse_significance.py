
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from scipy import stats

@dataclass
class Config:
    key_col_candidates: Tuple[str, ...] = ("image_id","image","frame","filename","file","path","name","id")
    rmse_aliases: Tuple[str, ...] = ("rmse","rms","root_mse","root_mean_squared_error","root_mean_square_error","root-mean-square-error","root_mean_square")
    rmse_abs_thr: float = 0.05      # 绝对差阈值（单位与RMSE一致）
    rmse_pct_thr: float = 0.10      # 相对涨幅阈值（10%）
    tail_percentile: float = 95.0   # 取差值的95分位作为尾部阈值
    topk: int = 50                  # 导出Top-K坏例
    greater_is_worse: bool = True   # RMSE越大越差

def _find_key_col(df: pd.DataFrame, cfg: Config) -> str:
    lowered = {c.lower(): c for c in df.columns}
    for cand in cfg.key_col_candidates:
        if cand in lowered:
            return lowered[cand]
    raise ValueError(f"未找到主键列，请包含其中之一: {cfg.key_col_candidates}")

def _find_rmse_col(df: pd.DataFrame, cfg: Config) -> str:
    lowered = {c.lower().replace(" ", "_"): c for c in df.columns}
    for cand in cfg.rmse_aliases:
        key = cand.lower().replace(" ", "_")
        if key in lowered:
            return lowered[key]
    raise ValueError(f"未找到RMSE列，请包含其中之一: {cfg.rmse_aliases}")

def _prep_detail(df: pd.DataFrame) -> pd.DataFrame:
    if "kind" in df.columns:
        return df[df["kind"].astype(str).str.lower().eq("detail")].copy()
    return df.copy()

def load_rmse(csv_path: str, cfg: Optional[Config]=None) -> Tuple[pd.DataFrame, str, str]:
    cfg = cfg or Config()
    df = pd.read_csv(csv_path)
    df = _prep_detail(df)
    key = _find_key_col(df, cfg)
    rmse_col = _find_rmse_col(df, cfg)
    out = df[[key, rmse_col]].copy()
    # 新版CSV包含param/summary行，会让RMSE列推断成object；强制转为数值
    out[rmse_col] = pd.to_numeric(out[rmse_col], errors="coerce")
    out = out.dropna(subset=[rmse_col])
    out = out.rename(columns={rmse_col: "rmse"})
    return out, key, "rmse"

def align(a: pd.DataFrame, b: pd.DataFrame, key: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a2 = a.set_index(key).sort_index()
    b2 = b.set_index(key).sort_index()
    common = a2.index.intersection(b2.index)
    if len(common) == 0:
        raise ValueError("两份CSV没有可对齐的样本（主键不相交）。")
    a2 = a2.loc[common].copy()
    b2 = b2.loc[common].copy()
    a2.reset_index(inplace=True)
    b2.reset_index(inplace=True)
    return a2, b2

def compute_diffs(a: pd.DataFrame, b: pd.DataFrame, key: str, greater_is_worse: bool=True) -> pd.DataFrame:
    merged = a.merge(b, on=key, suffixes=("_a","_b"))
    merged["rmse_delta"] = merged["rmse_b"] - merged["rmse_a"]
    with np.errstate(divide="ignore", invalid="ignore"):
        merged["rmse_pct"] = np.where(merged["rmse_a"]!=0, merged["rmse_delta"]/np.abs(merged["rmse_a"]), np.nan)
    # “更差”的方向
    merged["worse_direction"] = np.where(merged["rmse_delta"]>0, 1, -1) if greater_is_worse else np.where(merged["rmse_delta"]<0, 1, -1)
    return merged

def global_tests(diff_df: pd.DataFrame, greater_is_worse: bool=True) -> pd.DataFrame:
    d = diff_df["rmse_delta"].dropna().values
    if not greater_is_worse:
        d = -d  # 统一到“正值=更差”
    # t 检验（单侧，H1: 均值>0）
    tstat, p_t = stats.ttest_1samp(d, popmean=0.0, alternative='greater')
    # Wilcoxon 符号秩（单侧，H1: 中位数>0）
    if np.any(d!=0):
        wstat, p_w = stats.wilcoxon(d, alternative='greater', zero_method="zsplit", mode="auto")
    else:
        wstat, p_w = (np.nan, 1.0)
    # 效应量
    mean = float(np.nanmean(d)); std = float(np.nanstd(d, ddof=1)) if d.size>1 else np.nan
    cohend = mean / std if std not in (0.0, np.nan) else np.nan
    # rank-biserial r（由Wilcoxon可近似）
    n = diff_df.shape[0]
    r_rb = 1.0 - (2.0*wstat)/(n*(n+1)) if n>0 and pd.notna(wstat) else np.nan
    return pd.DataFrame([{
        "mean_delta(b-a)": float(np.nanmean(diff_df["rmse_delta"])),
        "median_delta(b-a)": float(np.nanmedian(diff_df["rmse_delta"])),
        "t_stat(one-sided b>a)": float(tstat),
        "p_ttest(one-sided b>a)": float(p_t),
        "wilcoxon_stat(one-sided b>a)": float(wstat),
        "p_wilcoxon(one-sided b>a)": float(p_w),
        "cohen_d_on_delta": float(cohend),
        "rank_biserial_r": float(r_rb),
        "N": int(n)
    }])

def tag_significant_worse(diff_df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[str,float]]:
    out = diff_df.copy()
    # 只考虑“更差方向”的正增量
    d = out["rmse_delta"].values
    tail = np.nanpercentile(d, cfg.tail_percentile) if out.shape[0]>0 else np.inf
    out["sig_rule_abs"]  = (out["rmse_delta"] >= cfg.rmse_abs_thr)
    out["sig_rule_pct"]  = (out["rmse_pct"]  >= cfg.rmse_pct_thr)
    out["sig_rule_tail"] = (out["rmse_delta"] >= tail) & (out["rmse_delta"]>0)
    out["sig_worse"] = out[["sig_rule_abs","sig_rule_pct","sig_rule_tail"]].any(axis=1)
    return out, {"rmse_abs_thr": cfg.rmse_abs_thr, "rmse_pct_thr": cfg.rmse_pct_thr, "tail_percentile": cfg.tail_percentile, "tail_value": float(tail)}

def make_report(csv_a: str, csv_b: str, out_dir: str, cfg: Optional[Config]=None) -> Dict[str,str]:
    cfg = cfg or Config()
    outp = Path(out_dir); outp.mkdir(parents=True, exist_ok=True)

    a, key, _ = load_rmse(csv_a, cfg)
    b, key2, _ = load_rmse(csv_b, cfg)
    assert key == key2, f"主键列不一致: {key} vs {key2}"

    a_aln, b_aln = align(a, b, key)
    diff = compute_diffs(a_aln, b_aln, key, cfg.greater_is_worse)
    tests = global_tests(diff, cfg.greater_is_worse)
    tagged, rule_info = tag_significant_worse(diff, cfg)

    # 输出文件（明确标识比较方向）
    diff_csv = outp / "all_samples_rmse_comparison.csv"
    topk_worse_csv = outp / "model_B_worse_than_A.csv"  # B比A差
    topk_better_csv = outp / "model_B_better_than_A.csv"  # B比A好
    tests_csv = outp / "statistical_significance_tests.csv"

    tagged.sort_values("rmse_delta", ascending=False).to_csv(diff_csv, index=False)
    tagged.query("sig_worse").sort_values("rmse_delta", ascending=False).head(cfg.topk).to_csv(topk_worse_csv, index=False)
    # Better: 选择rmse_delta < 0的样本（模型B比模型A好），按rmse_delta升序排列
    tagged.query("rmse_delta < 0").sort_values("rmse_delta", ascending=True).head(cfg.topk).to_csv(topk_better_csv, index=False)
    tests.to_csv(tests_csv, index=False)

    # Excel 汇总（更稳健：自动选择可用引擎，并打印明确日志）
    try:
        import importlib
        engine = None
        for eng in ("openpyxl", "xlsxwriter"):
            if importlib.util.find_spec(eng) is not None:
                engine = "openpyxl" if eng == "openpyxl" else "xlsxwriter"
                break

        if engine is None:
            print("[WARN] 未检测到 openpyxl/xlsxwriter，跳过 Excel 导出。已生成 CSV。")
        else:
            xlsx_path = outp / "model_comparison_report.xlsx"
            with pd.ExcelWriter(xlsx_path, engine=engine) as writer:
                tagged.to_excel(writer, sheet_name="all_samples_comparison", index=False)
                tagged.query("sig_worse").sort_values("rmse_delta", ascending=False).head(cfg.topk) \
                      .to_excel(writer, sheet_name="model_B_worse_than_A", index=False)
                tagged.query("rmse_delta < 0").sort_values("rmse_delta", ascending=True).head(cfg.topk) \
                      .to_excel(writer, sheet_name="model_B_better_than_A", index=False)
                tests.to_excel(writer, sheet_name="statistical_tests", index=False)
                pd.DataFrame([rule_info]).to_excel(writer, sheet_name="analysis_rules", index=False)
            print(f"[OK] Excel 汇总已导出：{xlsx_path}")
    except Exception as e:
        import traceback
        print(f"[ERROR] Excel 导出失败：{e}")
        traceback.print_exc()

    return {
        "diff_csv": str(diff_csv),
        "topk_worse_csv": str(topk_worse_csv),
        "topk_better_csv": str(topk_better_csv),
        "tests_csv": str(tests_csv),
        "report_xlsx": str(outp / "model_comparison_report.xlsx"),
        "key_col": key
    }

if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Compare RMSE between two CSVs and extract significantly worse samples (b vs a).")
    p.add_argument("--csv_a", default= "/home/yxt/文档/mono_result/eval/UAVid_china/monovit_vggt_rflow_tinj_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_o07y_weights_25/eval_results.csv", help="Model A CSV path")
    p.add_argument("--csv_b", default= "/home/yxt/文档/mono_result/eval/UAVid_china/monovit_uavid_tridataset_512x288_bs8_lr1e-04_e40_step20_dn2l_weights_38/eval_results.csv", help="Model B CSV path")
    p.add_argument("--out_dir", default="/home/yxt/文档/mono_result/eval/rmse_diff_report", help="Output directory")
    p.add_argument("--rmse_abs_thr", type=float, default=0.05)
    p.add_argument("--rmse_pct_thr", type=float, default=0.10)
    p.add_argument("--tail_percentile", type=float, default=95.0)
    p.add_argument("--topk", type=int, default=100)
    args = p.parse_args()

    cfg = Config(rmse_abs_thr=args.rmse_abs_thr, rmse_pct_thr=args.rmse_pct_thr,
                 tail_percentile=args.tail_percentile, topk=args.topk)
    info = make_report(args.csv_a, args.csv_b, args.out_dir, cfg)
    print(json.dumps(info, ensure_ascii=False, indent=2))

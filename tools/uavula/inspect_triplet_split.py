#!/usr/bin/env python3
"""
Utility to inspect which UAV triplet samples are filtered out by a split file.

Example
-------
python tools/uavula/inspect_triplet_split.py \
    --data-root /mnt/data_nvme3n1p1/dataset/UAV_ula/dataset \
    --triplet-root /mnt/data_nvme3n1p1/dataset/UAV_ula/tri_win1_disp0.03_lap0.05_U0.07_R0_Qn0_S0 \
    --split UAVula \
    --phase train
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect triplet samples removed by split filtering."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/data_nvme3n1p1/dataset/UAV_ula/dataset"),
        help="Root directory of the dataset (containing Train/ and Validation/).",
    )
    parser.add_argument(
        "--triplet-root",
        type=Path,
        default=Path("/mnt/data_nvme3n1p1/dataset/UAV_ula/tri_win1_disp0.03_lap0_U0_R0_Qn0_S0"),
        help="Root directory containing triplets.jsonl files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="UAVula",
        help="Name of the split (e.g. UAVula, UAVid2020_China).",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["train", "val", "both"],
        default="both",
        help="Which split file to use (train/val). Default: train.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=20,
        help="Maximum number of dropped samples to print per section.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/inspect_triplet_split_output.json"),
        help="Optional JSON file to dump full details about dropped samples.",
    )
    return parser


def read_split_lines(split_file: Path):
    lines = []
    with split_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return lines


def _resolve_seq_for_join(image_root: Path, raw_seq: str) -> str:
    seq_s = raw_seq.replace("\\", "/").lstrip("./")
    root_s = str(image_root).replace("\\", "/").rstrip("/")
    if root_s.endswith("/Train") and seq_s.startswith("Train/"):
        return seq_s[len("Train/") :]
    if root_s.endswith("/Validation") and seq_s.startswith("Validation/"):
        return seq_s[len("Validation/") :]
    return seq_s


def _normalize_seq(raw: str) -> str:
    value = raw.replace("\\", "/").strip()
    if value.startswith("Train/"):
        value = value[len("Train/") :]
    if value.startswith("Validation/"):
        value = value[len("Validation/") :]
    if value.startswith("./"):
        value = value[2:]
    return value


def parse_split_pairs(lines):
    pair_whitelist = set()
    for ln in lines:
        cleaned = ln.split("#", 1)[0].strip()
        if not cleaned:
            continue
        cleaned = cleaned.replace("\\", "/")
        try:
            seq_part, idx_str = cleaned.rsplit(maxsplit=1)
            seq = _normalize_seq(seq_part)
            idx = int(idx_str)
            pair_whitelist.add((seq, idx))
        except ValueError:
            continue
    return pair_whitelist


def _extract_center_idx(filename: str) -> int:
    stem = Path(filename).stem
    return int(stem.lstrip("0") or "0")


def load_triplets(image_root: Path, triplet_root: Path, seq_filter=None):
    samples = []
    for jsonl_path in sorted(triplet_root.rglob("triplets.jsonl")):
        with jsonl_path.open("r", encoding="utf-8") as f:
            for raw in f:
                if not raw.strip():
                    continue
                rec = json.loads(raw)
                center_info = rec.get("center") or {}
                center_file = center_info.get("file")
                if not center_file:
                    continue
                seq = _resolve_seq_for_join(image_root, rec.get("seq", ""))
                if seq_filter is not None and seq not in seq_filter:
                    continue
                center_idx = _extract_center_idx(center_file)
                samples.append(
                    {
                        "seq": seq,
                        "center_idx": center_idx,
                        "center_file": center_file,
                        "source_json": str(jsonl_path),
                    }
                )
    return samples


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    split_dir = project_root / "methods" / "splits" / args.split
    phase_requested = args.phase

    if not args.triplet_root.exists():
        raise FileNotFoundError(f"Triplet root not found: {args.triplet_root}")

    results = []

    def analyze_phase(phase_name: str):
        split_file = split_dir / f"{phase_name}_files.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        split_lines = read_split_lines(split_file)
        whitelist = parse_split_pairs(split_lines)
        seq_filter = {seq for seq, _ in whitelist}

        phase_dir = "Train" if phase_name == "train" else "Validation"
        data_path = args.data_root / phase_dir
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        samples = load_triplets(data_path, args.triplet_root, seq_filter=seq_filter)

        all_keys = {(sample["seq"], sample["center_idx"]) for sample in samples}
        dropped = [
            sample for sample in samples if (sample["seq"], sample["center_idx"]) not in whitelist
        ]
        kept = len(samples) - len(dropped)
        missing_in_triplets = sorted(whitelist - all_keys)

        seq_counter = Counter(sample["seq"] for sample in samples)
        dropped_seq_counter = Counter(sample["seq"] for sample in dropped)

        summary = {
            "phase": phase_name,
            "split_file": str(split_file),
            "data_path": str(data_path),
            "total_triplets": len(samples),
            "whitelist_pairs": len(whitelist),
            "pairs_kept": kept,
            "pairs_dropped": len(dropped),
            "missing_pairs": len(missing_in_triplets),
            "sequence_counts": dict(seq_counter),
            "dropped_sequence_counts": dict(dropped_seq_counter),
        }

        print(f"=== Split Filtering Report ({phase_name}) ===")
        print(f"Split file:           {split_file}")
        print(f"Triplet root:         {args.triplet_root}")
        print(f"Dataset phase dir:    {data_path}")
        print(f"Total triplets found: {len(samples)}")
        print(f"Whitelist pairs:      {len(whitelist)}")
        print(f"Pairs kept:           {kept}")
        print(f"Pairs dropped:        {len(dropped)}")
        print(f"Whitelist missing in triplets: {len(missing_in_triplets)}")

        if args.max_print > 0:
            if dropped:
                counter = Counter((sample["seq"], sample["center_idx"]) for sample in dropped)
                print("\nTop dropped samples (seq, idx):")
                for (seq, idx), count in counter.most_common(args.max_print):
                    print(f"  {seq} #{idx}: {count} occurrence(s)")

                print("\nTop dropped sequences:")
                for seq, count in dropped_seq_counter.most_common(args.max_print):
                    print(f"  {seq}: {count} sample(s)")

            if missing_in_triplets:
                print("\nPairs referenced in split but missing from triplets:")
                for seq, idx in missing_in_triplets[: args.max_print]:
                    print(f"  {seq} #{idx}")

        summary["dropped"] = [
            {
                "seq": sample["seq"],
                "center_idx": int(sample["center_idx"]),
                "center_file": sample["center_file"],
                "source_json": sample["source_json"],
            }
            for sample in dropped
        ]
        summary["missing_in_triplets"] = [
            {"seq": seq, "center_idx": idx} for seq, idx in missing_in_triplets
        ]
        results.append(summary)

    if phase_requested == "both":
        analyze_phase("train")
        print("\n")
        analyze_phase("val")
    else:
        analyze_phase(phase_requested)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"triplet_root": str(args.triplet_root), "results": results}
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed report saved to: {args.output}")


if __name__ == "__main__":
    main()

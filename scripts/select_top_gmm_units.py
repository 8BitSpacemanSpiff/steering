#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib

import pandas as pd


DEFAULT_COLUMNS = [
    "gmm_score",
    "gmm_ap",
    "gmm_auc",
    "ap",
    "gmm_minus_ap",
    "diff_mean",
    "pos_k",
    "neg_k",
    "on_mode_mean",
    "on_p50",
    "on_p90",
    "off_mean",
    "layer",
    "unit",
    "pos_means",
    "neg_means",
]


def _parse_columns(value: str) -> list[str]:
    return [col.strip() for col in value.split(",") if col.strip()]


def _filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if args.min_gmm_score is not None:
        out = out[out["gmm_score"] >= args.min_gmm_score]
    if args.min_ap is not None and "ap" in out.columns:
        out = out[out["ap"] >= args.min_ap]
    if args.max_ap is not None and "ap" in out.columns:
        out = out[out["ap"] < args.max_ap]
    if args.min_pos_k is not None:
        out = out[out["pos_k"] >= args.min_pos_k]
    if args.min_neg_k is not None:
        out = out[out["neg_k"] >= args.min_neg_k]
    if args.layer_contains:
        out = out[out["layer"].astype(str).str.contains(args.layer_contains, regex=True)]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="select_top_gmm_units.py",
        description=(
            "Print and optionally save the top units from a GMM expertise CSV. "
            "This ranks by GMM columns and does not use AP unless AP filters are "
            "explicitly passed."
        ),
    )
    parser.add_argument("--gmm-csv", type=pathlib.Path, required=True)
    parser.add_argument("--out-csv", type=pathlib.Path, default=None)
    parser.add_argument("--sort-by", type=str, default="gmm_score")
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-gmm-score", type=float, default=None)
    parser.add_argument("--min-ap", type=float, default=None)
    parser.add_argument("--max-ap", type=float, default=None)
    parser.add_argument("--min-pos-k", type=int, default=None)
    parser.add_argument("--min-neg-k", type=int, default=None)
    parser.add_argument(
        "--layer-contains",
        type=str,
        default=None,
        help="Regex/text pattern layer names must contain, e.g. 'gate_proj' or 'up_proj'.",
    )
    parser.add_argument(
        "--columns",
        type=str,
        default=",".join(DEFAULT_COLUMNS),
        help="Comma-separated columns to print/save.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.gmm_csv)
    if args.sort_by not in df.columns:
        raise RuntimeError(f"Sort column not found: {args.sort_by}")

    selected = _filter(df, args).sort_values(args.sort_by, ascending=args.ascending)
    if args.top_n > 0:
        selected = selected.head(args.top_n)

    columns = [col for col in _parse_columns(args.columns) if col in selected.columns]
    print("source:", args.gmm_csv)
    print("rows_in:", len(df))
    print("rows_selected:", len(selected))
    print("sort_by:", args.sort_by)
    print()
    print(selected[columns].to_string(index=False))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(args.out_csv, index=False)
        print()
        print("saved:", args.out_csv)


if __name__ == "__main__":
    main()

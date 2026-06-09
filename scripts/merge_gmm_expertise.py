#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib

import pandas as pd


GMM_COLUMNS = [
    "gmm_score",
    "gmm_ap",
    "gmm_auc",
    "gmm_minus_ap",
    "diff_mean",
    "pos_k",
    "neg_k",
    "pos_bic",
    "neg_bic",
    "on_mode_mean",
    "pos_means",
    "neg_means",
]


def _merge(ap_df: pd.DataFrame, gmm_df: pd.DataFrame, keep: str) -> pd.DataFrame:
    keys = ["layer", "unit"]
    cols = keys + [c for c in GMM_COLUMNS if c in gmm_df.columns]
    merged = ap_df.merge(gmm_df[cols], on=keys, how="left")

    if keep == "gmm":
        merged = merged[merged["gmm_score"].notna()].copy()
    elif keep == "all":
        pass
    else:
        raise ValueError(f"Unknown keep mode: {keep}")

    # Make generation with --metric gmm_score safe even when keep=all.
    if "gmm_score" in merged.columns:
        merged["gmm_score"] = merged["gmm_score"].fillna(-1.0)
    if "on_mode_mean" in merged.columns:
        merged["on_mode_mean"] = merged["on_mode_mean"].fillna(merged["on_p50"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="merge_gmm_expertise.py",
        description=(
            "Merge an experiment-side GMM expertise table into the original AP "
            "expertise table. This creates a generation-friendly CSV with AP and "
            "GMM columns together."
        ),
    )
    parser.add_argument("--expertise-csv", type=pathlib.Path, required=True)
    parser.add_argument("--gmm-csv", type=pathlib.Path, required=True)
    parser.add_argument("--out-csv", type=pathlib.Path, required=True)
    parser.add_argument(
        "--keep",
        choices=["gmm", "all"],
        default="gmm",
        help=(
            "'gmm' keeps only rows scored by GMM. 'all' keeps every AP row and "
            "fills unscored gmm_score with -1."
        ),
    )
    args = parser.parse_args()

    ap_df = pd.read_csv(args.expertise_csv)
    gmm_df = pd.read_csv(args.gmm_csv)
    merged = _merge(ap_df=ap_df, gmm_df=gmm_df, keep=args.keep)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    print("saved:", args.out_csv)
    print("rows:", len(merged))
    print("columns:", ", ".join(merged.columns))
    if "gmm_score" in merged.columns:
        print()
        print("top by gmm_score:")
        cols = [
            "ap",
            "gmm_score",
            "gmm_minus_ap",
            "on_p50",
            "on_mode_mean",
            "pos_k",
            "neg_k",
            "layer",
            "unit",
        ]
        print(merged.sort_values("gmm_score", ascending=False)[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()

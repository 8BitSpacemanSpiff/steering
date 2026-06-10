#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compare_gmm_tables.py",
        description="Compare two GMM expertise CSVs on matched layer/unit rows.",
    )
    parser.add_argument("--left-csv", type=pathlib.Path, required=True)
    parser.add_argument("--right-csv", type=pathlib.Path, required=True)
    parser.add_argument("--left-name", type=str, default="left")
    parser.add_argument("--right-name", type=str, default="right")
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    left = pd.read_csv(args.left_csv)
    right = pd.read_csv(args.right_csv)
    merged = left.merge(
        right,
        on=["layer", "unit"],
        suffixes=(f"_{args.left_name}", f"_{args.right_name}"),
    )
    if merged.empty:
        raise RuntimeError("No matched layer/unit rows.")

    score_left = f"gmm_score_{args.left_name}"
    score_right = f"gmm_score_{args.right_name}"
    ap_left = f"gmm_ap_{args.left_name}"
    ap_right = f"gmm_ap_{args.right_name}"

    print("left:", args.left_csv)
    print("right:", args.right_csv)
    print("left rows:", len(left))
    print("right rows:", len(right))
    print("matched rows:", len(merged))
    print()
    print("correlations:")
    print(merged[[score_left, score_right, ap_left, ap_right]].corr().to_string())
    print()

    left_top = set(
        tuple(row)
        for row in left.sort_values("gmm_score", ascending=False)[["layer", "unit"]]
        .head(args.top_n)
        .values
    )
    right_top = set(
        tuple(row)
        for row in right.sort_values("gmm_score", ascending=False)[["layer", "unit"]]
        .head(args.top_n)
        .values
    )
    overlap = left_top & right_top
    print(f"top_{args.top_n}_overlap:", len(overlap))
    print()

    cols = [
        score_left,
        score_right,
        f"ap_{args.left_name}",
        f"ap_{args.right_name}",
        f"pos_k_{args.left_name}",
        f"pos_k_{args.right_name}",
        f"neg_k_{args.left_name}",
        f"neg_k_{args.right_name}",
        "layer",
        "unit",
    ]
    cols = [col for col in cols if col in merged.columns]
    print(f"top {args.left_name}:")
    print(merged.sort_values(score_left, ascending=False)[cols].head(args.top_n).to_string(index=False))
    print()
    print(f"top {args.right_name}:")
    print(merged.sort_values(score_right, ascending=False)[cols].head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()

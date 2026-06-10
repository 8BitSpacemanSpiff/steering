#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import glob
import pathlib

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="merge_gmm_expertise_shards.py",
        description="Merge sharded compute_gmm_expertise.py outputs into one GMM CSV.",
    )
    parser.add_argument(
        "--shard-glob",
        type=str,
        required=True,
        help="Glob for shard CSVs, e.g. '$CONCEPT_DIR/expertise/gmm_all_shard*.csv'.",
    )
    parser.add_argument("--out-csv", type=pathlib.Path, required=True)
    parser.add_argument(
        "--sort-by",
        type=str,
        default="gmm_score",
        help="Column to sort descending after merging.",
    )
    args = parser.parse_args()

    paths = sorted(pathlib.Path(path) for path in glob.glob(args.shard_glob))
    if not paths:
        raise RuntimeError(f"No shard files matched: {args.shard_glob}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_shard_file"] = path.name
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["layer", "unit"], keep="first")
    after = len(merged)

    if args.sort_by in merged.columns:
        merged = merged.sort_values(args.sort_by, ascending=False)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)

    print("matched shards:", len(paths))
    for path in paths:
        print(" -", path)
    print("rows_before_dedup:", before)
    print("rows_after_dedup:", after)
    print("saved:", args.out_csv)
    print()
    cols = [
        "gmm_score",
        "gmm_ap",
        "ap",
        "gmm_minus_ap",
        "pos_k",
        "neg_k",
        "layer",
        "unit",
    ]
    cols = [col for col in cols if col in merged.columns]
    print("top merged rows:")
    print(merged[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()

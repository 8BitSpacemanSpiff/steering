#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib

import pandas as pd

from selfcond.gmm import class_conditional_gmm_score
from selfcond.responses import read_responses_from_cached


def _parse_k_values(value: str):
    return tuple(int(k.strip()) for k in value.split(",") if k.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="probe_gmm.py",
        description=(
            "Fit class-conditional GMMs for a small set of units and print a "
            "terminal-friendly table. This is a probe before adding GMM to the "
            "full expertise CSV."
        ),
    )
    parser.add_argument("--responses-dir", type=pathlib.Path, required=True)
    parser.add_argument("--expertise-csv", type=pathlib.Path, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--metric", type=str, default="ap")
    parser.add_argument("--k-values", type=str, default="1,2,3")
    parser.add_argument("--reg-covar", type=float, default=1e-4)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--out-csv", type=pathlib.Path, default=None)
    args = parser.parse_args()

    responses, labels, _ = read_responses_from_cached(args.responses_dir, args.concept)
    if labels is None:
        raise RuntimeError("Cached responses do not contain labels.")

    expertise = pd.read_csv(args.expertise_csv)
    selected = expertise.sort_values(args.metric, ascending=False).head(args.top_n)

    rows = []
    for _, unit_row in selected.iterrows():
        layer = unit_row["layer"]
        unit = int(unit_row["unit"])
        result = class_conditional_gmm_score(
            unit_response=responses[layer][unit],
            labels=labels,
            k_values=_parse_k_values(args.k_values),
            reg_covar=args.reg_covar,
            n_init=args.n_init,
            random_state=args.random_state,
        )
        rows.append(
            {
                "ap": float(unit_row["ap"]),
                "gmm_ap": result.ap_from_llr,
                "gmm_auc": result.auc_from_llr,
                "pos_k": result.pos_k,
                "neg_k": result.neg_k,
                "on_mode_mean": result.on_mode_mean,
                "on_p50": float(unit_row.get("on_p50", float("nan"))),
                "on_p90": float(unit_row.get("on_p90", float("nan"))),
                "off_mean": float(unit_row.get("off_mean", float("nan"))),
                "layer": layer,
                "unit": unit,
                "pos_means": ";".join(f"{x:.6g}" for x in result.pos_means),
                "neg_means": ";".join(f"{x:.6g}" for x in result.neg_means),
                "pos_bic": result.pos_bic,
                "neg_bic": result.neg_bic,
            }
        )

    out = pd.DataFrame(rows)
    display_cols = [
        "ap",
        "gmm_ap",
        "gmm_auc",
        "pos_k",
        "neg_k",
        "on_mode_mean",
        "on_p50",
        "on_p90",
        "layer",
        "unit",
        "pos_means",
        "neg_means",
    ]
    print(out[display_cols].to_string(index=False))
    print()
    print("rows:", len(out))
    print("pos_k counts:")
    print(out["pos_k"].value_counts().sort_index().to_string())
    print("neg_k counts:")
    print(out["neg_k"].value_counts().sort_index().to_string())

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"saved: {args.out_csv}")


if __name__ == "__main__":
    main()

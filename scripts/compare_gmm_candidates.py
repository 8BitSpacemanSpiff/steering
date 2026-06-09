#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib
import typing as t

import numpy as np
import pandas as pd

from selfcond.gmm import class_conditional_gmm_score
from selfcond.responses import read_responses_from_cached


def _parse_k_values(value: str) -> t.Tuple[int, ...]:
    return tuple(int(k.strip()) for k in value.split(",") if k.strip())


def _unit_diff_mean(unit_response: np.ndarray, labels: np.ndarray) -> float:
    pos = unit_response[labels == 1]
    neg = unit_response[labels == 0]
    return float(np.mean(pos) - np.mean(neg))


def _select_candidates(
    expertise: pd.DataFrame,
    top_n: int,
    random_n: int,
    bins: t.Sequence[float],
    per_bin: int,
    random_state: int,
) -> pd.DataFrame:
    rs = np.random.RandomState(random_state)
    pieces = []

    top = expertise.sort_values("ap", ascending=False).head(top_n).copy()
    top["candidate_source"] = "top_ap"
    pieces.append(top)

    random_n = min(random_n, len(expertise))
    random_rows = expertise.sample(n=random_n, replace=False, random_state=rs).copy()
    random_rows["candidate_source"] = "random"
    pieces.append(random_rows)

    for low, high in zip(bins[:-1], bins[1:]):
        in_bin = expertise[(expertise["ap"] >= low) & (expertise["ap"] < high)]
        if in_bin.empty:
            continue
        sample_n = min(per_bin, len(in_bin))
        sampled = in_bin.sample(n=sample_n, replace=False, random_state=rs).copy()
        sampled["candidate_source"] = f"ap_{low:.2f}_{high:.2f}"
        pieces.append(sampled)

    out = pd.concat(pieces, ignore_index=True)
    return out.drop_duplicates(subset=["layer", "unit"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compare_gmm_candidates.py",
        description=(
            "Run a cheap AP vs diff-mean vs GMM comparison on selected candidate "
            "units. This is the early Step 7 sanity check before full GMM scoring."
        ),
    )
    parser.add_argument("--responses-dir", type=pathlib.Path, required=True)
    parser.add_argument("--expertise-csv", type=pathlib.Path, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--random-n", type=int, default=100)
    parser.add_argument("--per-bin", type=int, default=50)
    parser.add_argument("--bins", type=str, default="0,0.5,0.7,0.8,0.9,1.01")
    parser.add_argument("--k-values", type=str, default="1,2,3")
    parser.add_argument("--reg-covar", type=float, default=1e-4)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--out-csv", type=pathlib.Path, required=True)
    args = parser.parse_args()

    responses, labels, _ = read_responses_from_cached(args.responses_dir, args.concept)
    if labels is None:
        raise RuntimeError("Cached responses do not contain labels.")
    labels = np.asarray(labels, dtype=int)

    expertise = pd.read_csv(args.expertise_csv)
    bins = [float(x.strip()) for x in args.bins.split(",") if x.strip()]
    candidates = _select_candidates(
        expertise=expertise,
        top_n=args.top_n,
        random_n=args.random_n,
        bins=bins,
        per_bin=args.per_bin,
        random_state=args.random_state,
    )

    rows = []
    for _, unit_row in candidates.iterrows():
        layer = unit_row["layer"]
        unit = int(unit_row["unit"])
        unit_response = responses[layer][unit]
        result = class_conditional_gmm_score(
            unit_response=unit_response,
            labels=labels,
            k_values=_parse_k_values(args.k_values),
            reg_covar=args.reg_covar,
            n_init=args.n_init,
            random_state=args.random_state,
        )
        rows.append(
            {
                "candidate_source": unit_row["candidate_source"],
                "ap": float(unit_row["ap"]),
                "diff_mean": _unit_diff_mean(unit_response, labels),
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
            }
        )

    out = pd.DataFrame(rows)
    out["rank_ap"] = out["ap"].rank(ascending=False, method="min")
    out["rank_diff_mean"] = out["diff_mean"].rank(ascending=False, method="min")
    out["rank_gmm_ap"] = out["gmm_ap"].rank(ascending=False, method="min")
    out["gmm_minus_ap"] = out["gmm_ap"] - out["ap"]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("saved:", args.out_csv)
    print("rows:", len(out))
    print()
    print("candidate sources:")
    print(out["candidate_source"].value_counts().to_string())
    print()
    print("correlations:")
    print(out[["ap", "diff_mean", "gmm_ap", "gmm_auc"]].corr().to_string())
    print()
    print("mode counts:")
    print(pd.crosstab(out["pos_k"], out["neg_k"]).to_string())
    print()
    print("top by GMM AP:")
    cols = [
        "ap",
        "gmm_ap",
        "gmm_minus_ap",
        "diff_mean",
        "pos_k",
        "neg_k",
        "layer",
        "unit",
        "pos_means",
        "neg_means",
    ]
    print(out.sort_values("gmm_ap", ascending=False)[cols].head(25).to_string(index=False))
    print()
    print("GMM higher than AP by largest margin:")
    print(out.sort_values("gmm_minus_ap", ascending=False)[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()

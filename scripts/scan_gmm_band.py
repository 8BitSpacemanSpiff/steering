#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib
import typing as t
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from selfcond.gmm import class_conditional_gmm_score
from selfcond.responses import read_responses_from_cached


_RESPONSES = None
_LABELS = None


def _parse_k_values(value: str) -> t.Tuple[int, ...]:
    return tuple(int(k.strip()) for k in value.split(",") if k.strip())


def _init_worker(responses, labels):
    global _RESPONSES, _LABELS
    _RESPONSES = responses
    _LABELS = labels


def _diff_mean(unit_response: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(unit_response[labels == 1]) - np.mean(unit_response[labels == 0]))


def _score_row(row_dict, k_values, reg_covar, n_init, random_state):
    layer = row_dict["layer"]
    unit = int(row_dict["unit"])
    unit_response = _RESPONSES[layer][unit]
    result = class_conditional_gmm_score(
        unit_response=unit_response,
        labels=_LABELS,
        k_values=k_values,
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
    )
    return {
        "ap": float(row_dict["ap"]),
        "diff_mean": _diff_mean(unit_response, _LABELS),
        "gmm_ap": result.ap_from_llr,
        "gmm_auc": result.auc_from_llr,
        "gmm_minus_ap": result.ap_from_llr - float(row_dict["ap"]),
        "pos_k": result.pos_k,
        "neg_k": result.neg_k,
        "on_mode_mean": result.on_mode_mean,
        "on_p50": float(row_dict.get("on_p50", float("nan"))),
        "on_p90": float(row_dict.get("on_p90", float("nan"))),
        "off_mean": float(row_dict.get("off_mean", float("nan"))),
        "layer": layer,
        "unit": unit,
        "pos_means": ";".join(f"{x:.6g}" for x in result.pos_means),
        "neg_means": ";".join(f"{x:.6g}" for x in result.neg_means),
    }


def _select_band(
    expertise: pd.DataFrame,
    min_ap: float,
    max_ap: float,
    max_units: int,
    random_state: int,
) -> pd.DataFrame:
    selected = expertise[(expertise["ap"] >= min_ap) & (expertise["ap"] < max_ap)].copy()
    selected = selected.sort_values("ap", ascending=False)
    if max_units > 0 and len(selected) > max_units:
        selected = selected.sample(
            n=max_units,
            replace=False,
            random_state=np.random.RandomState(random_state),
        ).sort_values("ap", ascending=False)
    return selected.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scan_gmm_band.py",
        description=(
            "Score every unit in an AP band with class-conditional GMMs. Use this "
            "to find medium-AP units that GMM ranks much higher."
        ),
    )
    parser.add_argument("--responses-dir", type=pathlib.Path, required=True)
    parser.add_argument("--expertise-csv", type=pathlib.Path, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--min-ap", type=float, default=0.75)
    parser.add_argument("--max-ap", type=float, default=0.9)
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--k-values", type=str, default="1,2,3")
    parser.add_argument("--reg-covar", type=float, default=1e-4)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--cpus", type=int, default=None)
    parser.add_argument("--out-csv", type=pathlib.Path, required=True)
    args = parser.parse_args()

    responses, labels, _ = read_responses_from_cached(args.responses_dir, args.concept)
    if labels is None:
        raise RuntimeError("Cached responses do not contain labels.")
    labels = np.asarray(labels, dtype=int)

    expertise = pd.read_csv(args.expertise_csv)
    selected = _select_band(
        expertise=expertise,
        min_ap=args.min_ap,
        max_ap=args.max_ap,
        max_units=args.max_units,
        random_state=args.random_state,
    )
    if selected.empty:
        raise RuntimeError(f"No units found with {args.min_ap} <= ap < {args.max_ap}")

    worker_count = min(cpu_count() - 1, 8) if args.cpus is None else args.cpus
    worker_count = max(1, worker_count)
    rows = selected.to_dict(orient="records")
    score_fn = partial(
        _score_row,
        k_values=_parse_k_values(args.k_values),
        reg_covar=args.reg_covar,
        n_init=args.n_init,
        random_state=args.random_state,
    )

    if worker_count == 1:
        _init_worker(responses, labels)
        scored = [
            score_fn(row)
            for row in tqdm(rows, total=len(rows), desc="GMM scoring selected units")
        ]
    else:
        with Pool(
            processes=worker_count,
            initializer=_init_worker,
            initargs=(responses, labels),
        ) as pool:
            scored = list(
                tqdm(
                    pool.imap(score_fn, rows, chunksize=16),
                    total=len(rows),
                    desc=f"GMM scoring selected units [{worker_count} workers]",
                )
            )

    out = pd.DataFrame(scored)
    out["rank_ap_in_scan"] = out["ap"].rank(ascending=False, method="min")
    out["rank_gmm_in_scan"] = out["gmm_ap"].rank(ascending=False, method="min")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("saved:", args.out_csv)
    print("rows:", len(out))
    print("ap band:", args.min_ap, "<= ap <", args.max_ap)
    print()
    print("correlations:")
    print(out[["ap", "diff_mean", "gmm_ap", "gmm_auc"]].corr().to_string())
    print()
    print("mode counts:")
    print(pd.crosstab(out["pos_k"], out["neg_k"]).to_string())
    print()
    print("gmm_ap thresholds:")
    for threshold in [0.95, 0.9, 0.85, 0.8]:
        print(f"gmm_ap > {threshold}: {int((out['gmm_ap'] > threshold).sum())}")
    print()
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
    print("top by GMM AP:")
    print(out.sort_values("gmm_ap", ascending=False)[cols].head(30).to_string(index=False))
    print()
    print("GMM higher than AP by largest margin:")
    print(out.sort_values("gmm_minus_ap", ascending=False)[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()

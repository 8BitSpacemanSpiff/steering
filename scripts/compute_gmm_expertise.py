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
from selfcond.gmm_torch import score_layer_units_torch_gpu
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
        "gmm_score": result.ap_from_llr,
        "gmm_ap": result.ap_from_llr,
        "gmm_auc": result.auc_from_llr,
        "gmm_minus_ap": result.ap_from_llr - float(row_dict["ap"]),
        "pos_k": result.pos_k,
        "neg_k": result.neg_k,
        "pos_bic": result.pos_bic,
        "neg_bic": result.neg_bic,
        "on_mode_mean": result.on_mode_mean,
        "on_p50": float(row_dict.get("on_p50", float("nan"))),
        "on_p90": float(row_dict.get("on_p90", float("nan"))),
        "off_mean": float(row_dict.get("off_mean", float("nan"))),
        "layer": layer,
        "unit": unit,
        "uuid": int(row_dict["uuid"]) if "uuid" in row_dict else -1,
        "concept": row_dict.get("concept", ""),
        "group": row_dict.get("group", ""),
        "pos_means": ";".join(f"{x:.6g}" for x in result.pos_means),
        "neg_means": ";".join(f"{x:.6g}" for x in result.neg_means),
    }


def _select_units(
    expertise: pd.DataFrame,
    min_ap: float,
    max_ap: float,
    max_units: int,
    random_state: int,
) -> pd.DataFrame:
    selected = expertise.copy()
    if min_ap is not None:
        selected = selected[selected["ap"] >= min_ap]
    if max_ap is not None:
        selected = selected[selected["ap"] < max_ap]
    selected = selected.sort_values("ap", ascending=False)
    if max_units > 0 and len(selected) > max_units:
        selected = selected.sample(
            n=max_units,
            replace=False,
            random_state=np.random.RandomState(random_state),
        ).sort_values("ap", ascending=False)
    return selected.reset_index(drop=True)


def _apply_shard(selected: pd.DataFrame, num_shards: int, shard_index: int) -> pd.DataFrame:
    if num_shards <= 1:
        return selected
    if shard_index < 0 or shard_index >= num_shards:
        raise RuntimeError(
            f"--shard-index must be in [0, {num_shards - 1}], got {shard_index}."
        )
    row_ids = np.arange(len(selected))
    return selected[row_ids % num_shards == shard_index].reset_index(drop=True)


def _score_rows(
    rows,
    responses,
    labels,
    k_values,
    reg_covar,
    n_init,
    random_state,
    cpus,
    chunksize,
) -> pd.DataFrame:
    worker_count = min(max(cpu_count() - 1, 1), 32) if cpus is None else cpus
    worker_count = max(1, worker_count)
    chunksize = max(1, chunksize)
    score_fn = partial(
        _score_row,
        k_values=k_values,
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
    )
    if worker_count == 1:
        _init_worker(responses, labels)
        scored = [
            score_fn(row)
            for row in tqdm(rows, total=len(rows), desc="GMM scoring units")
        ]
    else:
        with Pool(
            processes=worker_count,
            initializer=_init_worker,
            initargs=(responses, labels),
        ) as pool:
            scored = list(
                tqdm(
                    pool.imap(score_fn, rows, chunksize=chunksize),
                    total=len(rows),
                    desc=f"GMM scoring units [{worker_count} workers]",
                )
            )
    return pd.DataFrame(scored)


def _score_rows_torch_gpu(
    rows,
    responses,
    labels,
    k_values,
    reg_covar,
    n_init,
    batch_size,
    max_iter,
    tol,
    device,
) -> pd.DataFrame:
    scored = []
    rows_df = pd.DataFrame(rows)
    for layer, layer_df in rows_df.groupby("layer", sort=True):
        print(f"GPU scoring layer: {layer} rows={len(layer_df)}")
        scored.extend(
            score_layer_units_torch_gpu(
                layer_responses=responses[layer],
                layer_rows=layer_df.to_dict(orient="records"),
                labels=labels,
                k_values=k_values,
                reg_covar=reg_covar,
                n_init=n_init,
                batch_size=batch_size,
                max_iter=max_iter,
                tol=tol,
                device=device,
            )
        )
    return pd.DataFrame(scored)


def _print_summary(
    out: pd.DataFrame,
    min_ap: float,
    max_ap: float,
    num_shards: int,
    shard_index: int,
) -> None:
    print("rows:", len(out))
    if min_ap is None and max_ap is None:
        print("ap filter: none")
    else:
        print("ap filter:", min_ap, "<= ap <", max_ap)
    if num_shards > 1:
        print("shard:", shard_index, "of", num_shards)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compute_gmm_expertise.py",
        description=(
            "Compute class-conditional GMM expertise as an experiment-side table. "
            "The original AP expertise.csv is read but not modified."
        ),
    )
    parser.add_argument("--responses-dir", type=pathlib.Path, required=True)
    parser.add_argument("--expertise-csv", type=pathlib.Path, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--min-ap", type=float, default=None)
    parser.add_argument("--max-ap", type=float, default=None)
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--k-values", type=str, default="1,2,3")
    parser.add_argument("--reg-covar", type=float, default=1e-4)
    parser.add_argument(
        "--backend",
        choices=["sklearn-cpu", "torch-gpu"],
        default="sklearn-cpu",
        help="GMM fitting backend. Use torch-gpu on CUDA machines for all-neuron runs.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=3,
        help=(
            "Number of sklearn GMM initializations per k. Lower is faster; "
            "higher is more stable."
        ),
    )
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help=(
            "Number of multiprocessing workers. Default uses up to 32 CPU cores; "
            "pass $(nproc) on a large VM to use all cores."
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=64,
        help=(
            "Rows per multiprocessing task for sklearn-cpu. Larger values reduce "
            "CPU scheduling overhead."
        ),
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=2048,
        help="Units per CUDA batch for --backend torch-gpu.",
    )
    parser.add_argument(
        "--gpu-max-iter",
        type=int,
        default=50,
        help="Maximum EM iterations per k/init for --backend torch-gpu.",
    )
    parser.add_argument(
        "--gpu-tol",
        type=float,
        default=1e-3,
        help="EM log-likelihood tolerance for --backend torch-gpu.",
    )
    parser.add_argument(
        "--gpu-device",
        type=str,
        default="cuda",
        help="Torch device for --backend torch-gpu.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split selected units into this many deterministic shards.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard to score, zero-indexed.",
    )
    parser.add_argument("--out-csv", type=pathlib.Path, required=True)
    args = parser.parse_args()

    responses, labels, _ = read_responses_from_cached(args.responses_dir, args.concept)
    if labels is None:
        raise RuntimeError("Cached responses do not contain labels.")
    labels = np.asarray(labels, dtype=int)

    expertise = pd.read_csv(args.expertise_csv)
    selected = _select_units(
        expertise=expertise,
        min_ap=args.min_ap,
        max_ap=args.max_ap,
        max_units=args.max_units,
        random_state=args.random_state,
    )
    selected = _apply_shard(
        selected=selected,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    if selected.empty:
        raise RuntimeError("No units selected for GMM scoring.")
    print("selected units:", len(selected))
    if args.num_shards > 1:
        print("selected shard:", args.shard_index, "of", args.num_shards)

    rows = selected.to_dict(orient="records")
    if args.backend == "torch-gpu":
        out = _score_rows_torch_gpu(
            rows=rows,
            responses=responses,
            labels=labels,
            k_values=_parse_k_values(args.k_values),
            reg_covar=args.reg_covar,
            n_init=args.n_init,
            batch_size=args.gpu_batch_size,
            max_iter=args.gpu_max_iter,
            tol=args.gpu_tol,
            device=args.gpu_device,
        )
    else:
        out = _score_rows(
            rows=rows,
            responses=responses,
            labels=labels,
            k_values=_parse_k_values(args.k_values),
            reg_covar=args.reg_covar,
            n_init=args.n_init,
            random_state=args.random_state,
            cpus=args.cpus,
            chunksize=args.chunksize,
        )
    out["rank_ap"] = out["ap"].rank(ascending=False, method="min")
    out["rank_gmm"] = out["gmm_ap"].rank(ascending=False, method="min")
    out["rank_diff_mean"] = out["diff_mean"].rank(ascending=False, method="min")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("saved:", args.out_csv)
    _print_summary(out, args.min_ap, args.max_ap, args.num_shards, args.shard_index)


if __name__ == "__main__":
    main()

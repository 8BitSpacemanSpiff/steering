#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import math
import typing as t

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


def _parse_k_values(k_values: t.Sequence[int]) -> t.Tuple[int, ...]:
    return tuple(sorted(set(int(k) for k in k_values if int(k) > 0)))


def _initial_means(x: torch.Tensor, k: int, init_idx: int, n_init: int) -> torch.Tensor:
    if k == 1:
        return x.mean(dim=1, keepdim=True)

    x_sorted = torch.sort(x, dim=1).values
    n = x.shape[1]
    base = torch.linspace(
        1.0 / (k + 1),
        k / (k + 1),
        k,
        device=x.device,
        dtype=x.dtype,
    )
    if n_init > 1:
        offset = (init_idx - (n_init - 1) / 2.0) * (0.15 / max(k, 1))
        base = torch.clamp(base + offset, 0.05, 0.95)
    indices = torch.clamp((base * (n - 1)).round().long(), 0, n - 1)
    return x_sorted[:, indices]


def _log_normal(x: torch.Tensor, means: torch.Tensor, variances: torch.Tensor) -> torch.Tensor:
    return -0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(variances[:, None, :])
        + ((x[:, :, None] - means[:, None, :]) ** 2) / variances[:, None, :]
    )


def _score_samples(
    x: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
) -> torch.Tensor:
    log_weights = torch.log(torch.clamp(weights, min=1e-12))
    return torch.logsumexp(_log_normal(x, means, variances) + log_weights[:, None, :], dim=2)


def _fit_one_k(
    x: torch.Tensor,
    k: int,
    reg_covar: float,
    n_init: int,
    max_iter: int,
    tol: float,
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n = x.shape
    best_ll = torch.full((batch,), -torch.inf, device=x.device, dtype=x.dtype)
    best_weights = torch.empty((batch, k), device=x.device, dtype=x.dtype)
    best_means = torch.empty((batch, k), device=x.device, dtype=x.dtype)
    best_vars = torch.empty((batch, k), device=x.device, dtype=x.dtype)

    global_var = torch.clamp(x.var(dim=1, unbiased=False), min=reg_covar)
    for init_idx in range(max(1, n_init)):
        means = _initial_means(x, k=k, init_idx=init_idx, n_init=max(1, n_init))
        variances = global_var[:, None].repeat(1, k)
        weights = torch.full((batch, k), 1.0 / k, device=x.device, dtype=x.dtype)
        prev_ll = None

        for _ in range(max_iter):
            log_prob = _log_normal(x, means, variances) + torch.log(
                torch.clamp(weights, min=1e-12)
            )[:, None, :]
            log_norm = torch.logsumexp(log_prob, dim=2)
            resp = torch.exp(log_prob - log_norm[:, :, None])
            nk = torch.clamp(resp.sum(dim=1), min=1e-8)
            weights = nk / n
            means = (resp * x[:, :, None]).sum(dim=1) / nk
            variances = torch.clamp(
                (resp * ((x[:, :, None] - means[:, None, :]) ** 2)).sum(dim=1) / nk,
                min=reg_covar,
            )

            ll = log_norm.sum(dim=1)
            if prev_ll is not None and torch.max(torch.abs(ll - prev_ll)).item() < tol:
                break
            prev_ll = ll

        ll = _score_samples(x, weights, means, variances).sum(dim=1)
        better = ll > best_ll
        best_ll = torch.where(better, ll, best_ll)
        best_weights[better] = weights[better]
        best_means[better] = means[better]
        best_vars[better] = variances[better]

    param_count = (3 * k) - 1
    bic = -2.0 * best_ll + param_count * math.log(n)
    return best_weights, best_means, best_vars, bic, best_ll


def _fit_best_gmm_batch(
    x_class: torch.Tensor,
    x_all: torch.Tensor,
    k_values: t.Sequence[int],
    reg_covar: float,
    n_init: int,
    max_iter: int,
    tol: float,
) -> t.Dict[str, torch.Tensor]:
    batch, n_all = x_all.shape
    max_k = max(k_values)
    best_bic = torch.full((batch,), torch.inf, device=x_all.device, dtype=x_all.dtype)
    best_k = torch.ones((batch,), device=x_all.device, dtype=torch.int64)
    best_scores = torch.empty((batch, n_all), device=x_all.device, dtype=x_all.dtype)
    best_means = torch.full((batch, max_k), torch.nan, device=x_all.device, dtype=x_all.dtype)

    for k in k_values:
        if x_class.shape[1] < k:
            continue
        weights, means, variances, bic, _ = _fit_one_k(
            x=x_class,
            k=k,
            reg_covar=reg_covar,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
        )
        scores = _score_samples(x_all, weights, means, variances)
        better = bic < best_bic
        best_bic = torch.where(better, bic, best_bic)
        best_k = torch.where(better, torch.full_like(best_k, k), best_k)
        best_scores[better] = scores[better]
        padded_means = torch.full_like(best_means, torch.nan)
        padded_means[:, :k] = means
        best_means[better] = padded_means[better]

    return {
        "bic": best_bic,
        "k": best_k,
        "scores": best_scores,
        "means": best_means,
    }


def _rank_scores(
    llr: np.ndarray,
    labels: np.ndarray,
) -> t.Tuple[np.ndarray, np.ndarray]:
    ap = np.zeros(llr.shape[0], dtype=float)
    auc = np.zeros(llr.shape[0], dtype=float)
    for idx, scores in enumerate(llr):
        ap[idx] = average_precision_score(y_true=labels, y_score=scores)
        auc[idx] = roc_auc_score(y_true=labels, y_score=scores)
    return ap, auc


def score_layer_units_torch_gpu(
    layer_responses: np.ndarray,
    layer_rows,
    labels: np.ndarray,
    k_values: t.Sequence[int] = (1, 2, 3),
    reg_covar: float = 1e-4,
    n_init: int = 3,
    batch_size: int = 2048,
    max_iter: int = 50,
    tol: float = 1e-3,
    device: str = "cuda",
) -> t.List[t.Dict[str, t.Any]]:
    """Score selected units from one layer using batched 1D GMMs on CUDA."""
    k_values = _parse_k_values(k_values)
    labels = np.asarray(labels, dtype=int)
    pos_mask = torch.tensor(labels == 1, device=device)
    neg_mask = torch.tensor(labels == 0, device=device)

    rows = list(layer_rows)
    results: t.List[t.Dict[str, t.Any]] = []
    for start in tqdm(range(0, len(rows), batch_size), desc="GMM scoring layer on GPU"):
        batch_rows = rows[start : start + batch_size]
        units = [int(row["unit"]) for row in batch_rows]
        x_np = np.asarray(layer_responses[units], dtype=np.float32)
        x = torch.tensor(x_np, device=device)
        response_mean = x.mean(dim=1)
        response_std = x.std(dim=1, unbiased=False)
        response_std = torch.where(response_std < 1e-8, torch.ones_like(response_std), response_std)
        x_std = (x - response_mean[:, None]) / response_std[:, None]

        pos_fit = _fit_best_gmm_batch(
            x_class=x_std[:, pos_mask],
            x_all=x_std,
            k_values=k_values,
            reg_covar=reg_covar,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
        )
        neg_fit = _fit_best_gmm_batch(
            x_class=x_std[:, neg_mask],
            x_all=x_std,
            k_values=k_values,
            reg_covar=reg_covar,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
        )

        llr = (pos_fit["scores"] - neg_fit["scores"]).detach().cpu().numpy()
        ap, auc = _rank_scores(llr, labels)
        diff_mean = (
            x[:, pos_mask].mean(dim=1) - x[:, neg_mask].mean(dim=1)
        ).detach().cpu().numpy()

        response_mean_cpu = response_mean.detach().cpu().numpy()
        response_std_cpu = response_std.detach().cpu().numpy()
        pos_means_cpu = pos_fit["means"].detach().cpu().numpy()
        neg_means_cpu = neg_fit["means"].detach().cpu().numpy()
        pos_k_cpu = pos_fit["k"].detach().cpu().numpy()
        neg_k_cpu = neg_fit["k"].detach().cpu().numpy()
        pos_bic_cpu = pos_fit["bic"].detach().cpu().numpy()
        neg_bic_cpu = neg_fit["bic"].detach().cpu().numpy()

        for idx, row in enumerate(batch_rows):
            pos_means = sorted(
                float(mean * response_std_cpu[idx] + response_mean_cpu[idx])
                for mean in pos_means_cpu[idx, : pos_k_cpu[idx]]
                if not np.isnan(mean)
            )
            neg_means = sorted(
                float(mean * response_std_cpu[idx] + response_mean_cpu[idx])
                for mean in neg_means_cpu[idx, : neg_k_cpu[idx]]
                if not np.isnan(mean)
            )
            row_ap = float(row["ap"])
            results.append(
                {
                    "ap": row_ap,
                    "diff_mean": float(diff_mean[idx]),
                    "gmm_score": float(ap[idx]),
                    "gmm_ap": float(ap[idx]),
                    "gmm_auc": float(auc[idx]),
                    "gmm_minus_ap": float(ap[idx] - row_ap),
                    "pos_k": int(pos_k_cpu[idx]),
                    "neg_k": int(neg_k_cpu[idx]),
                    "pos_bic": float(pos_bic_cpu[idx]),
                    "neg_bic": float(neg_bic_cpu[idx]),
                    "on_mode_mean": max(pos_means) if pos_means else float(response_mean_cpu[idx]),
                    "on_p50": float(row.get("on_p50", float("nan"))),
                    "on_p90": float(row.get("on_p90", float("nan"))),
                    "off_mean": float(row.get("off_mean", float("nan"))),
                    "layer": row["layer"],
                    "unit": int(row["unit"]),
                    "uuid": int(row["uuid"]) if "uuid" in row else -1,
                    "concept": row.get("concept", ""),
                    "group": row.get("group", ""),
                    "pos_means": ";".join(f"{x:.6g}" for x in pos_means),
                    "neg_means": ";".join(f"{x:.6g}" for x in neg_means),
                }
            )
    return results

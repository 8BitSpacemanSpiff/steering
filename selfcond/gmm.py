#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import typing as t
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class UnitGMMResult:
    """Class-conditional GMM summary for one unit response vector."""

    ap_from_llr: float
    auc_from_llr: float
    pos_k: int
    neg_k: int
    pos_bic: float
    neg_bic: float
    pos_means: t.List[float]
    neg_means: t.List[float]
    on_mode_mean: float
    response_mean: float
    response_std: float


def _as_column(x: t.Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1, 1)


def standardize_unit_response(
    unit_response: t.Sequence[float],
    min_std: float = 1e-8,
) -> t.Tuple[np.ndarray, float, float]:
    """Standardize one unit's responses before GMM fitting."""
    x = _as_column(unit_response)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std < min_std:
        std = 1.0
    return (x - mean) / std, mean, std


def _fit_gmm(
    x: np.ndarray,
    k: int,
    reg_covar: float,
    n_init: int,
    random_state: int,
) -> GaussianMixture:
    return GaussianMixture(
        n_components=k,
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
    ).fit(x)


def fit_best_gmm(
    x: np.ndarray,
    k_values: t.Sequence[int] = (1, 2, 3),
    reg_covar: float = 1e-4,
    n_init: int = 3,
    random_state: int = 0,
) -> t.Tuple[GaussianMixture, float]:
    """Fit candidate GMMs and choose the one with the lowest BIC."""
    best_model = None
    best_bic = np.inf
    max_k = min(max(k_values), len(x))
    candidate_ks = [k for k in k_values if 1 <= k <= max_k]
    for k in candidate_ks:
        gm = _fit_gmm(
            x=x,
            k=k,
            reg_covar=reg_covar,
            n_init=n_init,
            random_state=random_state,
        )
        bic = float(gm.bic(x))
        if bic < best_bic:
            best_model = gm
            best_bic = bic
    assert best_model is not None
    return best_model, best_bic


def _unstandardized_component_means(
    gm: GaussianMixture,
    response_mean: float,
    response_std: float,
) -> t.List[float]:
    means = gm.means_.reshape(-1)
    means = means * response_std + response_mean
    return sorted(float(x) for x in means)


def class_conditional_gmm_score(
    unit_response: t.Sequence[float],
    labels: t.Sequence[int],
    k_values: t.Sequence[int] = (1, 2, 3),
    reg_covar: float = 1e-4,
    n_init: int = 3,
    random_state: int = 0,
) -> UnitGMMResult:
    """
    Fit one GMM to positive responses and one GMM to negative responses.

    The per-sentence score is:
        log p(response | positive GMM) - log p(response | negative GMM)

    If that score ranks positive sentences above negative sentences, the unit is
    concept-relevant according to the GMM density-ratio view.
    """
    labels_array = np.asarray(labels, dtype=int)
    x_std, response_mean, response_std = standardize_unit_response(unit_response)

    pos_x = x_std[labels_array == 1]
    neg_x = x_std[labels_array == 0]
    if len(pos_x) < 2 or len(neg_x) < 2:
        return UnitGMMResult(
            ap_from_llr=0.0,
            auc_from_llr=0.5,
            pos_k=1,
            neg_k=1,
            pos_bic=float("nan"),
            neg_bic=float("nan"),
            pos_means=[],
            neg_means=[],
            on_mode_mean=response_mean,
            response_mean=response_mean,
            response_std=response_std,
        )

    pos_gm, pos_bic = fit_best_gmm(
        pos_x,
        k_values=k_values,
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
    )
    neg_gm, neg_bic = fit_best_gmm(
        neg_x,
        k_values=k_values,
        reg_covar=reg_covar,
        n_init=n_init,
        random_state=random_state,
    )

    llr = pos_gm.score_samples(x_std) - neg_gm.score_samples(x_std)
    ap_from_llr = float(average_precision_score(y_true=labels_array, y_score=llr))
    auc_from_llr = float(roc_auc_score(y_true=labels_array, y_score=llr))

    pos_means = _unstandardized_component_means(pos_gm, response_mean, response_std)
    neg_means = _unstandardized_component_means(neg_gm, response_mean, response_std)

    return UnitGMMResult(
        ap_from_llr=ap_from_llr,
        auc_from_llr=auc_from_llr,
        pos_k=int(pos_gm.n_components),
        neg_k=int(neg_gm.n_components),
        pos_bic=pos_bic,
        neg_bic=neg_bic,
        pos_means=pos_means,
        neg_means=neg_means,
        on_mode_mean=max(pos_means) if pos_means else response_mean,
        response_mean=response_mean,
        response_std=response_std,
    )

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import json
import pathlib
import typing as t

import numpy as np

from selfcond.gmm import fit_best_gmm, standardize_unit_response
from selfcond.responses import read_responses_from_cached


def _parse_k_values(value: str) -> t.Tuple[int, ...]:
    return tuple(int(k.strip()) for k in value.split(",") if k.strip())


def _load_sentences_like_concept_dataset(
    json_file: pathlib.Path,
    num_per_concept: int,
    random_seed: int,
) -> t.Tuple[t.List[str], np.ndarray]:
    random_state = np.random.RandomState(random_seed)
    label_map = {"positive": 1, "negative": 0}

    with json_file.open("r") as fp:
        json_data = json.load(fp)

    sentences = []
    labels = []
    for label in sorted(json_data["sentences"].keys()):
        label_sentences = json_data["sentences"][label]
        if num_per_concept is not None and num_per_concept < len(label_sentences):
            idx = random_state.choice(len(label_sentences), num_per_concept, replace=False)
        else:
            idx = np.arange(len(label_sentences))
        sentences += [label_sentences[i] for i in idx]
        labels += [label_map[label]] * len(idx)
    return sentences, np.asarray(labels, dtype=int)


def _shorten(text: str, width: int) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _print_mode_examples(
    title: str,
    mode_ids: np.ndarray,
    responsibilities: np.ndarray,
    values: np.ndarray,
    sentences: t.Sequence[str],
    global_indices: np.ndarray,
    top_n: int,
    text_width: int,
) -> None:
    print()
    print(title)
    for mode in sorted(np.unique(mode_ids)):
        selected = np.where(mode_ids == mode)[0]
        order = selected[np.argsort(responsibilities[selected, mode])[::-1]]
        print(f"mode {mode}: count={len(selected)}")
        for local_idx in order[:top_n]:
            global_idx = int(global_indices[local_idx])
            print(
                f"  resp={responsibilities[local_idx, mode]:.3f} "
                f"value={values[local_idx]:.6g} "
                f"idx={global_idx} "
                f"text={_shorten(sentences[global_idx], text_width)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="inspect_gmm_unit.py",
        description=(
            "Inspect one unit's GMM modes by printing the sentences most strongly "
            "assigned to each positive and negative mode."
        ),
    )
    parser.add_argument("--responses-dir", type=pathlib.Path, required=True)
    parser.add_argument("--concept-json", type=pathlib.Path, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument("--unit", type=int, required=True)
    parser.add_argument("--k-values", type=str, default="1,2,3")
    parser.add_argument("--reg-covar", type=float, default=1e-4)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--dataset-random-seed", type=int, default=1234)
    parser.add_argument("--num-per-concept", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--text-width", type=int, default=180)
    args = parser.parse_args()

    responses, cached_labels, _ = read_responses_from_cached(args.responses_dir, args.concept)
    if cached_labels is None:
        raise RuntimeError("Cached responses do not contain labels.")
    cached_labels = np.asarray(cached_labels, dtype=int)

    sentences, json_labels = _load_sentences_like_concept_dataset(
        json_file=args.concept_json,
        num_per_concept=args.num_per_concept,
        random_seed=args.dataset_random_seed,
    )
    if len(sentences) != len(cached_labels):
        print(
            "[warning] Reconstructed sentence count does not match cached labels: "
            f"{len(sentences)} vs {len(cached_labels)}. Trimming to the shorter length."
        )
        n = min(len(sentences), len(cached_labels))
        sentences = sentences[:n]
        json_labels = json_labels[:n]
        cached_labels = cached_labels[:n]
    if not np.array_equal(json_labels, cached_labels):
        print("[warning] Reconstructed labels do not exactly match cached labels.")

    unit_response = responses[args.layer][args.unit][: len(cached_labels)]
    x_std, response_mean, response_std = standardize_unit_response(unit_response)

    k_values = _parse_k_values(args.k_values)
    pos_indices = np.where(cached_labels == 1)[0]
    neg_indices = np.where(cached_labels == 0)[0]

    pos_gm, pos_bic = fit_best_gmm(
        x_std[pos_indices],
        k_values=k_values,
        reg_covar=args.reg_covar,
        n_init=args.n_init,
        random_state=args.random_state,
    )
    neg_gm, neg_bic = fit_best_gmm(
        x_std[neg_indices],
        k_values=k_values,
        reg_covar=args.reg_covar,
        n_init=args.n_init,
        random_state=args.random_state,
    )

    pos_resp = pos_gm.predict_proba(x_std[pos_indices])
    neg_resp = neg_gm.predict_proba(x_std[neg_indices])
    pos_modes = np.argmax(pos_resp, axis=1)
    neg_modes = np.argmax(neg_resp, axis=1)

    pos_means = sorted((pos_gm.means_.reshape(-1) * response_std + response_mean).tolist())
    neg_means = sorted((neg_gm.means_.reshape(-1) * response_std + response_mean).tolist())

    print("layer:", args.layer)
    print("unit:", args.unit)
    print("response_mean:", response_mean)
    print("response_std:", response_std)
    print("pos_k:", pos_gm.n_components, "pos_bic:", pos_bic, "pos_means:", pos_means)
    print("neg_k:", neg_gm.n_components, "neg_bic:", neg_bic, "neg_means:", neg_means)

    _print_mode_examples(
        title="positive examples by positive-GMM mode",
        mode_ids=pos_modes,
        responsibilities=pos_resp,
        values=unit_response[pos_indices],
        sentences=sentences,
        global_indices=pos_indices,
        top_n=args.top_n,
        text_width=args.text_width,
    )
    _print_mode_examples(
        title="negative examples by negative-GMM mode",
        mode_ids=neg_modes,
        responsibilities=neg_resp,
        values=unit_response[neg_indices],
        sentences=sentences,
        global_indices=neg_indices,
        top_n=args.top_n,
        text_width=args.text_width,
    )


if __name__ == "__main__":
    main()

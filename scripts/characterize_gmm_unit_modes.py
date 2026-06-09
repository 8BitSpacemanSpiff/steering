#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import collections
import json
import pathlib
import re
import typing as t

import numpy as np

from selfcond.gmm import fit_best_gmm, standardize_unit_response
from selfcond.responses import read_responses_from_cached


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "his",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "which",
    "with",
}


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


def _tokens(text: str) -> t.List[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def _distinctive_terms(
    token_counts_by_mode: t.Mapping[int, collections.Counter],
    mode: int,
    top_terms: int,
    min_count: int = 3,
) -> str:
    mode_counts = token_counts_by_mode[mode]
    other_counts = collections.Counter()
    for other_mode, counts in token_counts_by_mode.items():
        if other_mode != mode:
            other_counts.update(counts)

    mode_total = sum(mode_counts.values())
    other_total = sum(other_counts.values())
    vocab = set(mode_counts) | set(other_counts)
    scored = []
    for term in vocab:
        count = mode_counts[term]
        if count < min_count:
            continue
        # Smoothed frequency ratio: larger means more specific to this mode.
        mode_freq = (count + 1.0) / (mode_total + len(vocab))
        other_freq = (other_counts[term] + 1.0) / (other_total + len(vocab))
        scored.append((mode_freq / other_freq, term, count))

    scored.sort(reverse=True)
    return ", ".join(f"{term}:{count}({ratio:.2f}x)" for ratio, term, count in scored[:top_terms])


def _mode_summary(
    mode_name: str,
    mode_ids: np.ndarray,
    responsibilities: np.ndarray,
    values: np.ndarray,
    sentences: t.Sequence[str],
    global_indices: np.ndarray,
    top_terms: int,
) -> None:
    print()
    print(mode_name)
    token_counts_by_mode = {}
    text_by_mode = {}
    for mode in sorted(np.unique(mode_ids)):
        selected = np.where(mode_ids == mode)[0]
        selected_global = global_indices[selected]
        text = [sentences[int(i)] for i in selected_global]
        text_by_mode[int(mode)] = text
        token_counts = collections.Counter()
        for sentence in text:
            token_counts.update(_tokens(sentence))
        token_counts_by_mode[int(mode)] = token_counts

    for mode in sorted(np.unique(mode_ids)):
        selected = np.where(mode_ids == mode)[0]
        text = text_by_mode[int(mode)]
        token_counts = token_counts_by_mode[int(mode)]
        lengths = np.asarray([len(sentence.split()) for sentence in text], dtype=float)
        digit_rate = float(np.mean([bool(re.search(r"\d", sentence)) for sentence in text]))
        football_rate = float(np.mean(["football" in sentence.lower() for sentence in text]))
        rugby_rate = float(np.mean(["rugby" in sentence.lower() for sentence in text]))
        soccer_rate = float(np.mean(["soccer" in sentence.lower() for sentence in text]))

        terms = ", ".join(
            f"{term}:{count}" for term, count in token_counts.most_common(top_terms)
        )
        print(f"mode {mode}")
        print(f"  count: {len(selected)}")
        print(f"  mean_activation: {float(np.mean(values[selected])):.6g}")
        print(f"  mean_responsibility: {float(np.mean(responsibilities[selected, mode])):.6g}")
        print(f"  mean_words: {float(np.mean(lengths)):.3f}")
        print(f"  has_digit_rate: {digit_rate:.3f}")
        print(f"  football_rate: {football_rate:.3f}")
        print(f"  rugby_rate: {rugby_rate:.3f}")
        print(f"  soccer_rate: {soccer_rate:.3f}")
        print(f"  top_terms: {terms}")
        print(
            "  distinctive_terms: "
            + _distinctive_terms(token_counts_by_mode, int(mode), top_terms=top_terms)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="characterize_gmm_unit_modes.py",
        description=(
            "Print simple text statistics and top terms for each GMM mode of one unit. "
            "This is a lightweight terminal-only check for mode artifacts vs themes."
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
    parser.add_argument("--top-terms", type=int, default=20)
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
    pos_gm, _ = fit_best_gmm(
        x_std[pos_indices],
        k_values=k_values,
        reg_covar=args.reg_covar,
        n_init=args.n_init,
        random_state=args.random_state,
    )
    neg_gm, _ = fit_best_gmm(
        x_std[neg_indices],
        k_values=k_values,
        reg_covar=args.reg_covar,
        n_init=args.n_init,
        random_state=args.random_state,
    )

    pos_resp = pos_gm.predict_proba(x_std[pos_indices])
    neg_resp = neg_gm.predict_proba(x_std[neg_indices])

    print("layer:", args.layer)
    print("unit:", args.unit)
    print("response_mean:", response_mean)
    print("response_std:", response_std)
    print("pos_k:", pos_gm.n_components)
    print("neg_k:", neg_gm.n_components)

    _mode_summary(
        mode_name="positive-mode text summaries",
        mode_ids=np.argmax(pos_resp, axis=1),
        responsibilities=pos_resp,
        values=unit_response[pos_indices],
        sentences=sentences,
        global_indices=pos_indices,
        top_terms=args.top_terms,
    )
    _mode_summary(
        mode_name="negative-mode text summaries",
        mode_ids=np.argmax(neg_resp, axis=1),
        responsibilities=neg_resp,
        values=unit_response[neg_indices],
        sentences=sentences,
        global_indices=neg_indices,
        top_terms=args.top_terms,
    )


if __name__ == "__main__":
    main()

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
import pandas as pd

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
        mode_freq = (count + 1.0) / (mode_total + len(vocab))
        other_freq = (other_counts[term] + 1.0) / (other_total + len(vocab))
        scored.append((mode_freq / other_freq, term, count))
    scored.sort(reverse=True)
    return ", ".join(f"{term}:{count}({ratio:.2f}x)" for ratio, term, count in scored[:top_terms])


def _mode_rows(
    unit_row: pd.Series,
    class_name: str,
    mode_ids: np.ndarray,
    responsibilities: np.ndarray,
    values: np.ndarray,
    sentences: t.Sequence[str],
    global_indices: np.ndarray,
    top_terms: int,
) -> t.List[t.Dict[str, t.Any]]:
    token_counts_by_mode = {}
    text_by_mode = {}
    for mode in sorted(np.unique(mode_ids)):
        selected = np.where(mode_ids == mode)[0]
        selected_global = global_indices[selected]
        text = [sentences[int(i)] for i in selected_global]
        text_by_mode[int(mode)] = text
        counts = collections.Counter()
        for sentence in text:
            counts.update(_tokens(sentence))
        token_counts_by_mode[int(mode)] = counts

    rows = []
    for mode in sorted(np.unique(mode_ids)):
        selected = np.where(mode_ids == mode)[0]
        text = text_by_mode[int(mode)]
        counts = token_counts_by_mode[int(mode)]
        lengths = np.asarray([len(sentence.split()) for sentence in text], dtype=float)
        rows.append(
            {
                "source_ap": float(unit_row["ap"]),
                "source_gmm_ap": float(unit_row["gmm_ap"]),
                "source_gmm_minus_ap": float(unit_row["gmm_minus_ap"]),
                "source_diff_mean": float(unit_row["diff_mean"]),
                "layer": unit_row["layer"],
                "unit": int(unit_row["unit"]),
                "class": class_name,
                "mode": int(mode),
                "count": int(len(selected)),
                "mean_activation": float(np.mean(values[selected])),
                "mean_responsibility": float(np.mean(responsibilities[selected, mode])),
                "mean_words": float(np.mean(lengths)),
                "has_digit_rate": float(np.mean([bool(re.search(r"\d", s)) for s in text])),
                "football_rate": float(np.mean(["football" in s.lower() for s in text])),
                "rugby_rate": float(np.mean(["rugby" in s.lower() for s in text])),
                "soccer_rate": float(np.mean(["soccer" in s.lower() for s in text])),
                "top_terms": ", ".join(
                    f"{term}:{count}" for term, count in counts.most_common(top_terms)
                ),
                "distinctive_terms": _distinctive_terms(
                    token_counts_by_mode,
                    int(mode),
                    top_terms=top_terms,
                ),
            }
        )
    return rows


def _select_units(scan_df: pd.DataFrame, sort_by: str, top_units: int) -> pd.DataFrame:
    selected = scan_df.sort_values(sort_by, ascending=False).head(top_units)
    return selected.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="summarize_gmm_scan_modes.py",
        description=(
            "Summarize GMM mode text statistics for the top units from a scan CSV. "
            "This creates one CSV row per unit/class/mode."
        ),
    )
    parser.add_argument("--responses-dir", type=pathlib.Path, required=True)
    parser.add_argument("--concept-json", type=pathlib.Path, required=True)
    parser.add_argument("--concept", type=str, required=True)
    parser.add_argument("--scan-csv", type=pathlib.Path, required=True)
    parser.add_argument("--sort-by", type=str, default="gmm_ap")
    parser.add_argument("--top-units", type=int, default=10)
    parser.add_argument("--k-values", type=str, default="1,2,3")
    parser.add_argument("--reg-covar", type=float, default=1e-4)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--dataset-random-seed", type=int, default=1234)
    parser.add_argument("--num-per-concept", type=int, default=1000)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--out-csv", type=pathlib.Path, required=True)
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

    scan_df = pd.read_csv(args.scan_csv)
    selected_units = _select_units(scan_df, sort_by=args.sort_by, top_units=args.top_units)
    k_values = _parse_k_values(args.k_values)
    pos_indices = np.where(cached_labels == 1)[0]
    neg_indices = np.where(cached_labels == 0)[0]

    rows = []
    for _, unit_row in selected_units.iterrows():
        layer = unit_row["layer"]
        unit = int(unit_row["unit"])
        unit_response = responses[layer][unit][: len(cached_labels)]
        x_std, _, _ = standardize_unit_response(unit_response)
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
        rows.extend(
            _mode_rows(
                unit_row=unit_row,
                class_name="positive",
                mode_ids=np.argmax(pos_resp, axis=1),
                responsibilities=pos_resp,
                values=unit_response[pos_indices],
                sentences=sentences,
                global_indices=pos_indices,
                top_terms=args.top_terms,
            )
        )
        rows.extend(
            _mode_rows(
                unit_row=unit_row,
                class_name="negative",
                mode_ids=np.argmax(neg_resp, axis=1),
                responsibilities=neg_resp,
                values=unit_response[neg_indices],
                sentences=sentences,
                global_indices=neg_indices,
                top_terms=args.top_terms,
            )
        )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("saved:", args.out_csv)
    print("units:", len(selected_units))
    print("mode rows:", len(out))
    display_cols = [
        "source_ap",
        "source_gmm_ap",
        "source_gmm_minus_ap",
        "layer",
        "unit",
        "class",
        "mode",
        "count",
        "mean_activation",
        "football_rate",
        "rugby_rate",
        "soccer_rate",
        "distinctive_terms",
    ]
    print(out[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()

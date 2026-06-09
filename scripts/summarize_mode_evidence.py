#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import pathlib
import re
import typing as t

import pandas as pd


SPORT_TERMS = {
    "team",
    "teams",
    "league",
    "championship",
    "tournament",
    "player",
    "players",
    "game",
    "games",
    "match",
    "season",
    "stadium",
    "scoring",
    "score",
    "won",
    "final",
    "rugby",
    "basketball",
    "ufc",
    "sevens",
    "ball",
}

GOVERNANCE_TERMS = {
    "governing",
    "fifa",
    "official",
    "body",
    "bodies",
    "federation",
    "association",
    "country",
    "states",
    "rules",
    "codes",
}

ACTION_TERMS = {
    "played",
    "playing",
    "play",
    "player",
    "players",
    "score",
    "scoring",
    "goals",
    "offside",
    "match",
    "season",
    "stadium",
    "signed",
    "moved",
    "win",
    "winning",
}

ARTIFACT_TERMS = {
    "including",
    "particularly",
    "especially",
    "such",
    "however",
    "specific",
    "reference",
}


def _term_counts(term_blob: str) -> t.Dict[str, int]:
    counts = {}
    for term, count in re.findall(r"([a-z0-9]+):([0-9]+)", str(term_blob).lower()):
        counts[term] = counts.get(term, 0) + int(count)
    return counts


def _score_terms(term_blob: str, term_set: t.Set[str]) -> int:
    counts = _term_counts(term_blob)
    return int(sum(count for term, count in counts.items() if term in term_set))


def _add_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    term_source = out["distinctive_terms"].fillna("") + ", " + out["top_terms"].fillna("")
    out["sports_term_score"] = term_source.map(lambda x: _score_terms(x, SPORT_TERMS))
    out["governance_term_score"] = term_source.map(lambda x: _score_terms(x, GOVERNANCE_TERMS))
    out["action_term_score"] = term_source.map(lambda x: _score_terms(x, ACTION_TERMS))
    out["artifact_term_score"] = term_source.map(lambda x: _score_terms(x, ARTIFACT_TERMS))
    return out


def _print_rows(title: str, df: pd.DataFrame, cols: t.Sequence[str], n: int) -> None:
    print()
    print(title)
    if df.empty:
        print("(none)")
        return
    print(df[cols].head(n).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="summarize_mode_evidence.py",
        description=(
            "Aggregate mode-summary CSV rows into simple evidence categories: "
            "sports confounds, football governance/history, football action, and artifacts."
        ),
    )
    parser.add_argument("--mode-summary-csv", type=pathlib.Path, required=True)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    df = _add_scores(pd.read_csv(args.mode_summary_csv))

    print("rows:", len(df))
    print("units:", df[["layer", "unit"]].drop_duplicates().shape[0])
    print()
    print("mode rows by class:")
    print(df["class"].value_counts().to_string())
    print()
    print("negative modes with sports_term_score > 0:", int(((df["class"] == "negative") & (df["sports_term_score"] > 0)).sum()))
    print("positive modes with governance_term_score > 0:", int(((df["class"] == "positive") & (df["governance_term_score"] > 0)).sum()))
    print("positive modes with action_term_score > 0:", int(((df["class"] == "positive") & (df["action_term_score"] > 0)).sum()))
    print("modes with artifact_term_score > 0:", int((df["artifact_term_score"] > 0).sum()))

    cols = [
        "source_ap",
        "source_gmm_ap",
        "source_gmm_minus_ap",
        "layer",
        "unit",
        "class",
        "mode",
        "count",
        "football_rate",
        "rugby_rate",
        "sports_term_score",
        "governance_term_score",
        "action_term_score",
        "artifact_term_score",
        "distinctive_terms",
    ]

    _print_rows(
        "strongest negative sports-confound modes",
        df[df["class"] == "negative"].sort_values("sports_term_score", ascending=False),
        cols,
        args.top_n,
    )
    _print_rows(
        "strongest positive governance/history/code modes",
        df[df["class"] == "positive"].sort_values("governance_term_score", ascending=False),
        cols,
        args.top_n,
    )
    _print_rows(
        "strongest positive match/player/action modes",
        df[df["class"] == "positive"].sort_values("action_term_score", ascending=False),
        cols,
        args.top_n,
    )
    _print_rows(
        "possible artifact-heavy modes",
        df.sort_values("artifact_term_score", ascending=False),
        cols,
        args.top_n,
    )


if __name__ == "__main__":
    main()

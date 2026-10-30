"""Compare two retrieval runs with paired reciprocal-rank bootstrapping.

    python -m scripts.compare_runs bge-small+hybrid bge-small
    python -m scripts.compare_runs bge-small+hybrid+hyde bge-small+hybrid --metric answer20

--metric answer20 compares "the answer chunk (every part, for multi) is in the
top 20" (1/0) instead of reciprocal rank; the mean difference is then the
answer-chunk recall@20 difference. Only questions labelled in both runs count.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from scripts.results_table import latest_runs

RESULTS = Path(__file__).resolve().parents[1] / "result"
METRICS = ("mrr", "answer20")


def reciprocal_rank(row: dict[str, Any]) -> float:
    rank = row.get("rank")
    return 1 / rank if isinstance(rank, int) and rank > 0 else 0.0


def answer20(row: dict[str, Any]) -> float:
    rank = row.get("answer_rank")
    return 1.0 if isinstance(rank, int) and 0 < rank <= 20 else 0.0


def _paired_reciprocal_ranks(
    first: dict[str, Any],
    second: dict[str, Any],
    only_repo: str | None,
    metric: str = "mrr",
) -> tuple[list[float], list[float]]:
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; expected one of {', '.join(METRICS)}")
    first_config = first.get("config", {})
    second_config = second.get("config", {})
    first_hash = first_config.get("golden_sha256")
    second_hash = second_config.get("golden_sha256")
    if first_hash and second_hash and first_hash != second_hash:
        raise ValueError("Runs use different golden sets and cannot be paired")

    def records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
        result = {}
        for row in payload.get("questions", []):
            if row.get("is_negative", row.get("type") == "negative"):
                continue
            expected = row.get("expected", [])
            repos = row.get("repos") or [
                item.get("repo")
                for item in expected
                if isinstance(item, dict)
            ]
            if only_repo and only_repo not in repos:
                continue
            result[str(row["id"])] = row
        return result

    first_rows = records(first)
    second_rows = records(second)
    question_ids = sorted(first_rows.keys() & second_rows.keys())
    if metric == "answer20":
        question_ids = [
            question_id
            for question_id in question_ids
            if first_rows[question_id].get("answer_labelled")
            and second_rows[question_id].get("answer_labelled")
        ]
    if not question_ids:
        raise ValueError(
            "The selected runs have no paired answerable questions"
            + (" with answer-chunk labels" if metric == "answer20" else "")
        )

    score = answer20 if metric == "answer20" else reciprocal_rank
    first_rr = [score(first_rows[question_id]) for question_id in question_ids]
    second_rr = [score(second_rows[question_id]) for question_id in question_ids]
    return first_rr, second_rr


def bootstrap_difference(
    first: list[float],
    second: list[float],
    *,
    boots: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    if len(first) != len(second) or not first:
        raise ValueError("Paired samples must be non-empty and equal in length")
    if boots < 1:
        raise ValueError("boots must be at least 1")
    differences = [left - right for left, right in zip(first, second)]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    samples = sorted(
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(boots)
    )
    low = samples[int(0.025 * (boots - 1))]
    high = samples[int(0.975 * (boots - 1))]
    return observed, low, high


def compare_runs(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    only_repo: str | None = None,
    boots: int = 10_000,
    seed: int = 0,
    metric: str = "mrr",
) -> dict[str, Any]:
    """Paired comparison; with metric="answer20" the "mrr_*" keys hold answer recall@20."""
    first_rr, second_rr = _paired_reciprocal_ranks(first, second, only_repo, metric)
    difference, low, high = bootstrap_difference(
        first_rr,
        second_rr,
        boots=boots,
        seed=seed,
    )
    paired_differences = [
        left - right for left, right in zip(first_rr, second_rr)
    ]
    return {
        "metric": metric,
        "n": len(first_rr),
        "mrr_first": sum(first_rr) / len(first_rr),
        "mrr_second": sum(second_rr) / len(second_rr),
        "mrr_difference": difference,
        "ci_95": [low, high],
        "wins_ties_losses": [
            sum(value > 0 for value in paired_differences),
            sum(value == 0 for value in paired_differences),
            sum(value < 0 for value in paired_differences),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", help="first run config name")
    parser.add_argument("second", help="second run config name")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--only-repo")
    parser.add_argument("--boots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="mrr",
        help="answer20: compare answer-chunk recall@20 instead of MRR",
    )
    args = parser.parse_args()
    runs = latest_runs(args.results_dir)
    missing = [name for name in (args.first, args.second) if name not in runs]
    if missing:
        parser.error(f"No saved run found for config(s): {', '.join(missing)}")
    try:
        result = compare_runs(
            runs[args.first],
            runs[args.second],
            only_repo=args.only_repo,
            boots=args.boots,
            seed=args.seed,
            metric=args.metric,
        )
    except ValueError as error:
        parser.error(str(error))
    wins, ties, losses = result["wins_ties_losses"]
    low, high = result["ci_95"]
    label = "MRR" if args.metric == "mrr" else "ansR@20"
    print(f"paired questions: {result['n']}")
    print(
        f"{args.first} {label}={result['mrr_first']:.3f}; "
        f"{args.second} {label}={result['mrr_second']:.3f}"
    )
    print(
        f"{label} difference ({args.first} - {args.second})="
        f"{result['mrr_difference']:+.3f}; bootstrap 95% CI "
        f"[{low:+.3f}, {high:+.3f}]"
    )
    print(f"wins/ties/losses: {wins}/{ties}/{losses}")


if __name__ == "__main__":
    main()

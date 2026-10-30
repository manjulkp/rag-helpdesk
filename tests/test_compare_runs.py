from __future__ import annotations

import pytest

from scripts.compare_runs import bootstrap_difference, compare_runs


def _run(name: str, ranks: list[int | None]) -> dict:
    return {
        "config": {"name": name, "golden_sha256": "same"},
        "questions": [
            {
                "id": f"q{index}",
                "type": "concept",
                "repos": ["prediction-markets" if index == 0 else "other"],
                "expected": [],
                "rank": rank,
                "is_negative": False,
            }
            for index, rank in enumerate(ranks)
        ],
    }


def test_bootstrap_difference_is_paired_and_reproducible() -> None:
    first = [1.0, 0.5, 0.0]
    second = [0.5, 0.5, 0.0]

    assert bootstrap_difference(first, second, boots=1000, seed=7) == (
        bootstrap_difference(first, second, boots=1000, seed=7)
    )
    assert bootstrap_difference(first, second, boots=1000, seed=7)[0] == pytest.approx(
        1 / 6
    )


def test_compare_runs_reports_paired_wins_ties_losses_and_repo_filter() -> None:
    result = compare_runs(
        _run("hybrid", [1, 2, None]),
        _run("vector", [2, 2, 1]),
        only_repo="prediction-markets",
        boots=100,
    )

    assert result["n"] == 1
    assert result["wins_ties_losses"] == [1, 0, 0]
    assert result["mrr_difference"] == pytest.approx(0.5)
    assert result["ci_95"] == [0.5, 0.5]


def test_compare_runs_answer20_uses_answer_rank_and_skips_unlabelled() -> None:
    first = _run("hyde", [1, 1, 1])
    second = _run("plain", [1, 1, 1])
    for row, (answer, labelled) in zip(
        first["questions"], [(3, True), (None, True), (2, False)]
    ):
        row["answer_rank"], row["answer_labelled"] = answer, labelled
    for row, (answer, labelled) in zip(
        second["questions"], [(25, True), (None, True), (None, False)]
    ):
        row["answer_rank"], row["answer_labelled"] = answer, labelled

    result = compare_runs(first, second, boots=100, metric="answer20")

    assert result["metric"] == "answer20"
    assert result["n"] == 2
    assert result["mrr_first"] == pytest.approx(0.5)
    assert result["mrr_second"] == 0.0
    assert result["wins_ties_losses"] == [1, 1, 0]


def test_compare_runs_rejects_different_golden_sets() -> None:
    first = _run("a", [1])
    second = _run("b", [1])
    second["config"]["golden_sha256"] = "different"

    with pytest.raises(ValueError, match="different golden sets"):
        compare_runs(first, second)

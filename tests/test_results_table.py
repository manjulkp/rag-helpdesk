from __future__ import annotations

import sys

from helpdesk.evaluate import QuestionResult, save_run
from scripts.results_table import main


def _result(question_id: str, repo: str, rank: int) -> QuestionResult:
    path = f"{question_id}.md"
    retrieved = [
        {"repo": "other", "path": "other.md", "score": 0.2, "title": "other"}
    ]
    retrieved.insert(
        rank - 1,
        {"repo": repo, "path": path, "score": 0.9, "title": path},
    )
    return QuestionResult(
        id=question_id,
        type="concept",
        repos=[repo],
        expected=[(repo, path)],
        retrieved=retrieved,
        rank=rank,
        all_found_rank=rank,
    )


def test_results_table_defaults_to_best_full_run_and_compares_pred_repo(
    tmp_path, monkeypatch, capsys
) -> None:
    save_run(
        [_result("q1", "prediction-markets", 2), _result("q2", "option-pricer", 2)],
        {"name": "lower-mrr"},
        tmp_path,
    )
    save_run(
        [_result("q1", "prediction-markets", 1), _result("q2", "option-pricer", 1)],
        {"name": "best-mrr"},
        tmp_path,
    )
    save_run(
        [_result("q1", "prediction-markets", 1)],
        {"name": "pred-v1", "only_repo": "prediction-markets"},
        tmp_path,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["results_table", "--results-dir", str(tmp_path)],
    )
    main()

    output = capsys.readouterr().out
    assert "## prediction-markets questions only (fair comparison with pred-v1)" in output
    assert "| best-mrr | 1 |" in output
    assert "## best-mrr: by type" in output
    assert "## lower-mrr: by type" not in output

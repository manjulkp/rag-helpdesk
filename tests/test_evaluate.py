from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpdesk.evaluate import (
    GoldenQuestion,
    QuestionResult,
    all_found_rank,
    answer_rank,
    first_relevant_rank,
    load_golden,
    mrr,
    recall_at_k,
    run,
    save_run,
    summarise,
)
from helpdesk.sources import REPOS


def question(
    question_id: str,
    expected: list[tuple[str, str]],
    question_type: str = "concept",
) -> GoldenQuestion:
    return GoldenQuestion(question_id, question_id, question_type, expected)


def hit(repo: str, path: str, score: float = 0.9) -> dict[str, object]:
    return {"repo": repo, "path": path, "score": score, "title": path}


def test_first_relevant_rank_matches_any_expected_file() -> None:
    retrieved = [
        hit("a", "unrelated.md"),
        hit("b", "docs/target.md"),
    ]

    assert first_relevant_rank(retrieved, [("b", "docs/target.md")]) == 2
    assert first_relevant_rank(retrieved, []) is None


def test_all_found_rank_requires_all_expected_files() -> None:
    retrieved = [
        hit("a", "one.md"),
        hit("a", "one.md"),
        hit("b", "two.md"),
    ]

    assert all_found_rank(retrieved, [("a", "one.md"), ("b", "two.md")]) == 3
    assert all_found_rank(retrieved[:2], [("a", "one.md"), ("b", "two.md")]) is None


def test_answer_rank_requires_every_title_group() -> None:
    retrieved = [
        {"repo": "a", "path": "one.md", "title": "API > Overview"},
        {"repo": "a", "path": "one.md", "title": "API > Retry"},
        {"repo": "b", "path": "two.md", "title": "Tests > Retry"},
    ]
    groups = [
        [("a", "one.md", "retry")],
        [("b", "two.md", "retry")],
    ]

    assert answer_rank(retrieved, groups) == 3
    assert answer_rank(retrieved[:2], groups) is None
    assert answer_rank(retrieved, []) is None


def test_answer_rank_needs_the_chunk_from_the_expected_file() -> None:
    retrieved = [
        {"repo": "x", "path": "docs/a.md", "title": "Retry"},  # wrong repo
        {"repo": "r", "path": "docs/a.md", "title": "Overview"},
        {"repo": "r", "path": "docs/a.md", "title": "A > Retry rules"},
        {"repo": "r", "path": "docs/b.md", "title": "anything"},
    ]
    groups = [[("r", "docs/a.md", "retry")]]

    assert answer_rank(retrieved, groups) == 3
    assert answer_rank(retrieved, groups + [[("r", "docs/b.md", "")]]) == 4
    assert answer_rank(retrieved[:3], groups + [[("r", "docs/b.md", "")]]) is None


def test_run_keeps_file_rank_separate_from_answer_rank() -> None:
    labelled = GoldenQuestion(
        "q1",
        "q1",
        "concept",
        [("repo", "expected.md")],
        expected_chunks=[[("repo", "expected.md", "retry")]],
    )
    retrieved = [
        {"repo": "repo", "path": "expected.md", "title": "Overview", "score": 0.9},
        {"repo": "repo", "path": "other.md", "title": "Retry", "score": 0.8},
        {"repo": "repo", "path": "expected.md", "title": "Retry", "score": 0.7},
    ]

    results = run(
        [labelled, question("q2", [("repo", "expected.md")])],
        lambda _text, _depth: retrieved,
    )
    summary = summarise(results)["overall"]

    assert results[0].rank == 1
    assert results[0].answer_rank == 3
    assert results[0].answer_labelled
    assert not results[1].answer_labelled
    assert summary["n_answer_labelled"] == 1
    assert summary["answer_recall@8"] == 1.0
    assert summary["answer_recall@20"] == 1.0


def test_negative_questions_are_excluded_from_recall_and_mrr() -> None:
    results = [
        QuestionResult(
            "q1", "concept", ["a"], [("a", "one.md")], [hit("a", "one.md")], 1, 1
        ),
        QuestionResult("q2", "negative", [], [], [hit("a", "one.md")], None, None, True),
    ]

    assert recall_at_k(results, 1) == 1.0
    assert mrr(results) == 1.0
    assert recall_at_k([results[1]], 1) is None
    assert mrr([results[1]]) is None


def test_run_records_ranks_and_top_hits() -> None:
    questions = [
        question("q1", [("repo", "expected.md")]),
        question("q2", [], "negative"),
    ]

    results = run(
        questions,
        lambda text, depth: (
            [hit("repo", "expected.md")] if text == "q1" else [hit("repo", "other.md", 0.2)]
        ),
    )

    assert results[0].rank == results[0].all_found_rank == 1
    assert results[1].rank is None
    assert results[1].top_score == pytest.approx(0.2)


def test_summary_includes_multi_recall_and_negative_score_spread() -> None:
    results = [
        QuestionResult(
            "q1",
            "multi",
            ["a", "b"],
            [("a", "one.md"), ("b", "two.md")],
            [hit("a", "one.md"), hit("b", "two.md")],
            1,
            2,
        ),
        QuestionResult(
            "q2", "negative", [], [], [hit("a", "no.md", 0.3)], None, None, True
        ),
    ]

    summary = summarise(results)

    assert summary["overall"]["n"] == 1
    assert summary["multi_all_found"]["1"] == 0.0
    assert summary["multi_all_found"]["3"] == 1.0
    assert summary["top1_score"]["negative"]["median"] == pytest.approx(0.3)


def test_load_golden_validates_duplicates_and_negative_expected_files(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [
                {"id": "q1", "question": "question", "type": "negative", "expected": []},
                {"id": "q1", "question": "duplicate", "type": "negative", "expected": []},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_golden(path)


def test_save_run_writes_config_summary_and_questions(tmp_path: Path) -> None:
    results = run(
        [question("q1", [("repo", "expected.md")])],
        lambda _query, _depth: [hit("repo", "expected.md")],
    )

    path = save_run(results, {"name": "test"}, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["config"]["name"] == "test"
    assert payload["summary"]["overall"]["recall@1"] == 1.0
    assert payload["questions"][0]["expected"] == [
        {"repo": "repo", "path": "expected.md"}
    ]


def test_real_golden_set_has_expected_mix_and_15_questions() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "golden.json"
    questions = load_golden(path)
    counts = {kind: sum(question.type == kind for question in questions) for kind in (
        "concept", "test", "identifier", "multi", "negative"
    )}

    assert len(questions) == 15
    assert counts == {
        "concept": 4,
        "test": 3,
        "identifier": 1,
        "multi": 2,
        "negative": 5,
    }


def test_real_golden_set_references_files_in_configured_repositories() -> None:
    missing_repositories = [
        str(path) for path in REPOS.values() if not path.is_dir()
    ]
    if missing_repositories:
        pytest.skip(
            "Source repositories are unavailable: "
            + ", ".join(missing_repositories)
        )

    from helpdesk.loader import load_documents

    path = Path(__file__).resolve().parents[1] / "data" / "golden.json"
    questions = load_golden(path)
    assert all(question.repos for question in questions if question.type != "negative")
    assert all(
        repo in REPOS
        for question in questions
        for repo, _ in question.expected
    )
    assert all(
        len(set(question.expected)) == 2
        for question in questions
        if question.type == "multi"
    )

    corpus = {(document.repo, document.path) for document in load_documents(REPOS)}
    missing_files = [
        (question.id, repo, expected_path)
        for question in questions
        for repo, expected_path in question.expected
        if (repo, expected_path) not in corpus
    ]
    assert not missing_files, f"Golden expected files are absent: {missing_files}"


def test_real_golden_chunk_titles_exist_in_source_chunks() -> None:
    if any(not path.is_dir() for path in REPOS.values()):
        pytest.skip("repositories are not available on this machine")

    from helpdesk.chunker import chunk_documents
    from helpdesk.loader import load_documents

    path = Path(__file__).resolve().parents[1] / "data" / "golden.json"
    questions = load_golden(path)
    chunks = chunk_documents(load_documents(REPOS), 400, lambda text: len(text.split()))
    available = {(chunk.repo, chunk.path, chunk.title.casefold()) for chunk in chunks}
    missing = [
        (question.id, repo, source_path, title)
        for question in questions
        for group in question.expected_chunks
        for repo, source_path, title in group
        if not any(
            actual_repo == repo
            and actual_path == source_path
            and title in actual_title
            for actual_repo, actual_path, actual_title in available
        )
    ]

    assert not missing, f"Golden chunk titles are absent: {missing}"

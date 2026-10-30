from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from helpdesk.generate import (
    ClaudeGenerator,
    ExtractiveGenerator,
    GeneratorError,
    markers,
)
from helpdesk.index import SearchHit
from scripts.eval_answers import summarise_answers
from scripts.grade_sheet import render_sheet


def hit(
    chunk_id: str,
    repo: str = "option-pricer",
    path: str = "docs/retry.md",
    text: str = "The retry policy uses exponential backoff.",
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=0.8,
        text=text,
        meta={
            "repo": repo,
            "path": path,
            "title": "Retry policy",
            "line_start": 3,
            "line_end": 8,
        },
    )


def test_markers_preserve_order_and_repeats() -> None:
    assert markers("Claim [2], then [1] and [2].") == [2, 1, 2]


def test_claude_generator_drops_invalid_citations(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        assert command[0] == "claude"
        assert "The retry policy uses exponential backoff." in kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "structured_output": {
                        "answered": True,
                        "answer": "It uses backoff [1] and extra detail [3].",
                        "cited": [1, 3],
                    },
                    "total_cost_usd": 0.001,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("helpdesk.generate.subprocess.run", fake_run)
    answer = ClaudeGenerator().answer("How does retry work?", [hit("chunk-1")])

    assert answer.answered
    assert answer.citations[0].path == "docs/retry.md"
    assert answer.dropped_citations == [3]
    assert "outside the retrieved source range" in answer.warnings[0]
    assert answer.cost_usd == pytest.approx(0.001)


def test_claude_answer_without_valid_citation_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(
        "helpdesk.generate.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "structured_output": {
                        "answered": True,
                        "answer": "It retries three times.",
                        "cited": [],
                    }
                }
            ),
            stderr="",
        ),
    )
    answer = ClaudeGenerator().answer("How many retries?", [hit("chunk-1")])

    assert not answer.answered
    assert answer.answer == "I couldn't find that in the docs."
    assert answer.citations == []
    assert any("no valid source citations" in warning for warning in answer.warnings)


def test_claude_generator_surfaces_cli_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "helpdesk.generate.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="not authenticated",
        ),
    )
    with pytest.raises(GeneratorError, match="not authenticated"):
        ClaudeGenerator().answer("Question?", [hit("chunk-1")])


def test_extractive_generator_uses_top_span_and_threshold() -> None:
    scores = iter(
        [
            {"answer": "three retries", "score": 0.4},
            {"answer": "exponential backoff", "score": 0.8},
        ]
    )
    generator = ExtractiveGenerator(
        threshold=0.5,
        reader=lambda **_kwargs: next(scores),
    )
    answer = generator.answer("What is the retry strategy?", [hit("1"), hit("2")])

    assert answer.answered
    assert answer.answer == "exponential backoff [2]"
    assert answer.citations[0].path == "docs/retry.md"
    assert answer.score == pytest.approx(0.8)

    below_threshold = ExtractiveGenerator(
        threshold=0.5,
        reader=lambda **_kwargs: {"answer": "guess", "score": 0.2},
    ).answer("Unrelated?", [hit("1")])
    assert not below_threshold.answered


def test_empty_retrieval_does_not_call_external_generator(monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("external generator should not run without hits")

    monkeypatch.setattr("helpdesk.generate.subprocess.run", fail_if_called)
    answer = ClaudeGenerator().answer("Question?", [])

    assert not answer.answered
    assert answer.citations == []


def test_answer_summary_counts_expected_citations_per_question() -> None:
    records = [
        {
            "type": "concept",
            "expected": [{"repo": "a", "path": "same.md"}],
            "answer": {
                "answered": True,
                "citations": [{"repo": "a", "path": "same.md"}],
                "dropped_citations": [],
                "latency_seconds": 1.0,
                "cost_usd": 0.01,
            },
        },
        {
            "type": "concept",
            "expected": [{"repo": "a", "path": "same.md"}],
            "answer": {
                "answered": True,
                "citations": [],
                "dropped_citations": [],
                "latency_seconds": 3.0,
                "cost_usd": 0.02,
            },
        },
        {
            "type": "negative",
            "expected": [],
            "answer": {
                "answered": False,
                "citations": [],
                "dropped_citations": [],
                "latency_seconds": 0.5,
                "cost_usd": None,
            },
        },
    ]

    summary = summarise_answers(records)

    assert summary["answer_rate"] == 1.0
    assert summary["refusal_rate_on_negatives"] == 1.0
    assert summary["citation_precision"] == 1.0
    assert summary["citation_recall"] == 0.5
    assert summary["citation_validity"] == 1.0
    assert summary["cost_usd_total"] == pytest.approx(0.03)
    assert summary["mean_latency_seconds"] == pytest.approx(1.5)


def test_grading_sheet_includes_answer_citation_context_and_checklist(
    tmp_path: Path,
) -> None:
    payload = {
        "config": {"generator": "haiku", "generator_model": "claude-haiku-5-5"},
        "questions": [
            {
                "id": "q1",
                "type": "concept",
                "question": "How does retry work?",
                "expected": [{"repo": "option-pricer", "path": "docs/retry.md"}],
                "answer": {
                    "answered": True,
                    "answer": "It uses backoff [1].",
                    "citations": [
                        {
                            "index": 1,
                            "repo": "option-pricer",
                            "path": "docs/retry.md",
                            "line_start": 3,
                            "line_end": 8,
                        }
                    ],
                },
                "retrieved": [
                    {
                        "repo": "option-pricer",
                        "path": "docs/retry.md",
                        "line_start": 3,
                        "line_end": 8,
                        "text": "The retry policy uses exponential backoff.",
                    }
                ],
            }
        ],
    }
    sheet = render_sheet(payload, tmp_path / "run.json")

    assert "It uses backoff [1]." in sheet
    assert "The retry policy uses exponential backoff." in sheet
    assert "[ ] Supported" in sheet

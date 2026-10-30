"""Golden-set retrieval metrics and reproducible run persistence.

``rank`` is the first chunk from any expected file. ``answer_rank`` is
stricter: the rank by which every answer-chunk group in golden_chunks.json
(every part, for multi) has a matching chunk, i.e. did the chunk that holds
the answer come back, not just its file.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

KS = (1, 3, 5, 8, 20)
ANSWER_KS = (8, 20)


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    type: str
    expected: list[tuple[str, str]]
    note: str = ""
    expected_chunks: list[list[tuple[str, str, str]]] = field(default_factory=list)

    @property
    def repos(self) -> list[str]:
        return sorted({repo for repo, _ in self.expected})


@dataclass
class QuestionResult:
    id: str
    type: str
    repos: list[str]
    expected: list[tuple[str, str]]
    retrieved: list[dict[str, Any]]
    rank: int | None
    all_found_rank: int | None
    is_negative: bool = False
    expected_chunks: list[list[tuple[str, str, str]]] = field(default_factory=list)
    answer_rank: int | None = None

    @property
    def answer_labelled(self) -> bool:
        return bool(self.expected_chunks)

    @property
    def top_score(self) -> float | None:
        return float(self.retrieved[0]["score"]) if self.retrieved else None


def load_golden(
    path: str | Path,
    chunks_path: str | Path | None = None,
) -> list[GoldenQuestion]:
    golden_path = Path(path)
    payload = json.loads(golden_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Golden set must be a JSON list")
    chunk_file = (
        Path(chunks_path)
        if chunks_path is not None
        else golden_path.with_name("golden_chunks.json")
    )
    chunk_payload: dict[str, Any] = {}
    if chunk_file.is_file():
        loaded_chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        if not isinstance(loaded_chunks, dict):
            raise ValueError("Golden chunk set must be a JSON object")
        chunk_payload = loaded_chunks
    questions: list[GoldenQuestion] = []
    seen_ids: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Every golden question must be a JSON object")
        question_id = row.get("id")
        prompt = row.get("question")
        question_type = row.get("type")
        raw_expected = row.get("expected", [])
        if not all(isinstance(value, str) and value.strip() for value in (
            question_id,
            prompt,
            question_type,
        )):
            raise ValueError("Golden questions need non-empty id, question, and type")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate golden question id: {question_id}")
        if not isinstance(raw_expected, list):
            raise ValueError(f"Expected files for {question_id!r} must be a list")
        expected: list[tuple[str, str]] = []
        for item in raw_expected:
            if not isinstance(item, dict) or not isinstance(item.get("repo"), str) or not isinstance(item.get("path"), str):
                raise ValueError(f"Expected entries for {question_id!r} need repo and path")
            expected.append((item["repo"], item["path"].replace("\\", "/")))
        if question_type == "negative" and expected:
            raise ValueError(f"Negative question {question_id!r} must not have expected files")
        if question_type != "negative" and not expected:
            raise ValueError(f"Answerable question {question_id!r} needs expected files")
        raw_groups = chunk_payload.get(question_id, [])
        if question_type == "negative" and raw_groups:
            raise ValueError(f"Negative question {question_id!r} cannot have expected chunks")
        if not isinstance(raw_groups, list):
            raise ValueError(f"Expected chunk groups for {question_id!r} must be a list")
        expected_chunks: list[list[tuple[str, str, str]]] = []
        for group in raw_groups:
            if not isinstance(group, list) or not group:
                raise ValueError(f"Every expected chunk group for {question_id!r} must be non-empty")
            targets: list[tuple[str, str, str]] = []
            for target in group:
                if (
                    not isinstance(target, dict)
                    or not isinstance(target.get("repo"), str)
                    or not isinstance(target.get("path"), str)
                    or not isinstance(target.get("title"), str)
                    or not target["repo"].strip()
                    or not target["path"].strip()
                    or not target["title"].strip()
                ):
                    raise ValueError(
                        f"Expected chunk targets for {question_id!r} need repo, path, and title"
                    )
                targets.append(
                    (
                        target["repo"],
                        _normal_path(target["path"]),
                        target["title"].casefold(),
                    )
                )
            expected_chunks.append(targets)
        seen_ids.add(question_id)
        questions.append(
            GoldenQuestion(
                id=question_id,
                question=prompt,
                type=question_type,
                expected=expected,
                note=str(row.get("note", "")),
                expected_chunks=expected_chunks,
            )
        )
    unknown_chunk_questions = set(chunk_payload) - seen_ids
    if unknown_chunk_questions:
        raise ValueError(
            "Golden chunk set contains unknown question IDs: "
            + ", ".join(sorted(unknown_chunk_questions))
        )
    return questions


def first_relevant_rank(
    retrieved: list[dict[str, Any]], expected: Iterable[tuple[str, str]]
) -> int | None:
    wanted = set(expected)
    if not wanted:
        return None
    for rank, item in enumerate(retrieved, start=1):
        if (item.get("repo"), _normal_path(item.get("path", ""))) in wanted:
            return rank
    return None


def all_found_rank(
    retrieved: list[dict[str, Any]], expected: Iterable[tuple[str, str]]
) -> int | None:
    wanted = set(expected)
    if not wanted:
        return None
    found: set[tuple[str, str]] = set()
    for rank, item in enumerate(retrieved, start=1):
        pair = (item.get("repo"), _normal_path(item.get("path", "")))
        if pair in wanted:
            found.add(pair)
            if found == wanted:
                return rank
    return None


def answer_rank(
    retrieved: list[dict[str, Any]],
    groups: list[list[tuple[str, str, str]]],
) -> int | None:
    """Return the rank where every required title group has a matching chunk."""
    if not groups:
        return None
    found = [False] * len(groups)
    for rank, item in enumerate(retrieved, start=1):
        repo = item.get("repo")
        path = _normal_path(item.get("path", ""))
        title = str(item.get("title", "")).casefold()
        for index, alternatives in enumerate(groups):
            if found[index]:
                continue
            found[index] = any(
                repo == expected_repo
                and path == expected_path
                and expected_title in title
                for expected_repo, expected_path, expected_title in alternatives
            )
        if all(found):
            return rank
    return None


def _normal_path(path: Any) -> str:
    return str(path).replace("\\", "/")


def _hit_dict(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        meta = hit.get("meta", hit.get("metadata", {}))
        if not meta:
            meta = hit
        score = hit.get("score", 0.0)
        chunk_id = hit.get("chunk_id", hit.get("id", ""))
    else:
        meta = getattr(hit, "meta", {})
        score = getattr(hit, "score", 0.0)
        chunk_id = getattr(hit, "chunk_id", getattr(hit, "id", ""))
    return {
        "id": str(chunk_id),
        "repo": str(meta.get("repo", "")),
        "path": _normal_path(meta.get("path", "")),
        "title": str(meta.get("title", "")),
        "score": round(float(score), 6),
    }


def run(
    questions: list[GoldenQuestion],
    retrieve: Callable[[str, int], list[Any]],
    depth: int = 20,
) -> list[QuestionResult]:
    if depth < 1:
        raise ValueError("depth must be at least 1")
    results: list[QuestionResult] = []
    for question in questions:
        hits = [_hit_dict(hit) for hit in retrieve(question.question, depth)]
        is_negative = question.type == "negative"
        results.append(
            QuestionResult(
                id=question.id,
                type=question.type,
                repos=question.repos,
                expected=question.expected,
                retrieved=hits,
                rank=None if is_negative else first_relevant_rank(hits, question.expected),
                all_found_rank=(
                    None if is_negative else all_found_rank(hits, question.expected)
                ),
                is_negative=is_negative,
                expected_chunks=question.expected_chunks,
                answer_rank=(
                    None if is_negative else answer_rank(hits, question.expected_chunks)
                ),
            )
        )
    return results


def recall_at_k(results: list[QuestionResult], k: int) -> float | None:
    answerable = [result for result in results if not result.is_negative]
    if not answerable:
        return None
    return sum(result.rank is not None and result.rank <= k for result in answerable) / len(answerable)


def mrr(results: list[QuestionResult]) -> float | None:
    answerable = [result for result in results if not result.is_negative]
    if not answerable:
        return None
    return sum(1 / result.rank if result.rank else 0 for result in answerable) / len(answerable)


def _score_row(results: list[QuestionResult]) -> dict[str, float | int | None]:
    row: dict[str, float | int | None] = {
        f"recall@{k}": recall_at_k(results, k) for k in KS
    }
    answerable = [result for result in results if not result.is_negative]
    row["n"] = len(answerable)
    row["mrr"] = mrr(results)
    # the chunk that holds the answer (all parts, for multi), not just its file
    labelled = [result for result in answerable if result.answer_labelled]
    for k in ANSWER_KS:
        row[f"answer_recall@{k}"] = (
            sum(result.answer_rank is not None and result.answer_rank <= k for result in labelled)
            / len(labelled)
            if labelled
            else None
        )
    row["n_answer_labelled"] = len(labelled)
    return row


def _spread(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def summarise(results: list[QuestionResult]) -> dict[str, Any]:
    by_type: dict[str, list[QuestionResult]] = defaultdict(list)
    by_repo: dict[str, list[QuestionResult]] = defaultdict(list)
    multi = [result for result in results if result.type == "multi"]
    for result in results:
        by_type[result.type].append(result)
        for repo in result.repos:
            by_repo[repo].append(result)

    multi_all_found = {
        str(k): (
            sum(
                result.all_found_rank is not None and result.all_found_rank <= k
                for result in multi
            )
            / len(multi)
            if multi
            else None
        )
        for k in KS
    }
    negative_scores = [
        result.top_score for result in results if result.is_negative and result.top_score is not None
    ]
    answerable_scores = [
        result.top_score for result in results if not result.is_negative and result.top_score is not None
    ]
    return {
        "overall": _score_row(results),
        "by_type": {name: _score_row(group) for name, group in sorted(by_type.items())},
        "by_repo": {name: _score_row(group) for name, group in sorted(by_repo.items())},
        "multi_all_found": multi_all_found,
        "top1_score": {
            "negative": _spread(negative_scores),
            "answerable_hit": _spread(answerable_scores),
        },
    }


def save_run(
    results: list[QuestionResult],
    config: dict[str, Any],
    out_dir: str | Path,
) -> Path:
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(config.get("name", "run"))
    ).strip("-") or "run"
    path = directory / f"{timestamp}-{safe_name}.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "config": config,
                "summary": summarise(results),
                "questions": [
                    {
                        **asdict(result),
                        "answer_labelled": result.answer_labelled,
                        "expected": [
                            {"repo": repo, "path": expected_path}
                            for repo, expected_path in result.expected
                        ],
                        "expected_chunks": [
                            [
                                {"repo": repo, "path": path, "title": title}
                                for repo, path, title in group
                            ]
                            for group in result.expected_chunks
                        ],
                    }
                    for result in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path

"""Answer-chunk recall on questions whose answer sits in a section the chunker
split into several pieces, vs the rest: does a context prefix (step 9) help a
later piece that has lost its heading text?

    python -m scripts.part_k bge-small+hybrid bge-small@header+hybrid

A split section is a (repo, path, title) with more than one chunk in the
index; its pieces share the title and are numbered by line order. A question
is "split" when every answer-chunk group (data/golden_chunks.json) matches
only split sections. For those it also prints the best rank of a matching
later piece (part >= 2), which carries no heading text of its own.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from helpdesk.embedder import DEFAULT_MODEL
from helpdesk.evaluate import GoldenQuestion, load_golden
from helpdesk.index import DEFAULT_PERSIST_DIR, get_collection
from scripts.results_table import latest_runs

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden.json"
RESULTS = ROOT / "result"

Section = tuple[str, str, str]  # repo, path, title (as stored)


def section_parts(metadatas: dict[str, dict[str, Any]]) -> dict[str, tuple[int, int]]:
    """Map chunk ID to (part number, parts in its section), by line order."""
    sections: dict[Section, list[tuple[int, str]]] = defaultdict(list)
    for chunk_id, meta in metadatas.items():
        key = (str(meta.get("repo")), str(meta.get("path")), str(meta.get("title")))
        sections[key].append((int(meta.get("line_start", 0)), chunk_id))
    parts: dict[str, tuple[int, int]] = {}
    for pieces in sections.values():
        for number, (_, chunk_id) in enumerate(sorted(pieces), start=1):
            parts[chunk_id] = (number, len(pieces))
    return parts


def matches(repo: str, path: str, title: str, group: list[tuple[str, str, str]]) -> bool:
    title = title.casefold()
    return any(repo == r and path == p and t in title for r, p, t in group)


def split_questions(
    golden: list[GoldenQuestion],
    metadatas: dict[str, dict[str, Any]],
    parts: dict[str, tuple[int, int]],
) -> list[str]:
    out = []
    for question in golden:
        if not question.expected_chunks:
            continue
        if all(
            (hits := [
                chunk_id
                for chunk_id, meta in metadatas.items()
                if matches(str(meta.get("repo")), str(meta.get("path")), str(meta.get("title")), group)
            ])
            and all(parts[chunk_id][1] >= 2 for chunk_id in hits)
            for group in question.expected_chunks
        ):
            out.append(question.id)
    return out


def best_later_part(
    row: dict[str, Any],
    groups: list[list[tuple[str, str, str]]],
    parts: dict[str, tuple[int, int]],
) -> int | None:
    for rank, hit in enumerate(row.get("retrieved", []), start=1):
        if parts.get(hit.get("id"), (1, 1))[0] >= 2 and any(
            matches(hit.get("repo", ""), hit.get("path", ""), hit.get("title", ""), group)
            for group in groups
        ):
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run names, e.g. bge-small+hybrid")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()

    payload = get_collection(DEFAULT_PERSIST_DIR, DEFAULT_MODEL).get(include=["metadatas"])
    metadatas = dict(zip(payload["ids"], payload["metadatas"]))
    parts = section_parts(metadatas)
    golden = load_golden(GOLDEN)
    groups = {question.id: question.expected_chunks for question in golden}
    split = split_questions(golden, metadatas, parts)
    rest = [question.id for question in golden if question.expected_chunks and question.id not in split]
    print(f"split-section questions ({len(split)}): {' '.join(split) or '-'}")
    if not split:
        print("no labelled answer sits in a split section; nothing to compare")

    runs = latest_runs(args.results_dir)
    for name in args.runs:
        if name not in runs:
            parser.error(f"no saved run named {name!r}")
        rows = {row["id"]: row for row in runs[name]["questions"]}

        def recall(ids: list[str], k: int) -> str:
            ids = [i for i in ids if i in rows]
            if not ids:
                return "  - "
            hit = sum(
                isinstance(rows[i].get("answer_rank"), int) and rows[i]["answer_rank"] <= k
                for i in ids
            )
            return f"{hit / len(ids):.2f}"

        print(
            f"{name:32} split ansR@8 {recall(split, 8)} @20 {recall(split, 20)} | "
            f"others ansR@8 {recall(rest, 8)} @20 {recall(rest, 20)}"
        )
        if split:
            print(
                f"{'':32} answer rank: "
                + " ".join(f"{i}:{rows[i].get('answer_rank') or '-'}" for i in split if i in rows)
            )
            print(
                f"{'':32} part>=2 rank: "
                + " ".join(
                    f"{i}:{best_later_part(rows[i], groups[i], parts) or '-'}"
                    for i in split
                    if i in rows
                )
            )


if __name__ == "__main__":
    main()

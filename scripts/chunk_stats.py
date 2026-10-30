"""Chunk the configured repositories and report token and fixture statistics.

Run from the project root with ``python -m scripts.chunk_stats``.
"""

from __future__ import annotations

import statistics
import time
from collections import Counter
from math import ceil

from helpdesk.chunker import DEFAULT_MAX_TOKENS, chunk_documents
from helpdesk.loader import load_documents
from helpdesk.sources import REPOS

MODEL = "BAAI/bge-small-en-v1.5"


def _percentile_95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[max(0, ceil(0.95 * len(ordered)) - 1)]


def _describe(label: str, sizes: list[int]) -> None:
    if not sizes:
        print(f"{label}: n=0")
        return
    print(
        f"{label}: n={len(sizes):4} min={min(sizes):4} "
        f"median={statistics.median(sizes):6.1f} "
        f"p95={_percentile_95(sizes):4} max={max(sizes):4}"
    )


def main() -> None:
    documents = load_documents(REPOS)
    if not documents:
        print("No source documents found; check the paths in helpdesk/sources.py.")
        return

    from sentence_transformers import SentenceTransformer
    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    model = SentenceTransformer(MODEL)
    tokenizer = model.tokenizer

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    started = time.perf_counter()
    chunks = chunk_documents(documents, DEFAULT_MAX_TOKENS, count_tokens)
    elapsed = time.perf_counter() - started

    counts = Counter((item.repo, item.kind) for item in chunks)
    print(
        f"model: {MODEL} | max sequence length: {model.max_seq_length} | "
        f"chunk budget: {DEFAULT_MAX_TOKENS}"
    )
    print(f"{'repo':32} {'doc':>6} {'test':>6} {'total':>6}")
    for repo in REPOS:
        docs = counts[(repo, "doc")]
        tests = counts[(repo, "test")]
        print(f"{repo:32} {docs:6} {tests:6} {docs + tests:6}")
    print(
        f"{'TOTAL':32} {sum(counts[(r, 'doc')] for r in REPOS):6} "
        f"{sum(counts[(r, 'test')] for r in REPOS):6} {len(chunks):6}"
    )

    _describe("all tokens/chunk", [item.n_tokens for item in chunks])
    for kind in ("doc", "test"):
        _describe(
            f"{kind} tokens/chunk",
            [item.n_tokens for item in chunks if item.kind == kind],
        )

    over = [item for item in chunks if item.n_tokens > DEFAULT_MAX_TOKENS]
    print(f"chunks over {DEFAULT_MAX_TOKENS} tokens: {len(over)}")
    for item in over:
        print(
            f"  {item.repo}/{item.path}:{item.line_start} "
            f"{item.title} ({item.n_tokens} tokens)"
        )

    small = [item for item in chunks if item.n_tokens < 20]
    split_counts: Counter[tuple[str, str, str]] = Counter(
        (item.repo, item.path, item.title) for item in chunks
    )
    split = [
        item
        for item in chunks
        if split_counts[(item.repo, item.path, item.title)] > 1
    ]
    fixture_chunks = [item for item in chunks if item.kind == "test" and item.fixtures]
    resolved_pointers = sum(
        "@" in fixture
        for item in fixture_chunks
        for fixture in item.fixtures
    )
    print(f"chunks under 20 tokens: {len(small)}")
    print(f"chunks that are part of a split: {len(split)}")
    print(
        f"test chunks naming fixtures: {len(fixture_chunks)} "
        f"({resolved_pointers} pointers resolved to a file)"
    )
    print(f"chunking time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from helpdesk.chunker import Chunk
from helpdesk.index import (
    IndexModelMismatch,
    build_index,
    collection_name,
    search,
)


class FakeEmbedder:
    """Deterministic bag-of-words vectors; no model download required."""

    def __init__(self, model_name: str = "fake/bow-64", dim: int = 64) -> None:
        self.model_name = model_name
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._vector(query)

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dim] += 1
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


def make_chunk(
    chunk_id: str,
    repo: str,
    text: str,
    *,
    path: str = "docs/guide.md",
    kind: str = "doc",
    title: str | None = None,
    line_start: int = 1,
    line_end: int = 1,
    fixtures: tuple[str, ...] = (),
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        repo=repo,
        path=path,
        kind=kind,
        title=title or chunk_id,
        line_start=line_start,
        line_end=line_end,
        last_commit="2026-10-01",
        n_tokens=len(text.split()),
        fixtures=fixtures,
    )


CHUNKS = [
    make_chunk(
        "id1",
        "unity-platform",
        "config engine heartbeat over websocket every thirty seconds",
        path="docs/config.md",
        title="Config engine",
    ),
    make_chunk(
        "id2",
        "option-pricer",
        "Mongo persistence handler writes snapshots and recovers on restart",
        path="docs/persistence.md",
        title="Mongo persistence",
    ),
    make_chunk(
        "id3",
        "option-pricer",
        "Retry policy makes three attempts with exponential backoff",
        path="tests/test_retry.py",
        kind="test",
        title="test_retry",
    ),
    make_chunk(
        "id4",
        "prediction-markets",
        "Redis cache stores short-lived keys for fast reads",
        path="docs/cache.md",
        title="Redis cache",
    ),
    make_chunk(
        "id5",
        "unity-platform",
        "Black Scholes call spread pricing for binary options",
        path="tests/test_pricing.py",
        kind="test",
        title="test_pricing",
    ),
]


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    build_index(CHUNKS, FakeEmbedder(), tmp_path)
    return tmp_path


def test_build_counts_chunks_and_is_idempotent(tmp_path: Path) -> None:
    embedder = FakeEmbedder()

    assert build_index(CHUNKS, embedder, tmp_path) == len(CHUNKS)
    assert build_index(CHUNKS, embedder, tmp_path) == len(CHUNKS)


def test_search_returns_expected_query_near_the_top(index_dir: Path) -> None:
    hits = search("config engine heartbeat", FakeEmbedder(), index_dir, k=3)

    assert hits
    assert hits[0].chunk_id == "id1"
    assert hits[0].score > 0


def test_rebuild_drops_chunks_that_disappeared(index_dir: Path) -> None:
    assert build_index(CHUNKS[:4], FakeEmbedder(), index_dir) == 4
    hits = search(CHUNKS[4].text, FakeEmbedder(), index_dir, k=8)

    assert len(hits) == 4
    assert "id5" not in [hit.chunk_id for hit in hits]


def test_where_filter_returns_only_that_repository(index_dir: Path) -> None:
    hits = search(
        "pricing",
        FakeEmbedder(),
        index_dir,
        k=8,
        where={"repo": "option-pricer"},
    )

    assert hits
    assert all(hit.meta["repo"] == "option-pricer" for hit in hits)


def test_multiple_where_fields_are_anded(index_dir: Path) -> None:
    hits = search(
        "retry",
        FakeEmbedder(),
        index_dir,
        k=8,
        where={"repo": "option-pricer", "kind": "doc"},
    )

    assert all(
        hit.meta["repo"] == "option-pricer" and hit.meta["kind"] == "doc"
        for hit in hits
    )
    assert "id3" not in [hit.chunk_id for hit in hits]


def test_exact_query_ranks_its_chunk_first(index_dir: Path) -> None:
    for chunk in CHUNKS:
        hits = search(chunk.text, FakeEmbedder(), index_dir, k=3)
        assert hits[0].chunk_id == chunk.id
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_collection_names_are_safe_and_normalized() -> None:
    assert collection_name("fake/bow") == collection_name("fake-bow")
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,61}[a-z0-9]", collection_name("fake/bow"))


def test_search_rejects_an_index_built_with_another_model(index_dir: Path) -> None:
    with pytest.raises(IndexModelMismatch, match="model"):
        search("heartbeat", FakeEmbedder("fake/other-model"), index_dir)


def test_metadata_is_flat_and_round_trips(tmp_path: Path) -> None:
    chunk = make_chunk(
        "fixture-chunk",
        "prediction-markets",
        "test uses the pricer fixture",
        path="tests/test_pricer.py",
        kind="test",
        title="test_pricer",
        line_start=19,
        line_end=22,
        fixtures=("pricer@tests/conftest.py", "tmp_path"),
    )
    build_index([chunk], FakeEmbedder(), tmp_path)

    hit = search(chunk.text, FakeEmbedder(), tmp_path, k=1)[0]

    assert hit.meta["fixtures"] == "pricer@tests/conftest.py, tmp_path"
    assert hit.meta["line_start"] == 19
    assert hit.meta["line_end"] == 22
    assert hit.meta["title"] == "test_pricer"

from __future__ import annotations

import pytest

from helpdesk.bm25 import BM25Index, tokenize
from helpdesk.hybrid import hybrid_search, rrf
from helpdesk.index import SearchHit


def hit(chunk_id: str, score: float = 1.0, text: str | None = None) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=score,
        text=text or chunk_id,
        meta={"repo": "repo", "path": f"{chunk_id}.md", "kind": "doc"},
    )


def test_tokenizer_keeps_identifiers_and_splits_components() -> None:
    tokens = tokenize("currentSource ITEM_DOWN FC.U")

    assert "currentsource" in tokens
    assert "current" in tokens
    assert "source" in tokens
    assert "item_down" in tokens
    assert "item" in tokens
    assert "down" in tokens
    assert "fc" in tokens
    assert "u" in tokens


def test_bm25_ranks_identifiers_and_applies_metadata_filters() -> None:
    index = BM25Index(
        ids=["one", "two", "three"],
        texts=[
            "ITEM_DOWN source fails over to another provider",
            "retry policy uses exponential backoff",
            "ITEM_DOWN is tested in the integration suite",
        ],
        metadatas=[
            {"repo": "alpha", "kind": "doc"},
            {"repo": "alpha", "kind": "test"},
            {"repo": "beta", "kind": "test"},
        ],
    )

    results = index.search("ITEM_DOWN", k=5, where={"repo": "alpha", "kind": "doc"})

    assert [result.chunk_id for result in results] == ["one"]
    assert results[0].score > 0


def test_bm25_returns_empty_for_empty_index_and_rejects_bad_inputs() -> None:
    index = BM25Index(ids=[], texts=[], metadatas=[])

    assert index.search("question") == []
    with pytest.raises(ValueError, match="query"):
        index.search(" ")
    with pytest.raises(ValueError, match="at least 1"):
        index.search("question", k=0)


def test_rrf_rewards_presence_in_both_rankings_and_is_deterministic() -> None:
    fused = rrf(
        [
            [hit("shared"), hit("vector-only")],
            [hit("shared"), hit("bm25-only")],
        ],
        k=60,
    )

    assert [item.chunk_id for item in fused] == [
        "shared",
        "bm25-only",
        "vector-only",
    ]
    assert fused[0].score == pytest.approx(2 / 61)
    assert fused[0].meta["rank_0"] == fused[0].meta["rank_1"] == 1
    with pytest.raises(ValueError, match="at least 1"):
        rrf([], k=0)


def test_hybrid_search_keyword_query_and_query_vector_override_each_side(monkeypatch) -> None:
    import helpdesk.hybrid as module

    seen = {}

    def fake_search(query, *_args, query_vector=None, **_kwargs):
        seen["vector"] = (query, query_vector)
        return [hit("vector-top")]

    class FakeBM25:
        def search(self, query, **_kwargs):
            seen["bm25"] = query
            return [hit("bm25-top")]

    monkeypatch.setattr(module, "search", fake_search)
    hybrid_search(
        "A hypothetical passage.",
        embedder=object(),
        bm25_index=FakeBM25(),
        k=2,
        candidates=2,
        query_vector=[0.1, 0.2],
        keyword_query="original question",
    )

    assert seen == {
        "vector": ("A hypothetical passage.", [0.1, 0.2]),
        "bm25": "original question",
    }


def test_hybrid_search_fuses_candidates_and_validates_depth() -> None:
    vector_hits = [hit("vector-top"), hit("shared")]
    bm25_hits = [hit("shared"), hit("bm25-top")]

    fused = hybrid_search(
        "query",
        embedder=object(),
        bm25_index=BM25Index([], [], []),
        k=2,
        candidates=3,
        rrf_k=60,
        vector_hits=vector_hits,
        bm25_hits=bm25_hits,
    )

    assert [item.chunk_id for item in fused] == ["shared", "vector-top"]
    with pytest.raises(ValueError, match="at least k"):
        hybrid_search(
            "query",
            embedder=object(),
            bm25_index=BM25Index([], [], []),
            k=3,
            candidates=2,
        )

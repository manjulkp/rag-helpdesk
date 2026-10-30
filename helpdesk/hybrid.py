"""Vector/BM25 hybrid retrieval using reciprocal-rank fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from helpdesk.bm25 import BM25Index
from helpdesk.index import DEFAULT_PERSIST_DIR, IndexModelMismatch, SearchHit, search

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATES = 50


def rrf(
    rankings: Sequence[Sequence[SearchHit]],
    k: int = DEFAULT_RRF_K,
) -> list[SearchHit]:
    """Fuse ranked hit lists using reciprocal-rank fusion.

    Chunks appearing in multiple rankings get a higher score. Ties are ordered
    by chunk ID for deterministic results.
    """
    if k < 1:
        raise ValueError("rrf k must be at least 1")
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    details: dict[str, dict[str, float | int]] = {}
    for ranking_name, ranking in enumerate(rankings):
        for rank, hit in enumerate(ranking, start=1):
            chunk_id = hit.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            hits.setdefault(chunk_id, hit)
            details.setdefault(chunk_id, {})[f"rank_{ranking_name}"] = rank
            details[chunk_id][f"score_{ranking_name}"] = float(hit.score)

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    fused: list[SearchHit] = []
    for chunk_id in ordered_ids:
        original = hits[chunk_id]
        metadata = {
            **original.meta,
            "rrf_score": scores[chunk_id],
            **details[chunk_id],
        }
        fused.append(
            SearchHit(
                chunk_id=chunk_id,
                score=scores[chunk_id],
                text=original.text,
                meta=metadata,
            )
        )
    return fused


def hybrid_search(
    query: str,
    embedder: Any,
    bm25_index: BM25Index,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    k: int = 8,
    candidates: int = DEFAULT_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
    where: dict[str, Any] | None = None,
    vector_hits: list[SearchHit] | None = None,
    bm25_hits: list[SearchHit] | None = None,
    reranker: Any | None = None,
    query_vector: Any | None = None,
    keyword_query: str | None = None,
    variant: str = "plain",
) -> list[SearchHit]:
    """Retrieve vector and BM25 candidates, fuse, optionally rerank, and return top-k.

    ``query_vector`` / ``keyword_query`` override what the vector and BM25 sides
    search with (HyDE: a passage vector, and the original question for BM25).
    ``variant`` picks the index variant; the BM25 index must be the same one.
    """
    bm25_variant = getattr(bm25_index, "variant", "plain")
    if bm25_variant != variant:
        # otherwise a header BM25 would silently fuse with a plain vector search
        raise IndexModelMismatch(
            f"BM25 index is the {bm25_variant!r} variant; hybrid search requested {variant!r}"
        )
    if not query.strip():
        raise ValueError("query must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")
    if candidates < 1:
        raise ValueError("candidates must be at least 1")
    if candidates < k:
        raise ValueError("candidates must be at least k")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    if vector_hits is None:
        vector_hits = search(
            query,
            embedder,
            persist_dir,
            k=candidates,
            where=where,
            query_vector=query_vector,
            variant=variant,
        )
    if bm25_hits is None:
        bm25_hits = bm25_index.search(keyword_query or query, k=candidates, where=where)
    fused = rrf([vector_hits, bm25_hits], k=rrf_k)
    if reranker is not None:
        fused = reranker.rerank(query, fused)
    return fused[:k]


def retrieve(
    query: str,
    embedder: Any,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    k: int = 8,
    where: dict[str, Any] | None = None,
    retriever: str = "vector",
    candidates: int = DEFAULT_CANDIDATES,
    rrf_k: int = DEFAULT_RRF_K,
    bm25_index: BM25Index | None = None,
    reranker: Any | None = None,
    variant: str = "plain",
) -> list[SearchHit]:
    """Dispatch to vector, BM25, or hybrid retrieval."""
    if retriever == "vector":
        return search(query, embedder, persist_dir, k=k, where=where, variant=variant)
    if retriever in {"bm25", "hybrid"}:
        index = bm25_index or BM25Index.from_chroma(
            persist_dir,
            str(embedder.model_name),
            variant=variant,
        )
        if retriever == "bm25":
            return index.search(query, k=k, where=where)
        return hybrid_search(
            query,
            embedder,
            index,
            persist_dir,
            k=k,
            candidates=candidates,
            rrf_k=rrf_k,
            where=where,
            reranker=reranker,
            variant=variant,
        )
    raise ValueError(f"Unknown retriever {retriever!r}; expected vector, bm25, or hybrid")

"""Search the local helpdesk index from the command line.

    python -m scripts.search "how does the config engine heartbeat work" -k 8 --repo X --kind doc
    python -m scripts.search "..." --no-instruction       # embed the query without bge's prefix
    python -m scripts.search "..." --retriever hybrid
    python -m scripts.search "..." --transform rewrite    # prints the transformed queries first
    python -m scripts.search "..." --variant header       # search the header index
"""

from __future__ import annotations

import argparse

from transformers.utils import logging as transformers_logging

from helpdesk.bm25 import BM25Index
from helpdesk.embedder import DEFAULT_MODEL, Embedder
from helpdesk.hybrid import DEFAULT_CANDIDATES, DEFAULT_RRF_K
from helpdesk.index import DEFAULT_PERSIST_DIR, VARIANTS
from helpdesk.query_transform import (
    DEFAULT_TRANSFORM_MODEL,
    TRANSFORMERS,
    TransformCache,
    make_transformer,
    retrieve_transformed,
)
from helpdesk.rerank import make_reranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=8)
    parser.add_argument("--variant", choices=VARIANTS, default="plain", help="index variant")
    parser.add_argument("--repo")
    parser.add_argument("--kind", choices=("doc", "test"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    parser.add_argument(
        "--retriever",
        choices=("vector", "bm25", "hybrid"),
        default="vector",
    )
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    parser.add_argument("--transform", choices=tuple(TRANSFORMERS), default="noop")
    parser.add_argument("--transform-model", default=DEFAULT_TRANSFORM_MODEL)
    parser.add_argument("--transform-cache")
    parser.add_argument(
        "--reranker",
        choices=("none", "cross-encoder", "claude", "gated-claude"),
        default="none",
    )
    parser.add_argument("--rerank-cutoff", type=float, default=0.1)
    parser.add_argument(
        "--no-instruction",
        action="store_true",
        help="don't prepend the model's query instruction",
    )
    args = parser.parse_args()
    if args.k < 1 or args.candidates < 1 or args.rrf_k < 1:
        parser.error("-k, --candidates, and --rrf-k must be at least 1")
    if args.retriever == "hybrid" and args.candidates < args.k:
        parser.error("--candidates must be at least -k for hybrid retrieval")

    filters = {
        key: value
        for key, value in (("repo", args.repo), ("kind", args.kind))
        if value is not None
    }
    transformers_logging.set_verbosity_error()
    embedder = Embedder(
        args.model,
        query_instruction="" if args.no_instruction else None,
    )
    transformer = make_transformer(
        args.transform,
        cache=TransformCache(args.transform_cache) if args.transform_cache else None,
        model=args.transform_model,
    )
    reranker = make_reranker(args.reranker, cutoff=args.rerank_cutoff)
    bm25_index = (
        BM25Index.from_chroma(args.persist_dir, args.model, variant=args.variant)
        if args.retriever in {"bm25", "hybrid"}
        else None
    )
    if args.transform != "noop":
        # cached, so retrieve_transformed below reuses these without another LLM call
        for number, query in enumerate(transformer.transform(args.query), start=1):
            print(f"query {number} ({query.kind}): {query.text}")
        print()
    hits = retrieve_transformed(
        args.query,
        transformer,
        embedder,
        args.persist_dir,
        k=args.k,
        where=filters or None,
        retriever=args.retriever,
        candidates=args.candidates,
        rrf_k=args.rrf_k,
        bm25_index=bm25_index,
        reranker=reranker,
        variant=args.variant,
    )
    for rank, hit in enumerate(hits, start=1):
        meta = hit.meta
        print(
            f"{rank}. {hit.score:.3f} "
            f"[{meta['repo']}] {meta['path']}:{meta['line_start']}-{meta['line_end']} "
            f"{meta['title']}"
        )
        print(hit.text)
        print()


if __name__ == "__main__":
    main()

"""Retrieve top-k chunks and answer one question with citations."""

from __future__ import annotations

import argparse
import logging

from transformers.utils import logging as transformers_logging

from helpdesk.bm25 import BM25Index
from helpdesk.embedder import DEFAULT_MODEL, Embedder
from helpdesk.generate import ClaudeGenerator, ExtractiveGenerator, Generator
from helpdesk.hybrid import DEFAULT_CANDIDATES, DEFAULT_RRF_K
from helpdesk.index import DEFAULT_PERSIST_DIR
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
    parser.add_argument("question")
    parser.add_argument(
        "--generator",
        choices=("haiku", "sonnet", "extractive"),
        default="haiku",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model")
    parser.add_argument("--generator-model", help="override the answer model")
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    parser.add_argument("-k", type=int, default=8)
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
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--show-sources", action="store_true")
    args = parser.parse_args()
    if args.k < 1 or args.candidates < 1 or args.rrf_k < 1:
        parser.error("-k, --candidates, and --rrf-k must be at least 1")
    if args.retriever == "hybrid" and args.candidates < args.k:
        parser.error("--candidates must be at least -k for hybrid retrieval")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    transformers_logging.set_verbosity_error()
    embedder = Embedder(args.model)
    transformer = make_transformer(
        args.transform,
        cache=TransformCache(args.transform_cache) if args.transform_cache else None,
        model=args.transform_model,
    )
    reranker = make_reranker(args.reranker, cutoff=args.rerank_cutoff)
    bm25_index = (
        BM25Index.from_chroma(args.persist_dir, args.model)
        if args.retriever in {"bm25", "hybrid"}
        else None
    )
    hits = retrieve_transformed(
        args.question,
        transformer,
        embedder,
        args.persist_dir,
        k=args.k,
        retriever=args.retriever,
        candidates=args.candidates,
        rrf_k=args.rrf_k,
        bm25_index=bm25_index,
        reranker=reranker,
    )
    generators: dict[str, Generator] = {
        "haiku": ClaudeGenerator(
            args.generator_model or "claude-haiku-5-5",
            name="haiku",
        ),
        "sonnet": ClaudeGenerator(
            args.generator_model or "claude-sonnet-5-5",
            name="sonnet",
        ),
        "extractive": ExtractiveGenerator(threshold=args.threshold),
    }
    answer = generators[args.generator].answer(args.question, hits)
    print(answer.answer)
    for citation in answer.citations:
        print(f"[{citation.index}] {citation.repo}/{citation.path}:{citation.line_start}-{citation.line_end}")
    if answer.warnings:
        for warning in answer.warnings:
            print(f"warning: {warning}")
    if answer.cost_usd is not None:
        print(f"cost_usd: {answer.cost_usd:.6f}")
    if args.show_sources:
        print("\nRetrieved sources:")
        for index, hit in enumerate(hits, start=1):
            print(
                f"[{index}] {hit.meta.get('repo')}/{hit.meta.get('path')} "
                f"{hit.meta.get('title', '')} score={hit.score:.3f}"
            )
            print(hit.text)


if __name__ == "__main__":
    main()

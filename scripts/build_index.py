"""Load, chunk, embed, and index the configured repositories.

    python -m scripts.build_index                      # plain
    python -m scripts.build_index --variant header     # "repo / path / heading" + chunk, free
    python -m scripts.build_index --variant llm        # Claude-written context + chunk (costs)

The llm variant caches contexts in data/context_cache.jsonl, so a rerun or a
restart after a crash only calls Claude for chunks without a cached context.
"""

from __future__ import annotations

import argparse
import time

from helpdesk.chunker import DEFAULT_MAX_TOKENS, chunk_documents
from helpdesk.contextualize import (
    DEFAULT_CONTEXT_CACHE,
    DEFAULT_CONTEXT_MODEL,
    ContextCache,
    LLMContextualizer,
    contexts_for,
)
from helpdesk.embedder import DEFAULT_MODEL, Embedder
from helpdesk.index import DEFAULT_PERSIST_DIR, VARIANTS, build_index, collection_name
from helpdesk.loader import load_documents
from helpdesk.sources import REPOS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="defaults to CUDA when available, otherwise CPU",
    )
    parser.add_argument("--persist-dir", type=str, default=str(DEFAULT_PERSIST_DIR))
    parser.add_argument("--variant", choices=VARIANTS, default="plain")
    parser.add_argument("--workers", type=int, default=4, help="parallel LLM calls (llm)")
    parser.add_argument("--context-model", default=DEFAULT_CONTEXT_MODEL)
    parser.add_argument("--context-cache", default=str(DEFAULT_CONTEXT_CACHE))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    total_started = time.perf_counter()

    load_started = time.perf_counter()
    documents = load_documents(REPOS)
    load_seconds = time.perf_counter() - load_started
    if not documents:
        print("No source documents found; check the paths in helpdesk/sources.py.")
        return

    from transformers.utils import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    embedder = Embedder(args.model, device=args.device)

    chunk_started = time.perf_counter()
    chunks = chunk_documents(documents, DEFAULT_MAX_TOKENS, embedder.count_tokens)
    chunk_seconds = time.perf_counter() - chunk_started

    context_started = time.perf_counter()
    contextualizer = None
    if args.variant == "llm":
        contextualizer = LLMContextualizer(
            {(doc.repo, doc.path): doc.text for doc in documents},
            cache=ContextCache(args.context_cache),
            model=args.context_model,
            workers=args.workers,
        )

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  contexts {done}/{total}", flush=True)

    contexts = contexts_for(
        args.variant, chunks, embedder.count_tokens, contextualizer, progress
    )
    context_seconds = time.perf_counter() - context_started

    embed_seconds = 0.0
    original_embed = embedder.embed_documents

    def timed_embed(texts: list[str]):
        nonlocal embed_seconds
        started = time.perf_counter()
        try:
            return original_embed(texts)
        finally:
            embed_seconds += time.perf_counter() - started

    embedder.embed_documents = timed_embed
    index_started = time.perf_counter()
    stats: dict[str, int] = {}
    count = build_index(
        chunks, embedder, args.persist_dir, variant=args.variant, contexts=contexts, stats=stats
    )
    index_seconds = time.perf_counter() - index_started
    total_seconds = time.perf_counter() - total_started

    print(f"model: {args.model} on {embedder.device}")
    print(f"collection: {collection_name(args.model, args.variant)}")
    print(f"documents: {len(documents)} | chunks: {len(chunks)} | indexed: {count}")
    print(f"re-embedded: {stats['embedded']} | metadata only: {stats['metadata_only']} | "
          f"unchanged: {stats['unchanged']} | deleted: {stats['deleted']}")
    print(f"load: {load_seconds:.2f}s")
    print(f"chunk: {chunk_seconds:.2f}s")
    if contexts is not None:
        print(f"contexts: {len(contexts)}/{len(chunks)} in {context_seconds:.2f}s")
    if contextualizer is not None:
        llm = contextualizer
        print(
            f"LLM calls: {llm.calls} | cache hits: {llm.cache_hits} | "
            f"failures: {len(llm.failures)} (embedded plain)"
        )
        if llm.calls:
            print(
                f"mean call {llm.latency_s / llm.calls:.1f}s | cost ${llm.cost_usd:.4f} "
                f"(${llm.cost_usd / llm.calls:.5f}/call)"
            )
            print(f"usage: {dict(llm.usage)}")
    print(f"embed: {embed_seconds:.2f}s")
    print(f"Chroma upsert and cleanup: {max(0.0, index_seconds - embed_seconds):.2f}s")
    print(f"total: {total_seconds:.2f}s")


if __name__ == "__main__":
    main()

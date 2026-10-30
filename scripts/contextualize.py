"""Generate (and cache) LLM contexts for some or all chunks, report the cost,
and estimate the cost of the rest. No index is built: build_index --variant
llm does that, reusing data/context_cache.jsonl.

    python -m scripts.contextualize --repo research-desk     # cost check on one repo
    python -m scripts.contextualize                          # everything

The estimate scales the measured cost per prompt token (document + chunk,
counted with bge's tokenizer, which is only a proxy for Claude's) to the
chunks not yet cached.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter

from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from helpdesk.chunker import DEFAULT_MAX_TOKENS, chunk_documents
from helpdesk.contextualize import (
    DEFAULT_CONTEXT_CACHE,
    DEFAULT_CONTEXT_MODEL,
    ContextCache,
    LLMContextualizer,
    content_hash,
)
from helpdesk.embedder import DEFAULT_MODEL
from helpdesk.loader import load_documents
from helpdesk.sources import REPOS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", choices=tuple(REPOS), help="only this repo's chunks")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=DEFAULT_CONTEXT_MODEL)
    parser.add_argument("--context-cache", default=str(DEFAULT_CONTEXT_CACHE))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    transformers_logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False, verbose=False))

    documents = load_documents(REPOS)
    chunks = chunk_documents(documents, DEFAULT_MAX_TOKENS, count)
    doc_text = {(doc.repo, doc.path): doc.text for doc in documents}
    doc_tokens = {key: count(text) for key, text in doc_text.items()}

    def prompt_tokens(chunk) -> int:
        return doc_tokens[(chunk.repo, chunk.path)] + chunk.n_tokens

    todo = [chunk for chunk in chunks if args.repo is None or chunk.repo == args.repo]
    if not todo:
        parser.error("no chunks to contextualize")
    print(
        f"{len(todo)} chunks in {len({(c.repo, c.path) for c in todo})} documents"
        + (f" of {args.repo}" if args.repo else "")
    )
    sizes = [prompt_tokens(c) for c in todo]
    print(f"prompt size (bge tokens): median {statistics.median(sizes):.0f}, max {max(sizes)}")

    cache = ContextCache(args.context_cache)
    llm = LLMContextualizer(doc_text, cache=cache, model=args.model, workers=args.workers)

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    started = time.perf_counter()
    out = llm.contextualize(todo, progress)
    wall = time.perf_counter() - started

    print(
        f"\ncalls {llm.calls} | cache hits {llm.cache_hits} | "
        f"failures {len(llm.failures)} | wall {wall:.0f}s"
    )
    if llm.calls:
        print(f"mean call {llm.latency_s / llm.calls:.1f}s")
        print(f"cost ${llm.cost_usd:.4f} (${llm.cost_usd / llm.calls:.5f}/call)")
    print(f"usage {dict(llm.usage)}")
    usage = llm.usage
    fresh = usage["input_tokens"] + usage["cache_creation_input_tokens"] + usage["cache_read_input_tokens"]
    if fresh:
        print(
            f"prompt caching: {usage['cache_read_input_tokens'] / fresh:.0%} of input tokens "
            f"read from cache, {usage['cache_creation_input_tokens'] / fresh:.0%} written to it"
        )

    # estimate everything not yet cached, from this run's cost per prompt token
    done_keys = set(cache.data)
    rest = [c for c in chunks if content_hash(c) not in done_keys]
    called = set(llm.called)
    if llm.calls and rest:
        per_token = llm.cost_usd / max(1, sum(prompt_tokens(c) for c in todo if c.id in called))
        by_repo: Counter[str] = Counter()
        for chunk in rest:
            by_repo[chunk.repo] += prompt_tokens(chunk) * per_token
        print(
            f"\nestimate for the {len(rest)} chunks not yet cached: "
            f"${sum(by_repo.values()):.2f} "
            f"({sum(prompt_tokens(c) for c in rest) / 1e6:.1f}M prompt tokens)"
        )
        for repo, usd in by_repo.most_common():
            print(f"  {repo:30} ${usd:.2f}")
        per_doc = Counter((c.repo, c.path) for c in rest)
        print("largest cost by document (doc tokens x chunks):")
        for key, n in sorted(per_doc.items(), key=lambda kv: -doc_tokens[kv[0]] * kv[1])[:5]:
            print(
                f"  {key[0]}/{key[1]}: {doc_tokens[key]} tokens x {n} chunks "
                f"= ${doc_tokens[key] * n * per_token:.2f}"
            )
    elif not rest:
        print("\nevery chunk has a cached context")

    for chunk in todo[:3]:
        print(f"\n{chunk.path} :: {chunk.title}\n  -> {out.get(chunk.id)}")


if __name__ == "__main__":
    main()

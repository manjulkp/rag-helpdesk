"""Run a golden retrieval set against one or more index configurations.

    python -m scripts.evaluate                                   # every config
    python -m scripts.evaluate bge-small --retriever hybrid
    python -m scripts.evaluate bge-small --retriever hybrid --transform hyde
    python -m scripts.evaluate bge-small --variant header --retriever hybrid

Runs are saved as e.g. "bge-small+bm25", "bge-small+hybrid", and with a
transform or reranker "bge-small+hybrid+hyde", "bge-small+hybrid+cross-encoder".
A non-plain index variant goes after "@": "bge-small@header+hybrid".
Besides file recall/MRR, prints answer-chunk recall@8/@20 against
data/golden_chunks.json.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from helpdesk.chunker import DEFAULT_MAX_TOKENS
from helpdesk.bm25 import BM25Index
from helpdesk.embedder import Embedder
from helpdesk.evaluate import load_golden, run, save_run, summarise
from helpdesk.hybrid import (
    DEFAULT_CANDIDATES,
    DEFAULT_RRF_K,
)
from helpdesk.index import DEFAULT_PERSIST_DIR, VARIANTS, collection_name
from helpdesk.query_transform import (
    DEFAULT_TRANSFORM_MODEL,
    TRANSFORMERS,
    TransformCache,
    make_transformer,
    retrieve_transformed,
)
from helpdesk.rerank import make_reranker

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden.json"
RESULTS = ROOT / "result"
PRED_DIR = ROOT / "data" / "chroma_pred"

CONFIGS: dict[str, dict[str, Any]] = {
    "bge-small": {"model": "BAAI/bge-small-en-v1.5"},
    "bge-small-noinstr": {
        "model": "BAAI/bge-small-en-v1.5",
        "query_instruction": "",
    },
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "qwen3-0.6b": {"model": "Qwen/Qwen3-Embedding-0.6B"},
    "pred-v1": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "persist_dir": str(PRED_DIR),
        "only_repo": "prediction-markets",
        "max_tokens": None,
    },
}


def _questions_for_config(questions, config: dict[str, Any]):
    only_repo = config.get("only_repo")
    if not only_repo:
        return questions
    return [
        question
        for question in questions
        if question.type == "negative" or only_repo in question.repos
    ]


def run_name(
    name: str, retriever: str, transform: str, reranker: str, variant: str = "plain"
) -> str:
    """Label a run so variant/transformed/reranked runs don't overwrite plain ones."""
    if variant != "plain":
        name += f"@{variant}"
    if retriever != "vector":
        name += f"+{retriever}"
    if transform != "noop":
        name += f"+{transform}"
    if reranker != "none":
        name += f"+{reranker}"
    return name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "configs",
        nargs="*",
        choices=tuple(CONFIGS),
        help="default: run every configured retrieval setup",
    )
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--variant", choices=VARIANTS, default="plain", help="index variant")
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
        "--workers",
        type=int,
        default=4,
        help="parallel transform calls to warm the cache before timing",
    )
    parser.add_argument(
        "--reranker",
        choices=("none", "cross-encoder", "claude", "gated-claude"),
        default="none",
    )
    parser.add_argument("--rerank-cutoff", type=float, default=0.1)
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.depth < 1 or args.candidates < 1 or args.rrf_k < 1 or args.workers < 1:
        parser.error("--depth, --candidates, --rrf-k, and --workers must be at least 1")
    if args.retriever == "hybrid" and args.candidates < args.depth:
        parser.error("--candidates must be at least --depth for hybrid retrieval")

    questions = load_golden(args.golden)
    names = args.configs or list(CONFIGS)
    transformer = make_transformer(
        args.transform,
        cache=TransformCache(args.transform_cache) if args.transform_cache else None,
        model=args.transform_model,
    )
    reranker = make_reranker(args.reranker, cutoff=args.rerank_cutoff)
    if args.transform != "noop":
        # fill the transform cache in parallel, so LLM calls stay out of the timing
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(transformer.transform, [q.question for q in questions]))
    for name in names:
        config = CONFIGS[name]
        selected = _questions_for_config(questions, config)
        embedder = Embedder(
            str(config["model"]),
            query_instruction=config.get("query_instruction"),
        )
        persist_dir = Path(config.get("persist_dir", DEFAULT_PERSIST_DIR))
        only_repo = config.get("only_repo")
        bm25_index = (
            BM25Index.from_chroma(persist_dir, str(config["model"]), variant=args.variant)
            if args.retriever in {"bm25", "hybrid"}
            else None
        )

        def retrieve_question(question: str, k: int):
            filters = {"repo": only_repo} if only_repo else None
            return retrieve_transformed(
                question,
                transformer,
                embedder,
                persist_dir,
                k=k,
                where=filters,
                retriever=args.retriever,
                candidates=max(args.candidates, k),
                rrf_k=args.rrf_k,
                bm25_index=bm25_index,
                reranker=reranker,
                variant=args.variant,
            )

        times: list[float] = []

        def timed(question: str, k: int):
            started = time.perf_counter()
            hits = retrieve_question(question, k)
            times.append(time.perf_counter() - started)
            return hits

        results = run(selected, timed, depth=args.depth)
        latency_ms = 1000 * sum(times) / len(times) if times else 0.0
        label = run_name(name, args.retriever, args.transform, args.reranker, args.variant)
        run_config = {
            **config,
            "name": label,
            "variant": args.variant,
            "collection": collection_name(str(config["model"]), args.variant),
            "retriever": args.retriever,
            "transform": args.transform,
            "reranker": args.reranker,
            "rerank_cutoff": args.rerank_cutoff if args.reranker == "gated-claude" else None,
            "candidates": args.candidates if args.retriever == "hybrid" else None,
            "rrf_k": args.rrf_k if args.retriever == "hybrid" else None,
            "depth": args.depth,
            "max_tokens": config.get("max_tokens", DEFAULT_MAX_TOKENS),
            "golden_sha256": hashlib.sha256(args.golden.read_bytes()).hexdigest(),
            "golden_chunks_sha256": (
                hashlib.sha256(
                    args.golden.with_name("golden_chunks.json").read_bytes()
                ).hexdigest()
                if args.golden.with_name("golden_chunks.json").is_file()
                else None
            ),
            "questions": len(selected),
            "device": embedder.device,
            "latency_ms_per_query": round(latency_ms, 1),
            "transform_fallbacks": getattr(transformer, "fallbacks", None),
            "rerank_skipped": getattr(reranker, "skipped", None),
        }
        result_path = save_run(results, run_config, args.results_dir)
        summary = summarise(results)["overall"]

        def fmt(value) -> str:
            return "n/a" if value is None else f"{value:.2f}"

        print(
            f"{label:28} n={summary['n']:2} "
            f"R@1 {fmt(summary['recall@1'])} R@5 {fmt(summary['recall@5'])} "
            f"R@8 {fmt(summary['recall@8'])} R@20 {fmt(summary['recall@20'])} "
            f"MRR {fmt(summary['mrr'])} "
            f"ansR@8 {fmt(summary['answer_recall@8'])} "
            f"ansR@20 {fmt(summary['answer_recall@20'])} "
            f"{latency_ms:.0f} ms/q -> {result_path.name}"
        )


if __name__ == "__main__":
    main()

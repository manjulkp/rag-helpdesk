"""Run the answerable golden set with each configured answer generator.

Examples:
    python -m scripts.eval_answers
    python -m scripts.eval_answers haiku
    python -m scripts.eval_answers --only concept-01,test-01
    python -m scripts.eval_answers haiku --retriever hybrid --transform hyde

Runs are labelled like haiku+hybrid+hyde (retriever, transform, reranker), and
haiku@header+hybrid for a non-plain index variant (--variant header|llm).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers.utils import logging as transformers_logging

from helpdesk.bm25 import BM25Index
from helpdesk.embedder import DEFAULT_MODEL, Embedder
from helpdesk.evaluate import GoldenQuestion, load_golden
from helpdesk.generate import (
    Answer,
    ClaudeGenerator,
    ExtractiveGenerator,
    Generator,
)
from helpdesk.hybrid import (
    DEFAULT_CANDIDATES,
    DEFAULT_RRF_K,
)
from helpdesk.index import DEFAULT_PERSIST_DIR, VARIANTS, SearchHit, collection_name
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
OUTPUT = ROOT / "results" / "answers"
GENERATORS = ("haiku", "sonnet", "extractive")
CLAUDE_MODELS = {
    "haiku": "claude-haiku-5-5",
    "sonnet": "claude-sonnet-5-5",
}


def make_generator(name: str, threshold: float) -> Generator:
    if name in CLAUDE_MODELS:
        return ClaudeGenerator(CLAUDE_MODELS[name], name=name)
    if name == "extractive":
        return ExtractiveGenerator(threshold=threshold)
    raise ValueError(f"Unknown generator: {name}")


def _question_record(
    question: GoldenQuestion,
    hits: list[SearchHit],
    answer: Answer,
) -> dict[str, Any]:
    return {
        "id": question.id,
        "type": question.type,
        "question": question.question,
        "expected": [
            {"repo": repo, "path": path} for repo, path in question.expected
        ],
        "expected_chunks": [
            [
                {"repo": repo, "path": path, "title": title}
                for repo, path, title in group
            ]
            for group in question.expected_chunks
        ],
        "answer": answer.to_dict(),
        "retrieved": [
            {
                "id": hit.chunk_id,
                "repo": str(hit.meta.get("repo", "")),
                "path": str(hit.meta.get("path", "")),
                "title": str(hit.meta.get("title", "")),
                "line_start": int(hit.meta.get("line_start", 1)),
                "line_end": int(hit.meta.get("line_end", 1)),
                "score": round(float(hit.score), 6),
                "text": hit.text,
            }
            for hit in hits
        ],
    }


def summarise_answers(records: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in records if row["type"] != "negative"]
    negatives = [row for row in records if row["type"] == "negative"]
    answerable_answers = [
        row for row in answerable if row["answer"]["answered"]
    ]
    answered_negatives = [
        row for row in negatives if row["answer"]["answered"]
    ]
    cited_count = sum(len(row["answer"]["citations"]) for row in answerable)
    invalid_citation_count = sum(
        len(row["answer"]["dropped_citations"]) for row in answerable
    )
    expected_count = sum(len(row["expected"]) for row in answerable)
    correct_citations = 0
    expected_cited_count = 0
    for row in answerable:
        expected = {(item["repo"], item["path"]) for item in row["expected"]}
        cited = {
            (item["repo"], item["path"])
            for item in row["answer"]["citations"]
        }
        correct_citations += sum(
            (item["repo"], item["path"]) in expected
            for item in row["answer"]["citations"]
        )
        expected_cited_count += len(expected & cited)
    latencies = [float(row["answer"]["latency_seconds"]) for row in records]
    costs = [
        float(row["answer"]["cost_usd"])
        for row in records
        if row["answer"]["cost_usd"] is not None
    ]
    return {
        "n": len(records),
        "answerable_n": len(answerable),
        "negative_n": len(negatives),
        "answer_rate": (
            len(answerable_answers) / len(answerable) if answerable else None
        ),
        "refusal_rate_on_negatives": (
            1 - len(answered_negatives) / len(negatives) if negatives else None
        ),
        "answered_negative_n": len(answered_negatives),
        "citation_precision": (
            correct_citations / cited_count if cited_count else None
        ),
        "citation_validity": (
            cited_count / (cited_count + invalid_citation_count)
            if cited_count + invalid_citation_count
            else None
        ),
        "citation_recall": (
            expected_cited_count / expected_count if expected_count else None
        ),
        "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
        "cost_usd_total": sum(costs) if costs else None,
    }


def run_label(
    generator: str, retriever: str, transform: str, reranker: str, variant: str = "plain"
) -> str:
    """e.g. haiku, haiku+hybrid, haiku+hybrid+hyde, haiku@header+hybrid."""
    parts = [generator if variant == "plain" else f"{generator}@{variant}"]
    if retriever != "vector":
        parts.append(retriever)
    if transform != "noop":
        parts.append(transform)
    if reranker != "none":
        parts.append(reranker)
    return "+".join(parts)


def _write_run(
    generator_name: str,
    generator: Generator,
    records: list[dict[str, Any]],
    *,
    golden_path: Path,
    model: str,
    k: int,
    threshold: float,
    output_dir: Path,
    retriever: str,
    candidates: int,
    rrf_k: int,
    transform: str,
    reranker: str,
    rerank_cutoff: float,
    transform_fallbacks: int | None = None,
    variant: str = "plain",
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    label = run_label(generator_name, retriever, transform, reranker, variant)
    run_config = {
        "label": label,
        "variant": variant,
        "collection": collection_name(model, variant),
        "generator": generator_name,
        "generator_model": getattr(generator, "model", None),
        "embedding_model": model,
        "top_k": k,
        "retriever": retriever,
        "transform": transform,
        "reranker": reranker,
        "rerank_cutoff": rerank_cutoff if reranker == "gated-claude" else None,
        "transform_fallbacks": transform_fallbacks,
        "candidates": candidates if retriever == "hybrid" else None,
        "rrf_k": rrf_k if retriever == "hybrid" else None,
        "extractive_threshold": threshold if generator_name == "extractive" else None,
        "golden_file": str(golden_path),
        "golden_sha256": hashlib.sha256(golden_path.read_bytes()).hexdigest(),
        "golden_chunks_sha256": (
            hashlib.sha256(
                golden_path.with_name("golden_chunks.json").read_bytes()
            ).hexdigest()
            if golden_path.with_name("golden_chunks.json").is_file()
            else None
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{timestamp}_{label}.json"
    payload = {
        "timestamp": timestamp,
        "config": run_config,
        "summary": summarise_answers(records),
        "questions": records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def evaluate_one(
    questions: list[GoldenQuestion],
    retrieved: dict[str, list[SearchHit]],
    generator: Generator,
    workers: int,
) -> list[dict[str, Any]]:
    def evaluate(question: GoldenQuestion) -> dict[str, Any]:
        hits = retrieved[question.id]
        answer = generator.answer(question.question, hits)
        return _question_record(question, hits, answer)

    if workers == 1:
        return [evaluate(question) for question in questions]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate, question) for question in questions]
        return [future.result() for future in futures]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("generators", nargs="*", choices=GENERATORS)
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("-k", type=int, default=8)
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
        "--reranker",
        choices=("none", "cross-encoder", "claude", "gated-claude"),
        default="none",
    )
    parser.add_argument("--rerank-cutoff", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--only",
        help="comma-separated question IDs to evaluate instead of the full golden set",
    )
    args = parser.parse_args()
    if args.k < 1 or args.workers < 1 or args.candidates < 1 or args.rrf_k < 1:
        parser.error("-k, --workers, --candidates, and --rrf-k must be at least 1")
    if args.retriever == "hybrid" and args.candidates < args.k:
        parser.error("--candidates must be at least -k for hybrid retrieval")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    transformers_logging.set_verbosity_error()
    questions = load_golden(args.golden)
    if args.only:
        selected_ids = {item.strip() for item in args.only.split(",") if item.strip()}
        questions = [question for question in questions if question.id in selected_ids]
        found_ids = {question.id for question in questions}
        missing_ids = selected_ids - found_ids
        if missing_ids:
            parser.error(f"Unknown golden question IDs: {', '.join(sorted(missing_ids))}")

    embedder = Embedder(args.model)
    bm25_index = (
        BM25Index.from_chroma(args.persist_dir, args.model, variant=args.variant)
        if args.retriever in {"bm25", "hybrid"}
        else None
    )
    transformer = make_transformer(
        args.transform,
        cache=TransformCache(args.transform_cache) if args.transform_cache else None,
        model=args.transform_model,
    )
    reranker = make_reranker(args.reranker, cutoff=args.rerank_cutoff)
    retrieved = {
        question.id: retrieve_transformed(
            question.question,
            transformer,
            embedder,
            args.persist_dir,
            k=args.k,
            retriever=args.retriever,
            candidates=args.candidates,
            rrf_k=args.rrf_k,
            bm25_index=bm25_index,
            reranker=reranker,
            variant=args.variant,
        )
        for question in questions
    }
    for name in args.generators or GENERATORS:
        generator = make_generator(name, args.threshold)
        records = evaluate_one(questions, retrieved, generator, args.workers)
        run_path = _write_run(
            name,
            generator,
            records,
            golden_path=args.golden,
            model=args.model,
            k=args.k,
            threshold=args.threshold,
            output_dir=args.output_dir,
            retriever=args.retriever,
            candidates=args.candidates,
            rrf_k=args.rrf_k,
            transform=args.transform,
            reranker=args.reranker,
            rerank_cutoff=args.rerank_cutoff,
            transform_fallbacks=getattr(transformer, "fallbacks", None),
            variant=args.variant,
        )
        summary = summarise_answers(records)
        label = run_label(name, args.retriever, args.transform, args.reranker, args.variant)
        print(
            f"{label}: answered={summary['answer_rate']} "
            f"negative_refusal={summary['refusal_rate_on_negatives']} "
            f"saved={run_path}"
        )


if __name__ == "__main__":
    main()

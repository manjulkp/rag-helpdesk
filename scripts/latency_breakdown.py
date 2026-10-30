"""Where a chat turn's time goes, stage by stage (13.6).

    python -m scripts.latency_breakdown            # 3 follow-up turns, median per stage

Mirrors pipeline.respond for a follow-up turn with every cache off, so each
LLM stage is a real Claude call: condense, query embedding, vector search,
BM25, fusion (RRF), gate (cross-encoder), policy scope check, Haiku answer.
The optional Claude rerank (off by default in the chat) is timed too and
reported separately. Models are loaded and warmed up before timing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from transformers.utils import logging as hf_logging

from helpdesk.bm25 import BM25Index
from helpdesk.conversation import Turn, format_history
from helpdesk.embedder import DEFAULT_MODEL, Embedder
from helpdesk.generate import ClaudeGenerator
from helpdesk.hybrid import rrf
from helpdesk.index import DEFAULT_PERSIST_DIR, search
from helpdesk.pipeline import gate_score
from helpdesk.policy import decide
from helpdesk.query_transform import ClaudeCLI, CondenseQuestion, TransformCache
from helpdesk.rerank import ClaudeReranker, CrossEncoderReranker

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["condense", "embed query", "vector search", "BM25", "fusion", "gate", "policy (scope)", "answer"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=3)
    args = parser.parse_args()
    hf_logging.set_verbosity_error()
    embedder = Embedder(DEFAULT_MODEL)
    bm25 = BM25Index.from_chroma(DEFAULT_PERSIST_DIR, DEFAULT_MODEL, variant="llm")
    gate = CrossEncoderReranker()
    condenser = CondenseQuestion(cache=TransformCache(None))
    cli = ClaudeCLI("claude-haiku-5-5")
    generator = ClaudeGenerator("claude-haiku-5-5", name="haiku")
    reranker = ClaudeReranker()
    # warm up the local models so loading isn't timed
    embedder.embed_query("warm up")
    gate._score([("warm up", "warm up")])

    convos = json.loads((ROOT / "data" / "golden_conversations.json").read_text(encoding="utf-8"))
    picked = [c for c in convos if c["type"] in ("pronoun", "ellipsis", "drilldown")][: args.turns]
    timings: list[dict[str, float]] = []
    for convo in picked:
        first, follow = convo["turns"][0]["question"], convo["turns"][1]["question"]
        history = [Turn("user", first), Turn("assistant", "(the previous answer)", action="answer")]
        t: dict[str, float] = {}

        def stage(name, fn):
            started = time.perf_counter()
            value = fn()
            t[name] = (time.perf_counter() - started) * 1000
            return value

        query = stage("condense", lambda: condenser.condense(follow, format_history(history)))
        vector = stage("embed query", lambda: embedder.embed_query(query))
        vec_hits = stage("vector search", lambda: search(query, embedder, DEFAULT_PERSIST_DIR, k=50,
                                                         query_vector=vector, variant="llm"))
        kw_hits = stage("BM25", lambda: bm25.search(query, k=50))
        fused = stage("fusion", lambda: rrf([vec_hits, kw_hits]))
        hits = fused[:8]
        score = stage("gate", lambda: gate_score(gate, query, hits))
        # a follow-up that names no project needs the scope call; force it for the timing
        stage("policy (scope)", lambda: decide(query.replace("trade desk", "it").replace("research desk", "it"),
                                               history, hits, score, llm=cli))
        stage("answer", lambda: generator.answer(follow, hits, format_history(history)))
        stage("Claude rerank (optional)", lambda: reranker.rerank(query, fused[:20]))
        timings.append(t)
        print(f"{convo['id']}: " + ", ".join(f"{k} {v:.0f} ms" for k, v in t.items()), flush=True)

    medians = {name: statistics.median(t[name] for t in timings) for name in timings[0]}
    total = sum(medians[s] for s in STAGES)
    print(f"\nmedian over {len(timings)} follow-up turns (shipped pipeline, no rerank): {total / 1000:.1f} s per turn")
    print(f"{'stage':26}{'ms':>10}{'share':>8}")
    for name in STAGES:
        print(f"{name:26}{medians[name]:>10.1f}{medians[name] / total:>8.1%}")
    rr = medians["Claude rerank (optional)"]
    print(f"{'Claude rerank (optional)':26}{rr:>10.1f}   (+{rr / total:.0%} if turned on)")
    out = ROOT / "results" / "faiss" / "latency_breakdown.json"
    out.write_text(json.dumps({"turns": timings, "median_ms": medians}, indent=2), encoding="utf-8")
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

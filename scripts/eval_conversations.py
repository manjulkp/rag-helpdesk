"""Evaluate follow-up retrieval on data/golden_conversations.json.

    python -m scripts.eval_conversations            # raw / concat / condense / condense+sticky
    python -m scripts.eval_conversations --answer   # + Haiku answers for the best config

Retrieval history is the earlier user questions (the golden set has no
assistant answers); the "sticky" sources are the previous turn's top 3 hits,
standing in for its citations. Reports answer-chunk recall@20 on turns 2+
(turn 1 is the same for every config), by conversation type; for topic
switches, the share of the top 8 that comes from the previous turn's files
("drag"); and prompt tokens over a 10-turn conversation with and without
summarising. --answer runs Haiku with the real history (questions and
answers) and writes a grading sheet for the follow-up answers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from transformers.utils import logging as transformers_logging

from helpdesk.bm25 import BM25Index
from helpdesk.conversation import (
    MODES,
    STICKY_SOURCES,
    Turn,
    estimate_tokens,
    format_history,
    load_golden_conversations,
    retrieve_turn,
)
from helpdesk.embedder import DEFAULT_MODEL, Embedder
from helpdesk.evaluate import _hit_dict, answer_rank, first_relevant_rank
from helpdesk.generate import ClaudeGenerator, GeneratorError, build_user_message
from helpdesk.hybrid import DEFAULT_CANDIDATES
from helpdesk.index import DEFAULT_PERSIST_DIR, VARIANTS
from helpdesk.query_transform import CondenseQuestion, NoopTransformer, retrieve_transformed

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden_conversations.json"
OUT = ROOT / "results" / "conversations"
DEPTH = 20
TOP = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="llm")
    parser.add_argument("--retriever", choices=("vector", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--answer", action="store_true", help="run Haiku on the best config")
    parser.add_argument("--from", dest="saved", type=Path,
                        help="only print the paired report for a saved results JSON")
    args = parser.parse_args()
    if args.saved:
        paired_report(json.loads(args.saved.read_text(encoding="utf-8"))["rows"])
        return
    transformers_logging.set_verbosity_error()

    convos = load_golden_conversations(GOLDEN)
    embedder = Embedder(DEFAULT_MODEL)
    bm25 = (
        BM25Index.from_chroma(DEFAULT_PERSIST_DIR, DEFAULT_MODEL, variant=args.variant)
        if args.retriever != "vector"
        else None
    )
    lock = Lock()

    def retrieve(query: str, depth: int):
        with lock:  # one embedder/Chroma client shared by the worker threads
            return retrieve_transformed(
                query, NoopTransformer(), embedder, DEFAULT_PERSIST_DIR, k=depth,
                retriever=args.retriever, candidates=max(DEFAULT_CANDIDATES, depth),
                bm25_index=bm25, variant=args.variant,
            )

    condenser = CondenseQuestion()

    def run_conversation(mode: str, convo) -> list[dict]:
        history: list[Turn] = []
        sticky: list = []
        rows = []
        for number, turn in enumerate(convo.turns, start=1):
            query, hits = retrieve_turn(
                mode, turn.question, history, retrieve, depth=DEPTH,
                condenser=condenser, sticky_hits=sticky,
            )
            retrieved = [_hit_dict(hit) for hit in hits]
            previous = set(convo.turns[number - 2].expected) if number > 1 else set()
            top_files = [(h["repo"], h["path"]) for h in retrieved[:TOP]]
            rows.append({
                "conversation": convo.id, "type": convo.type, "turn": number, "mode": mode,
                "question": turn.question, "query": query,
                "answerable": bool(turn.expected),
                "rank": first_relevant_rank(retrieved, turn.expected) if turn.expected else None,
                "answer_rank": answer_rank(retrieved, turn.answer_chunks),
                "drag": (
                    sum(f in previous and f not in set(turn.expected) for f in top_files) / len(top_files)
                    if number > 1 and top_files else None
                ),
                "top_score": retrieved[0]["score"] if retrieved else None,
                "retrieved": retrieved[:TOP],
            })
            history.append(Turn("user", turn.question))
            sticky = hits[:STICKY_SOURCES]
        return rows

    rows: list[dict] = []
    for mode in MODES:
        with ThreadPoolExecutor(args.workers) as pool:
            for result in pool.map(lambda c: run_conversation(mode, c), convos):
                rows.extend(result)

    def recall(selected: list[dict]) -> str:
        labelled = [r for r in selected if r["answerable"]]
        if not labelled:
            return "  - "
        return f"{sum(r['answer_rank'] is not None and r['answer_rank'] <= DEPTH for r in labelled) / len(labelled):.2f}"

    follow = [r for r in rows if r["turn"] > 1]
    types = sorted({c.type for c in convos})
    n = len([r for r in follow if r["mode"] == "raw" and r["answerable"]])
    print(f"\nanswer-chunk recall@{DEPTH} on follow-up turns (n={n} answerable) | {args.variant}+{args.retriever}")
    print(f"{'config':17}{'all':>6}" + "".join(f"{t[:11]:>13}" for t in types))
    for mode in MODES:
        mine = [r for r in follow if r["mode"] == mode]
        print(f"{mode:17}{recall(mine):>6}" + "".join(
            f"{recall([r for r in mine if r['type'] == t]):>13}" for t in types))

    paired_report(rows)

    print(f"\ntopic switches: share of the top {TOP} from the previous turn's files (drag), and ansR@{DEPTH}")
    for mode in MODES:
        switch = [r for r in follow if r["mode"] == mode and r["type"] == "topic_switch"]
        drag = sum(r["drag"] for r in switch) / len(switch)
        print(f"  {mode:17} drag {drag:.2f}  ansR@{DEPTH} {recall(switch)}  queries: "
              + " | ".join(r["query"][:70] for r in switch))

    unanswerable = [r for r in follow if not r["answerable"]]
    print("\nunanswerable follow-up, top-1 score: "
          + ", ".join(f"{r['mode']} {r['top_score']:.3f}" for r in unanswerable))

    # prompt tokens over one long 10-turn conversation
    long_questions = [c.turns[0].question for c in convos][:10]
    turns: list[Turn] = []
    growth = []
    for number, question in enumerate(long_questions, start=1):
        hits = retrieve(question, TOP)
        full = estimate_tokens(build_user_message(question, hits, format_history(turns, summarise=False)))
        short = estimate_tokens(build_user_message(question, hits, format_history(turns)))
        growth.append({"turn": number, "full": full, "summarised": short,
                       "history_full": estimate_tokens(format_history(turns, summarise=False)),
                       "history_summarised": estimate_tokens(format_history(turns))})
        # a stand-in answer: the first ~120 words of the top two chunks
        answer = " ".join(" ".join(h.text.split()[:60]) for h in hits[:2])
        turns += [Turn("user", question), Turn("assistant", answer)]
    print("\nprompt tokens (estimated, ~4 chars/token) by turn: full history vs summarised")
    for g in growth:
        print(f"  turn {g['turn']:2}: {g['full']:5} vs {g['summarised']:5}   (history {g['history_full']} vs {g['history_summarised']})")

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"config": {"variant": args.variant, "retriever": args.retriever, "depth": DEPTH,
                          "condense_calls": condenser.calls, "condense_fallbacks": condenser.fallbacks},
               "rows": rows, "token_growth": growth}

    if args.answer:
        payload["answers"] = answer_run(convos, retrieve, args.workers)
        sheet = OUT / f"grading_{stamp}.md"
        write_sheet(sheet, payload["answers"])
        print(f"\nHaiku answers -> {sheet.relative_to(ROOT)}")

    path = OUT / f"{stamp}_{args.variant}+{args.retriever}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {path.relative_to(ROOT)} | condense calls {condenser.calls}, fallbacks {condenser.fallbacks}")


PAIRS = (("condense", "raw"), ("condense", "concat"), ("condense+sticky", "condense"))


def paired_report(rows: list[dict]) -> None:
    """ansR@8 / ansR@20 / answer-chunk MRR per config on answerable follow-up
    turns, paired bootstrap 95% CIs for the PAIRS, and the turns where sticky
    sources change the answer chunk's rank."""
    from scripts.compare_runs import bootstrap_difference

    follow = [r for r in rows if r["turn"] > 1 and r["answerable"]]
    by_mode = {
        mode: {(r["conversation"], r["turn"]): r["answer_rank"] for r in follow if r["mode"] == mode}
        for mode in MODES
    }
    keys = sorted(by_mode["raw"])
    metrics = {
        "ansR@8": lambda rank: float(rank is not None and rank <= 8),
        f"ansR@{DEPTH}": lambda rank: float(rank is not None and rank <= DEPTH),
        "MRR": lambda rank: 1 / rank if rank else 0.0,
    }
    print(f"\nanswerable follow-up turns (n={len(keys)}): " + " | ".join(metrics))
    for mode in MODES:
        print(f"  {mode:17}" + "".join(
            f"{sum(f(by_mode[mode][k]) for k in keys) / len(keys):>10.2f}" for f in metrics.values()))
    print("paired bootstrap, 10,000 resamples: difference [95% CI], wins/ties/losses")
    for a, b in PAIRS:
        cells = []
        for name, f in metrics.items():
            first = [f(by_mode[a][k]) for k in keys]
            second = [f(by_mode[b][k]) for k in keys]
            diff, low, high = bootstrap_difference(first, second)
            wins = sum(x > y for x, y in zip(first, second))
            losses = sum(x < y for x, y in zip(first, second))
            cells.append(f"{name} {diff:+.2f} [{low:+.2f}, {high:+.2f}] {wins}/{len(keys) - wins - losses}/{losses}")
        print(f"  {a} - {b}: " + " | ".join(cells))
    moved = [(k, by_mode["condense"][k], by_mode["condense+sticky"][k]) for k in keys
             if by_mode["condense"][k] != by_mode["condense+sticky"][k]]
    print("sticky changes the answer chunk's rank on: "
          + (", ".join(f"{c} t{t}: {x} -> {y}" for (c, t), x, y in moved) or "no turn"))


def answer_run(convos, retrieve, workers: int) -> list[dict]:
    """condense+sticky retrieval and Haiku answers with the real history."""
    generator = ClaudeGenerator("claude-haiku-5-5", name="haiku")
    condenser = CondenseQuestion()

    def one(convo) -> list[dict]:
        history: list[Turn] = []
        sticky: list = []
        out = []
        for number, turn in enumerate(convo.turns, start=1):
            query, hits = retrieve_turn(
                "condense+sticky", turn.question, history, retrieve,
                depth=TOP, condenser=condenser, sticky_hits=sticky,
            )
            try:
                answer = generator.answer(turn.question, hits, format_history(history))
            except GeneratorError as error:  # one bad turn must not sink the run
                out.append({"conversation": convo.id, "type": convo.type, "turn": number,
                            "question": turn.question, "query": query, "answered": None,
                            "answer": f"ERROR: {str(error)[:200]}", "prompt_tokens": None,
                            "cost_usd": None, "cited": [],
                            "expected": [f"{r}/{p}" for r, p in turn.expected]})
                history.append(Turn("user", turn.question))
                sticky = []
                continue
            cited = [hits[c.index - 1] for c in answer.citations]
            out.append({
                "conversation": convo.id, "type": convo.type, "turn": number,
                "question": turn.question, "query": query, "answered": answer.answered,
                "answer": answer.answer, "prompt_tokens": answer.prompt_tokens,
                "cost_usd": answer.cost_usd,
                "cited": [f"{h.meta['repo']}/{h.meta['path']} :: {h.meta['title']}" for h in cited],
                "expected": [f"{r}/{p}" for r, p in turn.expected],
            })
            history += [Turn("user", turn.question),
                        Turn("assistant", answer.answer, sources=[h.chunk_id for h in cited])]
            sticky = cited[:STICKY_SOURCES]
        return out

    with ThreadPoolExecutor(workers) as pool:
        return [row for rows in pool.map(one, convos) for row in rows]


def write_sheet(path: Path, answers: list[dict]) -> None:
    lines = ["# Follow-up answers (condense+sticky, Haiku)", "",
             "Grade each follow-up: correct / partial / wrong / correctly refused.", ""]
    for row in answers:
        if row["turn"] == 1:
            continue
        lines += [
            f"## {row['conversation']} turn {row['turn']} ({row['type']})",
            f"- **Follow-up:** {row['question']}",
            f"- **Searched:** {row['query']}",
            f"- **Expected:** {', '.join(row['expected']) or '(nothing: should refuse)'}",
            f"- **Answered:** {row['answered']} | prompt tokens {row['prompt_tokens']}",
            f"- **Answer:** {row['answer']}",
            f"- **Cited:** {'; '.join(row['cited']) or '-'}",
            "- **Grade:** ",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Evaluate the reply policy on data/golden_policy.json.

    python -m scripts.eval_policy

Each case runs through the real pipeline (condense -> llm+hybrid retrieval ->
cross-encoder gate -> policy -> Haiku answer), once with the Haiku scope call
and once with rules only. The predicted action is the final reply's: an
in-scope question escalates when the gate or the answer step refuses, so the
answer step has to run. Escalation tickets go to results/policy/escalations/.

Reports a confusion matrix of expected vs predicted action, accuracy per
action, the over-clarifying rate (clarifying questions on clear, answerable
questions), under-escalation (in-scope questions with no answer that got a
refusal or an answer instead of a hand-off), grounded clarification options,
suggested contacts, and how many turns needed the LLM.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from transformers.utils import logging as transformers_logging

from helpdesk.bm25 import BM25Index
from helpdesk.conversation import Turn
from helpdesk.embedder import DEFAULT_MODEL, Embedder
from helpdesk.generate import Answer, ClaudeGenerator, GeneratorError
from helpdesk.hybrid import DEFAULT_CANDIDATES
from helpdesk.index import DEFAULT_PERSIST_DIR
from helpdesk.pipeline import respond
from helpdesk.policy import ACTIONS
from helpdesk.query_transform import (
    ClaudeCLI,
    CondenseQuestion,
    NoopTransformer,
    TransformCache,
    retrieve_transformed,
)
from helpdesk.rerank import CrossEncoderReranker

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "data" / "golden_policy.json"
OUT = ROOT / "results" / "policy"


class StubGenerator:
    """Stands in for Haiku: the policy is what's being evaluated."""

    def answer(self, question, hits, history=""):
        return Answer(answered=True, answer="(not generated in the policy eval)", citations=[],
                      generator="stub", latency_seconds=0.0)


def run(cases, retrieve, condenser, gate, policy_llm, policy_cache=None, generator=None) -> list[dict]:
    generator = generator or StubGenerator()
    rows = []
    for case in cases:
        history = [Turn(t["role"], t["text"], action=t.get("action")) for t in case.get("history", [])]
        for number, turn in enumerate(case["turns"], start=1):
            try:
                reply = respond(turn["user"], history, retrieve=retrieve, generator=generator,
                                condenser=condenser, gate=gate, policy_llm=policy_llm,
                                escalations_dir=OUT / "escalations", policy_cache=policy_cache)
            except GeneratorError as error:  # one failed Claude call must not sink the run
                rows.append({"case": case["id"], "category": case["category"], "turn": number,
                             "question": turn["user"], "query": turn["user"],
                             "expected": turn["expected_action"], "predicted": "error", "by": "-",
                             "reason": str(error)[:120], "gate": None, "options": [],
                             "options_grounded": True, "expected_options": turn.get("options", []),
                             "retrieved_expected": None, "contact": None})
                break  # the rest of this conversation has no valid history
            hit_repos = {h.meta["repo"] for h in reply.hits}
            files = {(h.meta["repo"], h.meta["path"]) for h in reply.hits}
            expected_files = {(e["repo"], e["path"]) for e in turn.get("expected", [])}
            rows.append({
                "case": case["id"], "category": case["category"], "turn": number,
                "question": turn["user"], "query": reply.query,
                "expected": turn["expected_action"], "predicted": reply.turn.action,
                "by": reply.decision.by, "reason": reply.decision.reason,
                "gate": reply.gate, "options": reply.decision.options,
                "options_grounded": all(o in hit_repos for o in reply.decision.options),
                "expected_options": turn.get("options", []),
                "retrieved_expected": bool(expected_files & files) if expected_files else None,
                "contact": reply.contact.name if reply.contact else None,
            })
            history += [Turn("user", turn["user"]), reply.turn]
    return rows


def report(name: str, rows: list[dict]) -> None:
    print(f"\n=== {name}: {sum(r['expected'] == r['predicted'] for r in rows)}/{len(rows)} turns correct")
    print("confusion (rows expected, columns predicted)")
    print(f"{'':14}" + "".join(f"{a[:12]:>13}" for a in ACTIONS))
    counts = Counter((r["expected"], r["predicted"]) for r in rows)
    for expected in ACTIONS:
        print(f"{expected:14}" + "".join(f"{counts[(expected, p)]:>13}" for p in ACTIONS))
    for action in ACTIONS:
        mine = [r for r in rows if r["expected"] == action]
        print(f"  {action:13} accuracy {sum(r['predicted'] == action for r in mine)}/{len(mine)}")
    answerable = [r for r in rows if r["expected"] == "answer"]
    over = [r["case"] for r in answerable if r["predicted"] == "clarify"]
    print(f"over-clarifying: {len(over)}/{len(answerable)} answerable turns got a clarifying question {over or ''}")
    escalate = [r for r in rows if r["expected"] == "escalate"]
    under = [(r["case"], r["predicted"]) for r in escalate if r["predicted"] != "escalate"]
    print(f"under-escalating: {len(under)}/{len(escalate)} {under or ''}")
    clar = [r for r in rows if r["predicted"] == "clarify"]
    print(f"grounded options: {sum(r['options_grounded'] for r in clar)}/{len(clar)} clarifications; "
          f"options match expected: {sum(set(r['options']) == set(r['expected_options']) for r in clar if r['expected_options'])}"
          f"/{sum(bool(r['expected_options']) for r in clar)}")
    after = [r for r in rows if r["category"] == "answer_after_clarification" and r["turn"] == 2]
    print(f"after a clarification: answered {sum(r['predicted'] == 'answer' for r in after)}/{len(after)}, "
          f"expected file retrieved {sum(bool(r['retrieved_expected']) for r in after)}/{len(after)}; "
          f"queries: {' | '.join(r['query'][:60] for r in after)}")
    esc = [r for r in rows if r["predicted"] == "escalate"]
    print(f"suggested contacts: {sum(r['contact'] is not None for r in esc)}/{len(esc)} escalations")
    by = Counter(r["by"] for r in rows)
    print(f"decided by rules {by['rule']}, by the LLM {by['llm']} ({by['llm'] / len(rows):.0%} of turns)")
    wrong = [(r["case"], r["turn"], r["expected"], r["predicted"], r["by"], r["reason"][:70])
             for r in rows if r["expected"] != r["predicted"]]
    for w in wrong:
        print(f"  miss {w[0]} t{w[1]}: expected {w[2]}, got {w[3]} ({w[4]}: {w[5]})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-only", action="store_true", help="skip the Haiku policy run")
    args = parser.parse_args()
    transformers_logging.set_verbosity_error()
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))
    embedder = Embedder(DEFAULT_MODEL)
    bm25 = BM25Index.from_chroma(DEFAULT_PERSIST_DIR, DEFAULT_MODEL, variant="llm")

    def retrieve(query: str, depth: int):
        return retrieve_transformed(query, NoopTransformer(), embedder, DEFAULT_PERSIST_DIR, k=depth,
                                    retriever="hybrid", candidates=max(DEFAULT_CANDIDATES, depth),
                                    bm25_index=bm25, variant="llm")

    condenser = CondenseQuestion()
    gate = CrossEncoderReranker()
    generator = ClaudeGenerator("claude-haiku-5-5", name="haiku")
    results = {"rules only": run(cases, retrieve, condenser, gate, None, generator=generator)}
    if not args.rules_only:
        results["rules + Haiku scope"] = run(cases, retrieve, condenser, gate,
                                             ClaudeCLI("claude-haiku-5-5"), TransformCache(),
                                             generator=generator)
    for name, rows in results.items():
        report(name, rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {path.relative_to(ROOT)} | condense calls {condenser.calls}")


if __name__ == "__main__":
    main()

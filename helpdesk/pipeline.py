"""One help desk turn: condense -> retrieve -> gate -> policy -> reply.

The reply is an answer, a clarifying question, an out-of-scope refusal or an
escalation, and comes back as the assistant ``Turn`` to append. Shared by
scripts/chat.py and scripts/eval_policy.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from helpdesk.conversation import Turn, format_history, retrieve_turn
from helpdesk.escalate import (
    DEFAULT_ESCALATIONS_DIR,
    candidate_files,
    escalation_text,
    suggested_contact,
    write_ticket,
)
from helpdesk.generate import Answer
from helpdesk.index import SearchHit
from helpdesk.access import filter_hits
from helpdesk.policy import GATE_CUTOFF, Decision, clarify_text, decide, out_of_scope_text


def gate_text(hit: SearchHit) -> str:
    """'repo / path / heading' header + chunk, whatever index the hit came from.
    Step 9: the header (not the LLM context) separates answerable questions
    from unanswerable ones best at the gate."""
    meta = hit.meta
    title = f"test: {meta.get('title')}" if meta.get("kind") == "test" else meta.get("title")
    return f"{meta.get('repo')} / {meta.get('path')} / {title}\n\n{hit.text}"


def gate_score(gate: Any | None, question: str, hits: list[SearchHit]) -> float | None:
    """Best cross-encoder score over the hits (header + chunk)."""
    if gate is None or not hits:
        return None
    return max(gate._score([(question, gate_text(hit)) for hit in hits]))


@dataclass
class Reply:
    turn: Turn  # the assistant turn to append
    decision: Decision
    query: str
    hits: list[SearchHit]
    gate: float | None
    answer: Answer | None = None
    ticket: Path | None = None
    contact: Any | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def respond(
    question: str,
    history: list[Turn],
    *,
    retrieve: Callable[[str, int], list[SearchHit]],
    generator: Any,
    condenser: Any | None = None,
    gate: Any | None = None,
    policy_llm: Any | None = None,
    sticky_hits: list[SearchHit] | None = None,
    mode: str = "condense+sticky",
    k: int = 8,
    reranker: Any | None = None,
    escalations_dir: Path = DEFAULT_ESCALATIONS_DIR,
    policy_cache: Any | None = None,
    user: Any | None = None,
) -> Reply:
    """``user`` (an access.User) limits every chunk path: retrieval, sticky
    sources, the reranker's candidates, the gate, the policy's hits and
    options, the generator and the hand-off. ``retrieve`` should already
    filter by repo; this is the second layer."""
    depth = max(k, 20) if reranker is not None else k

    def allowed_retrieve(query: str, depth: int) -> list[SearchHit]:
        return filter_hits(user, retrieve(query, depth))

    query, hits = retrieve_turn(mode, question, history, allowed_retrieve, depth=depth,
                                condenser=condenser, sticky_hits=filter_hits(user, sticky_hits or []))
    # the ambiguity check looks at the fused hits before reranking: a reranker
    # pulls its top k towards one project and hides the split a clarification is for
    spread = hits[:k]
    if reranker is not None:
        hits = reranker.rerank(query, hits)
    hits = hits[:k]
    score = gate_score(gate, query, hits)
    decision = decide(query, history, hits, score, llm=policy_llm, typed=question, cache=policy_cache,
                      spread_hits=spread)

    answer = None
    if decision.action == "answer":
        # escalation for an in-scope question comes after a refusal: the gate
        # found nothing (skip the generator), or the generator says "not found"
        if score is not None and score < GATE_CUTOFF:
            decision = Decision("escalate", f"in scope, but the gate found nothing relevant "
                                f"({score:.2f})", by=decision.by)
        else:
            answer = generator.answer(question, hits, format_history(history))
            if answer.answered:
                cited = [hits[c.index - 1].chunk_id for c in answer.citations]
                turn = Turn("assistant", answer.answer, citations=answer.citations, answered=True,
                            sources=cited, action="answer")
                return Reply(turn, decision, query, hits, score, answer=answer)
            decision = Decision("escalate", "in scope, but the answer step found nothing in the docs",
                                by=decision.by)
    if decision.action == "clarify":
        turn = Turn("assistant", clarify_text(decision.options), answered=False,
                    action="clarify", options=decision.options)
        return Reply(turn, decision, query, hits, score)
    if decision.action == "out_of_scope":
        visible = None if user is None or user.repos is None else user.repos
        turn = Turn("assistant", out_of_scope_text(visible), answered=False, action="out_of_scope")
        return Reply(turn, decision, query, hits, score)
    contact = suggested_contact(candidate_files(hits))
    ticket = write_ticket(
        question=question, condensed=query, summary=format_history(history),
        searched=query, hits=hits, reason=decision.reason, contact=contact, dir=escalations_dir,
    )
    turn = Turn("assistant", escalation_text(ticket, contact, decision.reason), answered=False,
                action="escalate")
    return Reply(turn, decision, query, hits, score, answer=answer, ticket=ticket, contact=contact)

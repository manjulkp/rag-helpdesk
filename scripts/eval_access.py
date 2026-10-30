"""Access control check: every golden question as every simulated user.

    python -m scripts.eval_access

Each run goes through the full pipeline (llm+hybrid retrieval with the user's
repo filter, gate, policy with the Haiku scope call, Haiku answers). Sticky
sources are deliberately poisoned with the unrestricted top 3 hits, so the
second layer (filter_hits) is exercised too.

A leak is a forbidden repo anywhere in: the hits, what the generator saw,
citations, clarification options, the hand-off ticket, or the reply text
(repo ID or project name). Leaks must be 0. Also reports the answer rate on
questions from allowed repos (against admin, who sees everything) and what a
user sees for questions from a forbidden repo.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from transformers.utils import logging as transformers_logging

from helpdesk.access import load_acl, repo_filter
from helpdesk.bm25 import BM25Index
from helpdesk.embedder import DEFAULT_MODEL, Embedder
from helpdesk.evaluate import load_golden
from helpdesk.generate import ClaudeGenerator, GeneratorError
from helpdesk.hybrid import DEFAULT_CANDIDATES
from helpdesk.index import DEFAULT_PERSIST_DIR
from helpdesk.pipeline import respond
from helpdesk.policy import ALIASES, PROJECTS
from helpdesk.query_transform import ClaudeCLI, NoopTransformer, TransformCache, retrieve_transformed
from helpdesk.rerank import CrossEncoderReranker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "access"


class SpyGenerator:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.seen: list = []

    def answer(self, question, hits, history=""):
        self.seen = list(hits)
        return self.inner.answer(question, hits, history)


def main() -> None:
    transformers_logging.set_verbosity_error()
    users = load_acl()
    questions = load_golden(ROOT / "data" / "golden.json")
    embedder = Embedder(DEFAULT_MODEL)
    bm25 = BM25Index.from_chroma(DEFAULT_PERSIST_DIR, DEFAULT_MODEL, variant="llm")
    gate = CrossEncoderReranker()
    policy_llm, policy_cache = ClaudeCLI("claude-haiku-5-5"), TransformCache()
    generator = SpyGenerator(ClaudeGenerator("claude-haiku-5-5", name="haiku"))
    all_repos = set(PROJECTS)

    def retriever(where):
        def retrieve(query: str, depth: int):
            return retrieve_transformed(query, NoopTransformer(), embedder, DEFAULT_PERSIST_DIR, k=depth,
                                        retriever="hybrid", candidates=max(DEFAULT_CANDIDATES, depth),
                                        bm25_index=bm25, variant="llm", where=where)
        return retrieve

    rows = []
    for name, user in sorted(users.items()):
        forbidden = all_repos - (user.repos if user.repos is not None else all_repos)
        forbidden_words = {w for repo in forbidden for w in (repo, PROJECTS[repo], *ALIASES[repo])}
        for q in questions:
            poisoned = retriever(None)(q.question, 3)  # unrestricted: may hold forbidden chunks
            generator.seen = []
            try:
                reply = respond(q.question, [], retrieve=retriever(repo_filter(user)), generator=generator,
                                gate=gate, policy_llm=policy_llm, policy_cache=policy_cache,
                                sticky_hits=poisoned, mode="condense+sticky", user=user,
                                escalations_dir=OUT / "escalations")
            except GeneratorError as error:
                rows.append({"user": name, "id": q.id, "action": "error", "leaks": [], "note": str(error)[:100]})
                continue
            leaks = []
            leaks += [f"hit:{h.meta['repo']}" for h in reply.hits if h.meta["repo"] in forbidden]
            leaks += [f"generator:{h.meta['repo']}" for h in generator.seen if h.meta["repo"] in forbidden]
            leaks += [f"citation:{c.repo}" for c in reply.turn.citations if c.repo in forbidden]
            leaks += [f"option:{o}" for o in reply.decision.options if o in forbidden]
            if reply.ticket is not None:
                # only the parts the system filled in: candidate files and the contact
                # (the Question lines echo what the user typed, which may name anything)
                ticket = reply.ticket.read_text(encoding="utf-8")
                filled = ticket.split("## Top candidate files", 1)[-1]
                leaks += [f"ticket:{repo}" for repo in forbidden if repo in filled]
            # a project name in the reply counts only if neither the user typed it nor an
            # allowed source the generator saw contains it (allowed docs may mention other projects)
            typed = q.question.casefold()
            seen_text = " ".join(h.text for h in generator.seen).casefold()
            leaks += [f"text:{w}" for w in forbidden_words
                      if w.casefold() in reply.turn.text.casefold()
                      and w.casefold() not in typed and w.casefold() not in seen_text]
            allowed_question = bool(q.repos) and (user.repos is None or set(q.repos) <= user.repos)
            rows.append({
                "user": name, "id": q.id, "type": q.type, "repos": q.repos,
                "allowed": allowed_question, "forbidden_question": bool(set(q.repos) & forbidden),
                "action": reply.turn.action, "leaks": leaks, "text": reply.turn.text[:200],
            })

    print(f"\nleaks: {sum(len(r['leaks']) for r in rows)} across {len(rows)} user x question runs")
    for r in rows:
        if r["leaks"]:
            print(f"  LEAK {r['user']} {r['id']}: {r['leaks']}")
    for name in sorted(users):
        mine = [r for r in rows if r["user"] == name]
        allowed = [r for r in mine if r.get("allowed")]
        forbidden = [r for r in mine if r.get("forbidden_question")]
        print(f"\n{name}: answered {sum(r['action'] == 'answer' for r in allowed)}/{len(allowed)} "
              f"questions from allowed repos; forbidden-repo questions -> "
              f"{dict(Counter(r['action'] for r in forbidden)) or '-'}; negatives -> "
              f"{dict(Counter(r['action'] for r in mine if not r.get('repos')))}")
        for r in forbidden[:3]:
            print(f"    {r['id']}: {r['action']}: {r['text'][:110]!r}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

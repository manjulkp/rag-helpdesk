"""Chat with the helpdesk: type a question, get a cited answer; follow-ups
use the conversation so far. Each turn goes condense -> retrieve -> gate ->
policy, and the reply is an answer, a clarifying question, an out-of-scope
refusal or an escalation (a local hand-off ticket in data/escalations/).

    python -m scripts.chat
    python -m scripts.chat --mode raw --variant plain      # no history in the search
    python -m scripts.chat --no-policy-llm                 # rules only, no Haiku policy call
    python -m scripts.chat --resume <session id>
    python -m scripts.chat --user rd-analyst               # sees only the research desk
    python -m scripts.chat --debug                         # show policy decisions

Commands: /new (start over), /history, /sources (last reply's chunks), /quit.
Sessions are saved as JSON in data/sessions/.
"""

from __future__ import annotations

import argparse
import logging
import uuid
from datetime import datetime

from transformers.utils import logging as transformers_logging

from helpdesk.access import load_acl, repo_filter
from helpdesk.bm25 import BM25Index
from helpdesk.conversation import (
    DEFAULT_SESSIONS_DIR,
    MODES,
    STICKY_SOURCES,
    Conversation,
    Turn,
    format_history,
)
from helpdesk.embedder import DEFAULT_MODEL, Embedder
from helpdesk.generate import ClaudeGenerator, GeneratorError
from helpdesk.hybrid import DEFAULT_CANDIDATES
from helpdesk.index import DEFAULT_PERSIST_DIR, VARIANTS, hits_by_id
from helpdesk.pipeline import respond
from helpdesk.query_transform import (
    ClaudeCLI,
    CondenseQuestion,
    NoopTransformer,
    TransformCache,
    retrieve_transformed,
)
from helpdesk.rerank import CrossEncoderReranker, make_reranker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, default="condense+sticky")
    parser.add_argument("--variant", choices=VARIANTS, default="llm")
    parser.add_argument("--retriever", choices=("vector", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--reranker", choices=("none", "cross-encoder", "claude", "gated-claude"),
                        default="none")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="embedding model")
    parser.add_argument("--generator-model", default="claude-haiku-5-5")
    parser.add_argument("--no-policy-llm", action="store_true", help="decide with rules only")
    parser.add_argument("-k", type=int, default=8)
    parser.add_argument("--resume", help="session id to continue (must be yours)")
    parser.add_argument("--user", default="admin", help="who you are, from data/acl.json")
    parser.add_argument("--debug", action="store_true",
                        help="show the policy decision and gate score (can name hidden projects)")
    args = parser.parse_args()
    users = load_acl()
    if args.user not in users:
        parser.error(f"unknown user {args.user!r}; data/acl.json has {', '.join(sorted(users))}")
    user = users[args.user]
    where = repo_filter(user)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    transformers_logging.set_verbosity_error()
    embedder = Embedder(args.model)
    bm25 = (
        BM25Index.from_chroma(DEFAULT_PERSIST_DIR, args.model, variant=args.variant)
        if args.retriever != "vector"
        else None
    )
    reranker = make_reranker(args.reranker)
    condenser = CondenseQuestion()
    generator = ClaudeGenerator(args.generator_model, name="haiku")
    gate = CrossEncoderReranker()
    policy_llm = None if args.no_policy_llm else ClaudeCLI("claude-haiku-5-5")
    policy_cache = TransformCache()

    def retrieve(query: str, depth: int):
        return retrieve_transformed(
            query, NoopTransformer(), embedder, DEFAULT_PERSIST_DIR, k=depth,
            retriever=args.retriever, candidates=max(DEFAULT_CANDIDATES, depth),
            bm25_index=bm25, variant=args.variant, where=where,
        )

    def new_session() -> Conversation:
        return Conversation(f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}", user=user.name)

    try:
        convo = Conversation.load(args.resume, user=user.name) if args.resume else new_session()
    except (PermissionError, FileNotFoundError):
        parser.error(f"no session {args.resume!r} for {user.name}")  # same message either way
    last_hits = []
    print(f"session {convo.id} | user {user.name} | mode {args.mode} | {args.variant}+{args.retriever} | "
          "/new /history /sources /quit")
    while True:
        try:
            # piped input from PowerShell starts with a byte-order mark
            question = input("\nyou> ").lstrip("﻿").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/new":
            convo, last_hits = new_session(), []
            print(f"new session {convo.id}")
            continue
        if question == "/history":
            print(format_history(convo.turns, summarise=False) or "(empty)")
            continue
        if question == "/sources":
            for number, hit in enumerate(last_hits, start=1):
                m = hit.meta
                print(f"[{number}] {m['repo']}/{m['path']}:{m['line_start']}-{m['line_end']} {m['title']}")
            if not last_hits:
                print("(no sources yet)")
            continue

        sticky = hits_by_id(convo.last_sources()[:STICKY_SOURCES], embedder, DEFAULT_PERSIST_DIR,
                            args.variant)
        try:
            reply = respond(
                question, list(convo.turns), retrieve=retrieve, generator=generator,
                condenser=condenser, gate=gate, policy_llm=policy_llm, sticky_hits=sticky,
                mode=args.mode, k=args.k, reranker=reranker, policy_cache=policy_cache, user=user,
            )
        except GeneratorError as error:
            print(f"error: {error}")
            continue
        last_hits = reply.hits
        if reply.query != question:
            print(f"(searching: {reply.query})")
        if args.debug:
            gate_text = "-" if reply.gate is None else f"{reply.gate:.2f}"
            print(f"(policy: {reply.decision.action} by {reply.decision.by}, gate {gate_text}: "
                  f"{reply.decision.reason})")
        print(f"\nhelpdesk> {reply.turn.text}")
        for citation in reply.turn.citations:
            print(f"  [{citation.index}] {citation.repo}/{citation.path}:{citation.line_start}-{citation.line_end}"
                  f" (last changed {citation.last_changed})")
        convo.turns.append(Turn("user", question))
        convo.turns.append(reply.turn)
        convo.save(DEFAULT_SESSIONS_DIR)


if __name__ == "__main__":
    main()

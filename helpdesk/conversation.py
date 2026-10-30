"""Multi-turn conversations: state, history for prompts, and follow-up retrieval.

A follow-up like "Does it retry?" is only searchable with its history. Four
ways to search a follow-up (``MODES``):

- ``raw``: the follow-up exactly as typed.
- ``concat``: earlier user questions and the follow-up joined together.
- ``condense``: an LLM rewrites history + follow-up into one standalone
  question (``CondenseQuestion`` in query_transform); unrelated follow-ups
  come back unchanged.
- ``condense+sticky``: as condense, plus the previous turn's cited chunks
  added to the candidates, so a drill-down still sees what was just discussed.

Prompts keep the last ``HISTORY_TURNS`` turns in full and squeeze older ones
into one short line each, so the history stays within ``HISTORY_TOKEN_BUDGET``.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from helpdesk.generate import Citation
from helpdesk.index import SearchHit

DEFAULT_SESSIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "sessions"
MODES = ("raw", "concat", "condense", "condense+sticky")
HISTORY_TURNS = 4  # two question/answer exchanges in full
HISTORY_TOKEN_BUDGET = 600
SUMMARY_WORDS = 20
STICKY_SOURCES = 3
_SENTENCE = re.compile(r"(?<=[.!?])\s")
_SAFE_ID = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str
    citations: list[Citation] = field(default_factory=list)
    answered: bool | None = None
    sources: list[str] = field(default_factory=list)  # cited chunk IDs (assistant turns)
    # assistant turns: "answer" | "not_found" | "clarify" | "out_of_scope" | "escalate"
    action: str | None = None
    options: list[str] = field(default_factory=list)  # clarification choices offered


@dataclass
class Conversation:
    id: str
    turns: list[Turn] = field(default_factory=list)
    user: str = ""  # the owner; loading it as anyone else fails

    def save(self, dir: Path = DEFAULT_SESSIONS_DIR) -> Path:
        directory = Path(dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe_id(self.id)}.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, id: str, dir: Path = DEFAULT_SESSIONS_DIR, user: str | None = None) -> "Conversation":
        """``user`` given: the session must belong to them (a session is bound
        to a user; resuming someone else's would show their questions and answers)."""
        payload = json.loads((Path(dir) / f"{_safe_id(id)}.json").read_text(encoding="utf-8"))
        owner = payload.get("user", "")
        if user is not None and owner != user:
            raise PermissionError(f"session {id!r} does not belong to {user!r}")
        return cls(
            user=owner,
            id=payload["id"],
            turns=[
                Turn(
                    role=turn["role"],
                    text=turn["text"],
                    citations=[Citation(**citation) for citation in turn.get("citations", [])],
                    answered=turn.get("answered"),
                    sources=list(turn.get("sources", [])),
                    action=turn.get("action"),
                    options=list(turn.get("options", [])),
                )
                for turn in payload.get("turns", [])
            ],
        )

    def user_questions(self) -> list[str]:
        return [turn.text for turn in self.turns if turn.role == "user"]

    def last_sources(self) -> list[str]:
        for turn in reversed(self.turns):
            if turn.role == "assistant":
                return turn.sources
        return []


def _safe_id(value: str) -> str:
    return _SAFE_ID.sub("-", value).strip("-") or "session"


def estimate_tokens(text: str) -> int:
    """Rough Claude token count (~4 characters per token), for budgeting only."""
    return (len(text) + 3) // 4


def _squeeze(text: str, words: int = SUMMARY_WORDS) -> str:
    first = _SENTENCE.split(text.strip(), maxsplit=1)[0]
    kept = first.split()
    return " ".join(kept[:words]) + (" ..." if len(kept) > words or first != text.strip() else "")


def format_history(
    turns: list[Turn],
    *,
    full_turns: int = HISTORY_TURNS,
    budget: int | None = HISTORY_TOKEN_BUDGET,
    summarise: bool = True,
) -> str:
    """History for a prompt: older turns one squeezed line each, the last
    ``full_turns`` in full. If that is still over ``budget`` tokens, more turns
    are squeezed, newest last. ``summarise=False`` keeps every turn in full."""
    if not turns:
        return ""

    def line(turn: Turn, full: bool) -> str:
        speaker = "User" if turn.role == "user" else "Assistant"
        return f"{speaker}: {turn.text.strip() if full else _squeeze(turn.text)}"

    if not summarise:
        return "\n".join(line(turn, True) for turn in turns)
    keep = min(full_turns, len(turns))
    while True:
        cut = len(turns) - keep
        older = [line(turn, False) for turn in turns[:cut]]
        recent = [line(turn, True) for turn in turns[cut:]]
        text = "\n".join(
            (["Earlier (summarised):", *older, "Recent:"] if older else []) + recent
        )
        if budget is None or estimate_tokens(text) <= budget or keep == 0:
            return text
        keep -= 1


@dataclass(frozen=True)
class GoldenTurn:
    question: str
    expected: list[tuple[str, str]]
    answer_chunks: list[list[tuple[str, str, str]]]


@dataclass(frozen=True)
class GoldenConversation:
    id: str
    type: str
    turns: list[GoldenTurn]


def load_golden_conversations(path: str | Path) -> list[GoldenConversation]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Golden conversations must be a JSON list")
    out: list[GoldenConversation] = []
    seen: set[str] = set()
    for row in payload:
        conversation_id = row.get("id")
        if not isinstance(conversation_id, str) or conversation_id in seen:
            raise ValueError(f"Golden conversation ids must be unique strings: {conversation_id!r}")
        seen.add(conversation_id)
        turns = row.get("turns")
        if not isinstance(turns, list) or not 2 <= len(turns) <= 4:
            raise ValueError(f"{conversation_id}: needs 2-4 turns")
        out.append(
            GoldenConversation(
                id=conversation_id,
                type=str(row.get("type", "")),
                turns=[
                    GoldenTurn(
                        question=turn["question"],
                        expected=[(item["repo"], item["path"]) for item in turn.get("expected", [])],
                        answer_chunks=[
                            [(item["repo"], item["path"], item["title"].casefold()) for item in group]
                            for group in turn.get("answer_chunks", [])
                        ],
                    )
                    for turn in turns
                ],
            )
        )
    return out


def search_query(
    mode: str,
    question: str,
    history: list[Turn],
    condenser: Any | None = None,
) -> str:
    """What to search with for a (follow-up) question in the given mode."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {', '.join(MODES)}")
    if mode == "raw" or not history:
        return question
    if mode == "concat":
        return " ".join([*(t.text for t in history if t.role == "user"), question])
    if condenser is None:
        raise ValueError(f"mode {mode!r} needs a condenser")
    return condenser.condense(question, format_history(history))


def with_sticky(
    hits: list[SearchHit],
    sticky: list[SearchHit],
    depth: int,
) -> list[SearchHit]:
    """Keep the top hits but make room at the end for the previous turn's
    sources (those not already retrieved)."""
    present = {hit.chunk_id for hit in hits[:depth]}
    extra = [hit for hit in sticky if hit.chunk_id not in present]
    return hits[: max(0, depth - len(extra))] + extra if extra else hits[:depth]


def retrieve_turn(
    mode: str,
    question: str,
    history: list[Turn],
    retrieve: Callable[[str, int], list[SearchHit]],
    *,
    depth: int,
    condenser: Any | None = None,
    sticky_hits: list[SearchHit] | None = None,
) -> tuple[str, list[SearchHit]]:
    """(search query, hits) for one turn."""
    query = search_query(mode, question, history, condenser)
    hits = retrieve(query, depth)
    if mode == "condense+sticky" and sticky_hits:
        hits = with_sticky(hits, sticky_hits, depth)
    return query, hits

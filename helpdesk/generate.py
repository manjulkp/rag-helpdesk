"""Citation-checked answer generation over retrieved helpdesk chunks."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from helpdesk.index import SearchHit

NOT_FOUND = "I couldn't find that in the docs."
MARKER = re.compile(r"\[(\d+)\]")
SYSTEM_PROMPT = """\
Answer questions using only the numbered sources provided. Do not use outside
knowledge or invent facts, paths, identifiers, or commands. Cite every factual
claim with one or more source markers such as [1]. If the sources do not answer
the question, return answered=false and say "I couldn't find that in the docs."
A conversation so far may be given: use it only to work out what the question
refers to, never as a source of facts, and never cite it. Answer only the
current question; do not comment on, re-check, or correct earlier answers.
Each source shows when it last changed. If sources disagree on a fact, say so:
cite both with their dates, prefer the newer one and say that is why. If the
dates are unknown or equal, say you can't tell which is current. Never
silently pick one.
Return only the requested JSON object.
"""
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answered": {"type": "boolean"},
        "answer": {"type": "string"},
        "cited": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answered", "answer", "cited"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Citation:
    index: int
    repo: str
    path: str
    line_start: int
    line_end: int
    # last commit date of the file, or "unknown" when it has no git history
    last_changed: str = "unknown"


@dataclass
class Answer:
    answered: bool
    answer: str
    citations: list[Citation]
    generator: str
    latency_seconds: float
    cost_usd: float | None = None
    score: float | None = None
    dropped_citations: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None  # input tokens the model saw (incl. cached)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Generator(Protocol):
    name: str

    def answer(self, question: str, hits: list[SearchHit], history: str = "") -> Answer: ...


class GeneratorError(RuntimeError):
    """Raised when an answer generator fails or returns malformed output."""


def markers(text: str) -> list[int]:
    """Return source-marker numbers in text, preserving their order."""
    return [int(value) for value in MARKER.findall(text)]


def location(index: int, hit: SearchHit) -> Citation:
    """Convert a retrieved hit's metadata to a citation."""
    return Citation(
        index=index,
        repo=str(hit.meta.get("repo", "")),
        path=str(hit.meta.get("path", "")),
        line_start=int(hit.meta.get("line_start", 1)),
        line_end=int(hit.meta.get("line_end", hit.meta.get("line_start", 1))),
        # untracked files show "unknown", not the file's modified time: copying or
        # checking out a file resets that, so it would look like freshness it isn't
        last_changed=str(hit.meta.get("last_commit") or "unknown"),
    )


def _checked_answer(
    *,
    text: str,
    answered: bool,
    hits: list[SearchHit],
    generator: str,
    latency_seconds: float,
    cost_usd: float | None = None,
    score: float | None = None,
    warnings: list[str] | None = None,
) -> Answer:
    if not answered:
        return Answer(
            answered=False,
            answer=NOT_FOUND,
            citations=[],
            generator=generator,
            latency_seconds=latency_seconds,
            cost_usd=cost_usd,
            score=score,
            warnings=list(warnings or []),
        )

    valid = set(range(1, len(hits) + 1))
    used: list[int] = []
    dropped: list[int] = []

    def keep_valid_marker(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if number not in valid:
            dropped.append(number)
            return ""
        if number not in used:
            used.append(number)
        return match.group(0)

    clean_text = MARKER.sub(keep_valid_marker, text).strip()
    answer_warnings = list(warnings or [])
    if dropped:
        answer_warnings.append(
            f"Removed citations outside the retrieved source range: {sorted(set(dropped))}"
        )
    if not used:
        answer_warnings.append("Answer had no valid source citations; treated as not found.")
        return Answer(
            answered=False,
            answer=NOT_FOUND,
            citations=[],
            generator=generator,
            latency_seconds=latency_seconds,
            cost_usd=cost_usd,
            score=score,
            dropped_citations=sorted(set(dropped)),
            warnings=answer_warnings,
        )
    return Answer(
        answered=True,
        answer=clean_text,
        citations=[location(number, hits[number - 1]) for number in used],
        generator=generator,
        latency_seconds=latency_seconds,
        cost_usd=cost_usd,
        score=score,
        dropped_citations=sorted(set(dropped)),
        warnings=answer_warnings,
    )


def build_user_message(question: str, hits: list[SearchHit], history: str = "") -> str:
    """Conversation first, then the sources, then the question last, so the
    question and the evidence sit at the ends of the prompt where long-context
    models attend best (Liu et al. 2023, Lost in the Middle)."""
    source_blocks = []
    for index, hit in enumerate(hits, start=1):
        source = location(index, hit)
        lines = (
            f":{source.line_start}-{source.line_end}"
            if source.line_start != source.line_end
            else f":{source.line_start}"
        )
        source_blocks.append(
            f"[{index}] {source.repo}/{source.path}{lines} (last changed {source.last_changed})\n"
            f"{hit.text.strip()}"
        )
    sources = "\n\n".join(source_blocks)
    conversation = f"CONVERSATION SO FAR:\n{history.strip()}\n\n" if history.strip() else ""
    return f"{conversation}SOURCES:\n{sources}\n\nQUESTION:\n{question}"


class ClaudeGenerator:
    """Use the installed Claude CLI in a temporary, tool-free working folder."""

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        *,
        name: str = "haiku",
        timeout_seconds: float = 120,
        executable: str = "claude",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        self.name = name
        self.timeout_seconds = timeout_seconds
        self.executable = executable

    def answer(self, question: str, hits: list[SearchHit], history: str = "") -> Answer:
        if not hits:
            return _checked_answer(
                text="",
                answered=False,
                hits=hits,
                generator=self.name,
                latency_seconds=0.0,
            )

        command = [
            self.executable,
            "-p",
            "--model",
            self.model,
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(ANSWER_SCHEMA),
            "--system-prompt",
            SYSTEM_PROMPT,
            "--tools",
            "",
        ]
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="helpdesk-claude-") as cwd:
                completed = subprocess.run(
                    command,
                    input=build_user_message(question, hits, history),
                    capture_output=True,
                    text=True,
                    # not the Windows code page: a non-cp1252 character in the
                    # prompt would otherwise lose the whole input
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    cwd=cwd,
                    check=False,
                )
        except FileNotFoundError as error:
            raise GeneratorError(
                f"Claude CLI executable {self.executable!r} was not found"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise GeneratorError(
                f"Claude CLI timed out after {self.timeout_seconds:g} seconds"
            ) from error
        latency = time.perf_counter() - started
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-1000:]
            raise GeneratorError(
                f"Claude CLI exited with {completed.returncode}: {detail}"
            )
        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise GeneratorError(
                f"Claude CLI returned invalid JSON: {completed.stdout[:300]!r}"
            ) from error
        if not isinstance(envelope, dict):
            raise GeneratorError("Claude CLI JSON response must be an object")
        if envelope.get("is_error"):
            raise GeneratorError(f"Claude reported an error: {envelope!r}")
        payload = envelope.get("structured_output", envelope.get("result"))
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as error:
                raise GeneratorError("Claude result field was not valid JSON") from error
        if not isinstance(payload, dict):
            raise GeneratorError("Claude response did not contain structured output")
        if not isinstance(payload.get("answered"), bool) or not isinstance(
            payload.get("answer"), str
        ):
            raise GeneratorError("Claude response is missing answered/answer fields")
        cited = payload.get("cited", [])
        if not isinstance(cited, list) or any(
            not isinstance(value, int) or isinstance(value, bool) for value in cited
        ):
            raise GeneratorError("Claude response cited field must be an integer list")
        warnings: list[str] = []
        answer_markers = set(markers(payload["answer"]))
        declared_markers = set(cited)
        if declared_markers != answer_markers:
            warnings.append("Claude's cited list did not match markers in its answer.")
        cost = envelope.get("total_cost_usd")
        if cost is not None and not isinstance(cost, (float, int)):
            raise GeneratorError("Claude total_cost_usd must be numeric")
        usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else {}
        prompt_tokens = sum(
            int(usage.get(key) or 0)
            for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        result = _checked_answer(
            text=payload["answer"],
            answered=payload["answered"],
            hits=hits,
            generator=self.name,
            latency_seconds=latency,
            cost_usd=float(cost) if cost is not None else None,
            warnings=warnings,
        )
        result.prompt_tokens = prompt_tokens or None
        return result


class ExtractiveGenerator:
    """Return the highest-scoring answer span from retrieved passages."""

    def __init__(
        self,
        model: str = "deepset/roberta-base-squad2",
        *,
        threshold: float = 0.05,
        reader: Any | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.model = model
        self.name = "extractive"
        self.threshold = threshold
        self._reader = reader

    def _load_reader(self) -> Any:
        if self._reader is None:
            from transformers import pipeline

            self._reader = pipeline(
                "question-answering",
                model=self.model,
                device=-1,
            )
        return self._reader

    def answer(self, question: str, hits: list[SearchHit], history: str = "") -> Answer:
        # an extractive reader can't use history; it answers the question as typed
        if not hits:
            return _checked_answer(
                text="",
                answered=False,
                hits=hits,
                generator=self.name,
                latency_seconds=0.0,
            )

        started = time.perf_counter()
        reader = self._load_reader()
        best: tuple[float, int, str] | None = None
        for index, hit in enumerate(hits, start=1):
            prediction = reader(question=question, context=hit.text)
            if not isinstance(prediction, dict):
                raise GeneratorError("Extractive reader must return a mapping")
            text = prediction.get("answer")
            score = prediction.get("score")
            if not isinstance(text, str) or not isinstance(score, (float, int)):
                raise GeneratorError("Extractive reader returned invalid answer/score")
            candidate = (float(score), index, text.strip())
            if best is None or candidate[0] > best[0]:
                best = candidate
        latency = time.perf_counter() - started
        if best is None or best[0] < self.threshold or not best[2]:
            return _checked_answer(
                text="",
                answered=False,
                hits=hits,
                generator=self.name,
                latency_seconds=latency,
                score=best[0] if best else None,
            )
        score, index, text = best
        return _checked_answer(
            text=f"{text} [{index}]",
            answered=True,
            hits=hits,
            generator=self.name,
            latency_seconds=latency,
            score=score,
        )

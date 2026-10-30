from __future__ import annotations

from pathlib import Path

import pytest

from helpdesk.bm25 import BM25Index
from helpdesk.chunker import DEFAULT_MAX_TOKENS, Chunk
from helpdesk.contextualize import (
    CONTEXT_TOKENS,
    ContextCache,
    LLMContextualizer,
    content_hash,
    contexts_for,
    fit_context,
    header_context,
)
from helpdesk.generate import GeneratorError
from helpdesk.hybrid import hybrid_search
from helpdesk.index import (
    IndexModelMismatch,
    build_index,
    collection_name,
    contextual_text,
    search,
)
from tests.test_index import CHUNKS, FakeEmbedder


def words(text: str) -> int:  # token counter for tests
    return len(text.split())


def chunk(
    cid: str = "c1",
    title: str = "Legacy Provenance > FeedClosePrice - full trace",
    text: str = "tier 1 lookup body",
    kind: str = "doc",
    path: str = "docs/legacy-provenance.md",
) -> Chunk:
    return Chunk(
        id=cid, text=text, repo="commodities-pricer", path=path, kind=kind, title=title,
        line_start=10, line_end=20, last_commit="2026-01-01", n_tokens=words(text),
    )


# 1. A chunk's header names its repo, path and heading path
def test_header_of_a_chunk() -> None:
    assert header_context(chunk()) == (
        "commodities-pricer / docs/legacy-provenance.md / "
        "Legacy Provenance > FeedClosePrice - full trace"
    )
    test = header_context(
        chunk(
            title="TestInstrumentConfig.test_window_is_naive_utc",
            kind="test",
            path="tests/integration/test_config_engine.py",
        )
    )
    assert test.endswith("/ test: TestInstrumentConfig.test_window_is_naive_utc")


# 2. Context + chunk stays inside the model's budget
def test_context_plus_chunk_fits_the_budget() -> None:
    long_ctx = " ".join(f"w{i}" for i in range(500))
    ctx = fit_context(long_ctx, words)
    assert words(ctx) == CONTEXT_TOKENS and long_ctx.startswith(ctx)
    body = " ".join(["x"] * DEFAULT_MAX_TOKENS)  # a chunk at the chunker's maximum
    assert words(contextual_text(ctx, body)) <= 510  # bge: 512 minus [CLS] and [SEP]
    assert fit_context("short context here", words) == "short context here"
    out = contexts_for("header", [chunk()], words)
    assert words(out["c1"]) <= CONTEXT_TOKENS
    assert contexts_for("plain", [chunk()], words) is None


@pytest.fixture
def header_index(tmp_path: Path) -> Path:
    contexts = {c.id: "zebra quartz context" for c in CHUNKS[:1]}  # words in no chunk
    build_index(CHUNKS, FakeEmbedder(), tmp_path, variant="header", contexts=contexts)
    return tmp_path


def test_variants_get_their_own_collection_names() -> None:
    assert collection_name("fake/bow-64") == "helpdesk-fake-bow-64"  # plain unchanged
    assert collection_name("fake/bow-64", "header") == "helpdesk-fake-bow-64-header"
    assert collection_name("fake/bow-64", "llm") == "helpdesk-fake-bow-64-llm"
    with pytest.raises(ValueError):
        collection_name("fake/bow-64", "fancy")


# 3. A header index searched with a plain setup is refused, by every retriever
def test_variant_mismatch_raises(header_index: Path) -> None:
    with pytest.raises(IndexModelMismatch, match="header"):
        search("mongo", FakeEmbedder(), header_index, variant="plain")
    with pytest.raises(IndexModelMismatch):
        BM25Index.from_chroma(header_index, "fake/bow-64")  # plain by default
    bm25 = BM25Index.from_chroma(header_index, "fake/bow-64", variant="header")
    with pytest.raises(IndexModelMismatch):
        hybrid_search("mongo", FakeEmbedder(), bm25, header_index, variant="plain")


def test_build_requires_contexts_exactly_for_contextual_variants(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_index(CHUNKS, FakeEmbedder(), tmp_path, variant="header")
    with pytest.raises(ValueError):
        build_index(CHUNKS, FakeEmbedder(), tmp_path, contexts={})


# 4. Hits carry the original chunk text; the context only changes what matches
def test_hit_text_is_the_original_chunk(header_index: Path) -> None:
    hits = search("zebra quartz", FakeEmbedder(), header_index, k=1, variant="header")
    assert hits[0].chunk_id == CHUNKS[0].id and hits[0].text == CHUNKS[0].text
    assert hits[0].meta["context"] == "zebra quartz context"
    bm25 = BM25Index.from_chroma(header_index, "fake/bow-64", variant="header")
    kw = bm25.search("zebra", k=3)
    assert [h.chunk_id for h in kw] == [CHUNKS[0].id] and kw[0].text == CHUNKS[0].text
    for h in hybrid_search("zebra", FakeEmbedder(), bm25, header_index, k=5,
                           candidates=5, variant="header"):
        assert h.text == next(c.text for c in CHUNKS if c.id == h.chunk_id)


class FakeLLM:
    def __init__(self, fail_after: int | None = None) -> None:
        self.prompts: list[str] = []
        self.systems: list[str] = []
        self.fail_after = fail_after

    def __call__(self, system: str, prompt: str):
        if self.fail_after is not None and len(self.prompts) >= self.fail_after:
            raise RuntimeError("simulated crash")
        self.systems.append(system)
        self.prompts.append(prompt)
        return f"context {len(self.prompts)}", 0.1, 0.001, {"input_tokens": 10}


def docs_for(chunks: list[Chunk]) -> dict[tuple[str, str], str]:
    return {(c.repo, c.path): f"whole document of {c.path}" for c in chunks}


# 5. A cache hit makes no call; changed chunk text does
def test_cache_hit_makes_no_call_and_changed_text_does(tmp_path: Path) -> None:
    cache = tmp_path / "ctx.jsonl"
    cs = [chunk(cid="a"), chunk(cid="b", title="Other")]
    first = FakeLLM()
    out = LLMContextualizer(docs_for(cs), cache=ContextCache(cache), call=first).contextualize(cs)
    assert len(first.prompts) == 2 and out["a"] and out["b"]
    # the document is in the (cacheable) system prompt, the chunk in the user message
    assert first.systems[0].endswith("<document>\nwhole document of docs/legacy-provenance.md\n</document>")
    assert first.systems[0] == first.systems[1]
    assert "<chunk>\ntier 1" in first.prompts[0] and "<document>" not in first.prompts[0]

    again = FakeLLM()
    llm = LLMContextualizer(docs_for(cs), cache=ContextCache(cache), call=again)
    assert llm.contextualize(cs) == out and again.prompts == [] and llm.cache_hits == 2

    edited = [chunk(cid="a", text="tier 1 lookup body, edited"), cs[1]]
    assert content_hash(edited[0]) != content_hash(cs[0])
    third = FakeLLM()
    LLMContextualizer(docs_for(edited), cache=ContextCache(cache), call=third).contextualize(edited)
    assert len(third.prompts) == 1 and "edited" in third.prompts[0]


# 6. After a crash, a restart only does the chunks that weren't finished
def test_restart_after_crash_does_only_the_rest(tmp_path: Path) -> None:
    cache = tmp_path / "ctx.jsonl"
    cs = [chunk(cid=f"c{i}", title=f"T{i}") for i in range(5)]
    with pytest.raises(RuntimeError):
        LLMContextualizer(
            docs_for(cs), cache=ContextCache(cache), call=FakeLLM(fail_after=3), workers=1
        ).contextualize(cs)
    with cache.open("a", encoding="utf-8") as stream:
        stream.write('{"key": "torn')  # killed mid-write
    rerun = FakeLLM()
    out = LLMContextualizer(docs_for(cs), cache=ContextCache(cache), call=rerun, workers=1).contextualize(cs)
    assert len(rerun.prompts) == 2 and len(out) == 5
    assert len(ContextCache(cache)) == 5


def test_first_chunk_of_each_document_runs_before_the_rest() -> None:
    cs = [
        chunk(cid="a1", path="a.md", title="A1"),
        chunk(cid="a2", path="a.md", title="A2"),
        chunk(cid="b1", path="b.md", title="B1"),
    ]
    llm = FakeLLM()
    LLMContextualizer(docs_for(cs), cache=ContextCache(None), call=llm, workers=1).contextualize(cs)
    order = ["a.md" if "a.md" in s else "b.md" for s in llm.systems]
    assert order == ["a.md", "b.md", "a.md"]


def test_failed_call_is_not_cached_and_chunk_goes_plain() -> None:
    def failing(system: str, prompt: str):
        raise GeneratorError("timeout")

    cache = ContextCache(None)
    llm = LLMContextualizer(docs_for([chunk()]), cache=cache, call=failing)
    assert contexts_for("llm", [chunk()], words, contextualizer=llm) == {}
    assert llm.failures == ["c1"] and len(cache) == 0

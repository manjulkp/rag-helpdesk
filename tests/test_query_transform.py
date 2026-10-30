from __future__ import annotations

import json

import pytest

from helpdesk.query_transform import (
    DecomposeTransformer,
    HyDETransformer,
    MultiQueryTransformer,
    Query,
    RewriteTransformer,
    TransformCache,
    identifiers_in,
    keep_identifiers,
    transformed_search,
)
from helpdesk.index import SearchHit


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {"queries": ["retry backoff policy"]}
        self.error = error
        self.calls = []

    def generate(self, prompt, *, system_prompt, schema):
        self.calls.append((prompt, system_prompt, schema))
        if self.error:
            raise self.error
        return self.payload


def hit(chunk_id: str, score: float = 1.0) -> SearchHit:
    return SearchHit(
        chunk_id,
        score,
        f"text {chunk_id}",
        {"repo": "repo", "path": f"{chunk_id}.md", "title": chunk_id},
    )


def test_identifiers_are_preserved_exactly_in_rewritten_queries() -> None:
    question = "Why does ITEM_DOWN switch currentSource?"
    assert identifiers_in(question) == ["ITEM_DOWN", "currentSource"]
    assert keep_identifiers(question, "provider failover") == (
        "provider failover ITEM_DOWN currentSource"
    )


def test_original_question_is_first_and_multiquery_is_bounded(tmp_path) -> None:
    client = FakeClient({"queries": ["first variant", "second variant", "third variant", "extra"]})
    transformer = MultiQueryTransformer(client, cache=TransformCache(tmp_path / "cache.json"))

    queries = transformer.transform("What is retry?")

    assert [query.text for query in queries] == [
        "What is retry?",
        "first variant",
        "second variant",
        "third variant",
    ]
    assert [query.kind for query in queries] == [
        "question",
        "query",
        "query",
        "query",
    ]


def test_hyde_keeps_original_for_keyword_search_and_caches_llm_output(tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    client = FakeClient({"queries": ["A passage describes exponential retries."]})
    cache = TransformCache(cache_path)
    transformer = HyDETransformer(client, cache=cache)

    first = transformer.transform("How does retry work?")
    second = HyDETransformer(FakeClient(), cache=TransformCache(cache_path)).transform(
        "How does retry work?"
    )

    assert first[0] == Query(
        "How does retry work?", kind="question", keyword_text="How does retry work?"
    )
    assert first[1].kind == "passage"
    assert first[1].keyword_text == "How does retry work?"
    assert second == first
    assert len(client.calls) == 1
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(payload) == 1


def test_transform_failure_returns_original_query(caplog, tmp_path) -> None:
    from helpdesk.generate import GeneratorError

    transformer = RewriteTransformer(
        FakeClient(error=GeneratorError("not authenticated")),
        cache=TransformCache(tmp_path / "cache.json"),
    )

    assert transformer.transform("What is retry?") == [
        Query("What is retry?", kind="question", keyword_text="What is retry?")
    ]
    assert "using original question" in caplog.text
    assert transformer.fallbacks == 1


def test_decompose_retains_original_and_tags_subquestions(tmp_path) -> None:
    transformer = DecomposeTransformer(
        FakeClient({"queries": ["How is retry configured?", "How is backoff applied?"]}),
        cache=TransformCache(tmp_path / "cache.json"),
    )

    queries = transformer.transform("How is retry configured and applied?")

    assert queries[0].kind == "question"
    assert [query.kind for query in queries[1:]] == ["subquestion", "subquestion"]


def test_bad_transform_payload_falls_back_to_original(tmp_path) -> None:
    transformer = RewriteTransformer(
        FakeClient({"queries": "not a list"}),
        cache=TransformCache(tmp_path / "cache.json"),
    )
    assert transformer.transform("Question?")[0].text == "Question?"


def test_transformed_search_fuses_unique_candidates_and_reranks() -> None:
    transformer = MultiQueryTransformer(
        FakeClient({"queries": ["alternate query"]}),
        cache=TransformCache(None),
    )
    calls = []

    def retrieve(query: Query, depth: int) -> list[SearchHit]:
        calls.append((query, depth))
        if query.text == "Question?":
            return [hit("shared"), hit("original")]
        return [hit("shared"), hit("alternate")]

    class Reverse:
        def rerank(self, query, hits):
            return list(reversed(hits))

    results = transformed_search(
        "Question?", transformer, retrieve, k=2, candidates=4, reranker=Reverse()
    )

    assert [call[1] for call in calls] == [4, 4]
    assert {item.chunk_id for item in results} == {"alternate", "original"}
    assert results[0].chunk_id == "original"


def test_hyde_uses_original_question_for_bm25_and_passage_for_vectors(monkeypatch) -> None:
    import helpdesk.query_transform as module

    transformer = HyDETransformer(
        FakeClient({"queries": ["A passage about retry policy."]}),
        cache=TransformCache(None),
    )
    vector_queries = []
    query_vectors = []
    bm25_queries = []
    hybrid_queries = []

    class FakeEmbedder:
        def embed_documents(self, texts):
            return [f"doc-vector:{text}" for text in texts]

    def fake_search(query, *_args, query_vector=None, **_kwargs):
        vector_queries.append(query)
        query_vectors.append(query_vector)
        return [hit(f"vector-{len(vector_queries)}")]

    class FakeBM25:
        def search(self, query, **_kwargs):
            bm25_queries.append(query)
            return [hit(f"bm25-{len(bm25_queries)}")]

    def fake_hybrid(query, *_args, vector_hits=None, bm25_hits=None, **_kwargs):
        hybrid_queries.append(query)
        assert vector_hits and bm25_hits
        return vector_hits + bm25_hits

    monkeypatch.setattr(module, "search", fake_search)
    monkeypatch.setattr(module, "hybrid_search", fake_hybrid)
    results = module.retrieve_transformed(
        "How does retry work?",
        transformer,
        embedder=FakeEmbedder(),
        retriever="hybrid",
        k=2,
        candidates=2,
        bm25_index=FakeBM25(),
    )

    assert vector_queries == [
        "How does retry work?",
        "A passage about retry policy.",
    ]
    # the question is embedded as a query; the HyDE passage as a document
    assert query_vectors == [None, "doc-vector:A passage about retry policy."]
    assert bm25_queries == ["How does retry work?", "How does retry work?"]
    assert hybrid_queries == vector_queries
    assert len(results) == 2


def test_cache_key_includes_transform_and_prompt_version(tmp_path) -> None:
    cache = TransformCache(tmp_path / "cache.json")
    cache.put("rewrite", "v1", "same question", {"queries": ["one"]})

    assert cache.get("rewrite", "v1", "same question") == {"queries": ["one"]}
    assert cache.get("multiquery", "v1", "same question") is None
    assert cache.get("rewrite", "v2", "same question") is None


def test_cache_rejects_non_object_json(tmp_path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        TransformCache(path).get("rewrite", "v1", "question")

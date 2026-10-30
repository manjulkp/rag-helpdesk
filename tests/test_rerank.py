from __future__ import annotations

import pytest

from helpdesk.generate import GeneratorError
from helpdesk.index import SearchHit
from helpdesk.rerank import (
    ClaudeReranker,
    CrossEncoderReranker,
    GatedReranker,
    not_found,
)


def hit(chunk_id: str, score: float = 0.5) -> SearchHit:
    return SearchHit(
        chunk_id,
        score,
        f"passage {chunk_id}",
        {"repo": "repo", "path": f"{chunk_id}.md", "title": chunk_id},
    )


def test_cross_encoder_rescores_and_sorts_candidates() -> None:
    reranker = CrossEncoderReranker(
        scorer=lambda pairs: [0.1 if "passage a" in pair[1] else 2.0 for pair in pairs]
    )

    ranked = reranker.rerank("question", [hit("a"), hit("b")])

    assert [item.chunk_id for item in ranked] == ["b", "a"]
    assert ranked[0].meta["retrieval_score"] == 0.5
    assert ranked[0].meta["rerank_score"] == 2.0
    assert ranked[0].score == 2.0


def test_cross_encoder_rejects_wrong_number_of_scores() -> None:
    with pytest.raises(ValueError, match="different number"):
        CrossEncoderReranker(scorer=lambda _pairs: []).rerank("q", [hit("a")])


def test_claude_reranker_discards_invalid_duplicate_ids_and_keeps_tail() -> None:
    class Client:
        def generate(self, *_args, **_kwargs):
            return {"ranking": [3, 0, 1, 3, 99]}

    ranked = ClaudeReranker(Client()).rerank("question", [hit("a"), hit("b"), hit("c")])
    assert [item.chunk_id for item in ranked] == ["c", "a", "b"]


def test_claude_reranker_surfaces_malformed_output() -> None:
    class Client:
        def generate(self, *_args, **_kwargs):
            return {"ranking": "first"}

    with pytest.raises(GeneratorError, match="integer candidate IDs"):
        ClaudeReranker(Client()).rerank("question", [hit("a")])


def test_not_found_respects_empty_results_and_cutoff() -> None:
    assert not_found([])
    assert not_found([hit("low", 0.09)], cutoff=0.1)
    assert not not_found([hit("high", 0.1)], cutoff=0.1)
    assert not not_found([hit("any")])


def test_gated_reranker_skips_expensive_stage_below_cutoff() -> None:
    class Gate:
        def rerank(self, _query, hits):
            return [hit(item.chunk_id, -1.0) for item in hits]

    class Expensive:
        calls = 0

        def rerank(self, _query, hits):
            self.calls += 1
            return hits

    expensive = Expensive()
    gated = GatedReranker(Gate(), expensive, cutoff=0.0)

    output = gated.rerank("question", [hit("a")])

    assert output[0].chunk_id == "a"
    assert gated.skipped == 1
    assert expensive.calls == 0
    assert not_found(output, cutoff=0.0)

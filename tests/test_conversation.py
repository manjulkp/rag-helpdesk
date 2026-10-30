from __future__ import annotations

from pathlib import Path

from helpdesk.conversation import (
    Conversation,
    Turn,
    estimate_tokens,
    format_history,
    load_golden_conversations,
    retrieve_turn,
    with_sticky,
)
from helpdesk.generate import Citation, build_user_message
from helpdesk.index import SearchHit
from helpdesk.query_transform import CondenseQuestion, TransformCache


class FakeLLM:
    def __init__(self, standalone: str = "", uses_history: bool = True) -> None:
        self.standalone = standalone
        self.uses_history = uses_history
        self.prompts: list[str] = []

    def generate(self, prompt, *, system_prompt, schema):
        self.prompts.append(prompt)
        return {"uses_history": self.uses_history, "standalone": self.standalone}


def hit(chunk_id: str) -> SearchHit:
    return SearchHit(chunk_id, 1.0, f"text {chunk_id}", {"repo": "r", "path": f"{chunk_id}.md",
                                                         "title": chunk_id, "line_start": 1, "line_end": 2})


# 1. The first turn makes no condense call
def test_first_turn_makes_no_condense_call() -> None:
    llm = FakeLLM("should not be used")
    condenser = CondenseQuestion(llm, cache=TransformCache(None))
    assert condenser.condense("How does the persistence layer write to Mongo?", "") == (
        "How does the persistence layer write to Mongo?"
    )
    query, _ = retrieve_turn("condense", "How does it work?", [], lambda q, d: [], depth=5,
                             condenser=condenser)
    assert query == "How does it work?" and llm.prompts == []


# 2. A condensed question keeps an identifier from an earlier turn
def test_condensed_question_keeps_identifier_from_earlier_turn() -> None:
    history = "User: Why does ITEM_DOWN switch currentSource to FC.U?"
    llm = FakeLLM("Does item_down retry before switching currentsource to fc.u?")
    condensed = CondenseQuestion(llm, cache=TransformCache(None)).condense("Does it retry?", history)
    assert "ITEM_DOWN" in condensed and "currentSource" in condensed and "FC.U" in condensed
    assert "Does it retry?" in llm.prompts[0] and history in llm.prompts[0]


def test_topic_switch_returns_follow_up_unchanged() -> None:
    llm = FakeLLM("How does Mongo persistence compute the news score?", uses_history=False)
    out = CondenseQuestion(llm, cache=TransformCache(None)).condense(
        "How is the news score computed?", "User: How does the persistence layer write to Mongo?"
    )
    assert out == "How is the news score computed?"


def test_condense_failure_searches_the_follow_up_as_typed() -> None:
    class Broken:
        def generate(self, *args, **kwargs):
            raise ValueError("bad output")

    condenser = CondenseQuestion(Broken(), cache=TransformCache(None))
    assert condenser.condense("Does it retry?", "User: Mongo?") == "Does it retry?"
    assert condenser.fallbacks == 1


# 3. A conversation saved and loaded back is identical
def test_conversation_save_and_load_round_trip(tmp_path: Path) -> None:
    convo = Conversation("s1", [
        Turn("user", "How does the persistence layer write to Mongo?"),
        Turn("assistant", "It upserts snapshots [1].",
             citations=[Citation(1, "r", "docs/mongo.md", 3, 9)], answered=True, sources=["c1"]),
    ])
    convo.save(tmp_path)
    assert Conversation.load("s1", tmp_path) == convo
    assert convo.last_sources() == ["c1"]


# 4. History over the budget gets summarised, and the prompt stays under the limit
def test_long_history_is_summarised_under_the_budget() -> None:
    turns = []
    for i in range(10):
        turns.append(Turn("user", f"Question {i} about the persistence layer and its retry policy?"))
        turns.append(Turn("assistant", f"Answer {i}. " + "Long explanation sentence. " * 60))
    full = format_history(turns, summarise=False)
    short = format_history(turns, budget=600)
    assert estimate_tokens(short) <= 600 < estimate_tokens(full)
    assert "Earlier (summarised):" in short and "Question 0" in short  # old turns kept as lines
    assert short.rstrip().endswith(turns[-1].text.strip()) or "Answer 9" in short
    prompt = build_user_message("Does it retry?", [hit("a")], short)
    assert prompt.startswith("CONVERSATION SO FAR:") and prompt.rstrip().endswith("Does it retry?")


# 5. The condense cache key changes when the history changes
def test_condense_cache_key_includes_history() -> None:
    cache = TransformCache(None)
    llm = FakeLLM("Does the persistence layer retry?")
    condenser = CondenseQuestion(llm, cache=cache)
    condenser.condense("Does it retry?", "User: How does the persistence layer write to Mongo?")
    condenser.condense("Does it retry?", "User: How does the persistence layer write to Mongo?")
    assert len(llm.prompts) == 1  # same history: cached
    condenser.condense("Does it retry?", "User: How does the websocket client reconnect?")
    assert len(llm.prompts) == 2  # new history: new call


def test_modes_build_the_right_search_query() -> None:
    history = [Turn("user", "How does the persistence layer write to Mongo?"), Turn("assistant", "It upserts.")]
    seen = []

    def retrieve(query, depth):
        seen.append(query)
        return [hit("x"), hit("y"), hit("z")]

    retrieve_turn("raw", "Does it retry?", history, retrieve, depth=3)
    retrieve_turn("concat", "Does it retry?", history, retrieve, depth=3)
    condenser = CondenseQuestion(FakeLLM("Does the Mongo persistence layer retry?"), cache=TransformCache(None))
    _, hits = retrieve_turn("condense+sticky", "Does it retry?", history, retrieve, depth=3,
                            condenser=condenser, sticky_hits=[hit("prev")])
    assert seen == [
        "Does it retry?",
        "How does the persistence layer write to Mongo? Does it retry?",
        "Does the Mongo persistence layer retry?",
    ]
    assert [h.chunk_id for h in hits] == ["x", "y", "prev"]  # sticky source takes the last slot
    assert [h.chunk_id for h in with_sticky([hit("a"), hit("prev")], [hit("prev")], 2)] == ["a", "prev"]


def test_golden_conversations_load() -> None:
    convos = load_golden_conversations(Path(__file__).resolve().parents[1] / "data" / "golden_conversations.json")
    assert len(convos) == 12
    counts = {}
    for convo in convos:
        counts[convo.type] = counts.get(convo.type, 0) + 1
        assert 2 <= len(convo.turns) <= 4
    assert counts == {"pronoun": 3, "ellipsis": 2, "drilldown": 2, "topic_switch": 2,
                      "correction": 2, "unanswerable": 1}

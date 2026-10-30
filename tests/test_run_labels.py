from __future__ import annotations

from scripts.eval_answers import run_label
from scripts.evaluate import run_name


def test_run_labels_include_retriever_transform_and_reranker() -> None:
    assert run_name("bge-small", "vector", "noop", "none") == "bge-small"
    assert run_name("bge-small", "hybrid", "hyde", "none") == "bge-small+hybrid+hyde"
    assert run_name("bge-small", "hybrid", "noop", "cross-encoder") == (
        "bge-small+hybrid+cross-encoder"
    )
    assert run_label("haiku", "hybrid", "hyde", "none") == "haiku+hybrid+hyde"
    assert run_label("haiku", "vector", "noop", "none") == "haiku"
    assert run_name("bge-small", "hybrid", "noop", "none", "header") == "bge-small@header+hybrid"
    assert run_label("haiku", "hybrid", "noop", "none", "header") == "haiku@header+hybrid"

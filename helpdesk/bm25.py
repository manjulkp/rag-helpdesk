"""Okapi BM25 search over the documents and metadata in a Chroma collection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from helpdesk.index import (
    DEFAULT_PERSIST_DIR,
    SearchHit,
    get_collection,
    indexed_text,
)

DEFAULT_BM25_K1 = 1.5
DEFAULT_BM25_B = 0.75

_TOKEN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+"
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9]+(?:[_./-][A-Za-z0-9]+)*")
STOP_WORDS = frozenset(
    "a an and are as at be been being by can could did do does for from had "
    "has have how i in into is it its may might of on or our should the their "
    "there these this those to was were what when where which who why will "
    "with would you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase searchable words while preserving code identifiers.

    Compound identifiers such as ``ITEM_DOWN`` and ``currentSource`` produce
    both their whole form and component tokens, so either query style matches.
    """
    tokens: list[str] = []
    for identifier in _IDENTIFIER.findall(text):
        normalized = identifier.casefold()
        parts = [part for part in re.split(r"[_./-]+", identifier) if part]
        camel_parts: list[str] = []
        for part in parts:
            words = [match.group(0).casefold() for match in _TOKEN.finditer(part)]
            if not words:
                words = [part.casefold()]
            camel_parts.extend(words)
        tokens.append(normalized)
        tokens.extend(part for part in camel_parts if part and part != normalized)
    return [token for token in tokens if token and token not in STOP_WORDS]


def _matches(meta: dict[str, Any], where: dict[str, Any] | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        clauses = where["$and"]
        return isinstance(clauses, list) and all(
            isinstance(clause, dict) and _matches(meta, clause)
            for clause in clauses
        )
    if "$or" in where:
        clauses = where["$or"]
        return isinstance(clauses, list) and any(
            isinstance(clause, dict) and _matches(meta, clause)
            for clause in clauses
        )
    for key, expected in where.items():
        actual = meta.get(key)
        if isinstance(expected, dict):
            if len(expected) != 1:
                raise ValueError(f"Unsupported BM25 filter expression for {key!r}")
            operator, value = next(iter(expected.items()))
            if operator == "$eq" and actual != value:
                return False
            if operator == "$ne" and actual == value:
                return False
            if operator == "$in" and actual not in value:
                return False
            if operator == "$nin" and actual in value:
                return False
            if operator not in {"$eq", "$ne", "$in", "$nin"}:
                raise ValueError(f"Unsupported BM25 filter operator: {operator}")
        elif actual != expected:
            return False
    return True


@dataclass
class BM25Index:
    """In-memory BM25 index backed by a model-specific Chroma collection.

    ``search_texts`` (default: ``texts``) is what gets scored, e.g. context +
    chunk for a contextual variant; hits still return the original ``texts``.
    """

    ids: list[str]
    texts: list[str]
    metadatas: list[dict[str, Any]]
    k1: float = DEFAULT_BM25_K1
    b: float = DEFAULT_BM25_B
    variant: str = "plain"
    search_texts: list[str] | None = None

    def __post_init__(self) -> None:
        if not (len(self.ids) == len(self.texts) == len(self.metadatas)):
            raise ValueError("BM25 ids, texts, and metadata must have equal lengths")
        if self.search_texts is not None and len(self.search_texts) != len(self.texts):
            raise ValueError("BM25 search_texts must match texts in length")
        if self.k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be between 0 and 1")
        self._tokens = [tokenize(text) for text in (self.search_texts or self.texts)]
        self._bm25 = (
            BM25Okapi(self._tokens, k1=self.k1, b=self.b)
            if any(self._tokens)
            else None
        )

    @classmethod
    def from_chroma(
        cls,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        model_name: str = "BAAI/bge-small-en-v1.5",
        *,
        k1: float = DEFAULT_BM25_K1,
        b: float = DEFAULT_BM25_B,
        variant: str = "plain",
    ) -> BM25Index:
        """Load documents from the exact collection used by vector search."""
        collection = get_collection(persist_dir, model_name, variant)
        payload = collection.get(include=["documents", "metadatas"])
        ids = payload.get("ids", [])
        texts = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        if not (len(ids) == len(texts) == len(metadatas)):
            raise ValueError("Chroma returned mismatched IDs, documents, and metadata")
        entries = [
            (str(chunk_id), str(text), dict(meta))
            for chunk_id, text, meta in zip(ids, texts, metadatas)
            if text is not None and meta is not None
        ]
        entries.sort(key=lambda item: item[0])
        return cls(
            ids=[item[0] for item in entries],
            texts=[item[1] for item in entries],
            metadatas=[item[2] for item in entries],
            k1=k1,
            b=b,
            variant=variant,
            search_texts=[indexed_text(item[1], item[2]) for item in entries],
        )

    def search(
        self,
        query: str,
        k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        if self._bm25 is None:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        query_term_set = set(query_tokens)
        candidates = [
            index
            for index, meta in enumerate(self.metadatas)
            if _matches(meta, where)
            and query_term_set.intersection(self._tokens[index])
        ]
        candidates.sort(key=lambda index: (-float(scores[index]), self.ids[index]))
        return [
            SearchHit(
                chunk_id=self.ids[index],
                score=float(scores[index]),
                text=self.texts[index],
                meta=self.metadatas[index],
            )
            for index in candidates[:k]
        ]


def bm25_search(
    query: str,
    persist_dir: str | Path,
    model_name: str,
    k: int = 8,
    where: dict[str, Any] | None = None,
    index: BM25Index | None = None,
    variant: str = "plain",
) -> list[SearchHit]:
    """Convenience function for a single BM25 query."""
    bm25_index = index or BM25Index.from_chroma(persist_dir, model_name, variant=variant)
    return bm25_index.search(query, k=k, where=where)

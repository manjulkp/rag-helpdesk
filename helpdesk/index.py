"""Persistent Chroma indexing and filtered semantic search.

An index comes in variants: ``plain`` embeds each chunk as-is, ``header`` and
``llm`` embed "context + chunk" (a repo / path / heading header, or an
LLM-written situating sentence). The document stored is always the original
chunk, so answers cite real text; the context is kept in ``meta["context"]``.
Each variant has its own collection, and searches refuse a mismatched one.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb
from chromadb.errors import NotFoundError

from helpdesk.chunker import Chunk
from helpdesk.embedder import DEFAULT_MODEL

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma"
INDEX_SCHEMA_VERSION = 1
UPSERT_BATCH_SIZE = 500
VARIANTS = ("plain", "header", "llm")
CONTEXTUAL_VARIANTS = ("header", "llm")


class IndexModelMismatch(ValueError):
    """Raised when search uses a model different from the stored index model."""


class IndexNotFoundError(FileNotFoundError):
    """Raised when no compatible index exists in the requested directory."""


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    text: str
    meta: dict[str, Any]


def collection_name(model_name: str = DEFAULT_MODEL, variant: str = "plain") -> str:
    """Return a Chroma-safe collection name for a model identifier and variant.

    ``plain`` keeps the original name, so indexes built before variants existed
    still work; other variants get a ``-<variant>`` suffix.
    """
    _check_variant(variant)
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", model_name).strip("-_").lower()
    slug = re.sub(r"[-_]{2,}", "-", slug)
    if not slug:
        slug = "default"
    suffix = "" if variant == "plain" else f"-{variant}"
    name = f"helpdesk-{slug}"[: 63 - len(suffix)].rstrip("-_")
    return name + suffix


def _check_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown index variant {variant!r}; expected one of {', '.join(VARIANTS)}")


def contextual_text(context: str, text: str) -> str:
    """What a contextual index embeds: the context, a blank line, then the chunk."""
    return f"{context}\n\n{text}" if context else text


def indexed_text(text: str, meta: dict[str, Any]) -> str:
    """Rebuild exactly what the vector side embedded for a stored chunk."""
    return contextual_text(str(meta.get("context") or ""), text)


def _client(persist_dir: str | Path) -> Any:
    return chromadb.PersistentClient(path=str(Path(persist_dir).expanduser()))


def _model_name(embedder: Any) -> str:
    value = getattr(embedder, "model_name", None)
    if not isinstance(value, str) or not value:
        raise ValueError("Embedder must expose a non-empty model_name")
    return value


def _metadata(chunk: Chunk) -> dict[str, str | int]:
    return {
        "repo": chunk.repo,
        "path": chunk.path,
        "kind": chunk.kind,
        "title": chunk.title,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "last_commit": chunk.last_commit,
        "n_tokens": chunk.n_tokens,
        "fixtures": ", ".join(chunk.fixtures),
    }


def _vectors(values: Any) -> list[list[float]]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [[float(value) for value in vector] for vector in values]


def build_index(
    chunks: Iterable[Chunk],
    embedder: Any,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    *,
    variant: str = "plain",
    contexts: dict[str, str] | None = None,
    stats: dict[str, int] | None = None,
) -> int:
    """Upsert chunks and remove stale IDs, returning the collection size.

    ``contexts`` maps chunk ID to context text and is required for the
    ``header`` and ``llm`` variants (a chunk missing from it is embedded plain,
    e.g. after a failed LLM call) and refused for ``plain``.

    Incremental: each chunk stores a hash of the text it was embedded from.
    Only new chunks and chunks whose embedded text changed are re-embedded; a
    chunk whose text is the same but whose metadata moved (line numbers after
    an edit above it) only gets its metadata updated. ``stats`` (if given) is
    filled with counts: embedded, metadata_only, unchanged, deleted.
    """
    _check_variant(variant)
    if variant == "plain" and contexts is not None:
        raise ValueError("the plain variant takes no contexts")
    if variant in CONTEXTUAL_VARIANTS and contexts is None:
        raise ValueError(f"the {variant} variant needs contexts")
    contexts = contexts or {}
    model_name = _model_name(embedder)
    client = _client(persist_dir)
    name = collection_name(model_name, variant)
    collection = client.get_or_create_collection(
        name=name,
        metadata={
            "model_name": model_name,
            "variant": variant,
            "schema_version": INDEX_SCHEMA_VERSION,
            "hnsw:space": "cosine",
        },
    )
    _check_collection(collection, model_name, variant)

    counts = {"embedded": 0, "metadata_only": 0, "unchanged": 0, "deleted": 0}
    chunk_list = list(chunks)
    if not chunk_list:
        existing_ids = collection.get(include=[]).get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)
        counts["deleted"] = len(existing_ids)
        if stats is not None:
            stats.clear()
            stats.update(counts)
        return 0

    ids = [chunk.id for chunk in chunk_list]
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk IDs must be unique within an index build")

    def metadata(item: Chunk) -> dict[str, Any]:
        embedded = contextual_text(contexts.get(item.id, ""), item.text)
        meta: dict[str, Any] = {**_metadata(item), "content_hash": _content_hash(embedded)}
        if variant in CONTEXTUAL_VARIANTS:
            meta["context"] = contexts.get(item.id, "")
        return meta

    stored = collection.get(include=["metadatas"])
    existing = dict(zip(stored.get("ids", []), stored.get("metadatas", [])))
    to_embed: list[tuple[Chunk, dict[str, Any]]] = []
    to_update: list[tuple[Chunk, dict[str, Any]]] = []
    for item in chunk_list:
        meta = metadata(item)
        old = existing.get(item.id)
        if old is None or old.get("content_hash") != meta["content_hash"]:
            to_embed.append((item, meta))
        elif any(old.get(key) != value for key, value in meta.items()):
            to_update.append((item, meta))
        else:
            counts["unchanged"] += 1

    for offset in range(0, len(to_embed), UPSERT_BATCH_SIZE):
        batch = to_embed[offset : offset + UPSERT_BATCH_SIZE]
        vectors = _vectors(
            embedder.embed_documents(
                [contextual_text(contexts.get(item.id, ""), item.text) for item, _ in batch]
            )
        )
        if len(vectors) != len(batch):
            raise ValueError("Embedder returned a different number of vectors than input texts")
        collection.upsert(
            ids=[item.id for item, _ in batch],
            embeddings=vectors,
            documents=[item.text for item, _ in batch],
            metadatas=[meta for _, meta in batch],
        )
    counts["embedded"] = len(to_embed)
    for offset in range(0, len(to_update), UPSERT_BATCH_SIZE):
        batch = to_update[offset : offset + UPSERT_BATCH_SIZE]
        collection.update(ids=[item.id for item, _ in batch], metadatas=[meta for _, meta in batch])
    counts["metadata_only"] = len(to_update)

    stale_ids = set(existing) - set(ids)
    if stale_ids:
        for offset in range(0, len(stale_ids), UPSERT_BATCH_SIZE):
            collection.delete(ids=sorted(stale_ids)[offset : offset + UPSERT_BATCH_SIZE])
    counts["deleted"] = len(stale_ids)
    if stats is not None:
        stats.clear()
        stats.update(counts)
    return collection.count()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _check_collection(collection: Any, model_name: str, variant: str) -> None:
    metadata = collection.metadata or {}
    stored_model = metadata.get("model_name")
    # collections from before variants existed are plain
    stored_variant = metadata.get("variant", "plain")
    if stored_model != model_name or stored_variant != variant:
        raise IndexModelMismatch(
            f"Index collection {collection.name!r} uses model {stored_model!r} "
            f"({stored_variant}); requested {model_name!r} ({variant}) — "
            "model or variant mismatch"
        )


def get_collection(
    persist_dir: str | Path,
    model_name: str,
    variant: str = "plain",
) -> Any:
    """Open the collection for a model and variant, shared by vector and BM25 search."""
    _check_variant(variant)
    client = _client(persist_dir)
    name = collection_name(model_name, variant)
    try:
        collection = client.get_collection(name=name)
    except NotFoundError as error:
        available = []
        for found in client.list_collections():
            found_name = found if isinstance(found, str) else found.name
            metadata = client.get_collection(name=found_name).metadata or {}
            available.append(
                f"{metadata.get('model_name')!r} ({metadata.get('variant', 'plain')})"
            )
        if available:
            raise IndexModelMismatch(
                f"No index for model {model_name!r} ({variant}) in {Path(persist_dir)}; "
                f"indexes there: {', '.join(sorted(available))}"
            ) from error
        raise IndexNotFoundError(
            f"No helpdesk index found in {Path(persist_dir)}"
        ) from error
    _check_collection(collection, model_name, variant)
    return collection


def hits_by_id(
    ids: list[str],
    embedder: Any,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    variant: str = "plain",
) -> list[SearchHit]:
    """Fetch stored chunks by ID, in the given order (score 0.0); unknown IDs are skipped."""
    if not ids:
        return []
    collection = get_collection(persist_dir, _model_name(embedder), variant)
    payload = collection.get(ids=list(ids), include=["documents", "metadatas"])
    found = {
        chunk_id: (text, meta)
        for chunk_id, text, meta in zip(payload["ids"], payload["documents"], payload["metadatas"])
    }
    return [
        SearchHit(chunk_id=chunk_id, score=0.0, text=found[chunk_id][0], meta=dict(found[chunk_id][1]))
        for chunk_id in ids
        if chunk_id in found
    ]


def search(
    query: str,
    embedder: Any,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    k: int = 8,
    where: dict[str, Any] | None = None,
    query_vector: Any | None = None,
    variant: str = "plain",
) -> list[SearchHit]:
    """Search the stored vectors and return cosine-ranked hits.

    ``query_vector`` replaces ``embed_query(query)``, e.g. a HyDE passage
    embedded as a document.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")

    collection = get_collection(persist_dir, _model_name(embedder), variant)
    if collection.count() == 0:
        return []
    vector = _vectors(
        [embedder.embed_query(query) if query_vector is None else query_vector]
    )[0]
    filters: dict[str, Any] | None = where
    if where and len(where) > 1:
        filters = {"$and": [{key: value} for key, value in where.items()]}
    results = collection.query(
        query_embeddings=[vector],
        n_results=k,
        where=filters,
        include=["documents", "metadatas", "distances"],
    )
    ids = results.get("ids", [[]])[0]
    texts = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    return [
        SearchHit(
            chunk_id=chunk_id,
            score=1.0 - float(distance),
            text=text,
            meta=metadata,
        )
        for chunk_id, text, metadata, distance in zip(ids, texts, metadatas, distances)
    ]

"""Embed every golden-set question once and save it, so the FAISS scale-up
runs without PyTorch in the process (FAISS and PyTorch each bring their own
OpenMP runtime on Windows, and together they made the 97-vector lab take
minutes instead of seconds).

    python -m scripts.faiss_queries    # writes results/faiss/queries.npy, questions.json,
                                       # vectors.npy and chunks.json (ids + metadata)
"""

from __future__ import annotations

import json
import time

import numpy as np

from scripts.faiss_lab import OUT, golden_questions


def main() -> None:
    from transformers.utils import logging as hf_logging

    from helpdesk.embedder import DEFAULT_MODEL, Embedder

    hf_logging.set_verbosity_error()
    started = time.perf_counter()
    questions = golden_questions()
    embedder = Embedder(DEFAULT_MODEL)
    xq = np.stack([embedder.embed_query(q) for q in questions]).astype("float32")
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "queries.npy", xq)
    (OUT / "questions.json").write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")

    from helpdesk.index import DEFAULT_PERSIST_DIR, get_collection

    payload = get_collection(DEFAULT_PERSIST_DIR, DEFAULT_MODEL).get(include=["embeddings", "metadatas"])
    xb = np.asarray(payload["embeddings"], dtype="float32")
    np.save(OUT / "vectors.npy", xb)
    (OUT / "chunks.json").write_text(json.dumps(
        [{"id": i, "repo": m["repo"], "path": m["path"], "title": m["title"]}
         for i, m in zip(payload["ids"], payload["metadatas"])], indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(questions)} questions embedded, {len(xb)} chunk vectors saved, "
          f"in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()

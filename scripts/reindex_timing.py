"""Full build vs incremental re-index after editing one document.

    python -m scripts.reindex_timing [--repo R --path P]

Builds the plain index into a temporary folder (the real index is untouched),
then appends a sentence to one document, re-chunks, and rebuilds. Only the
chunks whose text changed are re-embedded; chunks after the edit whose line
numbers moved get a metadata-only update.
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
import time

from transformers.utils import logging as transformers_logging

from helpdesk.chunker import DEFAULT_MAX_TOKENS, chunk_documents
from helpdesk.embedder import DEFAULT_MODEL, Embedder
from helpdesk.index import build_index
from helpdesk.loader import load_documents
from helpdesk.sources import REPOS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="two-agent-trader-riskmanger")
    parser.add_argument("--path", default="docs/interview-prep.md", help="the document to edit")
    args = parser.parse_args()
    transformers_logging.set_verbosity_error()
    embedder = Embedder(DEFAULT_MODEL)
    documents = load_documents(REPOS)
    target = next(d for d in documents if d.repo == args.repo and d.path == args.path)

    # Chroma keeps its files open, so Windows can't always delete the folder at the end
    with tempfile.TemporaryDirectory(prefix="reindex-", ignore_cleanup_errors=True) as folder:
        chunks = chunk_documents(documents, DEFAULT_MAX_TOKENS, embedder.count_tokens)
        stats: dict = {}
        started = time.perf_counter()
        build_index(chunks, embedder, folder, stats=stats)
        full = time.perf_counter() - started
        print(f"full build: {full:.2f}s, {stats}")

        started = time.perf_counter()
        build_index(chunks, embedder, folder, stats=stats)
        print(f"rebuild, nothing changed: {time.perf_counter() - started:.2f}s, {stats}")

        # edit one paragraph in the middle of the document
        lines = target.text.splitlines()
        middle = len(lines) // 2
        lines.insert(middle, "\nAn added sentence for the re-index timing check.\n")
        edited = dataclasses.replace(target, text="\n".join(lines))
        docs = [edited if d is target else d for d in documents]
        new_chunks = chunk_documents(docs, DEFAULT_MAX_TOKENS, embedder.count_tokens)
        doc_chunks = sum(c.repo == args.repo and c.path == args.path for c in new_chunks)
        started = time.perf_counter()
        build_index(new_chunks, embedder, folder, stats=stats)
        incremental = time.perf_counter() - started
        print(f"after editing {args.repo}/{args.path} ({doc_chunks} chunks): "
              f"{incremental:.2f}s, {stats}")
        print(f"incremental / full: {incremental / full:.0%}")


if __name__ == "__main__":
    main()

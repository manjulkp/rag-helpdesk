"""Rebuild the prediction-markets v1 index for a like-for-like baseline.

This adapter imports the original repository's ``collect_chunks`` and config,
but writes its Chroma index into this project's ``data/chroma_pred`` directory.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from transformers.utils import logging as transformers_logging

from helpdesk.chunker import Chunk
from helpdesk.embedder import Embedder
from helpdesk.index import build_index

PRED_ROOT = Path.home() / "pred" / "prediction-markets"
PERSIST_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_pred"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
KIND_BY_TYPE = {
    "documentation": "doc",
    "algorithm_interface": "code",
    "algorithm_impl": "code",
}


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _find_collect_chunks(root: Path) -> Path:
    skipped = {".git", ".venv", "venv", "node_modules", "__pycache__"}
    for path in sorted(root.rglob("*.py")):
        if any(part in skipped or part.endswith(".egg-info") for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, SyntaxError):
            continue
        if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "collect_chunks"
            for node in tree.body
        ):
            return path
    raise FileNotFoundError(f"No top-level collect_chunks function found under {root}")


def _to_chunk(raw: Any, repo_root: Path) -> Chunk:
    metadata = getattr(raw, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    file_path = Path(str(metadata.get("file_path", metadata.get("path", ""))))
    if not file_path.is_absolute():
        file_path = (repo_root / file_path).resolve()
    try:
        relative_path = file_path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        relative_path = file_path.as_posix()
    text = str(getattr(raw, "text", metadata.get("text", "")))
    raw_id = str(getattr(raw, "id", metadata.get("id", "")))
    if not raw_id:
        raise ValueError(f"Legacy chunk has no ID: {relative_path}")
    raw_type = str(metadata.get("type", "documentation"))
    return Chunk(
        id=raw_id,
        text=text,
        repo="prediction-markets",
        path=relative_path,
        kind=KIND_BY_TYPE.get(raw_type, raw_type),
        title=str(metadata.get("title", file_path.name)),
        line_start=int(metadata.get("line_start", 1)),
        line_end=int(metadata.get("line_end", metadata.get("line_start", 1))),
        last_commit="",
        n_tokens=0,
        fixtures=(),
    )


def main() -> None:
    if not PRED_ROOT.is_dir():
        raise FileNotFoundError(f"Prediction-markets repository not found: {PRED_ROOT}")

    config_path = PRED_ROOT / "config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"Legacy config not found: {config_path}")

    sys.path.insert(0, str(PRED_ROOT))
    config = _load_module("_pred_rag_config", config_path)
    source_path = _find_collect_chunks(PRED_ROOT)
    source_module = _load_module("_pred_rag_chunk_source", source_path)
    collect_chunks = getattr(source_module, "collect_chunks", None)
    if not callable(collect_chunks):
        raise TypeError(f"{source_path} does not expose collect_chunks()")

    repo_root = Path(getattr(config, "REPO_ROOT", PRED_ROOT)).resolve()
    model_name = str(getattr(config, "EMBEDDING_MODEL", MODEL))
    if not model_name.endswith(MODEL):
        raise ValueError(
            f"Expected the pred v1 baseline model {MODEL!r}, got {model_name!r}"
        )

    transformers_logging.set_verbosity_error()
    raw_chunks = collect_chunks()
    chunks = [_to_chunk(item, repo_root) for item in raw_chunks]
    embedder = Embedder(model_name)
    count = build_index(chunks, embedder, PERSIST_DIR)
    print(
        f"pred v1: {len(chunks)} chunks from "
        f"{len({item.path for item in chunks})} files indexed ({count}) in {PERSIST_DIR}"
    )


if __name__ == "__main__":
    main()

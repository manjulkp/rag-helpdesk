"""Sentence-transformer embeddings with model-aware query instructions."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTIONS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    """Thin wrapper around SentenceTransformer for document and query vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        query_instruction: str | None = None,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.query_instruction = (
            QUERY_INSTRUCTIONS.get(model_name, "")
            if query_instruction is None
            else query_instruction
        )
        self.batch_size = batch_size
        self._model: Any = SentenceTransformer(model_name, device=self.device)

    @property
    def max_seq_length(self) -> int:
        return int(self._model.max_seq_length)

    def count_tokens(self, text: str) -> int:
        return len(
            self._model.tokenizer.encode(
                text,
                add_special_tokens=False,
                verbose=False,
            )
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1000,
        )

    def embed_query(self, query: str) -> np.ndarray:
        text = f"{self.query_instruction}{query}"
        return self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

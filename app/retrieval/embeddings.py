"""Embedding providers behind one interface."""

from __future__ import annotations

import logging
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from app.config import Settings
from app.local_models import require_local_model
from app.text_utils import tokenize

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class SentenceTransformerEmbedder:
    """Local open-source embeddings (default: BAAI/bge-small-en-v1.5, 384-d, fast on CPU)."""

    def __init__(self, model: str, path: Path, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self.name = model
        self._model = SentenceTransformer(str(path), local_files_only=True)
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = dim_fn()
        self._batch = batch_size
        m = model.lower()
        # Asymmetric retrieval models expect an instruction / prefix on the query side.
        if "bge" in m and "-en" in m and "m3" not in m:
            self._q_prefix, self._d_prefix = "Represent this sentence for searching relevant passages: ", ""
        elif "e5" in m:
            self._q_prefix, self._d_prefix = "query: ", "passage: "
        else:
            self._q_prefix = self._d_prefix = ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            [self._d_prefix + t for t in texts],
            batch_size=self._batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self._model.encode(self._q_prefix + text, normalize_embeddings=True)


class HashingEmbedder:
    """Dependency-free feature-hashing embedder (unigrams + bigrams).

    NOT a semantic model - it exists so tests and CI run offline in milliseconds.
    """

    def __init__(self, dim: int = 768):
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        toks = tokenize(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        v = np.sign(v) * np.log1p(np.abs(v))
        return _l2(v)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


def build_embedder(s: Settings) -> Embedder:
    p = s.embedding_provider.lower()
    if p == "sentence-transformers":
        path = require_local_model(s.models_dir, s.embedding_model)
        return SentenceTransformerEmbedder(s.embedding_model, path, s.embedding_batch_size)
    if p == "hash":
        return HashingEmbedder()
    raise ValueError(f"Unknown EMBEDDING_PROVIDER '{s.embedding_provider}'")

"""In-memory BM25 keyword index, rebuilt lazily from SQLite when documents change.

Keyword search is what catches exact identifiers that embeddings blur together:
error codes ("E04"), clause numbers ("Section 9.2"), part numbers, acronyms.
For millions of chunks you would move this into Qdrant sparse vectors or OpenSearch.
"""

from __future__ import annotations

import threading

import numpy as np
from rank_bm25 import BM25Okapi

from app.schemas import Chunk
from app.text_utils import tokenize


class BM25Index:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._doc_ids: np.ndarray = np.array([])
        self.idf: dict[str, float] = {}

    def build(self, chunks: list[Chunk]) -> None:
        with self._lock:
            self._chunk_ids = [c.chunk_id for c in chunks]
            self._doc_ids = np.array([c.doc_id for c in chunks])
            if chunks:
                corpus = [tokenize(f"{c.section or ''} {c.text}") for c in chunks]
                self._bm25 = BM25Okapi(corpus)
                self.idf = dict(self._bm25.idf)
            else:
                self._bm25, self.idf = None, {}

    def search(self, query: str, limit: int, doc_ids: list[str] | None = None) -> list[tuple[str, float]]:
        with self._lock:
            if self._bm25 is None:
                return []
            q = tokenize(query)
            if not q:
                return []
            scores = self._bm25.get_scores(q)
            if doc_ids:
                scores = np.where(np.isin(self._doc_ids, doc_ids), scores, -np.inf)
            top = np.argsort(-scores)[:limit]
            return [(self._chunk_ids[i], float(scores[i])) for i in top if scores[i] > 0]

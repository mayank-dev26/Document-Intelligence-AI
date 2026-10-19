"""Qdrant wrapper.

* No QDRANT_URL  -> embedded Qdrant persisted on disk (local development).
* QDRANT_URL set -> Qdrant server (used by docker-compose).

The collection name includes the embedding model + dimension, so switching models never
mixes incompatible vectors; the pipeline re-embeds from SQLite automatically.
"""

from __future__ import annotations

import contextlib
import re
import threading
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.schemas import Chunk


def collection_name(base: str, model: str, dim: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")[-40:]
    return f"{base}_{slug}_{dim}"


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class VectorStore:
    def __init__(self, url: str | None, path: str, name: str, dim: int):
        self.client = QdrantClient(url=url) if url else QdrantClient(path=path)
        self.name = name
        self.dim = dim
        self._url = url
        self._lock = threading.RLock()  # the background worker and web requests share the client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        with self._lock:
            if self.client.collection_exists(self.name):
                return
            self.client.create_collection(
                self.name, vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)
            )
            if self._url:  # payload indexes only matter (and are only supported) on a real server
                self.client.create_payload_index(self.name, "doc_id", qm.PayloadSchemaType.KEYWORD)

    def reset(self) -> None:
        """Drop every vector (used when the app starts fresh)."""
        with self._lock:
            if self.client.collection_exists(self.name):
                self.client.delete_collection(self.name)
        self._ensure_collection()

    def count(self) -> int:
        with self._lock:
            return self.client.count(self.name, exact=True).count

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 256) -> None:
        with self._lock:
            for start in range(0, len(chunks), batch):
                part = chunks[start : start + batch]
                self.client.upsert(
                    self.name,
                    points=[
                        qm.PointStruct(
                            id=point_id(c.chunk_id),
                            vector=vectors[start + i].tolist(),
                            payload={"chunk_id": c.chunk_id, "doc_id": c.doc_id, "page": c.page},
                        )
                        for i, c in enumerate(part)
                    ],
                )

    def search(self, vector: np.ndarray, limit: int, doc_ids: list[str] | None = None) -> list[tuple[str, float]]:
        flt = None
        if doc_ids:
            flt = qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchAny(any=doc_ids))])
        with self._lock:
            res = self.client.query_points(
                self.name, query=vector.tolist(), limit=limit, query_filter=flt, with_payload=True
            )
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self.client.delete(
                self.name,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))])
                ),
            )

    def close(self, drop: bool = False) -> None:
        with self._lock, contextlib.suppress(Exception):  # best-effort on shutdown
            if drop and self._url:
                self.client.delete_collection(self.name)
            self.client.close()

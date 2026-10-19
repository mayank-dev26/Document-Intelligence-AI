"""The four retrieval strategies compared by this project."""

from __future__ import annotations

import time

from app.config import Settings
from app.ingestion.chunker import embedding_text
from app.retrieval.embeddings import Embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import Reranker
from app.schemas import RetrievalMode, RetrievedPassage
from app.store.bm25_index import BM25Index
from app.store.metadata import MetadataStore
from app.store.vector_store import VectorStore


class Retriever:
    def __init__(
        self,
        settings: Settings,
        meta: MetadataStore,
        vectors: VectorStore,
        bm25: BM25Index,
        embedder: Embedder,
        reranker: Reranker,
    ):
        self.s = settings
        self.meta = meta
        self.vectors = vectors
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(
        self, question: str, mode: RetrievalMode, top_k: int, doc_ids: list[str] | None = None
    ) -> tuple[list[RetrievedPassage], dict[str, float]]:
        """Return the top_k passages and stage timings in ms."""
        timings = {"retrieval_ms": 0.0, "rerank_ms": 0.0}
        cand_k = max(self.s.candidate_k, top_k)
        t0 = time.perf_counter()

        if mode == RetrievalMode.vector:
            ranked = [(cid, s, {"vector": s}) for cid, s in self._dense(question, top_k, doc_ids)]
        elif mode == RetrievalMode.bm25:
            ranked = [(cid, s, {"bm25": s}) for cid, s in self.bm25.search(question, top_k, doc_ids)]
        else:
            lists = {
                "vector": self._dense(question, cand_k, doc_ids),
                "bm25": self.bm25.search(question, cand_k, doc_ids),
            }
            ranked = [(cid, s, {**orig, "rrf": s}) for cid, s, orig in reciprocal_rank_fusion(lists, k=self.s.rrf_k)]
            if mode == RetrievalMode.hybrid:
                ranked = ranked[:top_k]
        timings["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        chunks = self.meta.get_chunks([cid for cid, _, _ in ranked])
        ranked = [r for r in ranked if r[0] in chunks]

        if mode == RetrievalMode.hybrid_rerank:
            t1 = time.perf_counter()
            texts = [embedding_text(chunks[cid]) for cid, _, _ in ranked]
            scores = self.reranker.score(question, texts)
            ranked = sorted(
                ((cid, sc, {**st, "rerank": sc}) for (cid, _, st), sc in zip(ranked, scores)),
                key=lambda x: -x[1],
            )[:top_k]
            timings["rerank_ms"] = (time.perf_counter() - t1) * 1000

        passages = [
            RetrievedPassage(chunk=chunks[cid], score=score, rank=i + 1, stage_scores=stages)
            for i, (cid, score, stages) in enumerate(ranked)
        ]
        return passages, timings

    def relevance(self, question: str, passages: list[RetrievedPassage]) -> float:
        """Calibrated 0..1 relevance of the best passage - the abstention gate.

        Re-uses rerank scores when available; otherwise scores the (few) final passages.
        """
        if not passages:
            return 0.0
        if all("rerank" in p.stage_scores for p in passages):
            return max(p.stage_scores["rerank"] for p in passages)
        scores = self.reranker.score(question, [embedding_text(p.chunk) for p in passages])
        for p, sc in zip(passages, scores):
            p.stage_scores["relevance"] = sc
        return max(scores) if scores else 0.0

    def _dense(self, question: str, limit: int, doc_ids: list[str] | None) -> list[tuple[str, float]]:
        return self.vectors.search(self.embedder.embed_query(question), limit, doc_ids)

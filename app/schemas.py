"""Pydantic models shared by the API, pipeline, UI and evaluation."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    vector = "vector"  # dense embeddings only
    bm25 = "bm25"  # keyword only
    hybrid = "hybrid"  # dense + BM25 fused with RRF
    hybrid_rerank = "hybrid_rerank"  # hybrid, then cross-encoder rerank


class Page(BaseModel):
    number: int  # 1-based
    text: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    chunk_index: int
    text: str
    char_start: int  # offset inside the page text
    char_end: int
    section: str | None = None


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    status: str  # processing | ready | failed
    error: str | None = None  # plain-language reason when failed
    num_pages: int | None = None
    num_chunks: int | None = None
    sha256: str
    uploaded_at: str
    # background whole-document overview: waiting | building | ready | failed | unavailable
    overview_status: str | None = None


class RetrievedPassage(BaseModel):
    chunk: Chunk
    score: float  # score from the final stage
    rank: int
    # per-stage scores, e.g. {"vector": .., "bm25": .., "rrf": .., "rerank": ..}
    stage_scores: dict[str, float] = Field(default_factory=dict)


class Citation(BaseModel):
    marker: str  # "S1" (a passage) or "N1" (overview notes for a page range), as written in the answer
    kind: str = "passage"  # passage | notes
    chunk_id: str
    doc_id: str
    doc_name: str
    page: int
    quote: str  # what the model claimed the source says
    verified: bool  # quote actually found in the source text
    match_score: float  # 0..1 fuzzy match ratio
    quote_start: int | None = None  # offset of the matched span inside the chunk
    quote_end: int | None = None
    page_end: int | None = None  # notes cover a page range
    notes: str | None = None  # the overview notes text, for kind == "notes"


class Turn(BaseModel):
    question: str
    answer: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: RetrievalMode = RetrievalMode.hybrid_rerank
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_ids: list[str] | None = None  # restrict to these documents
    history: list[Turn] = Field(default_factory=list, max_length=10)  # earlier turns, for follow-ups
    intent: str | None = None  # lookup | overview | brainstorm; None = decide automatically


class Timings(BaseModel):
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class Answer(BaseModel):
    question: str
    kind: str = "lookup"  # lookup (a fact) | overview (summary/insights) | brainstorm (ideas from the document)
    understood_as: str | None = None  # the question rewritten to stand alone (follow-ups)
    answer: str
    answerable: bool
    abstain_reason: str | None = None  # plain-language, shown to the user
    abstain_code: str | None = None  # no_evidence | below_threshold | model_refused | ungrounded
    note: str | None = None  # e.g. answered without the AI engine
    confidence: float  # calibrated relevance of evidence (0..1)
    citations: list[Citation]
    passages: list[RetrievedPassage]
    mode: RetrievalMode
    timings: Timings
    llm: str

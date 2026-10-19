"""Composition root: builds every component once and exposes the use-cases.

Uploads return immediately; one background worker reads, splits, embeds and indexes each file,
moving its status from `processing` to `ready` or `failed`. Unless KEEP_DOCUMENTS is set, all
stored documents are wiped when the app starts and when it shuts down, so nothing is left on disk.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pymupdf

from app.config import Settings, get_settings
from app.generation.answerer import Answerer
from app.generation.digest import Digester
from app.generation.llm import build_llm
from app.ingestion.chunker import chunk_pages, embedding_text
from app.ingestion.parser import SUPPORTED_EXTENSIONS, UnsupportedFileError, parse_file
from app.retrieval.embeddings import build_embedder
from app.retrieval.reranker import build_reranker
from app.retrieval.retriever import Retriever
from app.schemas import Answer, DocumentInfo, QueryRequest
from app.store.bm25_index import BM25Index
from app.store.metadata import MetadataStore
from app.store.vector_store import VectorStore, collection_name

log = logging.getLogger(__name__)

GENERIC_FAILURE = "This file couldn't be read. It may be damaged or password-protected."


class Pipeline:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        if not self.s.keep_documents:
            self._wipe_storage()
        self.s.ensure_dirs()
        self._write_lock = threading.Lock()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._jobs: dict[str, Future] = {}
        # Whole-document notes are written by a second worker, paused while a question is answered.
        self._digest_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="overview")
        self._asking = 0
        self._idle = threading.Condition()
        self._closing = False

        log.info("Loading embedding model (%s)...", self.s.embedding_model)
        self.embedder = build_embedder(self.s)
        self.meta = MetadataStore(self.s.sqlite_path)
        self.vectors = VectorStore(
            self.s.qdrant_url,
            str(self.s.qdrant_path),
            collection_name(self.s.qdrant_collection, self.embedder.name, self.embedder.dim),
            self.embedder.dim,
        )
        if not self.s.keep_documents:
            self.vectors.reset()  # also clears a Qdrant server in Docker
        self.bm25 = BM25Index()
        log.info("Loading reranking model (%s)...", self.s.reranker_model)
        self.reranker = build_reranker(self.s, idf_provider=lambda: self.bm25.idf)
        self.llm = build_llm(self.s)
        self.retriever = Retriever(self.s, self.meta, self.vectors, self.bm25, self.embedder, self.reranker)
        self.answerer = Answerer(self.s, self.llm, self.retriever)
        self.digester = Digester(self.meta, self.llm, self._wait_until_idle)

        self.bm25.build(self.meta.all_chunks())
        if self.s.keep_documents:
            self._sync_vectors()
            for d in self.meta.list_documents():
                if d.status == "ready" and d.overview_status != "ready":
                    self._schedule_overview(d.doc_id)

    # --- ingestion ---
    def submit_file(self, src: Path, filename: str | None = None) -> tuple[DocumentInfo, bool]:
        """Register a file and process it in the background. Returns (doc, created) immediately.

        Uploading identical content again returns the existing document (a failed one is retried).
        Raises UnsupportedFileError for file types the app can't read.
        """
        filename = filename or src.name
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFileError(
                "This file type isn't supported. Upload a PDF, Word (.docx), text or Markdown file."
            )
        sha = _sha256(src)
        if existing := self.meta.find_by_sha(sha):
            if existing.status != "failed":
                return existing, False
            self.delete_document(existing.doc_id)

        doc_id = sha[:16]
        stored = self.s.upload_dir / f"{doc_id}{ext}"
        if src.resolve() != stored.resolve():
            shutil.copyfile(src, stored)
        doc = self.meta.register_document(doc_id, filename, sha, stored)
        self._jobs[doc_id] = self._worker.submit(self._process, doc_id, filename, stored)
        return doc, True

    def ingest_file(self, src: Path, filename: str | None = None) -> tuple[DocumentInfo, bool]:
        """Like submit_file, but waits until the document is ready or failed (tests, benchmark)."""
        doc, created = self.submit_file(src, filename)
        if job := self._jobs.get(doc.doc_id):
            job.result()
        return self.meta.get_document(doc.doc_id), created  # type: ignore[return-value]

    def ingest_directory(self, folder: Path) -> list[DocumentInfo]:
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
        return [self.ingest_file(p)[0] for p in files]

    def _process(self, doc_id: str, filename: str, stored: Path) -> None:
        t0 = time.perf_counter()
        try:
            pages = parse_file(stored)
            chunks = chunk_pages(pages, doc_id, filename, self.s.chunk_size, self.s.chunk_overlap)
            vectors = self.embedder.embed_documents([embedding_text(c) for c in chunks])
            with self._write_lock:
                if not self.meta.get_document(doc_id):  # deleted while it was being read
                    return
                self.vectors.upsert(chunks, vectors)
                self.meta.complete_document(doc_id, pages, chunks)
                self.bm25.build(self.meta.all_chunks())
            self._schedule_overview(doc_id)
            log.info(
                "Ready: %s (%d pages, %d chunks, %.1fs)", filename, len(pages), len(chunks), time.perf_counter() - t0
            )
        except UnsupportedFileError as e:
            self.meta.fail_document(doc_id, str(e))
        except pymupdf.FileDataError:  # damaged, truncated or encrypted PDF
            log.warning("Couldn't open %s as a PDF", filename)
            self.meta.fail_document(doc_id, GENERIC_FAILURE)
        except Exception:  # noqa: BLE001 - anything else becomes a plain-language failure for the user
            log.exception("Processing failed for %s", filename)
            self.meta.fail_document(doc_id, GENERIC_FAILURE)
        finally:
            self._jobs.pop(doc_id, None)

    def delete_document(self, doc_id: str) -> bool:
        path = self.meta.get_document_path(doc_id)
        with self._write_lock:
            self.vectors.delete_document(doc_id)
            ok = self.meta.delete_document(doc_id)
            self.bm25.build(self.meta.all_chunks())
        if path and path.exists():
            path.unlink()
        return ok

    def _sync_vectors(self) -> None:
        """Re-embed from SQLite if the vector collection is behind (e.g. new embedding model)."""
        n_meta, n_vec = self.meta.count_chunks(), self.vectors.count()
        if n_meta and n_vec < n_meta:
            log.info("Vector index has %d/%d chunks - re-embedding from SQLite", n_vec, n_meta)
            chunks = self.meta.all_chunks()
            self.vectors.upsert(chunks, self.embedder.embed_documents([embedding_text(c) for c in chunks]))

    # --- whole-document overview ---
    def _schedule_overview(self, doc_id: str) -> None:
        if not self.s.prepare_overviews or self._closing:
            return
        if not self.digester.enabled:
            self.meta.set_overview_status(doc_id, "unavailable")
            return
        self.meta.set_overview_status(doc_id, "waiting")
        self._digest_worker.submit(self.digester.build, doc_id)

    def _retry_overviews(self) -> None:
        """The AI engine is back: prepare overviews that were postponed while it was off."""
        for d in self.meta.list_documents():
            if d.status == "ready" and d.overview_status == "unavailable":
                self._schedule_overview(d.doc_id)

    def _wait_until_idle(self) -> bool:
        """Block the overview worker while a question is being answered. False when closing."""
        with self._idle:
            while self._asking and not self._closing:
                self._idle.wait(0.5)
        return not self._closing

    # --- querying ---
    def ask(self, req: QueryRequest, abstain_threshold: float | None = None) -> Answer:
        final = None
        for event in self.ask_stream(req, abstain_threshold):
            if event["type"] == "final":
                final = event["answer"]
        return final  # type: ignore[return-value]

    def ask_stream(self, req: QueryRequest, abstain_threshold: float | None = None) -> Iterator[dict]:
        """Answer events as they happen: status, delta (answer text), reset, final."""
        with self._idle:
            self._asking += 1
        try:
            yield from self.answerer.stream(req, abstain_threshold)
        finally:
            with self._idle:
                self._asking -= 1
                self._idle.notify_all()

    # --- documents ---
    def list_documents(self) -> list[DocumentInfo]:
        return self.meta.list_documents()

    def page_text(self, doc_id: str, page: int) -> str | None:
        return self.meta.get_page(doc_id, page)

    def render_page(self, doc_id: str, page: int, highlight: str | None = None, dpi: int = 110) -> bytes | None:
        """PNG of a PDF page with the cited quote highlighted (None for non-PDF documents)."""
        path = self.meta.get_document_path(doc_id)
        if not path or path.suffix.lower() != ".pdf" or not path.exists():
            return None
        with pymupdf.open(path) as doc:
            if not 1 <= page <= doc.page_count:
                return None
            pg = doc[page - 1]
            if highlight:
                for rects in _find_rects(pg, highlight):
                    annot = pg.add_highlight_annot(rects)
                    annot.set_colors(stroke=(1.0, 0.85, 0.2))
                    annot.update()
            return pg.get_pixmap(dpi=dpi).tobytes("png")

    def status(self) -> dict:
        engine = self.llm.status()  # ready | offline | missing_model | extractive
        if engine == "ready":
            self._retry_overviews()
        docs = self.meta.list_documents()
        return {
            "ai_engine": engine,
            "model": getattr(self.llm, "model", None),
            "documents": len(docs),
            "ready": sum(d.status == "ready" for d in docs),
            "processing": sum(d.status == "processing" for d in docs),
            "preparing": sum(d.overview_status in ("waiting", "building") for d in docs),
            "keeps_documents": self.s.keep_documents,
        }

    def info(self) -> dict:
        """Technical details for the benchmark report and logs (not shown in the UI)."""
        return {
            "documents": len(self.meta.list_documents()),
            "chunks": self.meta.count_chunks(),
            "embedder": self.embedder.name,
            "reranker": self.reranker.name,
            "llm": self.llm.name,
            "vector_db": "qdrant-server" if self.s.qdrant_url else "qdrant-embedded",
            "abstain_threshold": self.s.abstain_threshold,
        }

    # --- lifecycle ---
    def close(self) -> None:
        self._closing = True
        self.digester.stop()
        with self._idle:
            self._idle.notify_all()
        self._worker.shutdown(wait=True, cancel_futures=True)
        # Don't wait for a model call in progress; the worker checks the stop flag before writing.
        self._digest_worker.shutdown(wait=False, cancel_futures=True)
        drop = not self.s.keep_documents
        self.vectors.close(drop=drop)
        self.meta.close()
        if drop:
            self._wipe_storage()

    def _wipe_storage(self) -> None:
        for path in (self.s.upload_dir, self.s.qdrant_path):
            shutil.rmtree(path, ignore_errors=True)
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(f"{self.s.sqlite_path}{suffix}").unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _find_rects(page: pymupdf.Page, quote: str) -> list[list]:
    """Find a quote on a PDF page; fall back to shorter pieces when line breaks/hyphenation differ."""
    quote = " ".join(quote.split())
    hits = page.search_for(quote, quads=True)
    if hits:
        return [hits]
    found = []
    words = quote.split()
    for size in (12, 8, 5):  # search the quote in overlapping word windows
        for i in range(0, max(1, len(words) - size + 1), max(1, size // 2)):
            piece = " ".join(words[i : i + size])
            if r := page.search_for(piece, quads=True):
                found.append(r)
        if found:
            return found
    return []

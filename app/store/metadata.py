"""SQLite store for documents, page text and chunks (the source of truth).

Each document has a status: `processing` while the background worker reads it, then
`ready` or `failed` (with a plain-language reason). The vector DB and BM25 index are derived
from this store. Every call goes through one lock because the background worker and the
web requests share the connection.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import Chunk, DocumentInfo, Page

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL, path TEXT NOT NULL,
    status TEXT NOT NULL, error TEXT, num_pages INTEGER, num_chunks INTEGER, uploaded_at TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    doc_id TEXT NOT NULL, page INTEGER NOT NULL, text TEXT NOT NULL,
    PRIMARY KEY (doc_id, page)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, doc_name TEXT NOT NULL, page INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL, text TEXT NOT NULL, char_start INTEGER, char_end INTEGER, section TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS notes (
    doc_id TEXT NOT NULL, idx INTEGER NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
    text TEXT NOT NULL, PRIMARY KEY (doc_id, idx)
);
"""
_ADDED_COLUMNS = {"overview": "TEXT", "overview_status": "TEXT"}  # added in v2.1


class MetadataStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            have = {r[1] for r in self._conn.execute("PRAGMA table_info(documents)")}
            for col, kind in _ADDED_COLUMNS.items():
                if col not in have:  # upgrade a database kept from an older version
                    self._conn.execute(f"ALTER TABLE documents ADD COLUMN {col} {kind}")

    def _all(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _one(self, sql: str, params: tuple | list = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    # --- documents ---
    def find_by_sha(self, sha: str) -> DocumentInfo | None:
        row = self._one("SELECT * FROM documents WHERE sha256=?", (sha,))
        return _doc(row) if row else None

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        row = self._one("SELECT * FROM documents WHERE doc_id=?", (doc_id,))
        return _doc(row) if row else None

    def get_document_path(self, doc_id: str) -> Path | None:
        row = self._one("SELECT path FROM documents WHERE doc_id=?", (doc_id,))
        return Path(row["path"]) if row else None

    def list_documents(self) -> list[DocumentInfo]:
        return [_doc(r) for r in self._all("SELECT * FROM documents ORDER BY uploaded_at, filename")]

    def ready_doc_ids(self) -> list[str]:
        return [r["doc_id"] for r in self._all("SELECT doc_id FROM documents WHERE status='ready'")]

    def register_document(self, doc_id: str, filename: str, sha: str, path: Path) -> DocumentInfo:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO documents (doc_id, filename, sha256, path, status, uploaded_at) VALUES (?,?,?,?,?,?)",
                (doc_id, filename, sha, str(path), "processing", now),
            )
        return self.get_document(doc_id)  # type: ignore[return-value]

    def complete_document(self, doc_id: str, pages: list[Page], chunks: list[Chunk]) -> bool:
        """Store pages and chunks and mark the document ready. False if it was deleted meanwhile."""
        with self._lock, self._conn:
            if not self._conn.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone():
                return False
            self._conn.executemany("INSERT INTO pages VALUES (?,?,?)", [(doc_id, p.number, p.text) for p in pages])
            self._conn.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        c.chunk_id,
                        c.doc_id,
                        c.doc_name,
                        c.page,
                        c.chunk_index,
                        c.text,
                        c.char_start,
                        c.char_end,
                        c.section,
                    )
                    for c in chunks
                ],
            )
            self._conn.execute(
                "UPDATE documents SET status='ready', error=NULL, num_pages=?, num_chunks=? WHERE doc_id=?",
                (len(pages), len(chunks), doc_id),
            )
        return True

    def fail_document(self, doc_id: str, message: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE documents SET status='failed', error=? WHERE doc_id=?", (message, doc_id))

    def delete_document(self, doc_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            self._conn.execute("DELETE FROM notes WHERE doc_id=?", (doc_id,))
        return cur.rowcount > 0

    # --- whole-document overview ---
    def set_overview_status(self, doc_id: str, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE documents SET overview_status=? WHERE doc_id=?", (status, doc_id))

    def set_overview(self, doc_id: str, text: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE documents SET overview=?, overview_status='ready' WHERE doc_id=?", (text, doc_id)
            )

    def get_overview(self, doc_id: str) -> str | None:
        row = self._one("SELECT overview FROM documents WHERE doc_id=?", (doc_id,))
        return row["overview"] if row else None

    def add_note(self, doc_id: str, idx: int, page_start: int, page_end: int, text: str) -> bool:
        with self._lock, self._conn:
            if not self._conn.execute("SELECT 1 FROM documents WHERE doc_id=?", (doc_id,)).fetchone():
                return False
            self._conn.execute(
                "INSERT OR REPLACE INTO notes VALUES (?,?,?,?,?)", (doc_id, idx, page_start, page_end, text)
            )
        return True

    def notes_for(self, doc_ids: list[str]) -> list[dict]:
        if not doc_ids:
            return []
        q = (
            "SELECT n.*, d.filename FROM notes n JOIN documents d USING (doc_id) "
            f"WHERE n.doc_id IN ({','.join('?' * len(doc_ids))}) ORDER BY d.uploaded_at, n.idx"
        )
        return [dict(r) for r in self._all(q, doc_ids)]

    def clear_notes(self, doc_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM notes WHERE doc_id=?", (doc_id,))

    def get_pages(self, doc_id: str) -> list[Page]:
        rows = self._all("SELECT page, text FROM pages WHERE doc_id=? ORDER BY page", (doc_id,))
        return [Page(number=r["page"], text=r["text"]) for r in rows]

    # --- pages / chunks ---
    def get_page(self, doc_id: str, page: int) -> str | None:
        row = self._one("SELECT text FROM pages WHERE doc_id=? AND page=?", (doc_id, page))
        return row["text"] if row else None

    def all_chunks(self) -> list[Chunk]:
        return [_chunk(r) for r in self._all("SELECT * FROM chunks ORDER BY doc_id, chunk_index")]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        q = f"SELECT * FROM chunks WHERE chunk_id IN ({','.join('?' * len(chunk_ids))})"
        return {r["chunk_id"]: _chunk(r) for r in self._all(q, chunk_ids)}

    def count_chunks(self) -> int:
        return self._one("SELECT COUNT(*) FROM chunks")[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _doc(r: sqlite3.Row) -> DocumentInfo:
    return DocumentInfo(
        doc_id=r["doc_id"],
        filename=r["filename"],
        status=r["status"],
        error=r["error"],
        num_pages=r["num_pages"],
        num_chunks=r["num_chunks"],
        sha256=r["sha256"],
        uploaded_at=r["uploaded_at"],
        overview_status=r["overview_status"],
    )


def _chunk(r: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=r["chunk_id"],
        doc_id=r["doc_id"],
        doc_name=r["doc_name"],
        page=r["page"],
        chunk_index=r["chunk_index"],
        text=r["text"],
        char_start=r["char_start"],
        char_end=r["char_end"],
        section=r["section"],
    )

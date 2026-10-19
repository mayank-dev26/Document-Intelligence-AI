"""Read each document in full, in the background, so whole-document questions can be answered.

After a document is ready for questions, the language model writes short page-referenced notes about
each section (~9,000 characters) and then an overview built from those notes. Summaries, insights
and brainstorming use these notes, so they cover the whole document rather than only the passages
whose words match the question.

The work pauses while a user question is being answered, stops when
the app closes, and is retried automatically if the AI engine was off when the document was added.
Status per document: waiting -> building -> ready | failed | unavailable.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable

from app.generation.llm import LLM, ExtractiveLLM, LLMUnavailableError
from app.generation.prompts import NOTES_SCHEMA, NOTES_SYSTEM, OVERVIEW_SYSTEM, build_notes_prompt
from app.schemas import Page
from app.store.metadata import MetadataStore

log = logging.getLogger(__name__)

SECTION_CHARS = 9000  # text read per notes call (fits the 8k-token context with room for the reply)
REDUCE_CHARS = 14000  # notes combined per overview call


class Digester:
    def __init__(self, meta: MetadataStore, llm: LLM, wait_until_idle: Callable[[], bool]):
        self.meta = meta
        self.llm = llm
        self._wait_until_idle = wait_until_idle  # returns False when the app is shutting down
        self._stop = threading.Event()

    @property
    def enabled(self) -> bool:
        return not isinstance(self.llm, ExtractiveLLM)

    def stop(self) -> None:
        self._stop.set()

    def build(self, doc_id: str) -> None:
        """Write notes and the overview for one document. Safe to call again (it starts over)."""
        if self._stop.is_set():
            return
        if not self.enabled:
            self.meta.set_overview_status(doc_id, "unavailable")
            return
        try:
            self.meta.set_overview_status(doc_id, "building")
            self.meta.clear_notes(doc_id)
            doc = self.meta.get_document(doc_id)
            if not doc:
                return
            notes: list[tuple[int, int, str]] = []
            for idx, (a, b, text) in enumerate(split_sections(self.meta.get_pages(doc_id))):
                note = self._call(NOTES_SYSTEM, build_notes_prompt(doc.filename, a, b, text))
                if note is None:
                    return
                if note and not self.meta.add_note(doc_id, idx, a, b, note):
                    return  # the document was deleted meanwhile
                notes.append((a, b, note))
            overview = self._overview(doc.filename, [n for n in notes if n[2]])
            if overview is None:
                return
            self.meta.set_overview(doc_id, overview)
            log.info("Overview ready: %s (%d sections)", doc.filename, len(notes))
        except LLMUnavailableError as e:
            log.info("Overview postponed, AI engine unavailable: %s", e)
            self._set_status(doc_id, "unavailable")
        except Exception:  # noqa: BLE001 - a failed overview never affects normal questions
            if not self._stop.is_set():
                log.exception("Overview failed for %s", doc_id)
                self._set_status(doc_id, "failed")

    def _overview(self, filename: str, notes: list[tuple[int, int, str]]) -> str | None:
        if not notes:
            return ""
        if len(notes) == 1:
            return notes[0][2]  # a short document: its notes are the overview
        blocks = [f"(pages {a}-{b}) {t}" if a != b else f"(page {a}) {t}" for a, b, t in notes]
        # Combine in rounds until everything fits in one call.
        while True:
            groups, cur, size = [], [], 0
            for blk in blocks:
                if cur and size + len(blk) > REDUCE_CHARS:
                    groups.append(cur)
                    cur, size = [], 0
                cur.append(blk)
                size += len(blk)
            groups.append(cur)
            results = []
            for g in groups:
                text = self._call(OVERVIEW_SYSTEM, f"DOCUMENT: {filename}\n\nNOTES:\n" + "\n\n".join(g))
                if text is None:
                    return None
                results.append(text)
            if len(results) == 1:
                return results[0]
            blocks = results

    def _call(self, system: str, user: str) -> str | None:
        """One model call, after any user question has finished. None means stop."""
        if self._stop.is_set() or not self._wait_until_idle():
            return None
        from app.generation.answerer import parse_model_json

        raw = parse_model_json(self.llm.complete(system, user, NOTES_SCHEMA))
        if self._stop.is_set():
            return None
        return str(raw.get("notes") or raw.get("answer") or "").strip()

    def _set_status(self, doc_id: str, status: str) -> None:
        with contextlib.suppress(Exception):  # the store may already be closed during shutdown
            self.meta.set_overview_status(doc_id, status)


def split_sections(pages: list[Page], limit: int = SECTION_CHARS) -> list[tuple[int, int, str]]:
    """Group consecutive pages into sections of about `limit` characters: (first page, last page, text).

    Each page's text is labelled with its number so the notes can reference pages. A single page
    longer than the limit is split into several sections.
    """
    sections: list[tuple[int, int, str]] = []
    cur: list[str] = []
    first = last = 0
    size = 0
    for p in pages:
        text = p.text.strip()
        if not text:
            continue
        pieces = [text[i : i + limit] for i in range(0, len(text), limit)]
        for piece in pieces:
            block = f"[Page {p.number}]\n{piece}"
            if cur and size + len(block) > limit:
                sections.append((first, last, "\n\n".join(cur)))
                cur, size = [], 0
            if not cur:
                first = p.number
            cur.append(block)
            last = p.number
            size += len(block) + 2
    if cur:
        sections.append((first, last, "\n\n".join(cur)))
    return sections

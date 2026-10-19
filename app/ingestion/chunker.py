"""Sentence-aware, page-aware chunking.

Design:
* Chunks never cross a page boundary -> every chunk maps to exactly one page citation.
* Chunks are packed from whole sentences up to `chunk_size` characters, with a
  sentence-level overlap so facts that straddle a boundary are still retrievable.
* We keep exact character offsets into the page text, so a cited quote can be
  located and highlighted precisely.
* The nearest section heading is tracked and stored; it is prepended to the text
  that gets embedded ("contextual chunk headers"), which noticeably helps dense
  retrieval on short chunks.
"""

from __future__ import annotations

import re

from app.schemas import Chunk, Page

_ABBREV = "".join(
    rf"(?<!\b{a}\.)"
    for a in ("Pvt", "Ltd", "Inc", "Co", "Rev", "No", "Nos", "Mr", "Ms", "Dr", "St", "vs", "Fig", "approx")
)
_SPLIT = re.compile(
    r"\n\s*\n"  # paragraph break
    r"|(?<!^\d\.)(?<!^\d\d\.)(?<!\be\.g\.)(?<!\bi\.e\.)"
    + _ABBREV  # not '2.' headings / abbreviations
    + r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])",  # sentence end followed by a capital/digit
    re.MULTILINE,
)
_NUMBERED_HEADING = re.compile(
    r"^(?:(?:section|article|chapter|part|clause)\s+\d+[.:]?\s+\S.*|\d+(?:\.\d+)*\.?\s+[A-Z].*)$", re.IGNORECASE
)
_CAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 &/,()'-]{3,80}$")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of sentences / paragraph units in `text`."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SPLIT.finditer(text):
        _add_span(text, start, m.start(), spans)
        start = m.end()
    _add_span(text, start, len(text), spans)
    return spans


def _add_span(text: str, s: int, e: int, out: list[tuple[int, int]]) -> None:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if e > s:
        out.append((s, e))


def _split_long(text: str, s: int, e: int, size: int) -> list[tuple[int, int]]:
    """Hard-split an over-long sentence on whitespace."""
    out, cur = [], s
    while e - cur > size:
        cut = text.rfind(" ", cur, cur + size)
        cut = cut if cut > cur else cur + size
        out.append((cur, cut))
        cur = cut + 1 if text[cut : cut + 1] == " " else cut
    if e > cur:
        out.append((cur, e))
    return out


def looks_like_heading(unit: str) -> bool:
    unit = unit.strip()
    if len(unit) > 90 or unit.endswith((".", ",", ";", ":")) or len(unit.split()) > 12:
        return False
    return bool(_NUMBERED_HEADING.match(unit) or _CAPS_HEADING.match(unit))


def _is_heading(text: str, span: tuple[int, int]) -> bool:
    return looks_like_heading(text[span[0] : span[1]])


def chunk_pages(
    pages: list[Page],
    doc_id: str,
    doc_name: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    section: str | None = None

    for page in pages:
        text = page.text
        units: list[tuple[int, int]] = []
        for s, e in sentence_spans(text):
            units.extend(_split_long(text, s, e, chunk_size) if e - s > chunk_size else [(s, e)])

        i = 0
        while i < len(units):
            # A heading-only unit opens a new section.
            if _is_heading(text, units[i]):
                section = text[units[i][0] : units[i][1]].strip()
            chunk_section = section

            # Greedily pack whole sentences up to chunk_size.
            j = i
            while j + 1 < len(units) and units[j + 1][1] - units[i][0] <= chunk_size:
                j += 1
            # Never end a chunk on a dangling heading - it belongs with the next chunk's content.
            while j > i and _is_heading(text, units[j]):
                j -= 1
            # Headings inside this chunk set the section for the chunks that follow.
            for u in units[i + 1 : j + 1]:
                if _is_heading(text, u):
                    section = text[u[0] : u[1]].strip()

            start, end = units[i][0], units[j][1]
            chunk_text = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{page.number}:{len(chunks)}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    page=page.number,
                    chunk_index=len(chunks),
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                    section=chunk_section,
                )
            )
            if j + 1 >= len(units):
                break
            i = _next_start(units, i, j, chunk_size, chunk_overlap)
    return chunks


def _next_start(units: list[tuple[int, int]], i: int, j: int, size: int, overlap: int) -> int:
    """Index of the first unit of the next chunk.

    Re-uses trailing units of the current chunk (up to `overlap` chars) as long as the
    next chunk can still fit the following new unit; always moves forward.
    """
    nxt = j + 1
    for k in range(j, i, -1):
        trailing = units[j][1] - units[k][0]
        fits_next = units[j + 1][1] - units[k][0] <= size
        if trailing <= overlap and fits_next:
            nxt = k
        else:
            break
    return nxt


def embedding_text(chunk: Chunk) -> str:
    """Text actually sent to the embedding model (contextual header + body)."""
    header = f"Document: {chunk.doc_name}"
    if chunk.section:
        header += f" | Section: {chunk.section}"
    return f"{header}\n{chunk.text}"

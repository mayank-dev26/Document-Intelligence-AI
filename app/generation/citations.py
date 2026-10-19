"""Citation verification: does the quoted text actually appear in the cited source?

LLMs sometimes "cite" a source while paraphrasing or inventing the quote. We check every
quote against the chunk text (exact match after normalisation, then a fuzzy fallback)
and record the matched character span so the UI can highlight it.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from app.text_utils import normalize, normalize_with_map


def locate_quote(quote: str, source: str) -> tuple[float, int | None, int | None]:
    """Return (match_score 0..1, start, end) of `quote` inside `source` (offsets in `source`)."""
    q = normalize(quote).strip(" \"'.")
    if len(q) < 3:
        return 0.0, None, None
    s_norm, idx = normalize_with_map(source)

    pos = s_norm.find(q)
    if pos >= 0:
        return 1.0, idx[pos], idx[pos + len(q) - 1] + 1

    # Fuzzy: align quote against source and measure how much of the quote is covered
    # by a compact region of the source (penalises scattered word-salad matches).
    sm = SequenceMatcher(None, s_norm, q, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 3]
    if not blocks:
        return 0.0, None, None
    # choose the densest window of blocks
    best = (0.0, None, None)
    for i in range(len(blocks)):
        matched = 0
        for j in range(i, len(blocks)):
            matched += blocks[j].size
            span = blocks[j].a + blocks[j].size - blocks[i].a
            score = matched / max(len(q), span)
            if score > best[0]:
                best = (score, blocks[i].a, blocks[j].a + blocks[j].size)
    score, a, b = best
    if a is None:
        return 0.0, None, None
    return round(score, 3), idx[a], idx[b - 1] + 1

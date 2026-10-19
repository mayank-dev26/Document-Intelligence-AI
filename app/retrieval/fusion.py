"""Reciprocal Rank Fusion (Cormack et al., 2009).

RRF only uses ranks, so it fuses BM25 scores (unbounded) and cosine similarities
(bounded) without any score normalisation or tuning - which is why it is the standard
default for hybrid search.
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float, dict[str, float]]]:
    """Fuse several ranked lists of (id, score).

    Returns (id, rrf_score, {list_name: original_score}) sorted best-first.
    """
    fused: dict[str, float] = defaultdict(float)
    origin: dict[str, dict[str, float]] = defaultdict(dict)
    for name, items in ranked_lists.items():
        for rank, (item_id, score) in enumerate(items, start=1):
            fused[item_id] += 1.0 / (k + rank)
            origin[item_id][name] = score
    return sorted(((i, s, origin[i]) for i, s in fused.items()), key=lambda x: -x[1])

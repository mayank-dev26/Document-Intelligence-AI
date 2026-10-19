"""Metric functions. Each is small and unit-testable on its own."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from app.schemas import Answer, Chunk
from app.text_utils import normalize, tokenize


def _norm_numbers(s: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3})", "", normalize(s))


# --- retrieval ---
def is_relevant(chunk: Chunk, item: dict) -> bool:
    """A chunk is relevant if it is from the gold doc+page and contains the evidence phrase."""
    if chunk.doc_name != item["doc"] or chunk.page not in item["pages"]:
        return False
    ev = item.get("evidence")
    return _norm_numbers(ev) in _norm_numbers(chunk.text) if ev else True


def recall_at_k(rel_flags: list[bool], k: int) -> float:
    """Hit rate: 1 if any relevant chunk is in the top-k (single-evidence questions)."""
    return 1.0 if any(rel_flags[:k]) else 0.0


def reciprocal_rank(rel_flags: list[bool]) -> float:
    for i, r in enumerate(rel_flags, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel_flags: list[bool], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rel_flags[:k]) if r)
    n_rel = sum(rel_flags)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_rel, k)))
    return dcg / idcg if idcg else 0.0


# --- answers ---
def key_fact_coverage(answer: str, key_facts: list[str]) -> float:
    """Fraction of required facts (numbers, names, terms) present in the answer."""
    if not key_facts:
        return 1.0
    a = _norm_numbers(answer)
    return sum(_norm_numbers(f) in a for f in key_facts) / len(key_facts)


def token_f1(prediction: str, reference: str) -> float:
    p, r = tokenize(prediction), tokenize(reference)
    if not p or not r:
        return 0.0
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def strip_markers(text: str) -> str:
    return re.sub(r"\[S\d+\]", "", text)


# --- citations ---
def citation_scores(ans: Answer, item: dict) -> dict:
    """Citation precision = verified citations that point at a gold page / all citations.

    Also reports quote verification rate (quotes found verbatim in the source) and whether
    at least one correct citation exists (citation recall for single-evidence questions).
    """
    cits = ans.citations
    if not cits:
        return {"citation_precision": 0.0, "quote_verified_rate": 0.0, "citation_recall": 0.0}
    gold = [c.verified and c.doc_name == item["doc"] and c.page in item["pages"] for c in cits]
    return {
        "citation_precision": sum(gold) / len(cits),
        "quote_verified_rate": sum(c.verified for c in cits) / len(cits),
        "citation_recall": 1.0 if any(gold) else 0.0,
    }


# --- abstention ---
def abstention_scores(rows: list[dict]) -> dict:
    """rows need 'answerable' (gold) and 'abstained' (system)."""
    tp = sum(1 for r in rows if not r["answerable"] and r["abstained"])  # correctly refused
    fp = sum(1 for r in rows if r["answerable"] and r["abstained"])  # wrongly refused
    fn = sum(1 for r in rows if not r["answerable"] and not r["abstained"])  # hallucinated answer
    n_ans = sum(1 for r in rows if r["answerable"])
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "abstain_precision": prec,
        "abstain_recall": rec,
        "abstain_f1": f1,
        "false_refusal_rate": fp / n_ans if n_ans else 0.0,
    }


def threshold_sweep(rows: list[dict]) -> list[dict]:
    """Abstention F1 / false refusals if the evidence gate were set at each threshold.

    Uses each question's recorded confidence, so it needs no extra model calls.
    """
    out = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        sim = [{"answerable": r["answerable"], "abstained": r["abstained"] or r["confidence"] < t} for r in rows]
        out.append({"threshold": float(t), **abstention_scores(sim)})
    return out


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0

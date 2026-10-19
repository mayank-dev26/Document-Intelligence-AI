"""Rerankers score (question, passage) pairs jointly - slower than bi-encoders but much more precise.

Every reranker returns a calibrated relevance in [0, 1]. That number doubles as the
"is there evidence for this question at all?" signal used for abstention.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.local_models import require_local_model
from app.text_utils import tokenize


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model: str, path: Path):
        from sentence_transformers import CrossEncoder

        self.name = model
        self._model = CrossEncoder(str(path), local_files_only=True)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        import torch

        pairs = [(query, p) for p in passages]
        identity = torch.nn.Identity()
        try:  # sentence-transformers >= 4
            logits = self._model.predict(pairs, activation_fn=identity, show_progress_bar=False)
        except TypeError:  # sentence-transformers 3.x
            logits = self._model.predict(pairs, activation_fct=identity, show_progress_bar=False)
        # Raw logits -> sigmoid gives a probability-like relevance we can threshold on.
        return [1.0 / (1.0 + math.exp(-float(x))) for x in logits]


class LexicalReranker:
    """IDF-weighted query-term coverage with a bigram bonus.

    A dependency-free fallback (and a useful baseline): no model download needed.
    """

    name = "lexical-coverage"

    def __init__(self, idf_provider=None, oov_weight: float | None = None):
        """oov_weight: weight for query terms absent from the corpus (None = treat as rarest term).

        Keeping OOV terms heavy is right for the abstention gate ("parental leave" should NOT
        match a corpus without 'parental'); down-weighting them suits sentence selection.
        """
        self._idf = idf_provider or dict
        self._oov = oov_weight

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = tokenize(query)
        if not q:
            return [0.0] * len(passages)
        idf = self._idf()
        default = (self._oov if self._oov is not None else max(idf.values())) if idf else 1.0
        w = {t: max(idf.get(t, default), 0.05) for t in set(q)}
        total = sum(w.values())
        q_bi = set(zip(q, q[1:]))
        out = []
        for p in passages:
            toks = tokenize(p)
            ts = set(toks)
            cov = sum(w[t] for t in w if t in ts) / total
            bi = len(q_bi & set(zip(toks, toks[1:]))) / len(q_bi) if q_bi else 0.0
            out.append(min(1.0, 0.85 * cov + 0.15 * bi))
        return out


class NoReranker:
    name = "none"

    def score(self, query: str, passages: list[str]) -> list[float]:
        return LexicalReranker().score(query, passages)  # still needed for the abstention gate


def build_reranker(s: Settings, idf_provider=None) -> Reranker:
    p = s.reranker_provider.lower()
    if p == "cross-encoder":
        return CrossEncoderReranker(s.reranker_model, require_local_model(s.models_dir, s.reranker_model))
    if p == "lexical":
        return LexicalReranker(idf_provider)
    if p == "none":
        return NoReranker()
    raise ValueError(f"Unknown RERANKER_PROVIDER '{s.reranker_provider}'")

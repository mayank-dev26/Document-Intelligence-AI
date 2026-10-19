"""Understand the question -> gather evidence -> generate (streamed) -> verify citations -> decide.

Every question is first understood: follow-ups are rewritten to stand alone (using the chat history),
and the question is routed to one of three kinds of answer:

  lookup     A specific fact. Retrieve passages and answer strictly from them, with verbatim quotes.
             Unanswerable questions are caught at three independent layers:
               1. Evidence gate:   best passage relevance < ABSTAIN_THRESHOLD -> the model isn't asked.
               2. Model refusal:   the model returns answerable=false when the passages lack the answer.
               3. Grounding check: none of the model's quotes are found in the passages -> withheld.
  overview   Summaries, key points, insights, risks. Uses notes written about every section of the
             document in the background (plus the most relevant passages), so the whole document is covered.
  brainstorm Ideas and recommendations, reasoned only from the documents' notes and passages; each idea
             must cite the part of the document it comes from. No outside knowledge.

`stream()` yields events for the UI: {"type": "status"|"delta"|"final", ...}. `answer()` returns the final one.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections.abc import Iterator

from app.config import Settings
from app.generation.citations import locate_quote
from app.generation.llm import LLM, ExtractiveLLM, LLMUnavailableError
from app.generation.prompts import (
    ANSWER_SCHEMA,
    BRAINSTORM_SYSTEM,
    NOT_FOUND,
    OVERVIEW_ANSWER_SYSTEM,
    SYSTEM_PROMPT,
    UNDERSTAND_SCHEMA,
    UNDERSTAND_SYSTEM,
    build_document_prompt,
    build_understand_prompt,
    build_user_prompt,
)
from app.ingestion.chunker import looks_like_heading, sentence_spans
from app.retrieval.reranker import LexicalReranker
from app.retrieval.retriever import Retriever
from app.schemas import Answer, Citation, QueryRequest, RetrievalMode, RetrievedPassage, Timings

log = logging.getLogger(__name__)
_MARKER = re.compile(r"\[([SN])(\d+)\]")
INTENTS = ("lookup", "overview", "brainstorm")

_OVERVIEW_WORDS = re.compile(
    r"\b(summar\w*|overview|gist|tl;?dr|key (?:points?|insights?|findings?|takeaways?|themes?)|insights?|"
    r"main (?:points?|ideas?|themes?|findings?|topics?)|takeaways?|highlights?|themes?|risks?|"
    r"what (?:is|'s) (?:this|the) (?:document|report|file|pdf|paper|project) about)\b",
    re.IGNORECASE,
)
_BRAINSTORM_WORDS = re.compile(
    r"\b(brainstorm\w*|ideas?|suggest\w*|recommend\w*|improv\w*|what could|how (?:could|might|can|should) "
    r"(?:we|i|they|the)|alternatives?|next steps?|opportunit\w+|what if|propos\w*|implications?)\b",
    re.IGNORECASE,
)

NOTES_BUDGET_CHARS = 7000  # notes given to the model for whole-document answers
OVERVIEW_BUDGET_CHARS = 4000  # overviews, shared between the documents asked about
DOC_PASSAGES = 4  # exact passages added to whole-document answers


def guess_intent(question: str) -> str:
    """Keyword routing, used when no language model is available (and as a safety net)."""
    if _BRAINSTORM_WORDS.search(question):
        return "brainstorm"
    if _OVERVIEW_WORDS.search(question):
        return "overview"
    return "lookup"


def partial_json_string(buf: str, field: str) -> str:
    """The value of a JSON string field decoded so far, from an incomplete JSON document (for streaming)."""
    m = re.search(rf'"{field}"\s*:\s*"', buf)
    if not m:
        return ""
    out, i = [], m.end()
    escapes = {"n": "\n", "t": "\t", "r": "", '"': '"', "\\": "\\", "/": "/", "b": "", "f": ""}
    while i < len(buf):
        ch = buf[i]
        if ch == '"':
            break
        if ch == "\\":
            if i + 1 >= len(buf):
                break
            nxt = buf[i + 1]
            if nxt == "u":
                if i + 6 > len(buf):
                    break
                with contextlib.suppress(ValueError):
                    out.append(chr(int(buf[i + 2 : i + 6], 16)))
                i += 6
                continue
            out.append(escapes.get(nxt, nxt))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


class Answerer:
    def __init__(self, settings: Settings, llm: LLM, retriever: Retriever):
        self.s = settings
        self.llm = llm
        self.retriever = retriever

    # --- public ---
    def answer(self, req: QueryRequest, abstain_threshold: float | None = None) -> Answer:
        """abstain_threshold overrides the configured gate (the evaluator passes 0 to observe the model)."""
        final = None
        for event in self.stream(req, abstain_threshold):
            if event["type"] == "final":
                final = event["answer"]
        return final

    def stream(self, req: QueryRequest, abstain_threshold: float | None = None) -> Iterator[dict]:
        t_start = time.perf_counter()
        yield {"type": "status", "text": "Understanding the question"}
        intent, question = self._understand(req)
        if intent == "lookup":
            gate = self.s.abstain_threshold if abstain_threshold is None else abstain_threshold
            yield from self._lookup(req, question, gate, t_start)
        else:
            yield from self._whole_document(req, question, intent, t_start)

    # --- understanding ---
    def _understand(self, req: QueryRequest) -> tuple[str, str]:
        if req.intent in INTENTS:
            return req.intent, req.question
        history = [(t.question, _clip(t.answer, 500)) for t in req.history[-3:]]
        if not isinstance(self.llm, ExtractiveLLM):
            try:
                raw = parse_model_json(
                    self.llm.complete(
                        UNDERSTAND_SYSTEM, build_understand_prompt(req.question, history), UNDERSTAND_SCHEMA
                    )
                )
                intent = raw.get("intent") if raw.get("intent") in INTENTS else guess_intent(req.question)
                # Only follow-ups need rewriting; a first question is used exactly as asked.
                question = (str(raw.get("question") or "").strip() if history else "") or req.question
                return intent, question
            except Exception as e:  # noqa: BLE001 - routing must never block an answer
                log.info("Question understanding fell back to keywords: %s", e)
        return guess_intent(req.question), req.question

    # --- lookup: one fact, strict quotes ---
    def _lookup(self, req: QueryRequest, question: str, gate: float, t_start: float) -> Iterator[dict]:
        yield {"type": "status", "text": "Finding the relevant pages"}
        top_k = req.top_k or self.s.top_k
        passages, stage = self.retriever.retrieve(question, req.mode, top_k, req.doc_ids)
        confidence = self.retriever.relevance(question, passages)
        state = {"note": None, "llm": self.llm.name}
        timings = Timings(**stage)

        def final(answer: str, answerable: bool, citations: list[Citation], reason=None, code=None) -> dict:
            timings.total_ms = (time.perf_counter() - t_start) * 1000
            return {
                "type": "final",
                "answer": Answer(
                    question=req.question,
                    kind="lookup",
                    understood_as=question if question != req.question else None,
                    answer=answer,
                    answerable=answerable,
                    abstain_reason=reason,
                    abstain_code=code,
                    note=state["note"],
                    confidence=round(confidence, 4),
                    citations=citations,
                    passages=passages,
                    mode=req.mode,
                    timings=timings,
                    llm=state["llm"],
                ),
            }

        # ---- Layer 1: evidence gate ----
        if not passages:
            yield final(NOT_FOUND, False, [], "Nothing in your documents matched this question.", "no_evidence")
            return
        if confidence < gate:
            log.info("Evidence relevance %.2f below threshold %.2f", confidence, gate)
            yield final(NOT_FOUND, False, [], "Your documents don't seem to cover this topic.", "below_threshold")
            return

        # ---- Generation ----
        yield {"type": "status", "text": "Writing the answer"}
        t_gen = time.perf_counter()
        raw = yield from self._generate(
            SYSTEM_PROMPT, build_user_prompt(question, passages), lambda: self._extractive(question, passages), state
        )
        timings.generation_ms = (time.perf_counter() - t_gen) * 1000

        # ---- Layer 2: model refusal ----
        if not raw.get("answerable", True):
            yield final(NOT_FOUND, False, [], "Your documents don't contain this information.", "model_refused")
            return

        answer_text = str(raw.get("answer", "")).strip()
        citations = self._verify(raw.get("citations") or [], answer_text, passages, [])

        # ---- Layer 3: grounding check ----
        if not any(c.verified for c in citations):
            yield final(
                NOT_FOUND,
                False,
                citations,
                "An answer was drafted, but its quotes couldn't be found in your documents, so it isn't shown.",
                "ungrounded",
            )
            return
        if raw.get("missing"):
            answer_text += f"\n\n_Not covered by the documents: {raw['missing']}_"
        yield final(answer_text, True, citations)

    # --- overview / brainstorm: whole document ---
    def _whole_document(self, req: QueryRequest, question: str, intent: str, t_start: float) -> Iterator[dict]:
        meta = self.retriever.meta
        ready = set(meta.ready_doc_ids())
        doc_ids = [d for d in (req.doc_ids or ready) if d in ready]
        if isinstance(self.llm, ExtractiveLLM) or not doc_ids:
            # Summaries and ideas need the language model; answer the closest fact instead.
            yield from self._lookup(req, question, self.s.abstain_threshold, t_start)
            return

        yield {"type": "status", "text": "Reviewing the whole document"}
        t0 = time.perf_counter()
        docs = {d.doc_id: d for d in meta.list_documents() if d.doc_id in doc_ids}
        per_doc = max(800, OVERVIEW_BUDGET_CHARS // len(doc_ids))  # several documents share the budget
        overviews = [(docs[d].filename, _clip(text, per_doc)) for d in doc_ids if (text := meta.get_overview(d))]
        notes = self._pick_notes(question, meta.notes_for(doc_ids))
        passages, stage = self.retriever.retrieve(question, RetrievalMode.hybrid_rerank, DOC_PASSAGES, doc_ids)
        timings = Timings(**stage)
        timings.retrieval_ms = (time.perf_counter() - t0) * 1000
        pending = [d for d in doc_ids if docs[d].overview_status in ("waiting", "building")]
        state = {"note": None, "llm": self.llm.name}
        if pending:
            state["note"] = (
                "The document is still being read in full, so this answer covers the parts read so far. "
                "Ask again in a minute for a complete picture."
            )

        system = OVERVIEW_ANSWER_SYSTEM if intent == "overview" else BRAINSTORM_SYSTEM
        history = [(t.question, _clip(t.answer, 500)) for t in req.history[-3:]]
        prompt = build_document_prompt(
            question,
            overviews,
            [(n["filename"], n["page_start"], n["page_end"], n["text"]) for n in notes],
            passages,
            history,
        )
        yield {"type": "status", "text": "Writing the answer"}
        t_gen = time.perf_counter()
        raw = yield from self._generate(system, prompt, lambda: self._extractive(question, passages), state)
        timings.generation_ms = (time.perf_counter() - t_gen) * 1000

        answer_text = str(raw.get("answer", "")).strip()
        citations = self._verify(raw.get("citations") or [], answer_text, passages, notes)
        grounded = any(c.kind == "notes" or c.verified for c in citations)
        answerable = bool(raw.get("answerable", True)) and bool(answer_text) and grounded
        reason, code = None, None
        if not answerable:
            answer_text = NOT_FOUND
            reason = "Your documents don't contain enough on this to answer it."
            code = "model_refused" if not raw.get("answerable", True) else "ungrounded"
        elif raw.get("missing"):
            answer_text += f"\n\n_Not covered by the documents: {raw['missing']}_"
        timings.total_ms = (time.perf_counter() - t_start) * 1000
        yield {
            "type": "final",
            "answer": Answer(
                question=req.question,
                kind=intent if state["llm"] == self.llm.name else "lookup",  # quoted fallback isn't a summary
                understood_as=question if question != req.question else None,
                answer=answer_text,
                answerable=answerable,
                abstain_reason=reason,
                abstain_code=code,
                note=state["note"],
                confidence=1.0,
                citations=citations if answerable else [],
                passages=passages,
                mode=RetrievalMode.hybrid_rerank,
                timings=timings,
                llm=state["llm"],
            ),
        }

    def _pick_notes(self, question: str, notes: list[dict]) -> list[dict]:
        """All notes if they fit; otherwise the most relevant ones, kept in document order."""
        if sum(len(n["text"]) for n in notes) <= NOTES_BUDGET_CHARS:
            return notes
        scores = self.retriever.reranker.score(question, [n["text"] for n in notes])
        chosen, used = set(), 0
        for i in sorted(range(len(notes)), key=lambda i: -scores[i]):
            if used + len(notes[i]["text"]) > NOTES_BUDGET_CHARS:
                continue
            chosen.add(i)
            used += len(notes[i]["text"])
        return [n for i, n in enumerate(notes) if i in chosen]

    # --- generation (streamed) ---
    def _generate(self, system: str, user: str, fallback, state: dict):
        """Stream the model's answer as 'delta' events; return the parsed JSON (yield from)."""
        if isinstance(self.llm, ExtractiveLLM):
            raw = fallback()
            if raw.get("answerable"):
                yield {"type": "delta", "text": _strip_markers(raw["answer"])}
            return raw
        buf, shown = "", ""
        try:
            for piece in self.llm.stream(system, user, ANSWER_SCHEMA):
                buf += piece
                text = partial_json_string(buf, "answer")
                if len(text) > len(shown):
                    yield {"type": "delta", "text": text[len(shown) :]}
                    shown = text
        except LLMUnavailableError as e:
            # Still useful without the model: quote the most relevant sentences instead.
            log.warning("Language model unavailable (%s); answering with quoted sentences", e)
            state["note"] = "The AI engine isn't running, so this answer quotes the most relevant sentences directly."
            state["llm"] = "extractive (fallback)"
            yield {"type": "reset"}
            raw = fallback()
            if raw.get("answerable"):
                yield {"type": "delta", "text": _strip_markers(raw["answer"])}
            return raw
        return parse_model_json(buf)

    # --- citations ---
    def _verify(
        self, raw_cits: list, answer_text: str, passages: list[RetrievedPassage], notes: list[dict]
    ) -> list[Citation]:
        out: list[Citation] = []
        seen: set[tuple[str, str]] = set()
        for rc in raw_cits:
            if not isinstance(rc, dict):
                continue
            m = re.match(r"\s*\[?([SN])(\d+)", str(rc.get("source", "")).upper())
            if not m:
                continue
            kind, n, quote = m.group(1), int(m.group(2)), str(rc.get("quote", "")).strip()
            if (kind, str(n)) in seen or (kind, quote) in seen:
                continue
            if kind == "S" and 1 <= n <= len(passages):
                out.append(self._citation(n, quote, passages[n - 1]))
            elif kind == "N" and 1 <= n <= len(notes):
                out.append(_note_citation(n, notes[n - 1]))
            else:
                continue
            seen.update({(kind, str(n)), (kind, quote)})

        # Markers used in the answer but missing from the citation list.
        cited = {c.marker for c in out}
        for kind, num in sorted(set(_MARKER.findall(answer_text)), key=lambda x: (x[0], int(x[1]))):
            n = int(num)
            if f"{kind}{n}" in cited:
                continue
            if kind == "S" and 1 <= n <= len(passages):
                out.append(self._citation(n, "", passages[n - 1]))
            elif kind == "N" and 1 <= n <= len(notes):
                out.append(_note_citation(n, notes[n - 1]))
        return out

    def _citation(self, n: int, quote: str, passage: RetrievedPassage) -> Citation:
        c = passage.chunk
        score, start, end = locate_quote(quote, c.text) if quote else (0.0, None, None)
        return Citation(
            marker=f"S{n}",
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            doc_name=c.doc_name,
            page=c.page,
            quote=quote,
            verified=score >= self.s.citation_match_threshold,
            match_score=score,
            quote_start=start,
            quote_end=end,
        )

    def _extractive(self, question: str, passages: list[RetrievedPassage]) -> dict:
        """No-LLM fallback: pick the most question-relevant sentences and quote them."""
        scorer = LexicalReranker(lambda: self.retriever.bm25.idf, oov_weight=0.1)
        cands: list[tuple[float, int, str]] = []
        for i, p in enumerate(passages[:4], start=1):
            context = p.chunk.section or ""
            for s, e in sentence_spans(p.chunk.text):
                sent = p.chunk.text[s:e].strip()
                if len(sent) >= 25 and not looks_like_heading(sent):
                    score = scorer.score(question, [f"{context} {sent}"])[0]
                    cands.append((score - 0.02 * (i - 1), i, sent))  # small prior for higher-ranked passages
        cands.sort(key=lambda x: -x[0])
        picked = [c for c in cands[:2] if c[0] >= 0.4 and c[0] >= 0.85 * cands[0][0]]
        if not picked:
            return {
                "answerable": False,
                "answer": NOT_FOUND,
                "citations": [],
                "missing": "No sentence in the retrieved passages addresses the question.",
            }
        return {
            "answerable": True,
            "answer": " ".join(f"{sent} [S{i}]" for _, i, sent in picked),
            "citations": [{"source": f"S{i}", "quote": sent} for _, i, sent in picked],
            "missing": None,
        }


def _note_citation(n: int, note: dict) -> Citation:
    return Citation(
        marker=f"N{n}",
        kind="notes",
        chunk_id="",
        doc_id=note["doc_id"],
        doc_name=note["filename"],
        page=note["page_start"],
        page_end=note["page_end"],
        quote="",
        verified=False,
        match_score=0.0,
        notes=note["text"],
    )


def _clip(text: str, limit: int) -> str:
    text = _strip_markers(text)
    return text if len(text) <= limit else text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _strip_markers(text: str) -> str:
    return re.sub(r"\s*\[[SN]\d+\]", "", text)


def parse_model_json(text: str) -> dict:
    """Extract the JSON object from a model reply (tolerates code fences and extra text)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)) if start >= 0 else []:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    log.warning("Model did not return valid JSON; treating output as un-cited text")
    return {"answerable": True, "answer": text, "citations": [], "missing": None}

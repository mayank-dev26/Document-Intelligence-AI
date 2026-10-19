"""Runs the benchmark: retrieval quality per mode, then end-to-end answer quality.

Dataset format (JSONL, one question per line):
  {"id", "question", "answerable": bool, "doc": filename, "pages": [int], "evidence": phrase,
   "reference": gold answer, "key_facts": [strings that must appear in a correct answer]}
Unanswerable questions only need id, question and answerable=false.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from statistics import mean

from app.generation.answerer import parse_model_json
from app.generation.llm import ExtractiveLLM
from app.pipeline import Pipeline
from app.schemas import QueryRequest, RetrievalMode
from evaluation import metrics as M

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
K_VALUES = (1, 3, 5, 10)

JUDGE_PROMPT = """You grade answers from a document QA system. Compare the SYSTEM ANSWER to the REFERENCE.
Score 1 if it is correct and complete, 0.5 if partially correct or missing a key detail, 0 if wrong.
Ignore citation markers like [S1] and differences in wording.
Return JSON only: {"score": 0 | 0.5 | 1, "reason": "short reason"}"""
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number", "enum": [0, 0.5, 1]}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
    "additionalProperties": False,
}


def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(mean(vals), 4) if vals else 0.0


def evaluate_retrieval(pipe: Pipeline, items: list[dict], mode: RetrievalMode) -> tuple[dict, list[dict]]:
    rows = []
    for it in (i for i in items if i["answerable"]):
        t0 = time.perf_counter()
        passages, _ = pipe.retriever.retrieve(it["question"], mode, max(K_VALUES))
        latency = (time.perf_counter() - t0) * 1000
        flags = [M.is_relevant(p.chunk, it) for p in passages]
        row = {
            "id": it["id"],
            "mode": mode.value,
            "latency_ms": latency,
            "mrr": M.reciprocal_rank(flags),
            "ndcg@5": M.ndcg_at_k(flags, 5),
            "first_relevant_rank": next((i + 1 for i, f in enumerate(flags) if f), None),
        }
        row.update({f"recall@{k}": M.recall_at_k(flags, k) for k in K_VALUES})
        rows.append(row)
    lat = [r["latency_ms"] for r in rows]
    summary = {f"recall@{k}": _avg(rows, f"recall@{k}") for k in K_VALUES}
    summary.update(
        {
            "mrr": _avg(rows, "mrr"),
            "ndcg@5": _avg(rows, "ndcg@5"),
            "latency_p50_ms": round(M.percentile(lat, 50), 2),
            "latency_p95_ms": round(M.percentile(lat, 95), 2),
            "n": len(rows),
        }
    )
    return summary, rows


def _judge(pipe: Pipeline, question: str, reference: str, answer: str) -> float | None:
    if isinstance(pipe.llm, ExtractiveLLM):
        return None
    try:
        raw = parse_model_json(
            pipe.llm.complete(
                JUDGE_PROMPT, f"QUESTION: {question}\nREFERENCE: {reference}\nSYSTEM ANSWER: {answer}", JUDGE_SCHEMA
            )
        )
        return float(raw.get("score", 0))
    except Exception as e:  # noqa: BLE001
        log.warning("Judge failed: %s", e)
        return None


def evaluate_generation(
    pipe: Pipeline,
    items: list[dict],
    mode: RetrievalMode,
    threshold: float,
    judge: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    rows = []
    failed: list[str] = []
    for n, it in enumerate(items, start=1):
        # Gate disabled (0.0) so we observe the model on every question; the gate is applied
        # post-hoc below, which also lets us sweep thresholds without re-running anything.
        try:
            ans = pipe.ask(QueryRequest(question=it["question"], mode=mode, intent="lookup"), abstain_threshold=0.0)
        except Exception as e:  # noqa: BLE001 - one failed LLM call must not lose the whole run
            failed.append(it["id"])
            log.warning("[%s] %s failed: %s", mode.value, it["id"], e)
            if progress:
                progress(f"[{mode.value}] {n}/{len(items)} {it['id']} FAILED: {e}")
            continue
        model_abstained = not ans.answerable
        abstained = model_abstained or ans.confidence < threshold
        row = {
            "id": it["id"],
            "mode": mode.value,
            "question": it["question"],
            "answerable": it["answerable"],
            "abstained": abstained,
            "model_abstained": model_abstained,
            "confidence": ans.confidence,
            "total_ms": ans.timings.total_ms,
            "generation_ms": ans.timings.generation_ms,
            "answer": ans.answer if not abstained else "[ABSTAINED] " + (ans.abstain_reason or "below gate threshold"),
            "citations": "; ".join(
                f"{c.marker}:{c.doc_name}#p{c.page}{'' if c.verified else '(unverified)'}" for c in ans.citations
            ),
        }
        if it["answerable"]:
            if abstained:
                row.update(
                    correctness=0.0,
                    f1=0.0,
                    citation_precision=0.0,
                    quote_verified_rate=0.0,
                    citation_recall=0.0,
                    judge=0.0 if judge else None,
                )
            else:
                clean = M.strip_markers(ans.answer)
                row.update(
                    correctness=M.key_fact_coverage(clean, it.get("key_facts", [])),
                    f1=M.token_f1(clean, it.get("reference", "")),
                    **M.citation_scores(ans, it),
                )
                row["judge"] = _judge(pipe, it["question"], it.get("reference", ""), clean) if judge else None
        else:
            row["correctness"] = 1.0 if abstained else 0.0
        rows.append(row)
        if progress:
            progress(f"[{mode.value}] {n}/{len(items)} {it['id']}")

    ans_rows = [r for r in rows if r["answerable"]]
    answered = [r for r in ans_rows if not r["abstained"]]
    lat = [r["total_ms"] for r in rows]
    summary = {
        "answer_correctness": _avg(ans_rows, "correctness"),  # key-fact coverage, abstentions count as 0
        "answer_f1": _avg(ans_rows, "f1"),
        "overall_accuracy": _avg(rows, "correctness"),  # incl. correct refusals on unanswerable
        "citation_precision": _avg(answered, "citation_precision"),
        "citation_recall": _avg(answered, "citation_recall"),
        "quote_verified_rate": _avg(answered, "quote_verified_rate"),
        **{k: round(v, 4) for k, v in M.abstention_scores(rows).items()},
        "latency_p50_ms": round(M.percentile(lat, 50), 1),
        "latency_p95_ms": round(M.percentile(lat, 95), 1),
        "n": len(rows),
    }
    if failed:
        summary["n_failed"] = len(failed)  # excluded from every metric above
    if judge:
        summary["llm_judge"] = _avg(ans_rows, "judge")
    sweep = M.threshold_sweep(rows)
    return summary, rows, sweep


def run_evaluation(
    pipe: Pipeline,
    dataset: Path,
    modes: list[RetrievalMode] | None = None,
    generate_modes: list[RetrievalMode] | None = None,
    judge: bool = False,
    progress: Callable[[str], None] | None = None,
    save: bool = True,
) -> dict:
    items = load_dataset(dataset)
    modes = modes or list(RetrievalMode)
    generate_modes = (
        generate_modes if generate_modes is not None else [RetrievalMode.vector, RetrievalMode.hybrid_rerank]
    )
    have = {d.filename for d in pipe.list_documents()}
    missing = sorted({i["doc"] for i in items if i["answerable"]} - have)

    report: dict = {
        "meta": {
            "dataset": dataset.name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "n_questions": len(items),
            "n_answerable": sum(i["answerable"] for i in items),
            "missing_documents": missing,
            **pipe.info(),
        },
        "retrieval": {},
        "generation": {},
        "threshold_sweep": {},
        "per_question": [],
    }
    for mode in modes:
        summary, rows = evaluate_retrieval(pipe, items, mode)
        report["retrieval"][mode.value] = summary
        report["per_question"].extend({"stage": "retrieval", **r} for r in rows)
        if progress:
            progress(f"retrieval [{mode.value}] recall@5={summary['recall@5']:.2f} mrr={summary['mrr']:.2f}")
    for mode in generate_modes:
        summary, rows, sweep = evaluate_generation(pipe, items, mode, pipe.s.abstain_threshold, judge, progress)
        report["generation"][mode.value] = summary
        report["threshold_sweep"][mode.value] = sweep
        report["per_question"].extend({"stage": "generation", **r} for r in rows)

    if save:
        save_report(report)
    return report


# --- persistence ---
def save_report(report: dict) -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = REPORTS_DIR / f"eval_{stamp}"
    paths = {
        "json": str(base.with_suffix(".json")),
        "markdown": str(base.with_suffix(".md")),
        "csv": str(base) + "_questions.csv",
    }
    report["meta"]["files"] = paths
    Path(paths["json"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(paths["markdown"]).write_text(to_markdown(report), encoding="utf-8")
    rows = report["per_question"]
    if rows:
        fields = sorted({k for r in rows for k in r})
        with open(paths["csv"], "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return paths


def list_reports() -> list[Path]:
    return sorted(REPORTS_DIR.glob("eval_*.json"), reverse=True) if REPORTS_DIR.exists() else []


def _table(header: list[str], rows: list[list]) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def to_markdown(report: dict) -> str:
    m = report["meta"]
    out = [
        f"# Evaluation report - {m['created']}",
        "",
        f"Dataset `{m['dataset']}`: {m['n_questions']} questions ({m['n_answerable']} answerable).  ",
        f"Embedder `{m['embedder']}` · Reranker `{m['reranker']}` · LLM `{m['llm']}` · "
        f"abstain threshold {m['abstain_threshold']}",
        "",
    ]
    if m.get("missing_documents"):
        out += [f"> Missing documents (ingest them first): {', '.join(m['missing_documents'])}", ""]
    if report["retrieval"]:
        cols = [f"recall@{k}" for k in K_VALUES] + ["mrr", "ndcg@5", "latency_p50_ms", "latency_p95_ms"]
        out += [
            "## Retrieval",
            "",
            _table(["mode"] + cols, [[mode] + [s[c] for c in cols] for mode, s in report["retrieval"].items()]),
            "",
        ]
    if report["generation"]:
        cols = [
            "answer_correctness",
            "answer_f1",
            "overall_accuracy",
            "citation_precision",
            "quote_verified_rate",
            "abstain_precision",
            "abstain_recall",
            "false_refusal_rate",
            "latency_p50_ms",
            "latency_p95_ms",
        ]
        if any("llm_judge" in s for s in report["generation"].values()):
            cols.insert(1, "llm_judge")
        out += [
            "## End-to-end answers",
            "",
            _table(
                ["mode"] + cols, [[mode] + [s.get(c, "") for c in cols] for mode, s in report["generation"].items()]
            ),
            "",
        ]
        for mode, s in report["generation"].items():
            if s.get("n_failed"):
                note = "some LLM calls failed"
                out += [f"> **{mode}**: {note}; {s.get('n_failed', 0)} question(s) excluded from the metrics.", ""]
        out += ["## Abstention threshold sweep", ""]
        for mode, sweep in report["threshold_sweep"].items():
            best = max(sweep, key=lambda r: (r["abstain_f1"], -r["false_refusal_rate"]))
            out.append(
                f"- **{mode}**: best abstain-F1 {best['abstain_f1']:.3f} at threshold {best['threshold']:.2f} "
                f"(false refusals {best['false_refusal_rate']:.1%})"
            )
        out.append("")
    return "\n".join(out)

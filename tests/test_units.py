"""Fast unit tests for the pure functions."""

from app.generation.answerer import parse_model_json
from app.generation.citations import locate_quote
from app.ingestion.chunker import chunk_pages, looks_like_heading, sentence_spans
from app.retrieval.fusion import reciprocal_rank_fusion
from app.schemas import Page
from evaluation import metrics as M

PAGE = (
    "1. Payment Terms\n\nClient will pay all undisputed invoices within 45 days of receipt. "
    "Late payments accrue interest at 1% per month.\n\n2. Term\n\nThe initial term is 36 months. "
    "Northwind Analytics Pvt. Ltd. is the Provider."
)


def test_sentence_split_keeps_numbered_headings_and_abbreviations():
    sents = [PAGE[s:e] for s, e in sentence_spans(PAGE)]
    assert "1. Payment Terms" in sents
    assert any(s.startswith("Northwind Analytics Pvt. Ltd.") for s in sents)


def test_heading_detection():
    assert looks_like_heading("6. Controller Error Codes")
    assert looks_like_heading("LIMITATION OF LIABILITY")
    assert not looks_like_heading("Client will pay all undisputed invoices within 45 days of receipt.")


def test_chunk_offsets_are_exact_and_pages_respected():
    pages = [Page(number=1, text=PAGE * 4), Page(number=2, text=PAGE)]
    chunks = chunk_pages(pages, "d1", "doc.pdf", chunk_size=200, chunk_overlap=60)
    assert len(chunks) > 3
    for c in chunks:
        assert pages[c.page - 1].text[c.char_start : c.char_end] == c.text
        assert len(c.text) <= 200
    assert {c.page for c in chunks} == {1, 2}
    assert chunks[0].section is None or "Payment" in chunks[0].section


def test_chunker_terminates_with_giant_sentence():
    pages = [Page(number=1, text="word " * 2000)]
    chunks = chunk_pages(pages, "d", "d.pdf", chunk_size=300, chunk_overlap=50)
    assert all(len(c.text) <= 300 for c in chunks) and len(chunks) >= 30


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion({"a": [("x", 9), ("y", 8)], "b": [("y", 0.9), ("z", 0.5)]}, k=60)
    assert fused[0][0] == "y"  # in both lists -> wins
    assert fused[0][2] == {"a": 8, "b": 0.9}


def test_quote_exact_fuzzy_and_fabricated():
    src = "Client will pay all undisputed invoices within 45 days of receipt."
    score, s, e = locate_quote("pay all undisputed invoices within 45 days", src)
    assert score == 1.0 and src[s:e] == "pay all undisputed invoices within 45 days"
    score, _, _ = locate_quote("Client will pay all  undisputed invoices within 45 days of reciept", src)
    assert 0.85 <= score < 1.0  # typo / whitespace -> still verified
    score, _, _ = locate_quote("The client may pay invoices whenever convenient.", src)
    assert score < 0.85  # paraphrase / invention -> not verified


def test_parse_model_json_handles_fences_and_chatter():
    assert parse_model_json('```json\n{"answerable": false, "answer": "x"}\n```')["answerable"] is False
    out = parse_model_json('Sure! {"answer": "a \\"quoted\\" {brace}", "citations": []} hope that helps')
    assert out["answer"] == 'a "quoted" {brace}'
    assert parse_model_json("plain text")["citations"] == []


def test_retrieval_metrics():
    flags = [False, True, False]
    assert M.recall_at_k(flags, 1) == 0 and M.recall_at_k(flags, 2) == 1
    assert M.reciprocal_rank(flags) == 0.5
    assert 0 < M.ndcg_at_k(flags, 3) < 1


def test_answer_metrics():
    assert M.key_fact_coverage("Every 4,000 operating hours", ["4000"]) == 1.0
    assert M.key_fact_coverage("24 months", ["24 months", "30 months"]) == 0.5
    assert M.token_f1("within 45 days", "within 45 days of receipt") > 0.7


def test_abstention_metrics():
    rows = [
        {"answerable": False, "abstained": True},
        {"answerable": False, "abstained": False},
        {"answerable": True, "abstained": True},
        {"answerable": True, "abstained": False},
    ]
    s = M.abstention_scores(rows)
    assert s["abstain_precision"] == 0.5 and s["abstain_recall"] == 0.5 and s["false_refusal_rate"] == 0.5


def test_sentences_split_across_pdf_blocks_are_rejoined():
    from app.ingestion.parser import _join_continuations

    parts = ["1. Working Hours", "Core hours are 10:00 to", "16:00.", "2. Annual Leave", "Employees receive 24 days."]
    assert _join_continuations(parts) == [
        "1. Working Hours",
        "Core hours are 10:00 to 16:00.",
        "2. Annual Leave",
        "Employees receive 24 days.",
    ]

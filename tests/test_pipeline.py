"""End-to-end behaviour of the pipeline with offline stand-ins."""

import time

import pytest

from app.pipeline import Pipeline
from app.schemas import QueryRequest, RetrievalMode
from tests.conftest import HANDBOOK, make_pdf, offline_settings


def _doc(pipeline, filename):
    return next(d for d in pipeline.list_documents() if d.filename == filename)


def test_documents_are_read_and_ready(pipeline):
    docs = {d.filename: d for d in pipeline.list_documents()}
    assert {d.status for d in docs.values()} == {"ready"}
    assert docs["handbook.pdf"].num_pages == 3 and docs["pump_manual.pdf"].num_pages == 2
    assert docs["meeting_notes.docx"].num_chunks >= 1


def test_running_footer_is_removed(pipeline):
    text = pipeline.page_text(_doc(pipeline, "pump_manual.pdf").doc_id, 2)
    assert "Rev C" not in text and "E04" in text


def test_reupload_returns_the_same_document(pipeline, corpus):
    before = len(pipeline.list_documents())
    doc, created = pipeline.ingest_file(corpus / "handbook.pdf")
    assert not created and doc.filename == "handbook.pdf" and len(pipeline.list_documents()) == before


@pytest.mark.parametrize("mode", list(RetrievalMode))
def test_answer_cites_the_right_page(pipeline, mode):
    a = pipeline.ask(QueryRequest(question="How many days of annual leave do employees get?", mode=mode))
    assert a.answerable and "24 days" in a.answer
    good = [c for c in a.citations if c.verified]
    assert good and good[0].doc_name == "handbook.pdf" and good[0].page == 1
    chunk = next(p.chunk for p in a.passages if p.chunk.chunk_id == good[0].chunk_id)
    assert chunk.text[good[0].quote_start : good[0].quote_end]


def test_exact_identifier_question(pipeline):
    a = pipeline.ask(QueryRequest(question="What does error code E04 mean?"))
    assert a.answerable and "Dry-run" in a.answer and a.citations[0].page == 2


def test_word_documents_are_searchable(pipeline):
    a = pipeline.ask(QueryRequest(question="How much marketing budget was approved?"))
    assert a.answerable and "40 lakh" in a.answer and a.citations[0].doc_name == "meeting_notes.docx"


def test_out_of_scope_question_is_refused_in_plain_words(pipeline):
    a = pipeline.ask(QueryRequest(question="What is the parental leave entitlement in Germany?"))
    assert not a.answerable and a.abstain_code in {"below_threshold", "model_refused", "no_evidence"}
    assert "threshold" not in a.abstain_reason.lower()  # no technical wording for the user


def test_scope_limits_search_to_one_document(pipeline):
    manual = _doc(pipeline, "pump_manual.pdf")
    a = pipeline.ask(QueryRequest(question="How many days of annual leave?", doc_ids=[manual.doc_id]))
    assert all(p.chunk.doc_id == manual.doc_id for p in a.passages)


def test_page_render_with_highlight(pipeline):
    doc = _doc(pipeline, "handbook.pdf")
    png = pipeline.render_page(doc.doc_id, 2, "Employees must give 30 days of written notice")
    assert png and png[:4] == b"\x89PNG"
    assert pipeline.render_page(doc.doc_id, 99) is None
    assert pipeline.render_page(_doc(pipeline, "meeting_notes.docx").doc_id, 1) is None  # PDFs only


# --- background processing & lifecycle ---
def test_upload_returns_immediately_then_becomes_ready(tmp_path):
    p = Pipeline(offline_settings(tmp_path / "data"))
    try:
        doc, created = p.submit_file(make_pdf(tmp_path / "h.pdf", HANDBOOK))
        assert created and doc.status == "processing"
        for _ in range(100):
            if p.meta.get_document(doc.doc_id).status != "processing":
                break
            time.sleep(0.05)
        assert p.meta.get_document(doc.doc_id).status == "ready"
    finally:
        p.close()


def test_unreadable_files_fail_with_a_plain_reason_and_can_be_retried(tmp_path):
    p = Pipeline(offline_settings(tmp_path / "data"))
    try:
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4 this is not a real pdf")
        doc, _ = p.ingest_file(broken)
        assert doc.status == "failed" and "damaged" in doc.error
        empty = make_pdf(tmp_path / "blank.pdf", [""])
        doc2, _ = p.ingest_file(empty)
        assert doc2.status == "failed" and "No readable text" in doc2.error
        again, created = p.ingest_file(broken)  # failed uploads are retried, not deduplicated
        assert created and again.status == "failed"
        with pytest.raises(Exception, match="isn't supported"):
            p.submit_file(tmp_path / "virus.exe")
    finally:
        p.close()


def test_documents_are_wiped_when_the_app_closes_and_restarts(tmp_path):
    data = tmp_path / "data"
    p = Pipeline(offline_settings(data))
    p.ingest_file(make_pdf(tmp_path / "h.pdf", HANDBOOK))
    assert any((data / "uploads").iterdir())
    p.close()
    assert not any(data.rglob("*.pdf")) and not (data / "metadata.db").exists()
    p2 = Pipeline(offline_settings(data))
    try:
        assert p2.list_documents() == []
    finally:
        p2.close()


def test_keep_documents_setting_persists_them(tmp_path):
    data = tmp_path / "data"
    p = Pipeline(offline_settings(data, keep_documents=True))
    p.ingest_file(make_pdf(tmp_path / "h.pdf", HANDBOOK))
    p.close()
    p2 = Pipeline(offline_settings(data, keep_documents=True))
    try:
        assert [d.status for d in p2.list_documents()] == ["ready"]
        assert p2.ask(QueryRequest(question="How many days of annual leave?")).answerable
    finally:
        p2.close()


def test_delete_removes_document_everywhere(tmp_path):
    p = Pipeline(offline_settings(tmp_path / "data"))
    try:
        doc, _ = p.ingest_file(make_pdf(tmp_path / "h.pdf", HANDBOOK))
        assert p.delete_document(doc.doc_id)
        assert p.list_documents() == [] and p.meta.count_chunks() == 0 and p.vectors.count() == 0
        assert not p.delete_document(doc.doc_id)
    finally:
        p.close()

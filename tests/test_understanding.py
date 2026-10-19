"""Question understanding, whole-document answers (summaries, ideas), background overviews and streaming."""

import json
import sqlite3
import time

import httpx

from app.config import Settings
from app.generation.answerer import guess_intent, partial_json_string
from app.generation.digest import Digester, split_sections
from app.generation.llm import OllamaLLM
from app.pipeline import Pipeline
from app.schemas import Page, QueryRequest, Turn
from app.store.metadata import MetadataStore
from tests.conftest import HANDBOOK, make_pdf, offline_settings


class DocLLM:
    """A fake local model that answers each kind of request the app makes."""

    name = "fake"

    def __init__(self, intent="lookup", rewrite=None, answer=None):
        self.intent, self.rewrite, self.answer = intent, rewrite, answer
        self.answer_prompts: list[tuple[str, str]] = []
        self.notes_calls = 0

    def complete(self, system, user, schema=None):
        props = (schema or {}).get("properties", {})
        if "intent" in props:
            latest = user.split("LATEST QUESTION:")[-1].split("\n\nReturn")[0].strip()
            return json.dumps({"intent": self.intent, "question": self.rewrite or latest})
        if "notes" in props:
            self.notes_calls += 1
            if "NOTES:" in user:
                return json.dumps({"notes": "OVERVIEW: an employee handbook covering hours, leave and notice."})
            return json.dumps({"notes": f"Section notes {self.notes_calls}: leave is 24 days (p. 1)."})
        self.answer_prompts.append((system, user))
        return json.dumps(self.answer(user) if callable(self.answer) else self.answer)

    def stream(self, system, user, schema=None):
        text = self.complete(system, user, schema)
        for i in range(0, len(text), 5):
            yield text[i : i + 5]

    def status(self):
        return "ready"


def fresh_pipeline(tmp_path, llm) -> tuple[Pipeline, str]:
    p = Pipeline(offline_settings(tmp_path / "data"))
    p.llm = p.answerer.llm = p.digester.llm = llm
    doc, _ = p.ingest_file(make_pdf(tmp_path / "handbook.pdf", HANDBOOK))
    return p, doc.doc_id


def wait_for_overview(p: Pipeline, doc_id: str, timeout=10.0) -> str:
    end = time.time() + timeout
    while time.time() < end:
        status = p.meta.get_document(doc_id).overview_status
        if status not in ("waiting", "building"):
            return status
        time.sleep(0.05)
    raise AssertionError("overview never finished")


# --- routing ---
def test_keyword_routing_when_no_model_is_available():
    assert guess_intent("Summarize this document") == "overview"
    assert guess_intent("What are the key insights?") == "overview"
    assert guess_intent("Brainstorm ideas to improve onboarding") == "brainstorm"
    assert guess_intent("What could we do about the risks?") == "brainstorm"
    assert guess_intent("How many days of annual leave do employees get?") == "lookup"


def test_partial_json_string_decodes_while_streaming():
    full = '{"answerable": true, "answer": "Line one\\nsays \\"hi\\" \\u00e9 [S1]", "citations": []}'
    assert partial_json_string(full[:10], "answer") == ""
    shown = [partial_json_string(full[:i], "answer") for i in range(len(full) + 1)]
    assert all(b.startswith(a) for a, b in zip(shown, shown[1:]))  # text only ever grows
    assert partial_json_string(full, "answer") == 'Line one\nsays "hi" é [S1]'
    # never shows half an escape sequence
    assert partial_json_string('{"answer": "a\\', "answer") == "a"
    assert partial_json_string('{"answer": "a\\u00', "answer") == "a"


# --- follow-ups ---
def test_follow_up_question_is_rewritten_before_searching(tmp_path):
    def answer(user):
        assert "QUESTION: How many days of annual leave do employees get?" in user
        return {
            "answerable": True,
            "answer": "24 days [S1].",
            "citations": [{"source": "S1", "quote": "Employees receive 24 days of paid annual leave"}],
            "missing": None,
        }

    llm = DocLLM(rewrite="How many days of annual leave do employees get?", answer=answer)
    p, _ = fresh_pipeline(tmp_path, llm)
    try:
        req = QueryRequest(
            question="And how many days is that?",
            history=[Turn(question="Is there annual leave?", answer="Yes, employees get paid annual leave.")],
        )
        a = p.ask(req)
    finally:
        p.close()
    assert a.answerable and a.understood_as == "How many days of annual leave do employees get?"
    assert a.citations[0].verified


# --- background overview ---
def test_split_sections_keeps_page_ranges():
    pages = [Page(number=i, text="x" * 4000) for i in range(1, 6)]
    sections = split_sections(pages, limit=9000)
    assert [(a, b) for a, b, _ in sections] == [(1, 2), (3, 4), (5, 5)]
    assert "[Page 3]" in sections[1][2]
    long = split_sections([Page(number=7, text="y" * 20000)], limit=9000)
    assert [(a, b) for a, b, _ in long] == [(7, 7)] * 3


def test_overview_is_prepared_in_the_background(tmp_path):
    p, doc_id = fresh_pipeline(tmp_path, DocLLM())
    try:
        assert wait_for_overview(p, doc_id) == "ready"
        notes = p.meta.notes_for([doc_id])
        assert notes and notes[0]["page_start"] == 1 and "24 days" in notes[0]["text"]
        assert p.meta.get_overview(doc_id)
    finally:
        p.close()


def test_overview_waits_while_a_question_is_answered(tmp_path):
    s = offline_settings(tmp_path / "d")
    meta = MetadataStore(s.sqlite_path)
    busy = {"asking": True}
    d = Digester(meta, DocLLM(), lambda: not busy["asking"])
    assert d._call("sys", "user") is None  # told to stop rather than compete with the user
    meta.close()


def test_overview_is_unavailable_without_the_ai_engine(pipeline):
    doc = next(d for d in pipeline.list_documents() if d.filename == "handbook.pdf")
    assert doc.overview_status == "unavailable"  # the test pipeline answers without a model


# --- summaries and ideas ---
def test_summary_uses_notes_from_the_whole_document(tmp_path):
    def answer(user):
        assert "[N1]" in user and "OVERVIEW OF handbook.pdf" in user
        return {
            "answerable": True,
            "answer": "The handbook covers hours and leave [N1].",
            "citations": [{"source": "N1", "quote": ""}],
            "missing": None,
        }

    llm = DocLLM(intent="overview", answer=answer)
    p, doc_id = fresh_pipeline(tmp_path, llm)
    try:
        wait_for_overview(p, doc_id)
        a = p.ask(QueryRequest(question="Give me the key points"))
    finally:
        p.close()
    assert a.kind == "overview" and a.answerable
    c = a.citations[0]
    assert c.kind == "notes" and c.page == 1 and c.notes and "24 days" in c.notes


def test_brainstorm_without_document_sources_is_withheld(tmp_path):
    llm = DocLLM(
        intent="brainstorm",
        answer={"answerable": True, "answer": "Adopt a four-day week.", "citations": [], "missing": None},
    )
    p, doc_id = fresh_pipeline(tmp_path, llm)
    try:
        wait_for_overview(p, doc_id)
        a = p.ask(QueryRequest(question="Brainstorm ways to improve the leave policy"))
    finally:
        p.close()
    assert a.kind == "brainstorm" and not a.answerable and a.abstain_code == "ungrounded"
    assert "BRAINSTORM" not in a.answer and a.citations == []
    system = llm.answer_prompts[0][0]
    assert "outside" in system  # the model is told not to use outside knowledge


def test_brainstorm_with_a_verified_quote_is_shown(tmp_path):
    def answer(user):
        return {
            "answerable": True,
            "answer": "- Remind staff to plan their 24 days early [S1].",
            "citations": [{"source": "S1", "quote": "Employees receive 24 days of paid annual leave"}],
            "missing": None,
        }

    p, doc_id = fresh_pipeline(tmp_path, DocLLM(intent="brainstorm", answer=answer))
    try:
        wait_for_overview(p, doc_id)
        a = p.ask(QueryRequest(question="Suggest improvements to how leave is planned"))
    finally:
        p.close()
    assert a.kind == "brainstorm" and a.answerable and a.citations[0].verified


def test_summary_without_a_model_answers_the_closest_fact(pipeline):
    a = pipeline.ask(QueryRequest(question="Summarize the annual leave rules"))
    assert a.kind == "lookup" and a.answerable


# --- streaming ---
def test_answer_streams_status_then_text_then_final(tmp_path):
    answer = {
        "answerable": True,
        "answer": "Employees get 24 days of paid leave [S1].",
        "citations": [{"source": "S1", "quote": "Employees receive 24 days of paid annual leave"}],
        "missing": None,
    }
    p, _ = fresh_pipeline(tmp_path, DocLLM(answer=answer))
    try:
        events = list(p.ask_stream(QueryRequest(question="How many days of annual leave do employees get?")))
    finally:
        p.close()
    kinds = [e["type"] for e in events]
    assert kinds[0] == "status" and kinds[-1] == "final" and kinds.count("delta") > 3
    assert "".join(e["text"] for e in events if e["type"] == "delta") == answer["answer"]
    assert events[-1]["answer"].answerable


def test_ollama_streaming_reads_ndjson():
    lines = [{"message": {"content": c}, "done": False} for c in ('{"answer": "', "hel", 'lo"}')]
    lines.append({"message": {"content": ""}, "done": True, "done_reason": "stop"})
    body = "\n".join(json.dumps(x) for x in lines).encode()

    def handler(req):
        assert json.loads(req.content)["stream"] is True
        return httpx.Response(200, content=body)

    s = Settings(_env_file=None, ollama_url="http://localhost:11434")
    llm = OllamaLLM(s, client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert "".join(llm.stream("S", "U")) == '{"answer": "hello"}'


def test_stream_endpoint(api_client, corpus):
    client = api_client
    with open(corpus / "handbook.pdf", "rb") as f:
        client.post("/documents", files=[("files", ("handbook.pdf", f, "application/pdf"))])
    for _ in range(100):
        if all(d["status"] != "processing" for d in client.get("/documents").json()):
            break
        time.sleep(0.05)
    r = client.post("/query/stream", json={"question": "How many days of annual leave do employees get?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in r.text.splitlines() if line]
    assert events[-1]["type"] == "final" and events[-1]["answer"]["answerable"]
    assert "24 days" in events[-1]["answer"]["answer"]


# --- storage upgrade ---
def test_database_from_an_older_version_is_upgraded(tmp_path):
    path = tmp_path / "old.sqlite3"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE documents (doc_id TEXT PRIMARY KEY, filename TEXT NOT NULL, sha256 TEXT UNIQUE NOT NULL, "
        "path TEXT NOT NULL, status TEXT NOT NULL, error TEXT, num_pages INTEGER, num_chunks INTEGER, "
        "uploaded_at TEXT)"
    )
    con.execute("INSERT INTO documents VALUES ('d1','a.pdf','s','p','ready',NULL,1,1,'2026-01-01')")
    con.commit()
    con.close()
    meta = MetadataStore(path)
    try:
        assert meta.get_document("d1").overview_status is None
        meta.set_overview(meta.get_document("d1").doc_id, "text")
        assert meta.get_overview("d1") == "text" and meta.get_document("d1").overview_status == "ready"
    finally:
        meta.close()

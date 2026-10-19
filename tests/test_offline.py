"""Privacy guarantee: the whole product flow works with every network connection blocked.

Any attempt to reach a non-local address fails the test. Connections to this computer
(Ollama on localhost) stay allowed, because that's where the language model runs.

On a machine where `python scripts/download_models.py` has been run, this test uses the REAL
embedding and reranking models from ./models, proving they load and run without internet.
Elsewhere (e.g. CI) it uses the offline stand-ins.
"""

import json
import re
import socket
import time

import httpx
import pytest

from app.config import ROOT, Settings, local_model_dir
from app.generation.llm import OllamaLLM
from app.pipeline import Pipeline
from app.schemas import QueryRequest
from tests.conftest import HANDBOOK, make_pdf

LOCAL = {"127.0.0.1", "::1", "localhost"}


class InternetBlocked(AssertionError):
    pass


@pytest.fixture
def no_internet(monkeypatch):
    attempts = []
    real_connect, real_create = socket.socket.connect, socket.create_connection

    def host_of(address):
        return address[0] if isinstance(address, tuple) else str(address)

    def guarded_connect(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6) and host_of(address) not in LOCAL:
            attempts.append(address)
            raise InternetBlocked(f"blocked network access to {address}")
        return real_connect(self, address)

    def guarded_create(address, *a, **kw):
        if host_of(address) not in LOCAL:
            attempts.append(address)
            raise InternetBlocked(f"blocked network access to {address}")
        return real_create(address, *a, **kw)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create)
    yield attempts
    assert not attempts, f"the app tried to reach the internet: {attempts}"


def _settings(tmp_path) -> Settings:
    s = Settings(_env_file=None)
    real = all((local_model_dir(s.models_dir, m)).is_dir() for m in (s.embedding_model, s.reranker_model))
    providers = {} if real else {"embedding_provider": "hash", "reranker_provider": "lexical"}
    return Settings(
        _env_file=None, data_dir=tmp_path / "data", qdrant_url=None, models_dir=ROOT / "models", **providers
    )


def test_full_flow_with_internet_blocked(tmp_path, no_internet):
    def fake_ollama(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        user = body["messages"][-1]["content"]
        props = (body.get("format") or {}).get("properties", {})
        if "intent" in props:  # question routing
            return reply({"intent": "lookup", "question": "What notice period applies during probation?"})
        if "notes" in props:  # background overview
            return reply({"notes": "Notice period during probation is 7 days (p. 2)."})
        quote = "During probation the notice period is 7 days."
        blocks = re.split(r"\n\n(?=\[S\d+\])", user.split("QUESTION:")[0])
        source = next(re.match(r"\[(S\d+)\]", b).group(1) for b in blocks if quote in " ".join(b.split()))
        answer = {
            "answerable": True,
            "answer": f"7 days [{source}]",
            "citations": [{"source": source, "quote": quote}],
            "missing": None,
        }
        return reply(answer)

    def reply(obj: dict) -> httpx.Response:
        # One NDJSON line is a valid reply for both streaming and non-streaming requests.
        return httpx.Response(200, json={"message": {"content": json.dumps(obj)}, "done": True, "done_reason": "stop"})

    s = _settings(tmp_path)
    p = Pipeline(s)
    try:
        llm = OllamaLLM(s, client=httpx.Client(transport=httpx.MockTransport(fake_ollama)))
        p.llm = p.answerer.llm = p.digester.llm = llm
        doc, _ = p.ingest_file(make_pdf(tmp_path / "handbook.pdf", HANDBOOK))
        assert doc.status == "ready"
        for _ in range(200):  # the whole-document overview is prepared in the background
            if p.meta.get_document(doc.doc_id).overview_status == "ready":
                break
            time.sleep(0.05)
        assert p.meta.get_overview(doc.doc_id)
        a = p.ask(QueryRequest(question="What notice period applies during probation?"))
        assert a.answerable and a.citations[0].verified and a.citations[0].page == 2
        assert p.render_page(doc.doc_id, 2, a.citations[0].quote)[:4] == b"\x89PNG"
    finally:
        p.close()


def test_the_guard_really_blocks_internet(no_internet):
    with pytest.raises(InternetBlocked):
        socket.create_connection(("example.com", 443), timeout=1)
    no_internet.clear()  # this attempt was intentional

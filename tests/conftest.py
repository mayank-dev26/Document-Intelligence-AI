"""Shared fixtures.

Tests create their own small documents in a temporary folder on every run (nothing is stored
in the repo) and use offline stand-ins: a hashing embedder, a lexical reranker and a scripted
or extractive answerer. They need no model downloads, no Ollama and no internet.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

HANDBOOK = [
    "Acme Employee Handbook\n\n1. Working Hours\n\nThe standard working week is 40 hours, Monday to Friday. "
    "Core hours are 10:00 to 16:00.\n\n2. Annual Leave\n\nEmployees receive 24 days of paid annual leave each year. "
    "Unused leave of up to 5 days may be carried over.",
    "3. Notice Period\n\nEmployees must give 30 days of written notice before resigning. During probation the notice "
    "period is 7 days.\n\n4. Remote Work\n\nEmployees may work remotely up to two days per week with approval.",
    "5. Expenses\n\nTravel expenses are reimbursed within 15 days of submitting receipts. Claims older than 60 days "
    "are not accepted.",
]
MANUAL = [
    "Pump X200 Manual\n\n1. Warranty\n\nThe pump is covered by a warranty of 24 months from installation. "
    "Seals are covered for 6 months only.",
    "2. Error Codes\n\nE01 - Low inlet pressure: open the suction valve fully.\n\nE04 - Dry-run detected: no flow for "
    "20 seconds, so the controller stops the pump.\n\nE05 - Excessive vibration: check the mounting bolts.",
]


def make_pdf(path: Path, pages: list[str], footer: str | None = None) -> Path:
    doc = pymupdf.open()
    for i, text in enumerate(pages, start=1):
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 740), text, fontsize=11)
        if footer:
            page.insert_text((72, 800), f"{footer} | Page {i}", fontsize=8)
    doc.save(path)
    return path


def make_docx(path: Path, paragraphs: list[str]) -> Path:
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(str(path))
    return path


def make_corpus(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    make_pdf(folder / "handbook.pdf", HANDBOOK)
    make_pdf(folder / "pump_manual.pdf", MANUAL, footer="Manual X200 Rev C")
    make_docx(folder / "meeting_notes.docx", ["Meeting notes", "The marketing budget of 40 lakh was approved."])
    return folder


def offline_settings(tmp: Path, **overrides):
    from app.config import Settings

    values = {
        "data_dir": tmp,
        "qdrant_url": None,
        "embedding_provider": "hash",
        "reranker_provider": "lexical",
        "llm_provider": "extractive",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(scope="session")
def corpus(tmp_path_factory) -> Path:
    return make_corpus(tmp_path_factory.mktemp("docs"))


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory, corpus):
    from app.pipeline import Pipeline

    p = Pipeline(offline_settings(tmp_path_factory.mktemp("data")))
    p.ingest_directory(corpus)
    yield p
    p.close()


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import config

    env = {
        "EMBEDDING_PROVIDER": "hash",
        "RERANKER_PROVIDER": "lexical",
        "LLM_PROVIDER": "extractive",
        "DATA_DIR": str(tmp_path / "data"),
        "QDRANT_URL": "",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    config.get_settings.cache_clear()
    from app.api.main import app

    with TestClient(app) as client:
        yield client
    config.get_settings.cache_clear()

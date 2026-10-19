"""API contract tests with FastAPI's TestClient."""

import time

from tests.conftest import HANDBOOK, make_pdf


def _upload(client, *paths):
    files = [("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in paths]
    return client.post("/documents", files=files)


def _wait_ready(client, doc_id, timeout=10):
    for _ in range(int(timeout / 0.05)):
        doc = next(d for d in client.get("/documents").json() if d["doc_id"] == doc_id)
        if doc["status"] != "processing":
            return doc
        time.sleep(0.05)
    raise AssertionError("document never finished processing")


def test_health_reports_plain_status(api_client):
    body = api_client.get("/health").json()
    assert body["status"] == "ok" and body["ai_engine"] == "extractive" and body["documents"] == 0
    assert body["keeps_documents"] is False


def test_upload_is_immediate_then_ask_render_delete(api_client, tmp_path):
    r = _upload(api_client, make_pdf(tmp_path / "handbook.pdf", HANDBOOK))
    result = r.json()[0]
    assert r.status_code == 200 and result["created"] and result["document"]["status"] in {"processing", "ready"}
    doc = _wait_ready(api_client, result["document"]["doc_id"])
    assert doc["status"] == "ready" and doc["num_pages"] == 3

    a = api_client.post("/query", json={"question": "What notice period applies during probation?"}).json()
    assert a["answerable"] and "7 days" in a["answer"]
    cit = a["citations"][0]
    assert cit["verified"] and cit["page"] == 2

    assert "7 days" in api_client.get(f"/documents/{doc['doc_id']}/pages/2").json()["text"]
    img = api_client.get(f"/documents/{doc['doc_id']}/pages/2/render", params={"highlight": cit["quote"]})
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"

    assert api_client.delete(f"/documents/{doc['doc_id']}").status_code == 200
    assert api_client.get("/documents").json() == []
    assert api_client.delete(f"/documents/{doc['doc_id']}").status_code == 404


def test_unsupported_and_invalid_input_get_plain_messages(api_client):
    r = api_client.post("/documents", files=[("files", ("setup.exe", b"MZ", "application/octet-stream"))])
    assert "isn't supported" in r.json()[0]["error"]
    assert api_client.post("/query", json={"question": "x"}).status_code == 422  # too short

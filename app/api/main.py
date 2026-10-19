"""FastAPI service.

Run:  uvicorn app.api.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import get_settings
from app.ingestion.parser import UnsupportedFileError
from app.pipeline import Pipeline
from app.schemas import Answer, DocumentInfo, QueryRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = Pipeline(get_settings())
    log.info("Ready: %s", app.state.pipeline.info())
    yield
    app.state.pipeline.close()  # also wipes stored documents unless KEEP_DOCUMENTS=true


app = FastAPI(
    title="Document Intelligence API",
    version="2.1.0",
    lifespan=lifespan,
    description="Private, offline question answering over your documents, with verified sources.",
)


def pipe(request: Request) -> Pipeline:
    return request.app.state.pipeline


@app.get("/health", tags=["system"])
def health(request: Request) -> dict:
    return {"status": "ok", **pipe(request).status()}


# --- documents ---
class UploadResult(BaseModel):
    filename: str
    document: DocumentInfo | None = None
    created: bool = False
    error: str | None = None


@app.post("/documents", response_model=list[UploadResult], tags=["documents"])
def upload_documents(request: Request, files: list[UploadFile] = File(...)) -> list[UploadResult]:
    """Accept files and return immediately; each document is read in the background (see its status)."""
    limit = get_settings().max_upload_mb
    results = []
    for f in files:
        name = Path(f.filename or "upload").name
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as tmp:
            shutil.copyfileobj(f.file, tmp)
            tmp_path = Path(tmp.name)
        try:
            if tmp_path.stat().st_size > limit * 1024 * 1024:
                raise UnsupportedFileError(f"This file is larger than {limit} MB.")
            doc, created = pipe(request).submit_file(tmp_path, name)
            results.append(UploadResult(filename=name, document=doc, created=created))
        except UnsupportedFileError as e:
            results.append(UploadResult(filename=name, error=str(e)))
        except Exception:  # noqa: BLE001
            log.exception("Upload failed for %s", name)
            results.append(UploadResult(filename=name, error="This file couldn't be uploaded. Please try again."))
        finally:
            tmp_path.unlink(missing_ok=True)
    return results


@app.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
def list_documents(request: Request) -> list[DocumentInfo]:
    return pipe(request).list_documents()


@app.delete("/documents/{doc_id}", tags=["documents"])
def delete_document(doc_id: str, request: Request) -> dict:
    if not pipe(request).delete_document(doc_id):
        raise HTTPException(404, "Document not found")
    return {"deleted": doc_id}


@app.get("/documents/{doc_id}/pages/{page}", tags=["documents"])
def get_page(doc_id: str, page: int, request: Request) -> dict:
    text = pipe(request).page_text(doc_id, page)
    if text is None:
        raise HTTPException(404, "Page not found")
    return {"doc_id": doc_id, "page": page, "text": text}


@app.get("/documents/{doc_id}/pages/{page}/render", tags=["documents"], responses={200: {"content": {"image/png": {}}}})
def render_page(
    doc_id: str,
    page: int,
    request: Request,
    highlight: str | None = Query(None, max_length=2000),
    dpi: int = Query(110, ge=50, le=200),
):
    """PNG of a PDF page with the cited quote highlighted."""
    png = pipe(request).render_page(doc_id, page, highlight, dpi)
    if png is None:
        raise HTTPException(404, "Page preview is only available for PDF files")
    return Response(content=png, media_type="image/png")


# --- questions ---
@app.post("/query", response_model=Answer, tags=["questions"])
def query(req: QueryRequest, request: Request) -> Answer:
    """Answer from the documents with verified sources, or say the documents don't cover it."""
    try:
        return pipe(request).ask(req)
    except Exception as e:  # noqa: BLE001
        log.exception("Query failed")
        raise HTTPException(500, "Something went wrong while answering. Please try again.") from e


@app.post("/query/stream", tags=["questions"])
def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    """The same answer, streamed as newline-delimited JSON events while it is written:

    {"type": "status", "text": ...}   what is happening ("Finding the relevant pages")
    {"type": "delta", "text": ...}    the next piece of the answer text
    {"type": "reset"}                 discard the text so far (the answer is being redone)
    {"type": "final", "answer": {...}} the complete, verified Answer
    {"type": "error", "text": ...}    something went wrong
    """
    pipeline = pipe(request)

    def events():
        try:
            for event in pipeline.ask_stream(req):
                if event["type"] == "final":
                    event = {"type": "final", "answer": event["answer"].model_dump(mode="json")}
                yield json.dumps(event) + "\n"
        except Exception:  # noqa: BLE001
            log.exception("Query failed")
            yield (
                json.dumps({"type": "error", "text": "Something went wrong while answering. Please try again."}) + "\n"
            )

    return StreamingResponse(events(), media_type="application/x-ndjson")

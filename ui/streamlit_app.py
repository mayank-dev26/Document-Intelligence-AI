"""Document Intelligence: private question answering over your own documents.

Run:  streamlit run ui/streamlit_app.py   (the API must be running: uvicorn app.api.main:app)
Theme and bundled fonts: .streamlit/config.toml and ui/static/fonts.

Answers are streamed from the API (/query/stream) and written as they arrive. The parts of the page
that change on their own (the document list, page previews) are fragments, so they update without
redrawing the whole page.
"""

from __future__ import annotations

import html
import json
import os
import re
import time

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
USER_AVATAR, ASSISTANT_AVATAR = ":material/person:", ":material/description:"
FILE_TYPES = ["pdf", "docx", "txt", "md"]

st.set_page_config(page_title="Document Intelligence", page_icon=":material/description:", layout="wide")
st.markdown(
    """
<style>
:root { --ink:#1B2A41; --slate:#5B6575; --line:#E3E7ED; --marker:#FFE45C; --marker-soft:#FFF3B0;
        --ok:#1F7A5A; --ok-soft:#E6F2EC; --warn:#B4690E; --warn-soft:#FBF0E1; }
.block-container { max-width: 960px; }

/* Document text: serif, to set it apart from the UI. */
.passage { font-family:'SourceSerif4', Georgia, serif; font-size:1.02rem; line-height:1.6; color:var(--ink);
           border-left:2px solid var(--line); padding:.15rem 0 .15rem 1rem; margin:.25rem 0 .5rem; }
.passage p { margin:0 0 .55rem; }
.passage p:last-child { margin-bottom:0; }

/* Highlight for the quoted words. */
mark { background:linear-gradient(180deg, transparent 12%, var(--marker) 12%, var(--marker) 90%, transparent 90%);
       color:inherit; padding:0 .12em; margin:0 -.12em; border-radius:.2em .35em .25em .4em;
       -webkit-box-decoration-break:clone; box-decoration-break:clone; }

.answer { font-size:1.05rem; line-height:1.65; margin-bottom:.4rem; }
.answer em { color:var(--slate); }
.cite { font-size:.68em; font-weight:600; line-height:1; padding:.05em .38em; margin-left:.12em; border-radius:.3em;
        background:var(--marker-soft); color:var(--ink); vertical-align:super; }
.chip { display:inline-block; font-size:.8rem; font-weight:600; padding:.12rem .6rem; border-radius:999px; margin-right:.55rem; }
.chip.ok { background:var(--ok-soft); color:var(--ok); }
.chip.warn { background:var(--warn-soft); color:var(--warn); }
.chip.idea { background:#ECEFF8; color:#33427A; }
.answer ul { margin:.2rem 0 .55rem 1.1rem; padding:0; }
.answer li { margin:.15rem 0; }
.answer p { margin:0 0 .55rem; }
.working { font-size:.9rem; color:var(--slate); }
.working::after { content:"…"; animation:dots 1.4s steps(4) infinite; display:inline-block; width:1em; overflow:hidden; vertical-align:bottom; }
.caret::after { content:""; display:inline-block; width:.5em; height:1.05em; margin-left:.1em; vertical-align:-.15em;
                background:var(--ink); opacity:.55; animation:blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity:0 } }
.notes { font-size:.95rem; line-height:1.6; color:var(--ink); border-left:2px solid var(--line); padding:.15rem 0 .15rem 1rem; }

/* Disable Streamlit's fade-out of stale elements during reruns. */
[data-stale="true"], .stale-element { opacity:1 !important; transition:none !important; }
.meta { font-size:.84rem; color:var(--slate); }

.status { font-size:.86rem; color:var(--slate); line-height:1.5; }
.dot { display:inline-block; width:.5rem; height:.5rem; border-radius:50%; margin-right:.45rem; vertical-align:.05rem; }
.dot.ok { background:var(--ok); } .dot.warn { background:var(--warn); }
.doc-name { font-weight:500; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.92rem; }
.doc-state { font-size:.8rem; color:var(--slate); }
.doc-state.failed { color:var(--warn); }
.doc-state.reading::after { content:"…"; animation:dots 1.4s steps(4) infinite; display:inline-block; width:1em; overflow:hidden; vertical-align:bottom; }
@keyframes dots { 0% { width:0 } 100% { width:1em } }
.empty { padding:2.5rem 0 1rem; max-width:34rem; }
.empty p { color:var(--slate); line-height:1.6; }

@media (prefers-reduced-motion: reduce) { * { animation:none !important; transition:none !important; } }
</style>""",
    unsafe_allow_html=True,
)


# --- API helpers ---
def api(method: str, path: str, timeout: float = 300, **kw):
    try:
        r = httpx.request(method, f"{API}{path}", timeout=timeout, **kw)
    except httpx.HTTPError:
        st.error("The app's engine isn't running yet. Start it, then refresh this page.")
        st.stop()
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(detail if isinstance(detail, str) else "Something went wrong. Please try again.")
        return None
    return r


@st.cache_data(ttl=3, show_spinner=False)
def get_status() -> dict | None:
    try:
        return httpx.get(f"{API}/health", timeout=5).json()
    except Exception:  # noqa: BLE001
        return None


def get_docs() -> list[dict]:
    r = api("GET", "/documents", timeout=10)
    return r.json() if r else []


@st.cache_data(show_spinner=False, max_entries=64)
def get_page_png(doc_id: str, page: int, highlight: str | None) -> bytes | None:
    r = httpx.get(
        f"{API}/documents/{doc_id}/pages/{page}/render",
        params={"highlight": highlight} if highlight else {},
        timeout=30,
    )
    return r.content if r.status_code == 200 else None


# --- text helpers ---
def is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


def location(filename: str, page: int) -> str:
    return f"page {page}" if is_pdf(filename) else f"part {page}"


_MARKER = re.compile(r"\s*\[([SN]\d+)\]")


def strip_markers(text: str) -> str:
    return _MARKER.sub("", text)


def render_answer_text(text: str, numbers: dict[str, int] | None = None) -> str:
    """Answer text as HTML: paragraphs, "- " bullet lists, and source markers as small numbers."""

    def inline(line: str) -> str:
        safe = html.escape(line)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        safe = re.sub(r"_(Not covered by the documents:.*?)_", r"<em>\1</em>", safe)
        if numbers is None:
            return _MARKER.sub("", safe)
        return _MARKER.sub(
            lambda m: f'<sup class="cite">{numbers[m.group(1)]}</sup>' if m.group(1) in numbers else "", safe
        )

    out, items, para = [], [], []

    def flush() -> None:
        if items:
            out.append("<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
            items.clear()
        if para:
            out.append("<p>" + "<br>".join(para) + "</p>")
            para.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r"^[-*•]\s+", stripped):
            if para:
                flush()
            items.append(inline(re.sub(r"^[-*•]\s+", "", stripped)))
        elif not stripped:
            flush()
        else:
            if items:
                flush()
            para.append(inline(stripped))
    flush()
    return "".join(out)


def passage_html(text: str, start: int | None = None, end: int | None = None) -> str:
    """Document text as serif paragraphs, with the quoted words marked by the highlighter.

    A quote can run across a paragraph break (a heading into its first sentence), so each
    paragraph highlights its own share of the quote; one <mark> can't span two <p> elements.
    """
    paras, pos = [], 0
    for m in re.finditer(r"\n\s*\n", text + "\n\n"):
        seg_start, seg_end = pos, m.start()
        pos = m.end()
        if not text[seg_start:seg_end].strip():
            continue
        if start is not None and end is not None and start < seg_end and end > seg_start:
            a, b = max(start, seg_start), min(end, seg_end)
            body = (
                html.escape(text[seg_start:a])
                + "<mark>"
                + html.escape(text[a:b])
                + "</mark>"
                + html.escape(text[b:seg_end])
            )
        else:
            body = html.escape(text[seg_start:seg_end])
        paras.append(f"<p>{body.strip()}</p>")
    return f'<div class="passage">{"".join(paras)}</div>'


# --- state ---
ss = st.session_state
ss.setdefault("chat", [])  # [{"q": str, "a": dict | None, "scope": str | None}]
ss.setdefault("uploader_key", 0)
ss.setdefault("upload_errors", [])
ss.setdefault("was_busy", False)

status = get_status()
if status is None:
    st.markdown(
        '<div class="empty"><h3>Starting up</h3><p>The app is getting ready. Refresh this page in a moment.</p></div>',
        unsafe_allow_html=True,
    )
    st.stop()

docs = get_docs()
ready_docs = [d for d in docs if d["status"] == "ready"]
busy = any(d["status"] == "processing" for d in docs)
preparing = any(d.get("overview_status") in ("waiting", "building") for d in docs)


def send_files(files) -> None:
    r = api("POST", "/documents", files=[("files", (f.name, f.getvalue())) for f in files])
    if r:
        ss.upload_errors = [f"{x['filename']}: {x['error']}" for x in r.json() if x["error"]]
    ss.uploader_key += 1  # clears the picker so the same files aren't sent twice
    st.rerun()


def uploader(container) -> None:
    files = container.file_uploader(
        "Add documents",
        type=FILE_TYPES,
        accept_multiple_files=True,
        key=f"uploader-{ss.uploader_key}",
        help="PDF, Word, text or Markdown files.",
    )
    if files:
        send_files(files)


# --- sidebar: your documents ---
def doc_state(d: dict) -> str:
    if d["status"] == "processing":
        return '<div class="doc-state reading">Reading</div>'
    if d["status"] == "failed":
        return f'<div class="doc-state failed">{html.escape(d.get("error") or "Could not be read")}</div>'
    pages = d.get("num_pages") or 0
    unit = "page" if is_pdf(d["filename"]) else "part"
    ready = f"Ready, {pages} {unit}{'s' if pages != 1 else ''}"
    if d.get("overview_status") in ("waiting", "building"):
        return f'<div class="doc-state reading">{ready}. Reading it in full for summaries</div>'
    return f'<div class="doc-state">{ready}</div>'


with st.sidebar:
    st.markdown("### Document Intelligence")
    engine = status.get("ai_engine")
    if engine == "ready":
        st.markdown('<div class="status"><span class="dot ok"></span>AI engine ready</div>', unsafe_allow_html=True)
    elif engine == "missing_model":
        st.markdown(
            '<div class="status"><span class="dot warn"></span>AI model not installed</div>', unsafe_allow_html=True
        )
        st.caption(
            f"In a terminal, run `ollama pull {status.get('model')}`, then refresh. "
            "Until then, answers quote your documents directly."
        )
    else:
        st.markdown('<div class="status"><span class="dot warn"></span>AI engine is off</div>', unsafe_allow_html=True)
        st.caption("Start Ollama for full answers. Until then, answers quote your documents directly.")

    if docs:
        st.divider()
        uploader(st)
        for err in ss.upload_errors:
            st.caption(f":orange[{err}]")

        # Only this list refreshes while documents are being read; the rest of the page stays put.
        @st.fragment(run_every="2s" if busy or preparing else None)
        def document_list() -> None:
            current = get_docs()
            still_busy = any(d["status"] == "processing" for d in current)
            for d in current:
                name, state = st.columns([5, 1], vertical_alignment="center")
                name.markdown(
                    f'<div class="doc-name" title="{html.escape(d["filename"])}">{html.escape(d["filename"])}</div>'
                    + doc_state(d),
                    unsafe_allow_html=True,
                )
                if state.button("", key=f"del-{d['doc_id']}", icon=":material/close:", help="Remove this document"):
                    api("DELETE", f"/documents/{d['doc_id']}")
                    st.rerun(scope="app")
            if ss.was_busy and not still_busy:
                ss.was_busy = False
                st.rerun(scope="app")  # a document just became ready: enable asking
            ss.was_busy = still_busy

        document_list()

    st.divider()
    if status.get("keeps_documents"):
        st.caption(":material/lock: Private. Your documents stay on this computer.")
    else:
        st.caption(":material/lock: Private. Your documents stay on this computer and are cleared when the app closes.")


# --- answers ---
CHIPS = {
    "overview": ("idea", "Summary of your document", "Written from notes on every part of your document."),
    "brainstorm": (
        "idea",
        "Ideas based on your document",
        "Suggestions drawn only from your document. Each one shows the part it comes from.",
    ),
}


def history_for(turns: list[dict]) -> list[dict]:
    """The last few answered turns, so follow-up questions ("and the second one?") make sense."""
    done = [t for t in turns if t["a"] and not t["a"].get("error")]
    return [{"question": t["q"], "answer": strip_markers(t["a"]["answer"])[:500]} for t in done[-3:]]


def stream_answer(turn: dict, history: list[dict]) -> dict:
    """Ask the API and write the answer into the page as it arrives. Returns the final answer."""
    live = st.empty()
    text, last_draw = "", 0.0
    body = {"question": turn["q"], "doc_ids": [turn["scope"]] if turn["scope"] else None, "history": history}
    live.markdown('<div class="working">Reading your question</div>', unsafe_allow_html=True)
    try:
        with httpx.stream("POST", f"{API}/query/stream", json=body, timeout=httpx.Timeout(300, connect=5)) as r:
            if r.status_code >= 400:
                return {"error": "Something went wrong while answering. Please try again."}
            for line in r.iter_lines():
                if not line.strip():
                    continue
                ev = json.loads(line)
                kind = ev.get("type")
                if kind == "status" and not text:
                    live.markdown(f'<div class="working">{html.escape(ev["text"])}</div>', unsafe_allow_html=True)
                elif kind == "delta":
                    text += ev["text"]
                    if time.monotonic() - last_draw > 0.05:  # redraw at most ~20 times a second
                        live.markdown(f'<div class="answer">{with_caret(text)}</div>', unsafe_allow_html=True)
                        last_draw = time.monotonic()
                elif kind == "reset":
                    text = ""
                elif kind == "final":
                    live.empty()
                    return ev["answer"]
                elif kind == "error":
                    live.empty()
                    return {"error": ev.get("text") or "Something went wrong while answering. Please try again."}
    except httpx.HTTPError:
        live.empty()
        return {"error": "The app's engine isn't running. Start it, then ask again."}
    live.empty()
    return {"error": "Something went wrong while answering. Please try again."}


def with_caret(text: str) -> str:
    """The answer so far, with a blinking cursor after the last word."""
    out = render_answer_text(text)
    cut = out.rfind("</li>") if out.endswith("</ul>") else out.rfind("</p>")
    caret = '<span class="caret"></span>'
    return out[:cut] + caret + out[cut:] if cut >= 0 else caret


@st.fragment
def page_preview(c: dict, quote: str | None, key: str) -> None:
    """The original PDF page with the quote highlighted. A fragment, so toggling it redraws only itself."""
    if st.toggle("Show on the page", key=key):
        png = get_page_png(c["doc_id"], c["page"], quote or None)
        if png:
            st.image(png, caption=f"{c['doc_name']}, page {c['page']}")


def pages_label(c: dict) -> str:
    a, b = c["page"], c.get("page_end") or c["page"]
    unit = "page" if is_pdf(c["doc_name"]) else "part"
    return f"{unit} {a}" if a == b else f"{unit}s {a}–{b}"


def show_answer(a: dict, key: str) -> None:
    if a.get("error"):
        st.markdown(f'<div class="answer">{html.escape(a["error"])}</div>', unsafe_allow_html=True)
        return
    if a.get("understood_as"):
        st.caption(f"Answering: {a['understood_as']}")
    if a.get("note"):
        st.caption(f":material/info: {a['note']}")
    if not a["answerable"]:
        st.markdown(f'<div class="answer">{html.escape(a["answer"])}</div>', unsafe_allow_html=True)
        reason = html.escape(a.get("abstain_reason") or "")
        st.markdown(
            f'<span class="chip warn">Not in your documents</span><span class="meta">{reason} '
            "Try rephrasing, or check you're asking about the right document.</span>",
            unsafe_allow_html=True,
        )
        return

    cits = a["citations"]
    numbers = {c["marker"]: i for i, c in enumerate(cits, start=1)}  # "S3", "N1" -> Source 1, 2, ...
    st.markdown(f'<div class="answer">{render_answer_text(a["answer"], numbers)}</div>', unsafe_allow_html=True)
    quotes = [c for c in cits if c.get("kind", "passage") == "passage"]
    verified = sum(c["verified"] for c in quotes)
    if a.get("kind") in CHIPS:
        style, chip, meta = CHIPS[a["kind"]]
        st.markdown(f'<span class="chip {style}">{chip}</span><span class="meta">{meta}</span>', unsafe_allow_html=True)
    elif verified == len(quotes):
        st.markdown(
            '<span class="chip ok">Sources checked</span>'
            '<span class="meta">Every quote was found word for word in your documents.</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="chip warn">Partly checked</span><span class="meta">{verified} of {len(quotes)} '
            "quotes were found word for word in your documents.</span>",
            unsafe_allow_html=True,
        )

    by_id = {p["chunk"]["chunk_id"]: p["chunk"] for p in a["passages"]}
    for i, c in enumerate(cits):
        n = numbers[c["marker"]]
        if c.get("kind") == "notes":
            label = f"Source {n}: {c['doc_name']}, {pages_label(c)} (summary notes)"
            with st.expander(label, expanded=False, icon=":material/summarize:"):
                st.caption("Notes the AI wrote while reading these pages.")
                st.markdown(
                    f'<div class="notes">{render_answer_text(c.get("notes") or "")}</div>', unsafe_allow_html=True
                )
            continue
        label = f"Source {n}: {c['doc_name']}, {location(c['doc_name'], c['page'])}"
        icon = ":material/check_circle:" if c["verified"] else ":material/error:"
        with st.expander(label, expanded=(i == 0 and a.get("kind", "lookup") == "lookup"), icon=icon):
            chunk = by_id.get(c["chunk_id"])
            if chunk:
                st.markdown(passage_html(chunk["text"], c["quote_start"], c["quote_end"]), unsafe_allow_html=True)
            if not c["verified"]:
                st.caption("This quote wasn't found word for word in the passage, so treat it with care.")
            if is_pdf(c["doc_name"]):
                quote = (
                    chunk["text"][c["quote_start"] : c["quote_end"]]
                    if chunk and c["quote_start"] is not None
                    else c["quote"]
                )
                page_preview(c, quote, f"pg-{key}-{i}")


# --- main: ask ---
if not docs:
    st.markdown(
        '<div class="empty"><h2>Ask questions about your documents</h2>'
        "<p>Add a PDF, Word or text file. It's read on this computer and never sent anywhere. "
        "Then ask anything, and every answer shows the exact lines it came from.</p></div>",
        unsafe_allow_html=True,
    )
    uploader(st)
    for err in ss.upload_errors:
        st.caption(f":orange[{err}]")
    st.stop()

scope = None
if len(ready_docs) > 1:
    names = {d["doc_id"]: d["filename"] for d in ready_docs}
    scope = st.selectbox("Ask about", [None, *names], format_func=lambda x: "All documents" if x is None else names[x])

# The chat box is pinned to the bottom wherever it's created. Reading it first means a new question is
# shown and answered in this same run, without redrawing the page first.
placeholder = "Ask a question about your documents" if ready_docs else "Available once your document is ready"
if question := st.chat_input(placeholder, disabled=not ready_docs):
    if len(question.strip()) < 3:
        st.toast("Please type a longer question.")
    else:
        ss.chat.append({"q": question.strip(), "a": None, "scope": scope})

if not ready_docs:
    st.markdown(
        '<div class="empty"><h3>Reading your document</h3>'
        "<p>This usually takes a few seconds. You can ask questions as soon as it's ready.</p></div>",
        unsafe_allow_html=True,
    )
elif not ss.chat:
    subject = html.escape(ready_docs[0]["filename"]) if len(ready_docs) == 1 else f"your {len(ready_docs)} documents"
    st.markdown(
        f'<div class="empty"><h3>Ask anything about {subject}</h3>'
        "<p>Look up a detail, ask for a summary or the key points, or brainstorm ideas. "
        "Everything comes only from your documents, with the exact lines shown. "
        "If the answer isn't there, you'll be told.</p></div>",
        unsafe_allow_html=True,
    )

for n, turn in enumerate(ss.chat):
    with st.chat_message("user", avatar=USER_AVATAR):
        st.write(turn["q"])
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        if turn["a"] is None:  # just asked: write the answer as it arrives
            turn["a"] = stream_answer(turn, history_for(ss.chat[:n]))
        show_answer(turn["a"], f"t{n}")

if ss.chat and st.button("New chat", icon=":material/add_comment:"):
    ss.chat = []
    st.rerun()

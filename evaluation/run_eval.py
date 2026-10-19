"""Benchmark the system on your own documents and labelled questions.

    python -m evaluation.run_eval --docs path/to/documents --dataset path/to/questions.jsonl
    python -m evaluation.run_eval --docs ... --dataset ... --generate vector hybrid_rerank --judge

It works in a temporary folder of its own, so it never touches the app's documents and can run
while the app is open. Reports are saved to reports/ (JSON, Markdown and a per-question CSV).
See the README for the question-file format.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from app.config import Settings
from app.pipeline import Pipeline
from app.schemas import RetrievalMode
from evaluation.runner import run_evaluation, to_markdown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, required=True, help="folder with the documents the questions are about")
    ap.add_argument("--dataset", type=Path, required=True, help="JSONL file of labelled questions")
    ap.add_argument(
        "--modes",
        nargs="+",
        default=[m.value for m in RetrievalMode],
        choices=[m.value for m in RetrievalMode],
        help="retrieval modes to benchmark",
    )
    ap.add_argument(
        "--generate",
        nargs="*",
        default=["vector", "hybrid_rerank"],
        choices=[m.value for m in RetrievalMode],
        help="modes to run end-to-end answers for (one model call per question); pass none to skip",
    )
    ap.add_argument("--judge", action="store_true", help="also grade answers with the language model as a judge")
    args = ap.parse_args()
    for path, kind in ((args.docs, "folder"), (args.dataset, "file")):
        if not path.exists():
            ap.error(f"{kind} not found: {path}")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    with tempfile.TemporaryDirectory(prefix="docintel-eval-") as tmp:
        pipe = Pipeline(Settings(data_dir=Path(tmp), qdrant_url=None, keep_documents=False, prepare_overviews=False))
        try:
            docs = pipe.ingest_directory(args.docs)
            for d in docs:
                print(f"  {d.filename}: {d.status}" + (f" ({d.error})" if d.error else ""))
            report = run_evaluation(
                pipe,
                args.dataset,
                [RetrievalMode(m) for m in args.modes],
                [RetrievalMode(m) for m in args.generate],
                args.judge,
                progress=print,
            )
        finally:
            pipe.close()
    print("\n" + to_markdown(report))
    print("Saved:", *report["meta"]["files"].values(), sep="\n  ")


if __name__ == "__main__":
    main()

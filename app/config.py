"""Central configuration. Values come from environment variables / .env.

Everything runs on this computer: the language model is served by Ollama on localhost,
and the embedding and reranking models are loaded from the local `models/` folder.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Storage. The app starts fresh: everything under data_dir is wiped on start-up and shutdown.
    data_dir: Path = ROOT / "data"
    keep_documents: bool = False  # set true to keep documents between restarts
    prepare_overviews: bool = True  # read each document in full in the background (summaries, ideas)
    qdrant_url: str | None = None  # e.g. http://qdrant:6333 in Docker; empty = embedded on disk
    qdrant_collection: str = "chunks"

    # Local models (downloaded once with scripts/download_models.py)
    models_dir: Path = ROOT / "models"
    embedding_provider: str = "sentence-transformers"  # sentence-transformers | hash (tests only)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    reranker_provider: str = "cross-encoder"  # cross-encoder | lexical (tests only)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Language model served by Ollama on this computer
    llm_provider: str = "ollama"  # ollama | extractive (no language model; quotes sentences)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:14b"
    ollama_context: int = 8192  # tokens the model can read at once (sent with every request)
    ollama_max_tokens: int = 1024  # longest answer, in tokens
    ollama_timeout_s: float = 180.0
    ollama_keep_alive: str = "30m"  # keep the model loaded on the GPU between questions

    # Chunking
    chunk_size: int = 600  # characters
    chunk_overlap: int = 100  # characters

    # Retrieval
    candidate_k: int = 30  # candidates pulled from each retriever before fusion / rerank
    top_k: int = 6  # passages sent to the language model
    rrf_k: int = 60  # Reciprocal Rank Fusion constant
    abstain_threshold: float = 0.30  # min calibrated relevance of best passage to attempt an answer
    citation_match_threshold: float = 0.85  # fuzzy ratio for a quote to count as verified

    # Upload limits
    max_upload_mb: int = 50

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "metadata.db"

    @property
    def qdrant_path(self) -> Path:
        return self.data_dir / "qdrant"

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)


def local_model_dir(models_dir: Path, name: str) -> Path:
    """Folder a downloaded model lives in: models/BAAI--bge-small-en-v1.5. A path is used as-is."""
    as_path = Path(name)
    if as_path.is_dir():
        return as_path
    return models_dir / name.replace("/", "--")


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s

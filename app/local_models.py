"""Load the embedding and reranking models from disk only, never from the internet.

Models are downloaded once with `python scripts/download_models.py`. At runtime the
Hugging Face libraries are switched to offline mode before they are imported, so no
model code can contact a server even by accident.
"""

from __future__ import annotations

import os
from pathlib import Path

from app.config import local_model_dir

OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
}


class ModelNotDownloadedError(RuntimeError):
    pass


def enable_offline_mode() -> None:
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value


def require_local_model(models_dir: Path, name: str) -> Path:
    enable_offline_mode()
    path = local_model_dir(models_dir, name)
    if not path.is_dir() or not any(path.iterdir()):
        raise ModelNotDownloadedError(
            f"The model '{name}' isn't on this computer yet (looked in {path}). "
            "Run `python scripts/download_models.py` once, with internet, then start the app again."
        )
    return path

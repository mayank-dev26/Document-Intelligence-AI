"""One-time setup: download the local models. This is the only step that needs internet.

    python scripts/download_models.py

It saves the embedding and reranking models into ./models and checks that Ollama has the
language model. After this, the app runs with the internet switched off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import get_settings, local_model_dir  # noqa: E402


def download(repo_id: str, target: Path) -> None:
    from huggingface_hub import snapshot_download

    if target.is_dir() and any(target.iterdir()):
        print(f"  already downloaded: {repo_id}")
        return
    from huggingface_hub import list_repo_files

    print(f"  downloading {repo_id} ...")
    # Only the PyTorch weights are needed. Skip other formats (ONNX, OpenVINO, TensorFlow, Flax),
    # and skip the older .bin copy when the same weights exist as .safetensors.
    ignore = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot", "tf_model*", "flax_model*"]
    if any(f.endswith(".safetensors") for f in list_repo_files(repo_id)):
        ignore.append("pytorch_model.bin")
    snapshot_download(repo_id=repo_id, local_dir=str(target), ignore_patterns=ignore)
    print(f"  saved to {target}")


def check_ollama(url: str, model: str) -> bool:
    try:
        tags = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=5).json().get("models", [])
    except httpx.HTTPError:
        print(f"  Ollama isn't running at {url}. Start Ollama, then run: ollama pull {model}")
        return False
    names = {m.get("name") for m in tags}
    if model in names or f"{model}:latest" in names:
        print(f"  Ollama has {model}")
        return True
    print(f"  Ollama is running but {model} isn't installed. Run: ollama pull {model}")
    return False


def main() -> None:
    s = get_settings()
    s.models_dir.mkdir(parents=True, exist_ok=True)
    print("Local models")
    for name in (s.embedding_model, s.reranker_model):
        download(name, local_model_dir(s.models_dir, name))
    print("Language model")
    ok = check_ollama(s.ollama_url, s.ollama_model)
    print("\nDone. You can now switch the internet off and start the app." if ok else "\nFinish the Ollama step above.")


if __name__ == "__main__":
    main()

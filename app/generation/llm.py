"""The language model, served by Ollama on this computer.

`OllamaLLM` uses Ollama's native chat API so each request can set everything the app relies on:
  * format     - a JSON schema, so the answer always parses (structured output)
  * think      - false: Qwen3-style "thinking" is skipped; retrieval already did the hard work
  * num_ctx    - the context window, sent per request so it never depends on Ollama's settings
  * keep_alive - keeps the model loaded on the GPU between questions

`ExtractiveLLM` is the no-model fallback: the answerer quotes the best sentences instead.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Protocol

import httpx

from app.config import Settings

log = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Ollama isn't running, or the model hasn't been pulled."""


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, schema: dict | None = None) -> str: ...

    def stream(self, system: str, user: str, schema: dict | None = None) -> Iterator[str]: ...


class OllamaLLM:
    def __init__(self, s: Settings, client: httpx.Client | None = None):
        self.name = f"ollama/{s.ollama_model}"
        self.model = s.ollama_model
        self._url = s.ollama_url.rstrip("/")
        self._options = {"num_ctx": s.ollama_context, "num_predict": s.ollama_max_tokens, "temperature": 0}
        self._keep_alive = s.ollama_keep_alive
        self._client = client or httpx.Client(timeout=s.ollama_timeout_s)

    def _body(self, system: str, user: str, schema: dict | None, stream: bool) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": stream,
            "think": False,
            "options": self._options,
            "keep_alive": self._keep_alive,
        }
        if schema:
            body["format"] = schema
        return body

    def stream(self, system: str, user: str, schema: dict | None = None) -> Iterator[str]:
        """Yield the reply piece by piece as the model writes it."""
        body = self._body(system, user, schema, stream=True)
        url = f"{self._url}/api/chat"
        for attempt in range(2):
            try:
                with self._client.stream("POST", url, json=body) as r:
                    if r.status_code == 404:
                        raise LLMUnavailableError(
                            f"The model '{self.model}' isn't installed. Run: ollama pull {self.model}"
                        )
                    if r.status_code == 400 and attempt == 0 and "think" in r.read().decode(errors="ignore").lower():
                        body.pop("think")  # models without a thinking mode reject the flag
                        continue
                    r.raise_for_status()
                    wrote = False
                    for line in r.iter_lines():
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        piece = (data.get("message") or {}).get("content") or ""
                        if piece:
                            wrote = wrote or bool(piece.strip())
                            yield piece
                        if data.get("done"):
                            if not wrote and data.get("done_reason") == "length":
                                raise RuntimeError(
                                    f"The answer was cut off at {self._options['num_predict']} tokens. "
                                    "Increase OLLAMA_MAX_TOKENS."
                                )
                            return
                    return
            except httpx.TransportError as e:
                raise LLMUnavailableError(f"Ollama isn't reachable at {self._url}. Is it running?") from e

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        body = self._body(system, user, schema, stream=False)
        try:
            r = self._client.post(f"{self._url}/api/chat", json=body)
        except httpx.TransportError as e:
            raise LLMUnavailableError(f"Ollama isn't reachable at {self._url}. Is it running?") from e
        if r.status_code == 404:
            raise LLMUnavailableError(f"The model '{self.model}' isn't installed. Run: ollama pull {self.model}")
        if r.status_code == 400 and "think" in r.text.lower():
            body.pop("think")  # models without a thinking mode reject the flag
            r = self._client.post(f"{self._url}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        text = (data.get("message") or {}).get("content") or ""
        if not text.strip() and data.get("done_reason") == "length":
            raise RuntimeError(
                f"The answer was cut off at {self._options['num_predict']} tokens. Increase OLLAMA_MAX_TOKENS."
            )
        return text

    def status(self) -> str:
        """'ready', 'missing_model' or 'offline' - shown to the user in plain words."""
        try:
            r = self._client.get(f"{self._url}/api/tags", timeout=3)
            r.raise_for_status()
        except httpx.HTTPError:
            return "offline"
        names = {m.get("name", "") for m in r.json().get("models", [])}
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return "ready" if wanted in names or self.model in names else "missing_model"


class ExtractiveLLM:
    """Marker class: the answerer builds an answer by quoting the best sentences. No model needed."""

    name = "extractive"

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:  # pragma: no cover
        raise NotImplementedError

    def stream(self, system: str, user: str, schema: dict | None = None) -> Iterator[str]:  # pragma: no cover
        raise NotImplementedError

    def status(self) -> str:
        return "extractive"


def build_llm(s: Settings) -> LLM:
    p = s.llm_provider.lower()
    if p == "ollama":
        return OllamaLLM(s)
    if p == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Unknown LLM_PROVIDER '{s.llm_provider}' (use 'ollama' or 'extractive')")

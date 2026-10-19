"""The Ollama client, exercised against a simulated Ollama server (no network, no GPU)."""

import json

import httpx
import pytest

from app.config import Settings
from app.generation.llm import LLMUnavailableError, OllamaLLM, build_llm
from app.generation.prompts import ANSWER_SCHEMA

REPLY = '{"answerable": true, "answer": "ok [S1]", "citations": [], "missing": null}'


def ollama(handler) -> OllamaLLM:
    s = Settings(_env_file=None, ollama_url="http://localhost:11434", ollama_model="qwen3:14b")
    return OllamaLLM(s, client=httpx.Client(transport=httpx.MockTransport(handler)))


def chat_reply(content: str = REPLY, done_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200, json={"message": {"role": "assistant", "content": content}, "done": True, "done_reason": done_reason}
    )


def test_request_has_schema_no_thinking_and_full_context():
    sent = []

    def handler(req: httpx.Request) -> httpx.Response:
        sent.append((req.url.path, json.loads(req.content)))
        return chat_reply()

    assert ollama(handler).complete("SYS", "USER", ANSWER_SCHEMA) == REPLY
    path, body = sent[0]
    assert path == "/api/chat" and body["model"] == "qwen3:14b" and body["stream"] is False
    assert body["format"] == ANSWER_SCHEMA and body["think"] is False
    assert body["options"]["num_ctx"] == 8192 and body["options"]["temperature"] == 0
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]


def test_model_without_thinking_mode_is_retried_without_the_flag():
    bodies = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        if "think" in bodies[-1]:
            return httpx.Response(400, json={"error": "model does not support thinking"})
        return chat_reply()

    assert ollama(handler).complete("S", "U") == REPLY
    assert len(bodies) == 2 and "think" not in bodies[1]


def test_ollama_not_running_is_reported_clearly():
    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(LLMUnavailableError, match="isn't reachable"):
        ollama(handler).complete("S", "U")
    assert ollama(handler).status() == "offline"


def test_missing_model_is_reported_with_the_fix():
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama3.1:8b"}]})
        return httpx.Response(404, json={"error": "model 'qwen3:14b' not found"})

    with pytest.raises(LLMUnavailableError, match="ollama pull qwen3:14b"):
        ollama(handler).complete("S", "U")
    assert ollama(handler).status() == "missing_model"


def test_status_ready_when_model_installed():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3:14b"}]})

    assert ollama(handler).status() == "ready"


def test_truncated_answer_gives_a_clear_error():
    with pytest.raises(RuntimeError, match="OLLAMA_MAX_TOKENS"):
        ollama(lambda req: chat_reply("", done_reason="length")).complete("S", "U", ANSWER_SCHEMA)


def test_only_local_providers_exist():
    assert build_llm(Settings(_env_file=None, llm_provider="extractive")).name == "extractive"
    with pytest.raises(ValueError, match="ollama"):
        build_llm(Settings(_env_file=None, llm_provider="openai"))

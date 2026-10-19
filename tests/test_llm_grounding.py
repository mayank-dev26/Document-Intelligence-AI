"""The LLM path, with a scripted fake model: proves the grounding checks catch bad model output."""

import json

import pytest

from app.schemas import QueryRequest


def understood(user: str, intent: str = "lookup") -> str:
    """What the routing step of a real model returns: the intent and the standalone question."""
    question = user.split("LATEST QUESTION:")[-1].split("\n\nReturn the JSON")[0].strip()
    return json.dumps({"intent": intent, "question": question})


class ScriptedLLM:
    name = "scripted"

    def __init__(self, response: dict | str, intent: str = "lookup"):
        self.response = response
        self.intent = intent
        self.calls = 0  # answer calls only (not question routing)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        if schema and "intent" in schema["properties"]:
            return understood(user, self.intent)
        self.calls += 1
        self.prompts.append(user)
        assert schema and schema["required"] == ["answerable", "answer", "citations", "missing"]
        assert "SOURCES:" in user and "[S1]" in user
        return self.response if isinstance(self.response, str) else json.dumps(self.response)

    def stream(self, system: str, user: str, schema: dict | None = None):
        text = self.complete(system, user, schema)
        for i in range(0, len(text), 7):  # arrives in small pieces, like a real model
            yield text[i : i + 7]


@pytest.fixture
def with_llm(pipeline):
    original = pipeline.answerer.llm

    def install(response):
        pipeline.answerer.llm = ScriptedLLM(response)
        return pipeline.answerer.llm

    yield install
    pipeline.answerer.llm = original


Q = QueryRequest(question="How many days of annual leave do employees get?")


def test_good_answer_is_verified(pipeline, with_llm):
    with_llm(
        {
            "answerable": True,
            "answer": "Employees get 24 days [S1].",
            "citations": [{"source": "S1", "quote": "Employees receive 24 days of paid annual leave"}],
            "missing": None,
        }
    )
    a = pipeline.ask(Q)
    assert a.answerable and a.citations[0].verified and a.citations[0].match_score == 1.0


def test_fabricated_quote_is_caught(pipeline, with_llm):
    with_llm(
        {
            "answerable": True,
            "answer": "Employees get 40 days [S1].",
            "citations": [{"source": "S1", "quote": "Every employee is entitled to forty days of holiday"}],
        }
    )
    a = pipeline.ask(Q)
    assert not a.answerable and a.abstain_code == "ungrounded"
    assert a.citations and not a.citations[0].verified  # shown to the user as unverified


def test_model_refusal_is_respected(pipeline, with_llm):
    with_llm({"answerable": False, "answer": "not found", "citations": [], "missing": "no leave policy"})
    a = pipeline.ask(Q)
    assert not a.answerable


def test_gate_skips_llm_call_for_irrelevant_questions(pipeline, with_llm):
    llm = with_llm({"answerable": True, "answer": "x [S1]", "citations": []})
    a = pipeline.ask(QueryRequest(question="What is the parental leave entitlement in Germany?"))
    assert not a.answerable and llm.calls == 0  # saved a model call


def test_out_of_range_markers_ignored_and_fenced_json(pipeline, with_llm):
    with_llm(
        '```json\n{"answerable": true, "answer": "24 days [S1][S99]", "citations": '
        '[{"source": "S99", "quote": "x"}, {"source": "S1", "quote": "24 days of paid annual leave"}]}\n```'
    )
    a = pipeline.ask(Q)
    assert a.answerable and [c.marker for c in a.citations] == ["S1"]


class UnavailableLLM:
    """Behaves like Ollama when it isn't running."""

    name = "ollama/qwen3:14b"

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        from app.generation.llm import LLMUnavailableError

        raise LLMUnavailableError("Ollama isn't reachable")

    def stream(self, system: str, user: str, schema: dict | None = None):
        yield from ()
        self.complete(system, user, schema)


def test_answers_still_work_when_the_ai_engine_is_off(pipeline):
    original, pipeline.answerer.llm = pipeline.answerer.llm, UnavailableLLM()
    try:
        a = pipeline.ask(Q)
    finally:
        pipeline.answerer.llm = original
    assert a.answerable and "24 days" in a.answer and a.citations[0].verified
    assert a.note and "isn't running" in a.note and a.llm == "extractive (fallback)"


class FlakyLLM:
    """Fails on the third call, like a crashed model, to test the benchmark keeps going."""

    name = "flaky"

    def __init__(self):
        self.calls = 0

    def complete(self, system: str, user: str, schema: dict | None = None) -> str:
        self.calls += 1
        if self.calls == 3:
            raise RuntimeError("model crashed")
        return json.dumps({"answerable": False, "answer": "", "citations": [], "missing": None})

    def stream(self, system: str, user: str, schema: dict | None = None):
        yield self.complete(system, user, schema)


def test_benchmark_survives_a_failed_model_call(pipeline, tmp_path):
    from app.schemas import RetrievalMode
    from evaluation.runner import run_evaluation

    dataset = tmp_path / "q.jsonl"
    rows = [
        {
            "id": f"q{i}",
            "question": "How many days of annual leave do employees get?",
            "answerable": True,
            "doc": "handbook.pdf",
            "pages": [1],
            "evidence": "24 days",
            "reference": "24 days",
            "key_facts": ["24"],
        }
        for i in range(5)
    ]
    dataset.write_text("\n".join(json.dumps(r) for r in rows))
    original, pipeline.answerer.llm = pipeline.answerer.llm, FlakyLLM()
    try:
        rep = run_evaluation(pipeline, dataset, [RetrievalMode.bm25], [RetrievalMode.hybrid], save=False)
    finally:
        pipeline.answerer.llm = original
    g = rep["generation"]["hybrid"]
    assert g["n"] == 4 and g["n_failed"] == 1

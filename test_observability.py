"""
Tests for observability.py (OpenTelemetry tracing helpers).

Strategy:
- Use ``InMemorySpanExporter`` so no real Langfuse / OTLP endpoint is needed.
- Each test reinitialises the OTEL provider via ``init_tracing`` with the
  in-memory exporter, then calls the function under test, and asserts that
  the expected spans / attributes were recorded.

Coverage:
1. ``@traced`` sync function creates a span with the right name.
2. ``@traced`` async function creates a span with the right name.
3. ``record_llm_call`` attaches attributes to the active span.
4. ``record_llm_call`` outside a span is a no-op (no exception).
5. ``LLMRouter.complete`` wraps the backend call in a span and records
   llm.* attributes automatically.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure OTEL SDK is importable before all tests run
# ---------------------------------------------------------------------------
pytest.importorskip("opentelemetry", reason="opentelemetry-sdk not installed")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace import ReadableSpan

from observability import init_tracing, traced, record_llm_call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_exporter() -> InMemorySpanExporter:
    """Create a fresh InMemorySpanExporter and register it as the global provider."""
    exp = InMemorySpanExporter()
    init_tracing("test-service", exporter=exp)
    return exp


def _span_names(exporter: InMemorySpanExporter) -> list[str]:
    return [s.name for s in exporter.get_finished_spans()]


def _find_span(exporter: InMemorySpanExporter, name: str) -> ReadableSpan | None:
    for s in exporter.get_finished_spans():
        if s.name == name:
            return s
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTracedDecorator:
    def test_sync_function_creates_span(self) -> None:
        exp = _make_exporter()

        @traced
        def my_sync_fn(x: int) -> int:
            return x * 2

        result = my_sync_fn(21)
        assert result == 42

        names = _span_names(exp)
        assert any("my_sync_fn" in n for n in names), f"span not found: {names}"

    def test_async_function_creates_span(self) -> None:
        exp = _make_exporter()

        @traced
        async def my_async_fn(x: int) -> int:
            return x + 1

        result = asyncio.run(my_async_fn(41))
        assert result == 42

        names = _span_names(exp)
        assert any("my_async_fn" in n for n in names), f"span not found: {names}"

    def test_span_name_uses_qualname(self) -> None:
        exp = _make_exporter()

        @traced
        def outer_fn() -> None:
            pass

        outer_fn()
        names = _span_names(exp)
        assert "outer_fn" in names[0]

    def test_traced_passes_return_value_through(self) -> None:
        _make_exporter()

        @traced
        def greet(name: str) -> str:
            return f"hello {name}"

        assert greet("world") == "hello world"

    def test_traced_propagates_exception(self) -> None:
        _make_exporter()

        @traced
        def boom() -> None:
            raise ValueError("explode")

        with pytest.raises(ValueError, match="explode"):
            boom()


class TestRecordLlmCall:
    def test_attaches_attributes_to_active_span(self) -> None:
        exp = _make_exporter()

        @traced
        def my_llm_wrapper() -> None:
            record_llm_call(
                model="test-model",
                tokens_in=10,
                tokens_out=20,
                cost=0.001,
                latency=150,
            )

        my_llm_wrapper()

        spans = exp.get_finished_spans()
        assert len(spans) == 1
        attrs = spans[0].attributes or {}
        assert attrs.get("llm.model") == "test-model"
        assert attrs.get("llm.tokens_in") == 10
        assert attrs.get("llm.tokens_out") == 20
        assert attrs.get("llm.cost_usd") == pytest.approx(0.001)
        assert attrs.get("llm.latency_ms") == 150

    def test_noop_outside_span(self) -> None:
        """Calling record_llm_call with no active span must not raise."""
        _make_exporter()
        # No @traced wrapper → no active span
        record_llm_call(model="x", tokens_in=1, tokens_out=1, cost=0.0, latency=5)


class TestLLMRouterInstrumentation:
    """Verify that LLMRouter.complete() creates a span with llm.* attributes."""

    def _make_mock_backend_fn(self) -> Any:
        def mock_request_fn(
            messages: list[dict[str, str]],
            model: str,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return {
                "content": "ok",
                "model_used": model,
                "tokens_in": 5,
                "tokens_out": 3,
                "cost_usd": 0.002,
            }
        return mock_request_fn

    def test_complete_creates_llm_span(self) -> None:
        exp = _make_exporter()

        from llm_router import LLMRouter, TaskType, LocalQwenBackend, GeminiBackend, ClaudeBackend

        fn = self._make_mock_backend_fn()
        router = LLMRouter(
            local_backend=LocalQwenBackend(request_fn=fn),
            gemini_backend=GeminiBackend(request_fn=fn),
            claude_backend=ClaudeBackend(request_fn=fn),
        )

        router.complete(TaskType.RETRIEVAL_REASONING, [{"role": "user", "content": "test"}])

        span = _find_span(exp, "llm.RETRIEVAL_REASONING")
        assert span is not None, f"expected span 'llm.RETRIEVAL_REASONING', got: {_span_names(exp)}"

    def test_complete_records_llm_attributes(self) -> None:
        exp = _make_exporter()

        from llm_router import LLMRouter, TaskType, LocalQwenBackend, GeminiBackend, ClaudeBackend

        fn = self._make_mock_backend_fn()
        router = LLMRouter(
            local_backend=LocalQwenBackend(request_fn=fn),
            gemini_backend=GeminiBackend(request_fn=fn),
            claude_backend=ClaudeBackend(request_fn=fn),
        )

        router.complete(TaskType.SYNTHESIS, [{"role": "user", "content": "hello"}])

        span = _find_span(exp, "llm.SYNTHESIS")
        assert span is not None
        attrs = span.attributes or {}
        assert "llm.model" in attrs
        assert attrs.get("llm.tokens_in") == 5
        assert attrs.get("llm.tokens_out") == 3
        assert attrs.get("llm.cost_usd") == pytest.approx(0.002)
        assert attrs.get("llm.completion_preview") == "ok"

    def test_intent_parsing_routes_to_local_backend(self) -> None:
        exp = _make_exporter()

        from llm_router import LLMRouter, TaskType, LocalQwenBackend, GeminiBackend, ClaudeBackend, DeepSeekBackend
        from llm_router import OllamaBackend

        fn = self._make_mock_backend_fn()
        router = LLMRouter(
            local_backend=LocalQwenBackend(request_fn=fn),
            gemini_backend=GeminiBackend(request_fn=fn),
            claude_backend=ClaudeBackend(request_fn=fn),
        )

        with patch.object(OllamaBackend, "is_available", return_value=True), \
             patch.object(DeepSeekBackend, "is_available", return_value=False):
            router.complete(TaskType.INTENT_PARSING, [{"role": "user", "content": "hi"}])

        span = _find_span(exp, "llm.INTENT_PARSING")
        assert span is not None
        attrs = span.attributes or {}
        # INTENT_PARSING chain: Ollama (alias LocalQwenBackend) → Gemini
        assert attrs.get("llm.backend.selected") == "OllamaBackend"
        attempted = attrs.get("llm.backend.attempted", "")
        assert "OllamaBackend:ok" in str(attempted)

    def test_critique_fallback_chain_uses_gemini_when_vllm_unavailable(self) -> None:
        """CRITIQUE chain is vLLM → Gemini → Ollama. With default empty VLLM_URL, Gemini wins."""
        import os

        exp = _make_exporter()

        from llm_router import LLMRouter, TaskType, LocalQwenBackend, GeminiBackend, ClaudeBackend

        fn = self._make_mock_backend_fn()
        router = LLMRouter(
            local_backend=LocalQwenBackend(request_fn=fn),
            gemini_backend=GeminiBackend(request_fn=fn),
            claude_backend=ClaudeBackend(request_fn=fn),
        )

        # Gemini availability is keyed off env; stabilize without relying on workstation config.
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-nonempty-key"}, clear=False):
            router.complete(TaskType.CRITIQUE, [{"role": "user", "content": "review this"}])

        span = _find_span(exp, "llm.CRITIQUE")
        assert span is not None
        attrs = span.attributes or {}
        assert attrs.get("llm.backend.selected") == "GeminiBackend"
        attempted = str(attrs.get("llm.backend.attempted", ""))
        assert "VLLMBackend:unavailable" in attempted
        assert "GeminiBackend:ok" in attempted

    def test_span_task_type_attribute(self) -> None:
        exp = _make_exporter()

        from llm_router import LLMRouter, TaskType, LocalQwenBackend, GeminiBackend, ClaudeBackend

        fn = self._make_mock_backend_fn()
        router = LLMRouter(
            local_backend=LocalQwenBackend(request_fn=fn),
            gemini_backend=GeminiBackend(request_fn=fn),
            claude_backend=ClaudeBackend(request_fn=fn),
        )

        router.complete(TaskType.EMBEDDING, [{"role": "user", "content": "embed me"}])

        span = _find_span(exp, "llm.EMBEDDING")
        assert span is not None
        attrs = span.attributes or {}
        assert attrs.get("llm.task_type") == "EMBEDDING"

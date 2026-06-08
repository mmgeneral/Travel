"""Tests for LLMRouter three-tier fallback logic.

Coverage
--------
* All backends available → first in chain is used.
* First backend unavailable → second is tried automatically.
* First backend raises CircuitBreakerOpenError → next backend is tried.
* All backends unavailable → RuntimeError raised.
* is_available() cache: second call within 30 s does NOT re-probe.
* is_available() cache: call after 30 s DOES re-probe.
* GeminiBackend.is_available() returns False when GEMINI_API_KEY is empty.
* ClaudeBackend.is_available() returns False when ANTHROPIC_API_KEY is empty.
* OllamaBackend (1080 Ti always-on) is first for INTENT_PARSING.
* VLLMBackend (4090 on-demand) is first for CRITIQUE.
* OllamaBackend is in CRITIQUE fallback chain.
* VLLM_URL empty → VLLMBackend.is_available() returns False immediately.
* VLLMBackend probes /v1/models endpoint.
* OllamaBackend probes /api/tags endpoint.
* OTEL span records llm.backend.attempted and llm.backend.selected.
* LocalQwenBackend alias works as OllamaBackend.
* LLMRouter backward-compat: local_backend kwarg still accepted.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from llm_router import (
    BackendTransientError,
    CircuitBreakerOpenError,
    ClaudeBackend,
    GeminiBackend,
    LLMResponse,
    LLMRouter,
    LocalQwenBackend,
    OllamaBackend,
    TaskType,
    VLLMBackend,
    _AvailabilityCache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_response(model: str = "test-model") -> LLMResponse:
    return LLMResponse(
        content="ok",
        model_used=model,
        tokens_in=10,
        tokens_out=5,
        latency_ms=100,
        cost_usd=0.0,
    )


def _mock_backend(
    available: bool = True,
    response: LLMResponse | None = None,
    raises: Exception | None = None,
    name: str = "MockBackend",
) -> MagicMock:
    b = MagicMock()
    b.is_available.return_value = available
    b.name = name
    if raises is not None:
        b.complete.side_effect = raises
    else:
        b.complete.return_value = response or _ok_response(name)
    return b


def _router(
    ollama: MagicMock | None = None,
    vllm: MagicMock | None = None,
    gemini: MagicMock | None = None,
    claude: MagicMock | None = None,
) -> LLMRouter:
    r = LLMRouter.__new__(LLMRouter)
    r.ollama_backend = ollama or _mock_backend(name="OllamaBackend")
    r.vllm_backend = vllm or _mock_backend(available=False, name="VLLMBackend")
    r.gemini_backend = gemini or _mock_backend(name="GeminiBackend")
    r.claude_backend = claude or _mock_backend(name="ClaudeBackend")
    r.deepseek_backend = _mock_backend(available=False, name="DeepSeekBackend")
    r.local_backend = r.ollama_backend
    return r


# ---------------------------------------------------------------------------
# 1. Routing: first available backend is used
# ---------------------------------------------------------------------------

class TestRoutingLogic:
    def test_intent_parsing_uses_ollama_first(self):
        ollama = _mock_backend(available=True, name="OllamaBackend")
        gemini = _mock_backend(available=True, name="GeminiBackend")
        r = _router(ollama=ollama, gemini=gemini)
        r.complete(TaskType.INTENT_PARSING, [{"role": "user", "content": "hi"}])
        ollama.complete.assert_called_once()
        gemini.complete.assert_not_called()

    def test_critique_uses_vllm_first(self):
        vllm = _mock_backend(available=True, name="VLLMBackend")
        gemini = _mock_backend(available=True, name="GeminiBackend")
        ollama = _mock_backend(available=True, name="OllamaBackend")
        r = _router(ollama=ollama, vllm=vllm, gemini=gemini)
        r.complete(TaskType.CRITIQUE, [{"role": "user", "content": "review"}])
        vllm.complete.assert_called_once()
        gemini.complete.assert_not_called()
        ollama.complete.assert_not_called()

    def test_retrieval_uses_ollama_first(self):
        ollama = _mock_backend(available=True, name="OllamaBackend")
        gemini = _mock_backend(available=True, name="GeminiBackend")
        r = _router(ollama=ollama, gemini=gemini)
        r.complete(TaskType.RETRIEVAL_REASONING, [{"role": "user", "content": "find"}])
        ollama.complete.assert_called_once()
        gemini.complete.assert_not_called()

    def test_synthesis_uses_ollama_first(self):
        ollama = _mock_backend(available=True, name="OllamaBackend")
        gemini = _mock_backend(available=True, name="GeminiBackend")
        r = _router(ollama=ollama, gemini=gemini)
        r.complete(TaskType.SYNTHESIS, [])
        ollama.complete.assert_called_once()
        gemini.complete.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Fallback when first backend is unavailable
# ---------------------------------------------------------------------------

class TestFallback:
    def test_falls_back_when_first_unavailable(self):
        """INTENT_PARSING: Ollama down → Gemini is used."""
        ollama = _mock_backend(available=False, name="OllamaBackend")
        gemini = _mock_backend(available=True, name="GeminiBackend")
        r = _router(ollama=ollama, gemini=gemini)
        resp = r.complete(TaskType.INTENT_PARSING, [{"role": "user", "content": "hi"}])
        ollama.complete.assert_not_called()
        gemini.complete.assert_called_once()
        assert resp.model_used == "GeminiBackend"

    def test_critique_fallback_vllm_down_uses_gemini(self):
        vllm = _mock_backend(available=False, name="VLLMBackend")
        gemini = _mock_backend(available=True, name="GeminiBackend")
        ollama = _mock_backend(available=True, name="OllamaBackend")
        r = _router(ollama=ollama, vllm=vllm, gemini=gemini)
        resp = r.complete(TaskType.CRITIQUE, [])
        vllm.complete.assert_not_called()
        gemini.complete.assert_called_once()
        assert resp.model_used == "GeminiBackend"

    def test_critique_full_fallback_to_ollama(self):
        """Both vLLM and Gemini down → Ollama handles CRITIQUE."""
        vllm = _mock_backend(available=False, name="VLLMBackend")
        gemini = _mock_backend(available=False, name="GeminiBackend")
        ollama = _mock_backend(available=True, name="OllamaBackend")
        r = _router(ollama=ollama, vllm=vllm, gemini=gemini)
        resp = r.complete(TaskType.CRITIQUE, [])
        assert resp.model_used == "OllamaBackend"

    def test_skips_circuit_breaker_open(self):
        """Backend with open circuit breaker is skipped, next is tried."""
        ollama = _mock_backend(
            available=True, raises=CircuitBreakerOpenError("open"), name="OllamaBackend"
        )
        gemini = _mock_backend(available=True, name="GeminiBackend")
        r = _router(ollama=ollama, gemini=gemini)
        resp = r.complete(TaskType.INTENT_PARSING, [])
        gemini.complete.assert_called_once()
        assert resp.model_used == "GeminiBackend"

    def test_skips_transient_error(self):
        """BackendTransientError causes fallback to next backend."""
        ollama = _mock_backend(
            available=True, raises=BackendTransientError("timeout"), name="OllamaBackend"
        )
        gemini = _mock_backend(available=True, name="GeminiBackend")
        r = _router(ollama=ollama, gemini=gemini)
        resp = r.complete(TaskType.INTENT_PARSING, [])
        assert resp.model_used == "GeminiBackend"

    def test_generic_exception_skips_to_next(self):
        """Any unexpected exception from a backend also triggers fallback."""
        ollama = _mock_backend(
            available=True, raises=RuntimeError("network error"), name="OllamaBackend"
        )
        gemini = _mock_backend(available=True, name="GeminiBackend")
        r = _router(ollama=ollama, gemini=gemini)
        resp = r.complete(TaskType.INTENT_PARSING, [])
        assert resp.model_used == "GeminiBackend"


# ---------------------------------------------------------------------------
# 3. All backends unavailable → RuntimeError
# ---------------------------------------------------------------------------

class TestAllUnavailable:
    def test_raises_when_all_unavailable(self):
        ollama = _mock_backend(available=False, name="OllamaBackend")
        gemini = _mock_backend(available=False, name="GeminiBackend")
        r = _router(ollama=ollama, gemini=gemini)
        with pytest.raises(RuntimeError, match="All backends exhausted"):
            r.complete(TaskType.INTENT_PARSING, [])

    def test_raises_when_all_circuit_broken(self):
        ollama = _mock_backend(
            available=True, raises=CircuitBreakerOpenError("open"), name="OllamaBackend"
        )
        gemini = _mock_backend(
            available=True, raises=CircuitBreakerOpenError("open"), name="GeminiBackend"
        )
        r = _router(ollama=ollama, gemini=gemini)
        with pytest.raises(RuntimeError, match="All backends exhausted"):
            r.complete(TaskType.INTENT_PARSING, [])

    def test_error_message_lists_attempted_backends(self):
        ollama = _mock_backend(available=False, name="OllamaBackend")
        gemini = _mock_backend(available=False, name="GeminiBackend")
        r = _router(ollama=ollama, gemini=gemini)
        with pytest.raises(RuntimeError) as exc_info:
            r.complete(TaskType.INTENT_PARSING, [])
        assert "unavailable" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. is_available() caching
# ---------------------------------------------------------------------------

class TestAvailabilityCache:
    def test_cache_is_used_within_ttl(self):
        """Second call within 30 s does NOT re-probe the health endpoint."""
        backend = OllamaBackend.__new__(OllamaBackend)
        backend._avail_cache = _AvailabilityCache(ttl_sec=30.0)
        backend._avail_cache.update(True)  # seed cache

        probe_calls = 0

        def counting_probe() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            return True

        backend._probe_availability = counting_probe
        result = backend.is_available()
        assert result is True
        assert probe_calls == 0, "Should use cache, not re-probe"

    def test_cache_expires_after_ttl(self):
        """Call after TTL triggers a fresh probe."""
        backend = OllamaBackend.__new__(OllamaBackend)
        backend._avail_cache = _AvailabilityCache(ttl_sec=0.0)  # zero TTL → always stale

        probe_calls = 0

        def counting_probe() -> bool:
            nonlocal probe_calls
            probe_calls += 1
            return True

        backend._probe_availability = counting_probe
        backend.is_available()
        assert probe_calls == 1

    def test_cache_ttl_is_30_seconds_by_default(self):
        cache = _AvailabilityCache()
        assert cache.ttl_sec == 30.0

    def test_fresh_cache_is_stale(self):
        cache = _AvailabilityCache()
        assert cache.is_stale()  # checked_at=0 → always stale

    def test_updated_cache_is_not_stale(self):
        cache = _AvailabilityCache(ttl_sec=30.0)
        cache.update(True)
        assert not cache.is_stale()


# ---------------------------------------------------------------------------
# 5. GeminiBackend / ClaudeBackend: key-based availability
# ---------------------------------------------------------------------------

class TestKeyBasedAvailability:
    def test_gemini_unavailable_when_key_empty(self):
        b = GeminiBackend()
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}, clear=False):
            b._avail_cache = _AvailabilityCache()  # reset cache
            assert b.is_available() is False

    def test_gemini_available_when_key_set(self):
        b = GeminiBackend()
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key-abc"}, clear=False):
            b._avail_cache = _AvailabilityCache()
            assert b.is_available() is True

    def test_claude_unavailable_when_key_empty(self):
        b = ClaudeBackend()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
            b._avail_cache = _AvailabilityCache()
            assert b.is_available() is False

    def test_claude_available_when_key_set(self):
        b = ClaudeBackend()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-xxx"}, clear=False):
            b._avail_cache = _AvailabilityCache()
            assert b.is_available() is True


# ---------------------------------------------------------------------------
# 6. VLLMBackend
# ---------------------------------------------------------------------------

class TestVLLMBackend:
    def test_unavailable_when_url_empty(self):
        b = VLLMBackend(base_url="")
        b._avail_cache = _AvailabilityCache()
        assert b.is_available() is False

    def test_unavailable_when_vllm_url_env_empty(self):
        b = VLLMBackend()
        with patch.dict("os.environ", {"VLLM_URL": ""}, clear=False):
            b._avail_cache = _AvailabilityCache()
            assert b.is_available() is False

    def test_probes_v1_models_endpoint(self):
        b = VLLMBackend(base_url="http://gpu-lab:8000")
        b._avail_cache = _AvailabilityCache()
        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            result = b.is_available()
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert "/v1/models" in call_url
        assert result is True

    def test_http_error_returns_false(self):
        b = VLLMBackend(base_url="http://gpu-lab:8000")
        b._avail_cache = _AvailabilityCache()
        with patch("requests.get", side_effect=ConnectionError("refused")):
            assert b.is_available() is False


# ---------------------------------------------------------------------------
# 7. OllamaBackend
# ---------------------------------------------------------------------------

class TestOllamaBackend:
    def test_probes_api_tags_endpoint(self):
        b = OllamaBackend(host="http://localhost:11434")
        b._avail_cache = _AvailabilityCache()
        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            result = b.is_available()
        call_url = mock_get.call_args[0][0]
        assert "/api/tags" in call_url
        assert result is True

    def test_connection_error_returns_false(self):
        b = OllamaBackend(host="http://localhost:11434")
        b._avail_cache = _AvailabilityCache()
        with patch("requests.get", side_effect=ConnectionError("refused")):
            assert b.is_available() is False

    def test_strips_api_chat_suffix_from_url(self):
        b = OllamaBackend(host="http://localhost:11434/api/chat")
        assert b._host == "http://localhost:11434"

    def test_local_qwen_backend_is_alias(self):
        b = LocalQwenBackend()
        assert isinstance(b, OllamaBackend)

    def test_reads_ollama_model_from_env(self):
        with patch.dict("os.environ", {"OLLAMA_MODEL": "llama3.1:8b"}):
            b = OllamaBackend()
        assert b.model == "llama3.1:8b"


# ---------------------------------------------------------------------------
# 8. LLMRouter backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_local_backend_kwarg_accepted(self):
        """Existing code that passes local_backend= must still work."""
        local = _mock_backend(name="OllamaBackend")
        r = LLMRouter(local_backend=local)
        assert r.local_backend is local
        assert r.ollama_backend is local

    def test_local_backend_attribute_present(self):
        r = LLMRouter()
        assert hasattr(r, "local_backend")
        assert r.local_backend is r.ollama_backend


# ---------------------------------------------------------------------------
# 9. OTEL span attributes
# ---------------------------------------------------------------------------

class TestOtelAttributes:
    def test_span_records_attempted_and_selected(self):
        """When OTEL is active, the span must record which backend was selected."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        import observability

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        # Inject provider into observability module
        observability._tracer = provider.get_tracer("test")
        try:
            ollama = _mock_backend(available=True, name="OllamaBackend")
            gemini = _mock_backend(available=True, name="GeminiBackend")
            r = _router(ollama=ollama, gemini=gemini)
            r.complete(TaskType.INTENT_PARSING, [{"role": "user", "content": "hi"}])
        finally:
            observability._tracer = None

        spans = exporter.get_finished_spans()
        assert spans, "No spans were recorded"
        attrs = spans[0].attributes or {}
        assert "llm.backend.attempted" in attrs
        assert "llm.backend.selected" in attrs
        assert "OllamaBackend" in str(attrs.get("llm.backend.selected", ""))

    def test_span_records_fallback_path(self):
        """When first backend fails, selected should show the fallback backend."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        import observability

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        observability._tracer = provider.get_tracer("test")
        try:
            ollama = _mock_backend(available=False, name="OllamaBackend")
            gemini = _mock_backend(available=True, name="GeminiBackend")
            r = _router(ollama=ollama, gemini=gemini)
            r.complete(TaskType.INTENT_PARSING, [])
        finally:
            observability._tracer = None

        spans = exporter.get_finished_spans()
        attrs = spans[0].attributes or {}
        selected = str(attrs.get("llm.backend.selected", ""))
        attempted = str(attrs.get("llm.backend.attempted", ""))
        assert "GeminiBackend" in selected
        assert "unavailable" in attempted  # OllamaBackend was skipped

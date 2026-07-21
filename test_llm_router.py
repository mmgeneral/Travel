from __future__ import annotations

import time

import pytest

from llm_router import (
    CircuitBreakerOpenError,
    ClaudeBackend,
    GeminiBackend,
    LLMRouter,
    LocalQwenBackend,
    TaskType,
)


def _ok_response(tag: str):
    return {
        "content": f"ok:{tag}",
        "model_used": f"mock-{tag}",
        "tokens_in": 3,
        "tokens_out": 5,
        "cost_usd": 0.001,
    }


def test_routing_logic_intent_goes_local() -> None:
    calls: list[str] = []
    local = LocalQwenBackend(request_fn=lambda **_: calls.append("local") or _ok_response("local"))
    gemini = GeminiBackend(request_fn=lambda **_: calls.append("gemini") or _ok_response("gemini"))
    claude = ClaudeBackend(request_fn=lambda **_: calls.append("claude") or _ok_response("claude"))
    r = LLMRouter(local_backend=local, gemini_backend=gemini, claude_backend=claude)

    from llm_router import DeepSeekBackend
    from unittest.mock import patch
    with patch.object(DeepSeekBackend, "is_available", return_value=False):
        out = r.complete(TaskType.INTENT_PARSING, [{"role": "user", "content": "hi"}])
    assert out.content == "ok:local"
    assert calls == ["local"]


def test_timeout_triggers_retry_then_success() -> None:
    state = {"n": 0}

    def flaky(**_):
        state["n"] += 1
        if state["n"] < 3:
            time.sleep(0.05)
        return _ok_response("eventual")

    backend = LocalQwenBackend(request_fn=flaky, timeout=0.01, max_retries=3)
    out = backend.complete([{"role": "user", "content": "ping"}])
    assert out.content == "ok:eventual"
    assert state["n"] == 3


def test_circuit_breaker_opens_after_five_failures() -> None:
    state = {"n": 0}

    def always_fail(**_):
        state["n"] += 1
        raise RuntimeError("boom")

    backend = LocalQwenBackend(
        request_fn=always_fail,
        timeout=0.02,
        max_retries=1,
        circuit_breaker_threshold=5,
        circuit_breaker_cooldown_sec=60,
    )
    for _ in range(5):
        with pytest.raises(Exception):
            backend.complete([{"role": "user", "content": "x"}])

    with pytest.raises(CircuitBreakerOpenError):
        backend.complete([{"role": "user", "content": "x"}])
    assert state["n"] == 5


def test_all_backends_support_mock_request_fn() -> None:
    local = LocalQwenBackend(request_fn=lambda **_: _ok_response("local"))
    gemini = GeminiBackend(request_fn=lambda **_: _ok_response("gemini"))
    claude = ClaudeBackend(request_fn=lambda **_: _ok_response("claude"))

    assert local.complete([{"role": "user", "content": "1"}]).content == "ok:local"
    assert gemini.complete([{"role": "user", "content": "2"}]).content == "ok:gemini"
    assert claude.complete([{"role": "user", "content": "3"}]).content == "ok:claude"

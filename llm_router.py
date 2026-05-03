"""LLMRouter — central dispatch for all LLM calls in the travel-agent system.

Backend priority (local-first, cloud-fallback):

    1. OllamaBackend   — always-on 1080 Ti (Qwen 7B)       → OLLAMA_URL
    2. VLLMBackend     — on-demand lab 4090 (Qwen 14B AWQ)  → VLLM_URL (skip if empty)
    3. GeminiBackend   — cloud free-tier fallback            → GEMINI_API_KEY
    4. ClaudeBackend   — optional paid tier                  → ANTHROPIC_API_KEY

Each backend exposes ``is_available()`` (cheap health-check, result cached 30 s).
``LLMRouter.complete()`` tries backends in task-specific order, skipping any that are
unavailable or have an open circuit-breaker, and raises only when the entire chain fails.

Environment variables (see .env.example)::

    OLLAMA_URL=http://localhost:11434   (default)
    OLLAMA_MODEL=qwen2.5:7b            (default)
    VLLM_URL=                          (empty → vLLM disabled)
    VLLM_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
    GEMINI_API_KEY=
    GEMINI_MODEL=gemini-2.0-flash
    ANTHROPIC_API_KEY=
    CLAUDE_MODEL=claude-3-5-sonnet-latest
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from observability import record_llm_call, _get_tracer
from tracing import (
    generation_update_error,
    generation_update_success,
    langfuse_tracing_active,
    llm_generation_context,
)


# ---------------------------------------------------------------------------
# Enums / dataclasses shared across all layers
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    INTENT_PARSING = "INTENT_PARSING"
    RETRIEVAL_REASONING = "RETRIEVAL_REASONING"
    CRITIQUE = "CRITIQUE"
    SYNTHESIS = "SYNTHESIS"
    EMBEDDING = "EMBEDDING"


@dataclass
class LLMResponse:
    content: str
    model_used: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float


@dataclass
class _AvailabilityCache:
    """30-second TTL cache for a single backend health check."""
    available: bool = False
    checked_at: float = 0.0
    ttl_sec: float = 30.0

    def is_stale(self) -> bool:
        return (time.monotonic() - self.checked_at) >= self.ttl_sec

    def update(self, available: bool) -> None:
        self.available = available
        self.checked_at = time.monotonic()


class CircuitBreakerOpenError(RuntimeError):
    pass


class BackendTransientError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# BaseBackend
# ---------------------------------------------------------------------------

class BaseBackend:
    """Abstract base with retry, circuit-breaker, timeout, and availability check."""

    _AVAILABILITY_TIMEOUT: float = 2.0  # max seconds for is_available() HTTP probe

    def __init__(
        self,
        model: str,
        *,
        timeout: float = 20.0,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_cooldown_sec: float = 30.0,
        request_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_cooldown_sec = circuit_breaker_cooldown_sec
        self.request_fn = request_fn
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._avail_cache = _AvailabilityCache()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the backend can accept requests.

        Result is cached for 30 s to avoid hammering health endpoints.
        """
        if not self._avail_cache.is_stale():
            return self._avail_cache.available
        result = self._probe_availability()
        self._avail_cache.update(result)
        return result

    def _probe_availability(self) -> bool:
        """Subclasses override this for their specific health check."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self._guard_circuit_breaker()
        started = time.perf_counter()
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=0.25, min=0.1, max=2),
                retry=retry_if_exception_type((TimeoutError, BackendTransientError)),
                reraise=True,
            )
            payload: dict[str, Any] | None = None
            for attempt in retryer:
                with attempt:
                    payload = self._invoke_with_timeout(messages, **kwargs)
            assert payload is not None
            self._consecutive_failures = 0
            latency_ms = int((time.perf_counter() - started) * 1000)
            content = str(payload.get("content", ""))
            tokens_in = int(payload.get("tokens_in", self._estimate_tokens(messages)))
            tokens_out = int(payload.get("tokens_out", self._estimate_tokens(content)))
            cost_usd = float(payload.get("cost_usd", 0.0))
            return LLMResponse(
                content=content,
                model_used=str(payload.get("model_used", self.model)),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
            )
        except Exception:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.circuit_breaker_threshold:
                self._circuit_open_until = time.monotonic() + self.circuit_breaker_cooldown_sec
            raise

    def _guard_circuit_breaker(self) -> None:
        if time.monotonic() < self._circuit_open_until:
            raise CircuitBreakerOpenError(f"{self.__class__.__name__} circuit breaker open")

    def _invoke_with_timeout(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self._call_provider, messages, **kwargs)
            try:
                return fut.result(timeout=self.timeout)
            except FutureTimeoutError as exc:
                raise TimeoutError(f"{self.__class__.__name__} timed out") from exc

    def _call_provider(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _estimate_tokens(v: Any) -> int:
        if isinstance(v, list):
            text = " ".join(str(x.get("content", "")) for x in v if isinstance(x, dict))
        else:
            text = str(v)
        return max(1, len(text) // 4)

    @property
    def name(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# OllamaBackend  (always-on, local 1080 Ti)
# ---------------------------------------------------------------------------

class OllamaBackend(BaseBackend):
    """Connects to a local Ollama server.

    Defaults: OLLAMA_URL=http://localhost:11434, OLLAMA_MODEL=qwen2.5:7b.
    Health check: GET {host}/api/tags, timeout=2 s.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        _host = (host or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        # Strip /api/chat suffix if user mistakenly included it
        if _host.endswith("/api/chat"):
            _host = _host[: -len("/api/chat")]
        self._host = _host
        _model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        super().__init__(model=_model, **kwargs)

    def _probe_availability(self) -> bool:
        try:
            r = requests.get(
                f"{self._host}/api/tags",
                timeout=self._AVAILABILITY_TIMEOUT,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _call_provider(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        if self.request_fn is not None:
            return self.request_fn(messages=messages, model=self.model, **kwargs)
        resp = requests.post(
            f"{self._host}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        obj = resp.json()
        msg = obj.get("message", {}) if isinstance(obj, dict) else {}
        return {
            "content": str(msg.get("content", "")),
            "model_used": self.model,
            "tokens_in": int(obj.get("prompt_eval_count", 0) or 0),
            "tokens_out": int(obj.get("eval_count", 0) or 0),
            "cost_usd": 0.0,
        }


# Backward-compat alias so existing code (agents/, tests/) keeps working
LocalQwenBackend = OllamaBackend


# ---------------------------------------------------------------------------
# VLLMBackend  (on-demand, lab 4090)
# ---------------------------------------------------------------------------

class VLLMBackend(BaseBackend):
    """Connects to a vLLM server serving an OpenAI-compatible /v1 API.

    Disabled when VLLM_URL is empty.
    Health check: GET {base_url}/v1/models, timeout=2 s.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        _url = (base_url or os.getenv("VLLM_URL", "")).rstrip("/")
        self._base_url = _url
        _model = model or os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
        super().__init__(model=_model, **kwargs)

    def _probe_availability(self) -> bool:
        if not self._base_url:
            return False  # not configured → always unavailable
        try:
            r = requests.get(
                f"{self._base_url}/v1/models",
                timeout=self._AVAILABILITY_TIMEOUT,
            )
            return r.status_code == 200
        except Exception:
            return False

    def _call_provider(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        if self.request_fn is not None:
            return self.request_fn(messages=messages, model=self.model, **kwargs)
        if not self._base_url:
            raise BackendTransientError("VLLM_URL not configured")
        resp = requests.post(
            f"{self._base_url}/v1/chat/completions",
            json={"model": self.model, "messages": messages},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        obj = resp.json()
        choice = (obj.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "")
        usage = obj.get("usage", {})
        return {
            "content": str(text),
            "model_used": obj.get("model", self.model),
            "tokens_in": int(usage.get("prompt_tokens", 0)),
            "tokens_out": int(usage.get("completion_tokens", 0)),
            "cost_usd": 0.0,
        }


# ---------------------------------------------------------------------------
# GeminiBackend  (cloud free-tier fallback)
# ---------------------------------------------------------------------------

class GeminiBackend(BaseBackend):
    """Google Gemini via google-genai SDK.

    Availability check: GEMINI_API_KEY env must be non-empty.
    (No HTTP probe — free-tier rate limits make live probing expensive.)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"), **kwargs)

    def _probe_availability(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY", "").strip())

    def _call_provider(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        if self.request_fn is not None:
            return self.request_fn(messages=messages, model=self.model, **kwargs)
        try:
            from google import genai  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise BackendTransientError("google-genai not installed") from exc
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise BackendTransientError("missing GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
        prompt = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in messages)
        out = client.models.generate_content(model=self.model, contents=prompt)
        text = getattr(out, "text", "") or ""
        return {"content": str(text), "model_used": self.model, "cost_usd": 0.0}


# ---------------------------------------------------------------------------
# ClaudeBackend  (optional paid tier)
# ---------------------------------------------------------------------------

class ClaudeBackend(BaseBackend):
    """Anthropic Claude via anthropic SDK.

    Availability check: ANTHROPIC_API_KEY env must be non-empty.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(model=os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest"), **kwargs)

    def _probe_availability(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())

    def _call_provider(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        if self.request_fn is not None:
            return self.request_fn(messages=messages, model=self.model, **kwargs)
        try:
            import anthropic  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise BackendTransientError("anthropic SDK not installed") from exc
        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise BackendTransientError("missing ANTHROPIC_API_KEY")
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(model=self.model, max_tokens=1024, messages=messages)
        text = ""
        if getattr(msg, "content", None):
            part = msg.content[0]
            text = str(getattr(part, "text", "") or "")
        usage = getattr(msg, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        return {
            "content": text,
            "model_used": self.model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": 0.0,
        }


# ---------------------------------------------------------------------------
# LLMRouter  — fallback-chain orchestrator
# ---------------------------------------------------------------------------

class LLMRouter:
    """Route LLM calls through a priority-ordered chain of backends.

    Priority (local-first):
    - INTENT_PARSING   → Ollama → Gemini
    - CRITIQUE         → vLLM → Gemini → Ollama  (needs strong reasoning)
    - everything else  → Ollama → Gemini

    Any backend that is unavailable (``is_available()`` returns False) or
    has an open circuit-breaker is silently skipped.  Only raises when the
    entire chain is exhausted.
    """

    def __init__(
        self,
        *,
        ollama_backend: OllamaBackend | None = None,
        vllm_backend: VLLMBackend | None = None,
        gemini_backend: GeminiBackend | None = None,
        claude_backend: ClaudeBackend | None = None,
        # Backward-compat kwarg name used by existing callsites
        local_backend: OllamaBackend | None = None,
    ) -> None:
        # local_backend is a legacy alias for ollama_backend
        self.ollama_backend: OllamaBackend = (
            ollama_backend or local_backend or OllamaBackend()
        )
        self.vllm_backend: VLLMBackend = vllm_backend or VLLMBackend()
        self.gemini_backend: GeminiBackend = gemini_backend or GeminiBackend()
        self.claude_backend: ClaudeBackend = claude_backend or ClaudeBackend()

        # Expose legacy attribute name so existing code doesn't break
        self.local_backend = self.ollama_backend

    def _backend_chain(self, task: TaskType) -> list[BaseBackend]:
        """Return ordered list of backends to try for a given task."""
        if task == TaskType.INTENT_PARSING:
            # Simple classification — local 7B is sufficient; cloud is overkill
            return [self.ollama_backend, self.gemini_backend]

        if task == TaskType.CRITIQUE:
            # Needs strong reasoning: on-demand 4090 > cloud > local 7B fallback
            return [self.vllm_backend, self.gemini_backend, self.ollama_backend]

        # RETRIEVAL_REASONING, SYNTHESIS, EMBEDDING, … → local-first
        return [self.ollama_backend, self.gemini_backend]

    def complete(
        self,
        task: TaskType,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Try each backend in priority order; return the first successful response."""
        chain = self._backend_chain(task)
        attempted: list[str] = []
        last_error: Exception | None = None

        def _try_chain() -> LLMResponse:
            nonlocal last_error
            for backend in chain:
                if backend is None:
                    continue
                bname = backend.name
                if not backend.is_available():
                    attempted.append(f"{bname}:unavailable")
                    continue
                attempted.append(f"{bname}:trying")
                try:
                    resp = backend.complete(messages=messages, **kwargs)
                    attempted.append(f"{bname}:ok")
                    return resp
                except CircuitBreakerOpenError as e:
                    attempted[-1] = f"{bname}:circuit_open"
                    last_error = e
                except BackendTransientError as e:
                    attempted[-1] = f"{bname}:transient_error"
                    last_error = e
                except Exception as e:
                    attempted[-1] = f"{bname}:error"
                    last_error = e
            raise RuntimeError(
                f"All backends exhausted for task={task.value}. "
                f"Attempted: {attempted}. Last error: {last_error}"
            )

        def _finalize_otel(resp: LLMResponse, span_obj: Any | None) -> LLMResponse:
            if span_obj is not None:
                selected = next(
                    (a.split(":")[0] for a in reversed(attempted) if a.endswith(":ok")),
                    "unknown",
                )
                span_obj.set_attribute("llm.backend.attempted", str(attempted))
                span_obj.set_attribute("llm.backend.selected", selected)
                span_obj.set_attribute("llm.model", resp.model_used)
            record_llm_call(
                model=resp.model_used,
                tokens_in=resp.tokens_in,
                tokens_out=resp.tokens_out,
                cost=resp.cost_usd,
                latency=resp.latency_ms,
                completion_preview=resp.content or None,
            )
            return resp

        def _call_with_optional_langfuse() -> LLMResponse:
            """LLM backends (required); optionally nest a Langfuse *generation* for the attempt."""
            if not langfuse_tracing_active():
                return _try_chain()
            wall0 = time.perf_counter()
            model_hint = os.getenv("GEMINI_MODEL") or os.getenv("OLLAMA_MODEL") or "llm_router"
            with llm_generation_context(
                name=f"llm.{task.value}",
                model_hint=model_hint,
                input_messages=messages,
            ) as gen:
                try:
                    resp = _try_chain()
                except BaseException as exc:
                    generation_update_error(
                        gen,
                        exc,
                        latency_ms=int((time.perf_counter() - wall0) * 1000),
                    )
                    raise
                generation_update_success(
                    gen,
                    output_text=resp.content,
                    model_used=resp.model_used,
                    tokens_in=resp.tokens_in,
                    tokens_out=resp.tokens_out,
                    latency_ms=resp.latency_ms,
                    cost_usd=resp.cost_usd,
                    extra_metadata={"attempted_backends": str(attempted)},
                )
                return resp

        tracer = _get_tracer()
        span_name = f"llm.{task.value}"
        if tracer is None:
            return _finalize_otel(_call_with_optional_langfuse(), None)
        with tracer.start_as_current_span(span_name) as span:
            span.set_attribute("llm.task_type", task.value)
            try:
                return _finalize_otel(_call_with_optional_langfuse(), span)
            except Exception as exc:
                span.set_attribute("llm.backend.attempted", str(attempted))
                span.set_attribute("llm.error", str(exc))
                raise

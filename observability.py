"""
OpenTelemetry tracing helpers for the Travel Agent service.

Usage:
    from observability import init_tracing, traced, record_llm_call

    # Once at startup (e.g. in api.py or __main__):
    init_tracing("travel-agent")

    # Decorate any sync or async function to get a span:
    @traced
    async def my_handler(...): ...

    # Inside an LLM wrapper after a completion:
    record_llm_call(
        model="gemini-2.0-flash",
        tokens_in=42,
        tokens_out=17,
        cost=0.0,
        latency=320,
        completion_preview="assistant text …",
    )

Design notes:
- All OTEL imports are guarded so the module loads fine even without the SDK installed.
- `init_tracing` accepts an optional `exporter` for tests (InMemorySpanExporter).
- `traced` works for both sync and async callables.
- `record_llm_call` maps tokens/cost onto the span and can attach a truncated ``llm.completion_preview``.
"""

from __future__ import annotations

import base64
import functools
import inspect
import os
from typing import Any, Callable, TypeVar

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False

_tracer: Any = None
_provider: Any = None  # kept for shutdown() in tests


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_tracing(
    service_name: str = "travel-agent",
    *,
    exporter: Any | None = None,
) -> None:
    """Initialise the module-level TracerProvider.

    Parameters
    ----------
    service_name:
        Value set on the ``service.name`` resource attribute.
    exporter:
        Optional custom ``SpanExporter`` (useful for tests with
        ``InMemorySpanExporter``).  When *None* the function builds an
        ``OTLPSpanExporter`` pointing at the local Langfuse instance.

    Note: we deliberately avoid ``trace.set_tracer_provider()`` because
    the OTEL SDK only permits that call once per process.  Instead we store
    the provider in a module-level variable and obtain tracers from it
    directly, which also makes per-test isolation straightforward.
    """
    global _tracer, _provider

    if not _OTEL_AVAILABLE:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    if exporter is not None:
        _attach_exporter(provider, exporter)
    else:
        _attach_langfuse_exporter(provider)

    # Keep module-level refs — do NOT call trace.set_tracer_provider() so
    # tests can call init_tracing() multiple times without hitting the
    # "Overriding of current TracerProvider is not allowed" guard.
    _provider = provider
    _tracer = provider.get_tracer(service_name)


def record_llm_call(
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    latency: int,
    *,
    completion_preview: str | None = None,
    preview_max_chars: int = 12_288,
) -> None:
    """Attach LLM call attributes to whichever span is currently active.

    Safe to call when there is no active span — the call becomes a no-op.

    Parameters
    ----------
    completion_preview:
        Assistant text truncated to ``preview_max_chars`` and stored as
        ``llm.completion_preview`` for Langfuse / OTLP consumers.
    preview_max_chars:
        Caps attribute payload size for OTLP backends.
    """
    if not _OTEL_AVAILABLE:
        return
    span = _otel_trace.get_current_span()
    # ``NonRecordingSpan`` is returned when there is no active span; it is
    # not an instance of ``sdk.trace.Span``, so we check ``is_recording()``.
    if not span.is_recording():
        return
    span.set_attribute("llm.model", model)
    span.set_attribute("llm.tokens_in", tokens_in)
    span.set_attribute("llm.tokens_out", tokens_out)
    span.set_attribute("llm.cost_usd", cost)
    span.set_attribute("llm.latency_ms", latency)
    if completion_preview:
        cap = max(256, preview_max_chars)
        text = completion_preview.strip()
        if len(text) > cap:
            text = text[: cap - 1] + "…"
        span.set_attribute("llm.completion_preview", text)


F = TypeVar("F", bound=Callable[..., Any])


def traced(fn: F) -> F:
    """Decorator that wraps a sync *or* async function in an OTEL span.

    The span name is ``<qualname>`` of the decorated function.
    Falls back to a transparent pass-through when OTEL is unavailable or
    tracing has not been initialised.
    """
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
            t = _get_tracer()
            if t is None:
                return await fn(*args, **kwargs)
            with t.start_as_current_span(fn.__qualname__):
                return await fn(*args, **kwargs)

        return _async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        t = _get_tracer()
        if t is None:
            return fn(*args, **kwargs)
        with t.start_as_current_span(fn.__qualname__):
            return fn(*args, **kwargs)

    return _sync_wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _get_tracer() -> Any:
    """Return the module-level tracer.

    Returns ``None`` when OTEL is not installed or ``init_tracing`` has not
    yet been called, so callers can skip span creation gracefully.
    """
    global _tracer
    if _tracer is not None:
        return _tracer
    if _OTEL_AVAILABLE and _provider is not None:
        return _provider.get_tracer("travel-agent")
    return None


def _attach_exporter(provider: Any, exporter: Any) -> None:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    provider.add_span_processor(SimpleSpanProcessor(exporter))


def _attach_langfuse_exporter(provider: Any) -> None:
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:  # pragma: no cover
        return

    langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    if not public_key or not secret_key or not langfuse_host:
        return  # 沒設 key 就不建立 exporter
    credentials = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()

    otlp_exporter = OTLPSpanExporter(
        endpoint=f"{langfuse_host}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {credentials}"},
    )
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

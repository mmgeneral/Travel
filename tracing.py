"""
Centralized Langfuse client, root trace / stage spans, and LLM generation helpers.

- One root observation per agent run: ``name="agent_run"``, trace ``id`` aligned with
  ``agent_run_id`` (32-char hex or ``create_trace_id(seed=…)``).
- Stages ``retriever``, ``planner``, ``critic`` are child spans with input / output /
  latency / error recorded on the active Langfuse observation (no ``print``).

LLM calls route through :func:`llm_generation_context` (``as_type="generation"``).

Tracing is optional: without ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` installed
and set, all helpers no-op. OpenTelemetry helpers in ``observability`` remain for tests
and non-Langfuse setups.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

_client: Any = None
_client_failed: bool = False

# Active root ``agent_run`` Langfuse span (for optional output patch-up).
_current_agent_root: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "travel_current_agent_root_span", default=None
)


def ensure_langfuse_env() -> None:
    """Map legacy ``LANGFUSE_HOST`` → ``LANGFUSE_BASE_URL`` if the latter is unset."""
    base = (os.getenv("LANGFUSE_BASE_URL") or "").strip()
    host = (os.getenv("LANGFUSE_HOST") or "").strip().rstrip("/")
    if not base and host:
        os.environ["LANGFUSE_BASE_URL"] = host


def credentials_configured() -> bool:
    return bool(
        (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
        and (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    )


def _try_get_client() -> Any | None:
    global _client, _client_failed
    if _client_failed:
        return None
    if _client is not None:
        return _client
    if not credentials_configured():
        return None
    ensure_langfuse_env()
    try:
        from langfuse import get_client as _get_client  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - import-time
        logger.warning("Langfuse SDK import failed; tracing disabled: %s", exc)
        _client_failed = True
        return None
    try:
        _client = _get_client()
        return _client
    except Exception as exc:  # pragma: no cover
        logger.warning("Langfuse get_client() failed; tracing disabled: %s", exc)
        _client_failed = True
        return None


def langfuse_tracing_active() -> bool:
    """True when Langfuse client is initialised and usable for this process."""
    return _try_get_client() is not None


def init_langfuse() -> None:
    """Eagerly initialise the Langfuse singleton (safe to call multiple times)."""
    _try_get_client()


def flush() -> None:
    client = _try_get_client()
    if client is None:
        return
    try:
        if hasattr(client, "flush"):
            client.flush()
    except Exception:
        logger.exception("Langfuse flush failed")


def shutdown_langfuse() -> None:
    """Shutdown / flush Langfuse (FastAPI lifespan teardown, workers)."""
    client = _try_get_client()
    if client is None:
        return
    try:
        if hasattr(client, "shutdown"):
            client.shutdown()
            return
        if hasattr(client, "flush"):
            client.flush()
    except Exception:
        logger.exception("Langfuse shutdown failed")


def resolve_trace_id(agent_run_id: str) -> str:
    """Normalise external id to Langfuse OpenTelemetry trace id (32 lowercase hex)."""
    s = (agent_run_id or "").strip().lower()
    if len(s) == 32 and re.fullmatch(r"[0-9a-f]{32}", s):
        return s
    lf = _try_get_client()
    if lf is not None and hasattr(lf, "create_trace_id"):
        try:
            return str(lf.create_trace_id(seed=(agent_run_id or "travel")))
        except Exception:
            pass
    return uuid.uuid4().hex


@contextmanager
def agent_run(
    agent_run_id: str,
    *,
    trace_input: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    """Root span for a full agent invocation; trace id follows ``agent_run_id``."""
    lf = _try_get_client()
    if lf is None:
        yield None
        return

    trace_hex = resolve_trace_id(agent_run_id)
    try:
        cm = lf.start_as_current_observation(
            as_type="span",
            name="agent_run",
            trace_context={"trace_id": trace_hex},
            input=trace_input,
            metadata=metadata or {},
        )
    except Exception:
        logger.exception("Langfuse agent_run span start failed")
        yield None
        return

    token: contextvars.Token[Any | None] | None = None
    with cm as root:
        try:
            token = _current_agent_root.set(root)
            t0 = time.perf_counter()
            try:
                yield root
            except BaseException as exc:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                try:
                    root.update(
                        level="ERROR",
                        status_message=str(exc)[:8000],
                        metadata={"latency_ms": latency_ms, "error_type": type(exc).__name__},
                    )
                except Exception:
                    logger.debug("Langfuse root.update (error) failed", exc_info=True)
                raise
            else:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                try:
                    md: dict[str, Any] = dict(metadata or {})
                    md["latency_ms"] = latency_ms
                    root.update(metadata=md)
                except Exception:
                    logger.debug("Langfuse root metadata update failed", exc_info=True)
        finally:
            if token is not None:
                _current_agent_root.reset(token)


def update_agent_run_output(summary: Any | None) -> None:
    """Attach final output to the current ``agent_run`` root (call before leaving ``agent_run()``)."""
    root = _current_agent_root.get()
    if root is None or summary is None:
        return
    try:
        root.update(output=summary)
    except Exception:
        logger.debug("Langfuse agent_run output update failed", exc_info=True)


def _sanitize_state(stage: str, state: dict[str, Any]) -> dict[str, Any]:
    """Compact state excerpts for tracing (avoid megabyte blobs)."""
    base: dict[str, Any] = {
        "query": (state.get("query") or "")[:1200],
        "agent_run_id": state.get("agent_run_id"),
        "checkpoint_thread_id": state.get("checkpoint_thread_id"),
    }
    err = state.get("error")
    if err:
        base["error"] = err
    if stage == "retriever":
        hist = state.get("retrieval_history") or []
        pool = state.get("dynamic_shop_pool") or []
        base["retrieval_rounds"] = len(hist) if isinstance(hist, list) else 0
        base["dynamic_shop_pool_size"] = len(pool) if isinstance(pool, list) else 0
    elif stage == "planner":
        fi = state.get("final_itinerary")
        cards = state.get("ui_cards") or []
        base["final_itinerary_len"] = len(str(fi)) if fi is not None else 0
        base["ui_cards_count"] = len(cards) if isinstance(cards, list) else 0
        base["auditor_feedback_excerpt"] = str(state.get("auditor_feedback") or "")[:500]
    elif stage == "critic":
        ch = state.get("critique_history") or []
        base["critique_rounds"] = len(ch) if isinstance(ch, list) else 0
    return base


def _finalize_stage_span(span: Any, *, out: dict[str, Any], stage: str, t0: float) -> None:
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if span is None:
        return
    try:
        span.update(
            output=_sanitize_state(stage, out),
            metadata={"latency_ms": latency_ms},
        )
    except Exception:
        logger.debug("Langfuse stage span update failed", exc_info=True)


def _fail_stage_span(span: Any, exc: BaseException, t0: float) -> None:
    latency_ms = int((time.perf_counter() - t0) * 1000)
    if span is None:
        return
    try:
        span.update(
            level="ERROR",
            status_message=str(exc)[:8000],
            metadata={"latency_ms": latency_ms, "error_type": type(exc).__name__},
        )
    except Exception:
        logger.debug("Langfuse stage span error update failed", exc_info=True)


def trace_agent_stage(stage: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for graph nodes: records ``retriever`` / ``planner`` / ``critic`` spans."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                state = args[0] if args else None
                if not isinstance(state, dict):
                    return await fn(*args, **kwargs)
                lf = _try_get_client()
                if lf is None:
                    return await fn(*args, **kwargs)
                inp = _sanitize_state(stage, state)
                t0 = time.perf_counter()
                try:
                    cm = lf.start_as_current_observation(
                        as_type="span",
                        name=stage,
                        input=inp,
                    )
                except Exception:
                    return await fn(*args, **kwargs)
                with cm as span:
                    try:
                        out = await fn(*args, **kwargs)
                    except BaseException as exc:
                        _fail_stage_span(span, exc, t0)
                        raise
                    if isinstance(out, dict):
                        _finalize_stage_span(span, out=out, stage=stage, t0=t0)
                    return out

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def swrapper(*args: Any, **kwargs: Any) -> Any:
            state = args[0] if args else None
            if not isinstance(state, dict):
                return fn(*args, **kwargs)
            lf = _try_get_client()
            if lf is None:
                return fn(*args, **kwargs)
            inp = _sanitize_state(stage, state)
            t0 = time.perf_counter()
            try:
                cm = lf.start_as_current_observation(as_type="span", name=stage, input=inp)
            except Exception:
                return fn(*args, **kwargs)
            with cm as span:
                try:
                    out = fn(*args, **kwargs)
                except BaseException as exc:
                    _fail_stage_span(span, exc, t0)
                    raise
                if isinstance(out, dict):
                    _finalize_stage_span(span, out=out, stage=stage, t0=t0)
                return out

        return swrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def llm_generation_context(
    *,
    name: str,
    model_hint: str,
    input_messages: Iterable[dict[str, str]] | Any,
    model_parameters: dict[str, Any] | None = None,
):
    """Context manager for a Langfuse *generation* (nested under the current trace)."""
    lf = _try_get_client()
    if lf is None:
        yield None
        return
    try:
        cm = lf.start_as_current_observation(
            as_type="generation",
            name=name,
            model=model_hint,
            input=list(input_messages) if input_messages is not None else None,
            model_parameters=model_parameters,
        )
    except Exception:
        yield None
        return
    with cm as gen:
        yield gen


def generation_update_success(
    gen: Any,
    *,
    output_text: str,
    model_used: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: int,
    cost_usd: float,
    extra_metadata: dict[str, Any] | None = None,
) -> None:
    if gen is None:
        return
    try:
        meta: dict[str, Any] = {
            "latency_ms": int(latency_ms),
            "cost_usd": float(cost_usd),
        }
        if extra_metadata:
            meta.update(extra_metadata)
        gen.update(
            output=output_text,
            model=model_used,
            usage_details={
                "input": int(tokens_in),
                "output": int(tokens_out),
                "total": int(tokens_in + tokens_out),
            },
            metadata=meta,
        )
    except Exception:
        logger.debug("Langfuse generation.update failed", exc_info=True)


def generation_update_error(gen: Any, exc: BaseException, *, latency_ms: int) -> None:
    if gen is None:
        return
    try:
        gen.update(
            level="ERROR",
            status_message=str(exc)[:8000],
            metadata={"latency_ms": latency_ms, "error_type": type(exc).__name__},
        )
    except Exception:
        logger.debug("Langfuse generation error update failed", exc_info=True)

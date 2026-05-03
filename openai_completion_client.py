"""
Async HTTP adapter for OpenAI Chat Completions (transport only; no web framework).

Inject via ``AgentState["runtime_services"]["openai_chat_client"]`` to override defaults.
"""

from __future__ import annotations

import os
import time
from contextlib import nullcontext
from typing import Any

import httpx

from tracing import (
    generation_update_error,
    generation_update_success,
    langfuse_tracing_active,
    llm_generation_context,
)


class OpenAIChatCompletionClient:
    """Minimal ``/v1/chat/completions`` client for agent nodes."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        default_model: str | None = None,
        timeout_connect_s: float = 10.0,
        timeout_total_s: float = 45.0,
    ) -> None:
        self._api_key = (api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")).strip()
        self._default_model = (
            default_model if default_model is not None else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        ).strip()
        self._timeout = httpx.Timeout(timeout_total_s, connect=timeout_connect_s)

    def is_configured(self) -> bool:
        return bool(self._api_key)

    async def chat_completion_payload(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if not self._api_key:
            raise ValueError("OpenAI API key is not configured")
        use_model = (model or self._default_model).strip()
        wall0 = time.perf_counter()
        ctx = (
            llm_generation_context(
                name="openai.chat_completions",
                model_hint=use_model,
                input_messages=messages,
            )
            if langfuse_tracing_active()
            else nullcontext(None)
        )
        with ctx as gen:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self._api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": use_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                        },
                    )
                    resp.raise_for_status()
                    payload = resp.json()
            except BaseException as exc:
                if gen is not None:
                    generation_update_error(
                        gen,
                        exc,
                        latency_ms=int((time.perf_counter() - wall0) * 1000),
                    )
                raise

            usage = payload.get("usage") or {}
            if gen is not None:
                uit = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                uot = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                txt = ""
                try:
                    ch0 = (payload.get("choices") or [{}])[0]
                    txt = str((ch0.get("message") or {}).get("content") or "")[:32000]
                except Exception:
                    pass
                generation_update_success(
                    gen,
                    output_text=txt,
                    model_used=str(payload.get("model") or use_model),
                    tokens_in=uit,
                    tokens_out=uot,
                    latency_ms=int((time.perf_counter() - wall0) * 1000),
                    cost_usd=0.0,
                )
            return payload

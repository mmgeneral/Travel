"""LangGraph checkpoint I/O helpers shared by ``api.py`` and ``agent_orchestrator``."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _graph_supports_checkpointing(graph: object) -> bool:
    return hasattr(graph, "aget_state_history") or hasattr(graph, "get_state_history")


def _graph_has_checkpoint_state_reader(graph: object) -> bool:
    return hasattr(graph, "aget_state") or hasattr(graph, "get_state")


async def _graph_history_chronological(graph: object, thread_id: str) -> list[Any]:
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        if hasattr(graph, "aget_state_history"):
            hist: list[Any] = []
            async for snap in graph.aget_state_history(cfg):  # type: ignore[union-attr]
                hist.append(snap)
        else:
            hist = await asyncio.to_thread(lambda: list(graph.get_state_history(cfg)))  # type: ignore[union-attr]
    except Exception:
        logger.exception("aget_state_history/get_state_history failed thread_id=%s", thread_id)
        return []
    return list(reversed(hist))


async def _graph_resolve_checkpoint_snapshot(graph: Any, thread_id: str, ref: str) -> Any | None:
    ref_st = ref.strip()
    hist = await _graph_history_chronological(graph, thread_id)
    if ref_st.lower().startswith("cp_"):
        suffix = ref_st[3:]
        if suffix.isdigit():
            idx = int(suffix) - 1
            if 0 <= idx < len(hist):
                return hist[idx]
    for sn in hist:
        conf = (getattr(sn, "config", None) or {}) or {}
        cfg = conf.get("configurable") or {}
        cid = cfg.get("checkpoint_id")
        if str(cid) == ref_st:
            return sn
    return None


async def _graph_get_state(graph: Any, cfg: dict[str, Any]) -> Any:
    if hasattr(graph, "aget_state"):
        return await graph.aget_state(cfg)  # type: ignore[union-attr]
    return await asyncio.to_thread(graph.get_state, cfg)  # type: ignore[union-attr]


async def _graph_update_state(graph: Any, config: Any, values: dict[str, Any]) -> Any:
    if hasattr(graph, "aupdate_state"):
        return await graph.aupdate_state(config, values)  # type: ignore[union-attr]
    return await asyncio.to_thread(graph.update_state, config, values)  # type: ignore[union-attr]

"""LangGraph checkpoint I/O helpers shared by ``api.py`` and ``orchestrator.py``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import MutableMapping
from typing import Any

logger = logging.getLogger(__name__)


def _checkpoint_id_from_runnable_config(config: Any) -> str | None:
    if config is None or not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if isinstance(configurable, dict):
        raw = configurable.get("checkpoint_id")
        if raw is not None:
            return str(raw)
        for legacy_key in (
            "__pregel_checkpoint_id",
            "__pregel_resume_checkpoint_id",
        ):
            v = configurable.get(legacy_key)
            if v is not None:
                return str(v)
    return None


def _checkpoint_id_from_state_snapshot(snap: Any) -> str | None:
    """Return the checkpoint UUID string for ``snap`` returned by ``aget_state``."""
    if snap is None:
        return None
    cid = _checkpoint_id_from_runnable_config(getattr(snap, "config", None))
    if cid:
        return cid
    return None


def turn_checkpoints_trimmed_to_checkpoint_id(
    head_turn_checkpoints: Any, rewind_checkpoint_id: str
) -> list[str] | None:
    """Return a prefix of ``head_turn_checkpoints`` ending at ``rewind_checkpoint_id`` (inclusive).

    ``None`` means the rewind id is not found on the head list — caller should keep snapshot values.
    """
    rew = str(rewind_checkpoint_id or "").strip()
    if not rew:
        return None
    tcp = [str(x).strip() for x in (head_turn_checkpoints or []) if x is not None and str(x).strip()]
    if not tcp:
        return None
    for i, cid in enumerate(tcp):
        if cid == rew:
            return tcp[: i + 1]
    return None


def extend_turn_checkpoint_in_state(state_vals: MutableMapping[str, Any], config: Any) -> None:
    """Append ``configurable.checkpoint_id`` to ``turn_checkpoints`` if new (mutates ``state_vals``).

    Used by :func:`node_plan` after itinerary synthesis so each completed planning turn has one
    bookmark id (not every LangGraph node checkpoint). ``synthesizer`` does not call this.
    """
    cid = _checkpoint_id_from_runnable_config(config)
    if not cid:
        return
    raw_tcp = state_vals.get("turn_checkpoints") or []
    tcp = [str(x) for x in raw_tcp if x is not None]
    if tcp and tcp[-1] == cid:
        return
    tcp.append(cid)
    state_vals["turn_checkpoints"] = tcp


async def _graph_persist_turn_checkpoint(graph: Any, cfg: dict[str, Any]) -> None:
    """Append the graph head checkpoint id to ``state['turn_checkpoints']`` once per completed run."""
    if not _graph_has_checkpoint_state_reader(graph):
        return
    try:
        snap = await _graph_get_state(graph, cfg)
    except Exception:
        logger.exception("_graph_persist_turn_checkpoint get_state failed")
        return
    cid = _checkpoint_id_from_state_snapshot(snap)
    if not cid:
        logger.warning(
            "_graph_persist_turn_checkpoint: missing checkpoint_id on snapshot (thread cfg keys=%s)",
            list((getattr(snap, "config", None) or {}).get("configurable", {}).keys()),
        )
        return
    vals = dict(getattr(snap, "values", None) or {})
    raw_tcp = vals.get("turn_checkpoints") or []
    tcp = [str(x) for x in raw_tcp if x is not None]
    if tcp and tcp[-1] == cid:
        return
    tcp.append(cid)
    try:
        await _graph_update_state(graph, snap.config, {"turn_checkpoints": tcp})
    except Exception:
        logger.exception("_graph_persist_turn_checkpoint update_state failed")


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
    if ref_st.lower().startswith("cp_"):
        suffix = ref_st[3:]
        if suffix.isdigit():
            idx_turn = int(suffix) - 1
            turn_cp_ids: list[str] = []
            try:
                snap_cur = await _graph_get_state(graph, {"configurable": {"thread_id": thread_id}})
                vals_cur = dict(getattr(snap_cur, "values", None) or {})
                turn_cp_ids = [
                    str(x) for x in (vals_cur.get("turn_checkpoints") or []) if x is not None
                ]
            except Exception:
                logger.exception("resolve cp_*: failed to read turn_checkpoints thread_id=%s", thread_id)
                turn_cp_ids = []
            if turn_cp_ids:
                if 0 <= idx_turn < len(turn_cp_ids):
                    ref_st = str(turn_cp_ids[idx_turn])
                else:
                    return None
            else:
                hist_fb = await _graph_history_chronological(graph, thread_id)
                if 0 <= idx_turn < len(hist_fb):
                    return hist_fb[idx_turn]
                return None

    hist = await _graph_history_chronological(graph, thread_id)
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

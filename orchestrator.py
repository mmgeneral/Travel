"""
Agent orchestration: LangGraph ``astream_events`` (v2) and transport-neutral dict events.

SSE framing lives in ``api.py`` only.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import traceback
import uuid
from typing import Any, AsyncIterator

from agent import build_graph, make_initial_state
from tracing import agent_run, update_agent_run_output
from graph_checkpoint_utils import (
    _graph_get_state,
    _graph_resolve_checkpoint_snapshot,
    _graph_supports_checkpointing,
    _graph_update_state,
    turn_checkpoints_trimmed_to_checkpoint_id,
)

logger = logging.getLogger(__name__)

# Nodes registered on the main travel graph (``agent.build_graph``).
_KNOWN_GRAPH_NODES: frozenset[str] = frozenset(
    {
        "route_intent",
        "clarify_constraint",
        "retriever",
        "researcher",
        "critic",
        "synthesizer",
        "collect_feedback",
        "plan",
    }
)

_OUTPUT_LOG_MAX_CHARS = 24_000


def _truncate_log(s: str, max_chars: int = _OUTPUT_LOG_MAX_CHARS) -> str:
    return s if len(s) <= max_chars else f"{s[: max_chars - 3]}..."


def _serialize_output_for_log(obj: Any) -> str:
    try:
        serialized = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        serialized = repr(obj)
    return _truncate_log(serialized)


def _final_itinerary_excerpt(output: Any) -> str | None:
    """If ``final_itinerary`` appears anywhere in ``output``, return a short excerpt."""
    if isinstance(output, dict):
        raw = output.get("final_itinerary")
        if raw is not None:
            hit = "" if raw is None else str(raw)
            hit = hit.strip()
            return hit[:320] + ("..." if len(hit) > 320 else "")
        for v in output.values():
            ex = _final_itinerary_excerpt(v)
            if ex is not None:
                return ex
    if isinstance(output, (list, tuple)):
        for v in output:
            ex = _final_itinerary_excerpt(v)
            if ex is not None:
                return ex
    return None


async def _probe_checkpoint_sqlite(
    *,
    checkpointer: Any | None,
    checkpoint_conn: Any | None,
) -> str | None:
    """Return a user-visible error message if SQLite is not usable before streaming."""
    if checkpointer is None:
        return None
    conn = checkpoint_conn if checkpoint_conn is not None else getattr(checkpointer, "conn", None)
    if conn is None:
        logger.warning("Checkpointer configured but SQLite connection probe skipped (no conn handle)")
        return None
    try:
        await conn.execute("SELECT 1")
        return None
    except Exception:
        logger.exception("AsyncSqliteSaver connection probe failed before graph stream")
        return "資料庫檢查失敗，請稍後重試或聯繫管理人員。"


def _merge_continuation_invoke_state(
    prev: dict[str, Any],
    *,
    query: str,
    agent_run_id: str,
    checkpoint_thread_id: str,
    dietary_profile: dict[str, Any] | None,
    advanced_mode: bool,
    user_locale: str | None,
    user_lat: float | None,
    user_lng: float | None,
) -> dict[str, Any]:
    """Fresh planning turn on an existing thread — keep learned weights & pool, reset loop fields."""
    dietary = dietary_profile if dietary_profile is not None else (prev.get("dietary_profile") or None)
    ctid = (checkpoint_thread_id or "").strip()
    merged = make_initial_state(
        query=query,
        dietary_profile=dietary if isinstance(dietary, dict) else None,
        agent_run_id=agent_run_id,
        advanced_mode=advanced_mode,
        user_locale=(user_locale if user_locale is not None else (prev.get("user_locale") or None)),
        user_lat=user_lat if user_lat is not None else prev.get("user_lat"),
        user_lng=user_lng if user_lng is not None else prev.get("user_lng"),
        checkpoint_thread_id=ctid,
    )
    for key in ("learned_weight_profile", "feedback_updates", "dynamic_shop_pool"):
        v = prev.get(key)
        if v:
            merged[key] = v
    rs = prev.get("runtime_services")
    if isinstance(rs, dict):
        merged["runtime_services"] = copy.deepcopy(rs)
    dr = prev.get("dietary_clarification_resolved")
    if isinstance(dr, dict) and dr:
        merged["dietary_clarification_resolved"] = copy.deepcopy(dr)
    pend = prev.get("pending_dietary_clarification")
    if isinstance(pend, dict) and pend:
        merged["pending_dietary_clarification"] = copy.deepcopy(pend)
    if prev.get("awaiting_dietary_clarification"):
        merged["awaiting_dietary_clarification"] = True
    ih = list(prev.get("intent_history") or [])
    pi = prev.get("intent")
    if isinstance(pi, dict) and pi:
        ih.append(copy.deepcopy(pi))
    merged["intent_history"] = ih
    merged["intent"] = None
    merged["researcher_candidate_names"] = []
    merged["researcher_notes"] = ""
    merged["auditor_feedback"] = ""
    merged["auditor_rejected"] = False
    merged["research_iteration"] = 0
    merged["retrieval_history"] = []
    merged["critique_history"] = []
    prev_itinerary = prev.get("final_itinerary") or ""
    if prev_itinerary:
        merged["prev_itinerary"] = prev_itinerary
    tcp_prev = prev.get("turn_checkpoints")
    merged["turn_checkpoints"] = [str(x) for x in tcp_prev] if isinstance(tcp_prev, list) else []
    return merged


def _transport_error_validation(thread_id: str, code: str, user_message: str) -> dict[str, Any]:
    return {
        "event": "error",
        "data": {"error": code, "thread_id": thread_id, "user_message": user_message},
    }


def _transport_error_stream(
    thread_id: str,
    exc: BaseException,
    *,
    hint: str | None = None,
) -> dict[str, Any]:
    """Error envelope for the frontend (includes message + optional traceback detail)."""
    tb = "".join(traceback.format_exception(exc)).strip()[-24000:]
    payload: dict[str, Any] = {
        "event": "error",
        "message": str(exc),
        "thread_id": thread_id,
        "exc_type": type(exc).__name__,
        "detail": tb,
    }
    if hint:
        payload["hint"] = hint
    return {"event": "error", "data": payload}


def _event_for_producer_failure(thread_id: str, exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, asyncio.InvalidStateError):
        return _transport_error_stream(thread_id, exc, hint="checkpoint_transport_error")
    if isinstance(exc, (OSError, ConnectionError)):
        return _transport_error_stream(thread_id, exc, hint="checkpoint_db_error")
    return _transport_error_stream(thread_id, exc)


def _state_excerpt(node_state: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in node_state.items()
        if k
        in (
            "research_log",
            "transit_audit",
            "rollback_occurred",
            "saga_snapshot_idx",
            "agent_run_id",
            "error",
        )
    }


def _coerce_agent_state_fragment(output: Any) -> dict[str, Any] | None:
    """Extract an AgentState-like dict from LangGraph Runnable / chain output."""
    if not isinstance(output, dict):
        return None
    if isinstance(output.get("research_log"), list) and "query" in output:
        return output
    for nk in _KNOWN_GRAPH_NODES:
        if nk in output and isinstance(output[nk], dict):
            cand: dict[str, Any] = output[nk]
            if isinstance(cand.get("research_log"), list) or "query" in cand:
                return cand
    return None


class AgentOrchestrator:
    """Runs the compiled LangGraph via ``astream_events(version='v2')``; yields transport dicts."""

    graph: Any

    async def stream_events(
        self,
        *,
        query: str,
        day: int,
        thread_id: str | None,
        rewind_to_checkpoint: str | None = None,
        dietary_profile: dict | None = None,
        advanced_mode: bool = False,
        user_locale: str | None = None,
        user_lat: float | None = None,
        user_lng: float | None = None,
        checkpointer: Any | None = None,
        checkpoint_conn: Any | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        requested_tid = (thread_id or "").strip()
        tid = requested_tid or uuid.uuid4().hex
        try:
            async for ev in self._stream_events_guarded(
                tid=tid,
                requested_tid=requested_tid,
                query=query,
                day=day,
                rewind_to_checkpoint=rewind_to_checkpoint,
                dietary_profile=dietary_profile,
                advanced_mode=advanced_mode,
                user_locale=user_locale,
                user_lat=user_lat,
                user_lng=user_lng,
                checkpointer=checkpointer,
                checkpoint_conn=checkpoint_conn,
            ):
                yield ev
        except Exception as exc:
            logger.exception("AgentOrchestrator.stream_events fatal thread=%s", tid)
            yield _transport_error_stream(tid, exc)

    async def _stream_events_guarded(
        self,
        *,
        tid: str,
        requested_tid: str,
        query: str,
        day: int,
        rewind_to_checkpoint: str | None,
        dietary_profile: dict | None,
        advanced_mode: bool,
        user_locale: str | None,
        user_lat: float | None,
        user_lng: float | None,
        checkpointer: Any | None,
        checkpoint_conn: Any | None,
    ) -> AsyncIterator[dict[str, Any]]:
        self.graph = build_graph(checkpointer=checkpointer)
        graph = self.graph
        graph_cfg: dict[str, Any] = {"configurable": {"thread_id": tid}}
        chk_ok = _graph_supports_checkpointing(graph)

        stream_aborted = False
        stream_cfg: Any = graph_cfg
        stream_input: dict[str, Any] | None = None

        if rewind_to_checkpoint and not rewind_to_checkpoint.strip():
            yield _transport_error_validation(tid, "invalid_rewind", "rewind_to_checkpoint 不可為空白")
            stream_aborted = True
        elif rewind_to_checkpoint and not (requested_tid or "").strip():
            yield _transport_error_validation(
                tid,
                "rewind_requires_thread_id",
                "time-travel 需要已有 thread_id，請先發起對話並帶同一個 thread_id",
            )
            stream_aborted = True

        if not stream_aborted:
            if rewind_to_checkpoint:
                if not chk_ok:
                    yield _transport_error_validation(
                        tid,
                        "checkpointing_disabled",
                        "此環境無 LangGraph SqliteSaver，無法 rewind",
                    )
                    stream_aborted = True
                else:
                    snap = await _graph_resolve_checkpoint_snapshot(graph, tid, rewind_to_checkpoint)
                    if snap is None:
                        yield _transport_error_validation(
                            tid,
                            "checkpoint_not_found",
                            f"找不到 checkpoint：`{rewind_to_checkpoint.strip()}`",
                        )
                        stream_aborted = True
                    else:
                        fork_base = dict(snap.values or {})
                        fork_base["query"] = query
                        fork_base["agent_run_id"] = tid
                        fork_base.setdefault("final_itinerary", "")
                        fork_base.setdefault("checkpoint_thread_id", requested_tid or tid)
                        fork_base.setdefault("ui_cards", [])
                        if dietary_profile is not None:
                            fork_base["dietary_profile"] = dietary_profile
                        fork_base["advanced_mode"] = advanced_mode
                        if user_locale is not None:
                            fork_base["user_locale"] = user_locale or ""
                        rew_id = rewind_to_checkpoint.strip()
                        try:
                            head_snap = await _graph_get_state(graph, graph_cfg)
                            hv = dict(getattr(head_snap, "values", None) or {})
                            trimmed_tcp = turn_checkpoints_trimmed_to_checkpoint_id(
                                hv.get("turn_checkpoints"), rew_id
                            )
                            if trimmed_tcp is not None:
                                fork_base["turn_checkpoints"] = trimmed_tcp
                        except Exception:
                            logger.exception(
                                "rewind prune turn_checkpoints failed thread=%s (non-fatal)", tid
                            )
                        try:
                            stream_cfg = await _graph_update_state(graph, snap.config, fork_base)
                            stream_input = None
                        except Exception as exc:
                            logger.exception("update_state rewind fork failed thread=%s", tid)
                            yield _transport_error_validation(tid, "fork_failed", f"time-travel fork 失敗：{exc!s}")
                            stream_aborted = True
            elif chk_ok and requested_tid:
                snap = await _graph_get_state(graph, graph_cfg)
                prev_vals: dict[str, Any] | None = None
                if snap is not None and getattr(snap, "values", None):
                    raw = snap.values
                    prev_vals = dict(raw) if isinstance(raw, dict) else None
                if prev_vals:
                    stream_input = _merge_continuation_invoke_state(
                        prev_vals,
                        query=query,
                        agent_run_id=tid,
                        checkpoint_thread_id=tid,
                        dietary_profile=dietary_profile,  # type: ignore[arg-type]
                        advanced_mode=advanced_mode,
                        user_locale=user_locale,
                        user_lat=user_lat,
                        user_lng=user_lng,
                    )
                else:
                    stream_input = make_initial_state(
                        query=query,
                        dietary_profile=dietary_profile,
                        agent_run_id=tid,
                        advanced_mode=advanced_mode,
                        user_locale=user_locale,
                        user_lat=user_lat,
                        user_lng=user_lng,
                        checkpoint_thread_id=requested_tid,
                    )
            else:
                stream_input = make_initial_state(
                    query=query,
                    dietary_profile=dietary_profile,
                    agent_run_id=tid,
                    advanced_mode=advanced_mode,
                    user_locale=user_locale,
                    user_lat=user_lat,
                    user_lng=user_lng,
                    checkpoint_thread_id=requested_tid or None,
                )

        start_body = {
            "day": day,
            "query": query,
            "agent_run_id": tid,
            "thread_id": tid,
            "rewind_to_checkpoint": rewind_to_checkpoint.strip() if rewind_to_checkpoint else None,
        }

        stream_failed = False
        final_state: dict[str, Any] | None = None

        if not stream_aborted:
            db_probe = await _probe_checkpoint_sqlite(
                checkpointer=checkpointer,
                checkpoint_conn=checkpoint_conn,
            )
            if db_probe is not None:
                yield _transport_error_validation(tid, "checkpoint_db_unavailable", db_probe)
                stream_aborted = True

        emit_success_done = False
        if not stream_aborted:
            yield {"event": "start", "data": start_body}

            trace_input = {
                "query": query,
                "day": day,
                "thread_id": tid,
                "advanced_mode": advanced_mode,
                "rewind_to_checkpoint": rewind_to_checkpoint.strip() if rewind_to_checkpoint else None,
            }
            with agent_run(tid, trace_input=trace_input, metadata={"source": "AgentOrchestrator"}):
                astream_events_fn = getattr(graph, "astream_events", None)
                if astream_events_fn is None:
                    err = RuntimeError("Compiled graph has no astream_events; upgrade langgraph")
                    logger.error("%s thread=%s", err, tid)
                    yield _transport_error_stream(tid, err)
                    stream_failed = True
                else:
                    try:
                        inp = stream_input if stream_input is not None else {}
                        agen = astream_events_fn(inp, stream_cfg, version="v2")
                        it = agen.__aiter__()
                        final_itinerary_seen_in_stream = False
                        while True:
                            try:
                                lg_ev = await asyncio.wait_for(it.__anext__(), timeout=15.0)
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                yield {"event": "ping", "data": {}}
                                continue

                            ev_type = str(lg_ev.get("event") or "")
                            meta = lg_ev.get("metadata") or {}
                            data = lg_ev.get("data") or {}
                            node_name = meta.get("langgraph_node") or lg_ev.get("name") or ""
                            out = data.get("output")
                            if out is None and isinstance(data.get("chunk"), dict):
                                out = data.get("chunk")

                            if ev_type == "on_chain_end":
                                fi_excerpt = _final_itinerary_excerpt(out) if out is not None else None
                                if fi_excerpt is not None:
                                    final_itinerary_seen_in_stream = True
                                    logger.info(
                                        "astream_events FINAL_ITINERARY_SEEN thread=%s langgraph_node=%s excerpt=%r",
                                        tid,
                                        node_name or "?",
                                        fi_excerpt[:200],
                                    )

                                if node_name in _KNOWN_GRAPH_NODES:
                                    payload = _serialize_output_for_log(out) if out is not None else "(no output)"
                                    fi_len = 0
                                    if isinstance(out, dict) and "final_itinerary" in out:
                                        fi_v = out.get("final_itinerary")
                                        fi_len = len(str(fi_v)) if fi_v is not None else 0
                                    logger.info(
                                        "astream_events NODE_OUTPUT thread=%s node=%s final_itinerary_len=%s output=%s",
                                        tid,
                                        node_name,
                                        fi_len if fi_len > 0 else (len(fi_excerpt or "") if fi_excerpt else 0),
                                        payload,
                                    )

                            if ev_type == "on_chain_end" and node_name in _KNOWN_GRAPH_NODES:
                                frag = _coerce_agent_state_fragment(out)
                                if frag is not None:
                                    final_state = frag
                                    yield {
                                        "event": "node",
                                        "data": {
                                            "node": node_name,
                                            "agent_run_id": tid,
                                            "thread_id": tid,
                                            "state": _state_excerpt(frag),
                                        },
                                    }
                                    if (
                                        node_name == "clarify_constraint"
                                        and isinstance(frag.get("clarification_broadcast"), dict)
                                        and frag["clarification_broadcast"].get("question")
                                    ):
                                        yield {
                                            "event": "clarification",
                                            "data": dict(frag["clarification_broadcast"]),
                                        }
                                    await asyncio.sleep(0)

                        logger.info(
                            "astream_events SUMMARY thread=%s final_itinerary_seen_any_moment=%s",
                            tid,
                            final_itinerary_seen_in_stream,
                        )
                    except Exception as exc:
                        logger.error(
                            "AgentOrchestrator astream_events failed thread=%s err=%s",
                            tid,
                            type(exc).__name__,
                            exc_info=True,
                        )
                        yield _transport_error_stream(tid, exc)
                        stream_failed = True

                emit_success_done = not stream_failed

                if emit_success_done:
                    # Turn checkpoints are appended only by ``node_plan`` (see ``extend_turn_checkpoint_in_state``).
                    if stream_input is not None:
                        fallback_initial = stream_input
                    else:
                        fallback_initial = make_initial_state(
                            query=query,
                            dietary_profile=dietary_profile,
                            agent_run_id=tid,
                            advanced_mode=advanced_mode,
                            user_locale=user_locale,
                            user_lat=user_lat,
                            user_lng=user_lng,
                            checkpoint_thread_id=requested_tid or None,
                        )
                    resolved_final = final_state if isinstance(final_state, dict) else fallback_initial

                    done_body = {
                        "itinerary": resolved_final.get("final_itinerary", ""),
                        "ui_cards": resolved_final.get("ui_cards", []),
                        "snapshot_idx": resolved_final.get("saga_snapshot_idx", -1),
                        "outcome": "SUCCESS",
                        "semantic_status": "RUN_COMPLETED",
                        "agent_run_id": tid,
                        "thread_id": tid,
                    }
                    if resolved_final.get("awaiting_dietary_clarification"):
                        done_body["awaiting_clarification"] = True
                    err = resolved_final.get("error")
                    if err:
                        done_body["error"] = err
                        done_body["outcome"] = "FAILED"
                        done_body["semantic_status"] = "RUN_AGENT_ERROR"
                    update_agent_run_output(
                        {
                            "outcome": done_body["outcome"],
                            "semantic_status": done_body["semantic_status"],
                            "final_itinerary_len": len(str(done_body.get("itinerary") or "")),
                            "ui_cards_count": len(done_body.get("ui_cards") or []),
                        }
                    )
                    yield {"event": "done", "data": done_body}

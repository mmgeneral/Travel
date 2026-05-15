"""
FastAPI wrapper for agent.py — exposes the LangGraph agent over HTTP/SSE.

Graph scheduling and normalized event payloads are implemented in ``orchestrator.py``
(``AgentOrchestrator``); this module only frames those dicts as SSE for the HTTP response.

Why SSE instead of WebSocket?
  - One-way streaming (agent → browser) matches our use case
  - Works through any HTTP proxy without special config
  - Browser's EventSource API is built-in, no library needed

Routes:
  GET  /healthz       — liveness probe for Docker / browser offline detection
  POST /agent/query   — fire a query, receive SSE stream of agent events (thread_id / rewind optional)
  GET  /agent/history/{thread_id} — list checkpoints for time-travel (requires Sqlite checkpointer)
  POST /agent/rollback — rollback to a snapshot_idx
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import os
import httpx
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from observability import init_tracing, traced

init_tracing("travel-agent")

from tracing import ensure_langfuse_env, init_langfuse, shutdown_langfuse

from agent import build_graph, make_initial_state, node_synthesizer
from orchestrator import AgentOrchestrator
from agents.synthesizer import SynthesisReport
from graph_checkpoint_utils import (
    _graph_get_state,
    _graph_resolve_checkpoint_snapshot,
    _graph_has_checkpoint_state_reader,
    _graph_supports_checkpointing,
    _graph_update_state,
)
from saga import SagaEngine
from duffel import DuffelService
import h3
import redis.asyncio as aioredis
from acl import ActionOutcome, SagaActionResult

# Align with agent._SAGA_DIR (graph SQLite lives here, not saga_holds).
_GRAPH_SAGA_DIR = Path(os.getenv("SAGA_PERSIST_DIR", str(Path.home() / ".travel_agent" / "saga")))
_GRAPH_SAGA_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

_ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if x.strip()]
_API_TOKEN = os.getenv("TRAVEL_AGENT_API_TOKEN", "")
_DEV_MODE = os.getenv("DEV_MODE", "0") == "1"
_ENV = os.getenv("ENV", "development").lower()
_APP_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Taipei"))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_langfuse_env()
    init_langfuse()
    try:
        # Shared async HTTP client avoids per-request TLS handshakes; never block with sync requests.*.
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as http_client:
            app.state.http_client = http_client
            try:
                redis_client = aioredis.Redis(host="redis", port=6379, decode_responses=True)
                app.state.redis_client = redis_client
            except Exception:
                logger.warning("Redis unavailable – falling back to direct GraphHopper", exc_info=True)
                app.state.redis_client = None
            app.state.checkpoint_conn = None
            async with aiosqlite.connect(str(_GRAPH_SAGA_DIR / "graph_checkpoints.sqlite")) as conn:
                app.state.checkpoint_conn = conn
                app.state.checkpointer = AsyncSqliteSaver(conn)
                yield
            app.state.checkpoint_conn = None
            app.state.checkpointer = None
            if app.state.redis_client is not None:
                await app.state.redis_client.close()
            app.state.redis_client = None
            app.state.http_client = None
    finally:
        shutdown_langfuse()


app = FastAPI(title="Travel Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


def _now_iso() -> str:
    return datetime.now(_APP_TZ).isoformat()


def _require_api_token(x_api_token: str) -> None:
    # Production is always authenticated even if DEV_MODE is accidentally set.
    if _DEV_MODE and _ENV != "production":
        return
    if not _API_TOKEN:
        raise HTTPException(status_code=503, detail="Server auth not configured")
    if not hmac.compare_digest(x_api_token, _API_TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _snapshot_ts_iso(snapshot: Any) -> str:
    chk = getattr(snapshot, "checkpoint", None)
    if isinstance(chk, dict):
        ts = chk.get("ts")
        if ts is not None:
            try:
                tnum = float(ts)
                secs = tnum / 1e9 if tnum > 10**12 else tnum
                return (
                    datetime.fromtimestamp(secs, tz=timezone.utc)
                    .astimezone(_APP_TZ)
                    .isoformat(timespec="seconds")
                )
            except Exception:
                pass
    return _now_iso()




def _turn_checkpoint_summary(vals: dict[str, Any]) -> str:
    """One-line recap for undo / timeline UI (city, user query excerpt, itinerary preview)."""
    q = str(vals.get("query") or "").strip().replace("\n", " ")
    if len(q) > 120:
        q = q[:117] + "..."
    intent = vals.get("intent")
    city = ""
    if isinstance(intent, dict):
        city = str(intent.get("city") or "").strip()
    excerpt = ""
    fi = vals.get("final_itinerary")
    if fi is not None:
        excerpt = str(fi).strip().replace("\n", " ")
        if len(excerpt) > 80:
            excerpt = excerpt[:77] + "..."
    bits: list[str] = []
    if city and q:
        bits.append(f"{city}: {q}")
    elif q:
        bits.append(q)
    elif city:
        bits.append(city)
    if excerpt:
        bits.append(excerpt)
    return " · ".join(bits) if bits else ""


def _transport_event_to_sse(msg: dict[str, Any]) -> str:
    """Map orchestrator JSON to a single SSE frame (transport layer only)."""
    ev = str(msg.get("event") or "message")
    if ev == "ping":
        return ": heartbeat\n\n"
    payload = msg.get("data")
    if payload is None:
        payload = {}
    return f"event: {ev}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    day: int = 1
    thread_id: str | None = None
    rewind_to_checkpoint: str | None = None
    dietary_profile: dict | None = None
    advanced_mode: bool = False
    user_locale: str | None = None
    user_lat: float | None = None
    user_lng: float | None = None


QueryRequest = AgentQueryRequest  # backwards-compatible alias


class RollbackRequest(BaseModel):
    agent_run_id: str
    snapshot_idx: int


class DietaryProfileRequest(BaseModel):
    ethics: str = "omnivore"
    allergens: list[str] = []
    religious: str = "none"
    medical: list[str] = []


class TravelTimeRequest(BaseModel):
    start: list[float] = Field(..., min_items=2, max_items=2)
    end: list[float] = Field(..., min_items=2, max_items=2)
    mode: str = Field(default="car")


async def _do_tte(req: TravelTimeRequest, profile: str, client: httpx.AsyncClient, redis_client: aioredis.Redis | None = None) -> dict:
    start_h3 = h3.geo_to_cell(req.start[0], req.start[1], 9)
    end_h3 = h3.geo_to_cell(req.end[0], req.end[1], 9)
    cache_key = f"TTE:{profile}:{start_h3}:{end_h3}"

    # Try cache
    if redis_client is not None:
        try:
            cached = await redis_client.get(cache_key)
        except Exception:
            cached = None
        if cached is not None:
            result = json.loads(cached)
            result["cache"] = "hit"
            return result

    base_url = os.getenv("GRAPHHOPPER_BASE_URL", "http://localhost:8989")
    params = {
        "point": [f"{req.start[0]},{req.start[1]}", f"{req.end[0]},{req.end[1]}"],
        "profile": profile,
    }
    try:
        resp = await client.get(f"{base_url}/route", params=params)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"GraphHopper unreachable: {exc}")
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"GraphHopper returned {resp.status_code}: {resp.text}",
        )
    data = resp.json()
    paths = data.get("paths")
    if not paths:
        raise HTTPException(status_code=404, detail="No route found")

    path = paths[0]
    time_ms = path.get("time")
    distance_m = path.get("distance")
    if time_ms is None or distance_m is None:
        raise HTTPException(
            status_code=502, detail="Missing time/distance in GraphHopper response"
        )

    estimated_seconds = int(time_ms / 1000)
    distance_meters = int(distance_m)
    if distance_meters < 5000:
        confidence = "high"
    elif distance_meters < 20000:
        confidence = "medium"
    else:
        confidence = "low"
    result = {
        "estimated_seconds": estimated_seconds,
        "distance_meters": distance_meters,
        "confidence": confidence,
        "cache": "miss",
    }

    # Store in cache
    if redis_client is not None:
        try:
            await redis_client.setex(cache_key, 3600, json.dumps(result))
        except Exception:
            logger.warning("Failed to store TTE in Redis", exc_info=True)

    return result


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(request: Request) -> dict:
    checks: dict[str, str] = {}
    try:
        _SAGA_DIR.mkdir(parents=True, exist_ok=True)
        probe = _SAGA_DIR / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks["saga_dir_writable"] = "ok"
    except Exception as exc:
        checks["saga_dir_writable"] = f"error:{exc}"
    checks["indexes_loaded"] = "skipped"
    try:
        token = os.getenv("DUFFEL_ACCESS_TOKEN", "")
        if not token:
            checks["duffel_connectivity"] = "skipped:no_token"
        else:
            client = getattr(request.app.state, "http_client", None)
            headers = {"Authorization": f"Bearer {token}", "Duffel-Version": "v2", "Accept": "application/json"}
            if client is not None:
                resp = await client.get("https://api.duffel.com/air/airports?limit=1", headers=headers)
            else:
                async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as c:
                    resp = await c.get("https://api.duffel.com/air/airports?limit=1", headers=headers)
            checks["duffel_connectivity"] = "ok" if resp.status_code < 500 else f"error:{resp.status_code}"
    except Exception as exc:
        checks["duffel_connectivity"] = f"error:{exc.__class__.__name__}"
    ready = checks["saga_dir_writable"] == "ok"
    return {"status": "ready" if ready else "degraded", "checks": checks}


@app.post("/api/v1/tte")
async def travel_time_estimate(
    req: TravelTimeRequest,
    request: Request,
    x_api_token: str = Header(default=""),
) -> dict:
    _require_api_token(x_api_token)

    # 1. mode validation
    valid_modes = {"car", "bike", "foot"}
    if req.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{req.mode}'. Allowed: {', '.join(sorted(valid_modes))}"
        )

    # 2. coordinate range validation (Hsinchu bounding box)
    lat_min, lat_max = 24.5, 25.0
    lon_min, lon_max = 120.7, 121.2
    for name, coord in [("start", req.start), ("end", req.end)]:
        lat, lon = coord[0], coord[1]
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{name} coordinate ({lat}, {lon}) is outside the allowed"
                    f" Hsinchu bounding box:"
                    f" lat {lat_min}..{lat_max}, lon {lon_min}..{lon_max}"
                )
            )

    # 3. profile mapping (safe after validation)
    mode_map = {"car": "car", "bike": "bike", "foot": "foot"}
    profile = mode_map[req.mode]

    client = getattr(request.app.state, "http_client", None)
    redis_client = getattr(request.app.state, "redis_client", None)
    if client is None:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as temp_client:
            return await _do_tte(req, profile, temp_client, redis_client=redis_client)
    return await _do_tte(req, profile, client, redis_client=redis_client)


async def _stream_agent(
    *,
    query: str,
    day: int,
    user_id: str,
    thread_id: str | None,
    rewind_to_checkpoint: str | None = None,
    dietary_profile: dict | None = None,
    advanced_mode: bool = False,
    user_locale: str | None = None,
    user_lat: float | None = None,
    user_lng: float | None = None,
    checkpointer: Any | None = None,
    checkpoint_conn: Any | None = None,
) -> AsyncIterator[str]:
    """Delegate LangGraph execution to ``AgentOrchestrator``; emit SSE only here."""
    orch = AgentOrchestrator()
    async for msg in orch.stream_events(
        query=query,
        day=day,
        thread_id=thread_id,
        rewind_to_checkpoint=rewind_to_checkpoint,
        dietary_profile=dietary_profile,
        advanced_mode=advanced_mode,
        user_locale=user_locale,
        user_lat=user_lat,
        user_lng=user_lng,
        checkpointer=checkpointer,
        checkpoint_conn=checkpoint_conn,
    ):
        if msg.get("event") == "done":
            data = msg.get("data") or {}
            _save_itinerary(
                user_id=user_id,
                day=day,
                itinerary_text=str(data.get("itinerary") or ""),
                semantic_status=str(data.get("semantic_status") or "RUN_COMPLETED"),
                ui_cards_json=json.dumps(data.get("ui_cards") or [], ensure_ascii=False),
            )
        yield _transport_event_to_sse(msg)


@app.post("/agent/query")
@traced
async def agent_query(
    req: AgentQueryRequest,
    request: Request,
    x_user_id: str = Header(default=""),
    x_api_token: str = Header(default=""),
) -> StreamingResponse:
    """Stream agent SSE. ``thread_id`` is stored in LangGraph ``configurable`` for checkpoint I/O."""
    _require_api_token(x_api_token)
    user_id = x_user_id.strip() or "anonymous"
    cp = getattr(request.app.state, "checkpointer", None)
    cconn = getattr(request.app.state, "checkpoint_conn", None)
    return StreamingResponse(
        _stream_agent(
            query=req.query,
            day=req.day,
            user_id=user_id,
            thread_id=req.thread_id,
            rewind_to_checkpoint=req.rewind_to_checkpoint,
            dietary_profile=req.dietary_profile,
            advanced_mode=req.advanced_mode,
            user_locale=req.user_locale,
            user_lat=req.user_lat,
            user_lng=req.user_lng,
            checkpointer=cp,
            checkpoint_conn=cconn,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/agent/history/{thread_id}")
async def agent_history(
    thread_id: str,
    request: Request,
    x_api_token: str = Header(default=""),
) -> dict[str, Any]:
    """Timeline bookmarked **turns** for ``thread_id`` (oldest → newest).

    Lists only IDs stored under ``AgentState.turn_checkpoints`` from the thread head —
    never raw LangGraph node-level history (often 100+ entries).

    ``label`` is ``cp_XXX`` (1-based) for UX. ``rewind_to_checkpoint`` should use ``checkpoint_id``.
    """
    _require_api_token(x_api_token)
    graph = build_graph(checkpointer=getattr(request.app.state, "checkpointer", None))
    checkpoints: list[dict[str, Any]] = []
    if _graph_supports_checkpointing(graph) and _graph_has_checkpoint_state_reader(graph):
        cfg = {"configurable": {"thread_id": thread_id}}
        try:
            snap_head = await _graph_get_state(graph, cfg)
        except Exception:
            snap_head = None
        vals = dict(getattr(snap_head, "values", None) or {}) if snap_head else {}
        turn_ids = [str(x) for x in (vals.get("turn_checkpoints") or []) if x is not None]
        for i, cid in enumerate(turn_ids, start=1):
            resolved = await _graph_resolve_checkpoint_snapshot(graph, thread_id, cid)
            prev = dict(getattr(resolved, "values", None) or {}) if resolved else {}
            checkpoints.append(
                {
                    "checkpoint_id": cid,
                    "label": f"cp_{i:03d}",
                    "ts": _snapshot_ts_iso(resolved) if resolved else _now_iso(),
                    "summary": _turn_checkpoint_summary(prev),
                    "turn": str(i),
                }
            )
    return {"thread_id": thread_id, "checkpoint_source": "turn_checkpoints", "checkpoints": checkpoints}


@app.post("/agent/pause/{thread_id}")
async def agent_pause(
    thread_id: str,
    request: Request,
    x_api_token: str = Header(default=""),
) -> dict:
    """Trigger SynthesizerAgent on the paused thread and return a SynthesisReport.

    The compiled graph uses the FastAPI lifespan ``AsyncSqliteSaver``.
    Pause builds synthesis from the latest checkpoint snapshot; it does not require
    a LangGraph interrupt.  If no checkpoint exists for the thread, returns an empty
    synthesis with a 200 so the UI can display
    "nothing to synthesise yet".
    """
    _require_api_token(x_api_token)
    graph = build_graph(checkpointer=getattr(request.app.state, "checkpointer", None))

    if not _graph_has_checkpoint_state_reader(graph):
        raise HTTPException(
            status_code=503,
            detail="Pause/resume requires a LangGraph checkpointer (API lifespan).",
        )

    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = await _graph_get_state(graph, cfg)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Thread not found: {exc}") from exc

    state_dict: dict = dict(snapshot.values) if snapshot and snapshot.values else {}
    if not state_dict:
        return {
            "thread_id": thread_id,
            "synthesis": SynthesisReport(
                consensus=["No agent state found for this thread."],
                unresolved=[],
                frontier=[],
                discussion_freshness=1.0,
            ).as_dict(),
            "paused_at": None,
        }

    # Run the synthesizer node directly on the current state (does NOT resume the graph)
    updated_state = node_synthesizer(dict(state_dict))
    report_dict = (updated_state.get("synthesis_history") or [{}])[-1]

    # Persist the updated synthesis_history back to the checkpoint
    try:
        await _graph_update_state(graph, cfg, {"synthesis_history": updated_state.get("synthesis_history", [])})
    except Exception:
        pass  # non-fatal — UI still gets the report

    paused_at = list(snapshot.next) if snapshot and snapshot.next else []
    return {
        "thread_id": thread_id,
        "synthesis": report_dict,
        "paused_at": paused_at,
    }


@app.post("/agent/resume/{thread_id}")
async def agent_resume(
    thread_id: str,
    request: Request,
    x_api_token: str = Header(default=""),
) -> StreamingResponse:
    """Resume a paused graph thread from its last checkpoint / next step, streaming SSE events."""
    _require_api_token(x_api_token)
    graph = build_graph(checkpointer=getattr(request.app.state, "checkpointer", None))

    if not _graph_has_checkpoint_state_reader(graph):
        raise HTTPException(
            status_code=503,
            detail="Pause/resume requires a LangGraph checkpointer (API lifespan).",
        )

    cfg = {"configurable": {"thread_id": thread_id}}

    async def _stream_resume() -> AsyncIterator[str]:
        yield f"event: resume_start\ndata: {json.dumps({'thread_id': thread_id})}\n\n"
        q: asyncio.Queue[dict | None] = asyncio.Queue()
        fatal: list[BaseException | None] = [None]

        async def _produce() -> None:
            try:
                async for step in graph.astream(None, config=cfg):
                    await q.put(step)
            except asyncio.InvalidStateError as e:
                logger.warning(
                    "Resume stream InvalidStateError thread=%s", thread_id, exc_info=True
                )
                fatal[0] = e
            except (OSError, ConnectionError, RuntimeError) as e:
                logger.warning(
                    "Resume stream backend failure thread=%s err=%s",
                    thread_id,
                    e.__class__.__name__,
                    exc_info=True,
                )
                fatal[0] = e
            except Exception as e:  # pragma: no cover
                fatal[0] = e
            finally:
                await q.put(None)

        producer_task = asyncio.create_task(_produce())
        try:
            while True:
                try:
                    step = await asyncio.wait_for(q.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if step is None:
                    break
                for node_name, node_state in step.items():
                    payload = {
                        "node": node_name,
                        "thread_id": thread_id,
                        "state": {k: v for k, v in node_state.items()
                                  if k in ("research_log", "transit_audit", "critique_history",
                                           "synthesis_history", "final_itinerary")},
                    }
                    yield f"event: step\ndata: {json.dumps(payload, default=str)}\n\n"
        finally:
            await producer_task
        if fatal[0] is not None:
            err = fatal[0]
            if isinstance(err, asyncio.InvalidStateError):
                payload = json.dumps(
                    {
                        "error": "checkpoint_transport_error",
                        "thread_id": thread_id,
                        "user_message": "檢查點／串流狀態異常，請重新發送請求或使用新 thread。",
                    },
                    ensure_ascii=False,
                )
            elif isinstance(err, (OSError, ConnectionError, RuntimeError)):
                payload = json.dumps(
                    {
                        "error": "checkpoint_db_error",
                        "thread_id": thread_id,
                        "user_message": "持久化連線中斷，請稍後重試。",
                    },
                    ensure_ascii=False,
                )
            else:
                payload = json.dumps(
                    {
                        "error": "stream_failed",
                        "thread_id": thread_id,
                        "user_message": "恢復串流失敗，請稍後重試。",
                    },
                    ensure_ascii=False,
                )
            yield f"event: error\ndata: {payload}\n\n"
            return
        yield f"event: done\ndata: {json.dumps({'thread_id': thread_id})}\n\n"

    return StreamingResponse(
        _stream_resume(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/agent/itinerary")
async def get_itinerary(day: int = 1, x_user_id: str = Header(default=""), x_api_token: str = Header(default="")) -> dict:
    _require_api_token(x_api_token)
    user_id = x_user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-Id header")
    row = _get_itinerary(user_id=user_id, day=day)
    if row is None:
        return {"synced": False, "day": int(day), "itinerary": "", "semantic_status": "NOT_SYNCED"}
    return {
        "synced": True,
        "day": int(row["day"]),
        "updated_at": row["updated_at"],
        "itinerary": row["itinerary_text"],
        "semantic_status": row["semantic_status"],
        "ui_cards": json.loads(row["ui_cards_json"] or "[]"),
    }


@app.get("/profile/onboarding")
async def get_onboarding_profile(x_user_id: str = Header(default=""), x_api_token: str = Header(default="")) -> dict:
    _require_api_token(x_api_token)
    user_id = x_user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-Id header")
    return {"user_id": user_id, "profile": _get_onboarding_profile(user_id)}


@app.put("/profile/onboarding")
async def put_onboarding_profile(payload: dict, x_user_id: str = Header(default=""), x_api_token: str = Header(default="")) -> dict:
    _require_api_token(x_api_token)
    user_id = x_user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-Id header")
    return {"ok": True, "user_id": user_id, "profile": _save_onboarding_profile(user_id, payload or {})}


@app.post(
    "/agent/rollback",
    deprecated=True,
    summary="[DEPRECATED] Trigger agent rollback via Saga",
    description="Rollback now handled by LangGraph checkpointer thread history. See ADR in README.md.",
)
async def agent_rollback(req: RollbackRequest, x_api_token: str = Header(default="")) -> dict:
    _require_api_token(x_api_token)
    raise HTTPException(
        status_code=501,
        detail=(
            "Rollback endpoint moved to LangGraph checkpointer thread history. "
            f"Use thread_id={req.agent_run_id} with graph checkpoints."
        ),
    )


# ── Duffel hold-and-cancel endpoints ──────────────────────────────────────────

_duffel = DuffelService()
_SAGA_DIR = Path(os.getenv("SAGA_PERSIST_DIR", str(Path.home() / ".travel_agent" / "saga_holds")))
_SAGA_DIR.mkdir(parents=True, exist_ok=True)
_HOLDS_DB = _SAGA_DIR / "holds.sqlite3"


def _db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_HOLDS_DB), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _init_hold_store() -> None:
    with _db_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hold_intents (
                intent_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL,
                offer_id TEXT NOT NULL,
                passenger_id TEXT NOT NULL,
                idem_key TEXT NOT NULL,
                saga_id TEXT NOT NULL,
                order_id TEXT,
                hold_trace_id TEXT,
                receipt_json TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hold_intents_order_id ON hold_intents(order_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_hold_intents_status ON hold_intents(status)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cancel_receipts (
                order_id TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                receipt_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dietary_profiles (
                user_id TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                profile_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS onboarding_profiles (
                user_id TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                profile_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS itineraries (
                user_id TEXT NOT NULL,
                day INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                itinerary_text TEXT NOT NULL,
                semantic_status TEXT NOT NULL,
                ui_cards_json TEXT DEFAULT '[]',
                PRIMARY KEY(user_id, day)
            )
            """
        )
        # Migration for existing databases that lack ui_cards_json column
        try:
            conn.execute("ALTER TABLE itineraries ADD COLUMN ui_cards_json TEXT DEFAULT '[]'")
        except Exception:
            pass


def _insert_intent(intent_id: str, offer_id: str, passenger_id: str, idem_key: str, saga_id: str) -> None:
    now = _now_iso()
    with _db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO hold_intents(intent_id, created_at, updated_at, status, offer_id, passenger_id, idem_key, saga_id)
            VALUES(?, ?, ?, 'holding', ?, ?, ?, ?)
            """,
            (intent_id, now, now, offer_id, passenger_id, idem_key, saga_id),
        )
        conn.execute("COMMIT")


def _mark_intent_held(intent_id: str, order_id: str, hold_trace_id: str, receipt: dict) -> None:
    now = _now_iso()
    with _db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE hold_intents
            SET updated_at=?, status='held', order_id=?, hold_trace_id=?, receipt_json=?
            WHERE intent_id=?
            """,
            (now, order_id, hold_trace_id, json.dumps(receipt, ensure_ascii=False), intent_id),
        )
        conn.execute("COMMIT")


def _get_cancel_receipt(order_id: str) -> dict | None:
    with _db_conn() as conn:
        row = conn.execute("SELECT receipt_json FROM cancel_receipts WHERE order_id=?", (order_id,)).fetchone()
    if row is None:
        return None
    return json.loads(row["receipt_json"])


def _get_hold_by_order(order_id: str) -> dict | None:
    with _db_conn() as conn:
        row = conn.execute("SELECT * FROM hold_intents WHERE order_id=? ORDER BY created_at DESC LIMIT 1", (order_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["receipt"] = json.loads(out["receipt_json"]) if out.get("receipt_json") else {}
    return out


def _mark_cancelled(order_id: str, receipt: dict) -> None:
    now = _now_iso()
    with _db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO cancel_receipts(order_id, updated_at, receipt_json)
            VALUES(?, ?, ?)
            ON CONFLICT(order_id) DO UPDATE SET updated_at=excluded.updated_at, receipt_json=excluded.receipt_json
            """,
            (order_id, now, json.dumps(receipt, ensure_ascii=False)),
        )
        conn.execute(
            "UPDATE hold_intents SET status='cancelled', updated_at=? WHERE order_id=?",
            (now, order_id),
        )
        conn.execute("COMMIT")


def _recover_pending_holds() -> None:
    with _db_conn() as conn:
        intents = conn.execute(
            """
            SELECT intent_id, offer_id, passenger_id, idem_key
            FROM hold_intents
            WHERE status='holding' AND (order_id IS NULL OR order_id='')
            ORDER BY created_at ASC
            """
        ).fetchall()
    for row in intents:
        result = _duffel.hold_order(row["offer_id"], row["passenger_id"], idempotency_key=row["idem_key"])
        if result.outcome != ActionOutcome.SUCCESS:
            continue
        order = result.raw_metadata.get("order", {})
        order_id = order.get("id")
        if not order_id:
            continue
        receipt = {
            "order_id": order_id,
            "idem_key": row["idem_key"],
            "held_at": _now_iso(),
            "hold_trace_id": result.trace_id or "",
            "hold_raw_metadata": result.raw_metadata,
        }
        _mark_intent_held(row["intent_id"], order_id, receipt["hold_trace_id"], receipt)


def _normalize_profile(p: dict | None) -> dict:
    profile = p or {}
    ethics = str(profile.get("ethics", "unspecified")).strip().lower()
    if ethics not in {"unspecified", "omnivore", "pescatarian", "vegetarian", "vegan"}:
        ethics = "unspecified"

    religious = str(profile.get("religious", "none")).strip().lower()
    if religious not in {"none", "halal", "kosher", "hindu_veg"}:
        religious = "none"

    alias = {
        "shrimp": "shellfish",
        "prawn": "shellfish",
        "crab": "shellfish",
        "lobster": "shellfish",
        "nuts": "tree_nuts",
        "nut": "tree_nuts",
        "milk": "dairy",
    }
    allowed_allergens = {"shellfish", "peanuts", "tree_nuts", "egg", "dairy", "soy", "wheat", "fish", "sesame"}
    allergens_raw = profile.get("allergens", [])
    allergens: list[str] = []
    for item in allergens_raw if isinstance(allergens_raw, list) else []:
        token = alias.get(str(item).strip().lower(), str(item).strip().lower())
        if token in allowed_allergens and token not in allergens:
            allergens.append(token)

    allowed_medical = {"pregnant", "low_sodium", "low_sugar", "low_purine", "gluten_sensitive"}
    medical_raw = profile.get("medical", [])
    medical: list[str] = []
    for item in medical_raw if isinstance(medical_raw, list) else []:
        token = str(item).strip().lower()
        if token in allowed_medical and token not in medical:
            medical.append(token)

    return {
        "ethics": ethics,
        "allergens": allergens,
        "religious": religious,
        "medical": medical,
    }


def _get_dietary_profile(user_id: str) -> dict:
    with _db_conn() as conn:
        row = conn.execute("SELECT profile_json FROM dietary_profiles WHERE user_id=?", (user_id,)).fetchone()
    if row is None:
        return _normalize_profile(None)
    try:
        return _normalize_profile(json.loads(row["profile_json"]))
    except Exception:
        return _normalize_profile(None)


def _save_dietary_profile(user_id: str, profile: dict) -> dict:
    normalized = _normalize_profile(profile)
    now = _now_iso()
    with _db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO dietary_profiles(user_id, updated_at, profile_json)
            VALUES(?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET updated_at=excluded.updated_at, profile_json=excluded.profile_json
            """,
            (user_id, now, json.dumps(normalized, ensure_ascii=False)),
        )
        conn.execute("COMMIT")
    return normalized


def _save_itinerary(user_id: str, day: int, itinerary_text: str, semantic_status: str = "RUN_COMPLETED", ui_cards_json: str = "[]") -> None:
    now = _now_iso()
    with _db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO itineraries(user_id, day, updated_at, itinerary_text, semantic_status, ui_cards_json)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, day) DO UPDATE SET
                updated_at=excluded.updated_at,
                itinerary_text=excluded.itinerary_text,
                semantic_status=excluded.semantic_status,
                ui_cards_json=excluded.ui_cards_json
            """,
            (user_id, int(day), now, itinerary_text, semantic_status, ui_cards_json),
        )
        conn.execute("COMMIT")


def _get_itinerary(user_id: str, day: int) -> dict | None:
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT user_id, day, updated_at, itinerary_text, semantic_status, ui_cards_json FROM itineraries WHERE user_id=? AND day=?",
            (user_id, int(day)),
        ).fetchone()
    return dict(row) if row is not None else None


def _save_onboarding_profile(user_id: str, profile: dict) -> dict:
    normalized = {
        "start_date": str(profile.get("start_date", "")).strip(),
        "city": str(profile.get("city", "")).strip(),
        "party_size": max(1, int(profile.get("party_size", 2))),
        "days": max(1, min(14, int(profile.get("days", 3)))),
    }
    now = _now_iso()
    with _db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO onboarding_profiles(user_id, updated_at, profile_json)
            VALUES(?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                profile_json=excluded.profile_json
            """,
            (user_id, now, json.dumps(normalized, ensure_ascii=False)),
        )
        conn.execute("COMMIT")
    return normalized


def _get_onboarding_profile(user_id: str) -> dict | None:
    with _db_conn() as conn:
        row = conn.execute("SELECT profile_json FROM onboarding_profiles WHERE user_id=?", (user_id,)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row["profile_json"])
    except Exception:
        return None


@app.on_event("startup")
async def recover_pending_holds() -> None:
    _init_hold_store()
    _recover_pending_holds()


@app.get("/profile/dietary")
async def get_dietary_profile(x_user_id: str = Header(default=""), x_api_token: str = Header(default="")) -> dict:
    _require_api_token(x_api_token)
    user_id = x_user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-Id header")
    return {"user_id": user_id, "dietary_profile": _get_dietary_profile(user_id)}


@app.put("/profile/dietary")
async def put_dietary_profile(
    req: DietaryProfileRequest,
    x_user_id: str = Header(default=""),
    x_api_token: str = Header(default=""),
) -> dict:
    _require_api_token(x_api_token)
    user_id = x_user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="Missing X-User-Id header")
    saved = _save_dietary_profile(user_id, req.model_dump(mode="json"))
    return {"ok": True, "user_id": user_id, "dietary_profile": saved}


class SearchRequest(BaseModel):
    origin:      str = "TPE"
    destination: str = "NRT"
    date:        str   # YYYY-MM-DD


class HoldRequest(BaseModel):
    offer_id:     str
    passenger_id: str


@app.post(
    "/duffel/search",
    deprecated=True,
    summary="[DEPRECATED] Search Duffel flight offers",
    description=(
        "Kept as an engineering reference for the distributed-transaction pattern. "
        "Flight booking is not part of the main agent flow. See ADR in README.md."
    ),
)
async def duffel_search(req: SearchRequest, x_api_token: str = Header(default="")) -> dict:
    _require_api_token(x_api_token)
    result = _duffel.search_offers(req.origin, req.destination, req.date)
    if not result["offers"]:
        raise HTTPException(status_code=404, detail="No offers found")
    first = result["offers"][0]
    return {
        "offer_id":     first["id"],
        "passenger_id": result["passenger_id"],
        "price":        first.get("total_amount"),
        "currency":     first.get("total_currency"),
    }


@app.post(
    "/duffel/hold",
    deprecated=True,
    summary="[DEPRECATED] Hold a Duffel flight order (Saga step)",
    description=(
        "Demonstrates idempotent 2-phase hold with Saga compensation. "
        "Not invoked by the main agent flow. See ADR in README.md."
    ),
)
async def duffel_hold(req: HoldRequest, x_api_token: str = Header(default="")) -> dict:
    _require_api_token(x_api_token)
    request_trace_id = str(uuid.uuid4())
    saga = SagaEngine(kind="transactional", persist_path=str(_SAGA_DIR / f"hold_{request_trace_id}.json"))
    hold_key = _duffel.build_idempotency_key(saga.log.saga_id, "duffel_hold")
    intent_id = request_trace_id
    _insert_intent(intent_id, req.offer_id, req.passenger_id, hold_key, saga.log.saga_id)
    hold_result: SagaActionResult | None = None
    for attempt in range(3):
        hold_result = _duffel.hold_order(req.offer_id, req.passenger_id, idempotency_key=hold_key)
        if hold_result.outcome == ActionOutcome.SUCCESS:
            break
        if hold_result.outcome != ActionOutcome.RETRYABLE or attempt == 2:
            raise HTTPException(
                status_code=502,
                detail=f"{hold_result.semantic_status}: {hold_result.error_message or hold_result.raw_metadata}",
            )
        await asyncio.sleep(float(2**attempt))

    assert hold_result is not None
    order = hold_result.raw_metadata.get("order", {})
    order_id = order.get("id")
    if not order_id:
        raise HTTPException(status_code=502, detail="Duffel hold missing order id")

    receipt = {
        "order_id": order_id,
        "idem_key": hold_key,
        "held_at": _now_iso(),
        "hold_trace_id": hold_result.trace_id or request_trace_id,
        "hold_raw_metadata": hold_result.raw_metadata,
    }
    _mark_intent_held(intent_id, order_id, receipt.get("hold_trace_id", request_trace_id), receipt)

    return {"order_id": order_id, "status": "held", "idem_key": hold_key}


@app.delete(
    "/duffel/orders/{order_id}",
    deprecated=True,
    summary="[DEPRECATED] Cancel / compensate a held Duffel order",
    description=(
        "Demonstrates idempotent Saga compensation via cached cancel receipts. "
        "Not invoked by the main agent flow. See ADR in README.md."
    ),
)
async def duffel_cancel(order_id: str, x_api_token: str = Header(default="")) -> dict:
    _require_api_token(x_api_token)
    cached_cancel = _get_cancel_receipt(order_id)
    hold_ctx = _get_hold_by_order(order_id)
    if cached_cancel is not None:
        return {
            "outcome": "SUCCESS",
            "semantic_status": cached_cancel.get("semantic_status", "ALREADY_CANCELLED_SUCCESS"),
            "compensated": True,
            "order_id": order_id,
            "trace_id": cached_cancel.get("trace_id", ""),
            "raw_metadata": cached_cancel.get("raw_metadata", {}),
            "cancel_status": cached_cancel.get("cancel_status"),
            "idempotent": True,
        }

    if hold_ctx is None:
        live = _duffel.get_order(order_id)
        status = live.get("status", "unknown")
        if not live.get("exists", True):
            raise HTTPException(status_code=410, detail=f"Order {order_id} no longer exists downstream")
        if status in {"cancelled", "canceled", "expired"}:
            return {
                "outcome": "SUCCESS",
                "semantic_status": "RECONCILED_OUT_OF_BAND",
                "compensated": True,
                "order_id": order_id,
                "trace_id": live.get("duffel_request_id", ""),
                "raw_metadata": live,
                "cancel_status": None,
                "idempotent": True,
            }
        raise HTTPException(status_code=409, detail=f"Order {order_id} exists downstream (status={status}) but not in local pending index")

    step_id = f"{hold_ctx.get('saga_id','system')}:duffel_hold"
    result = _duffel.cancel_order(order_id, step_id=step_id)
    receipt = dict(hold_ctx.get("receipt", {}))
    receipt["cancel_outcome"] = result.outcome.value
    receipt["semantic_status"] = result.semantic_status
    receipt["trace_id"] = result.trace_id
    receipt["raw_metadata"] = result.raw_metadata
    receipt["cancel_status"] = result.raw_metadata.get("http_status", 0)
    receipt["cancelled_at"] = _now_iso()
    _mark_cancelled(order_id, receipt)

    return {
        "outcome": receipt.get("cancel_outcome", "FAILED"),
        "semantic_status": receipt.get("semantic_status", "UNKNOWN"),
        "compensated":   True,
        "order_id":      order_id,
        "trace_id": receipt.get("trace_id", ""),
        "raw_metadata":  receipt.get("raw_metadata", {}),
        "cancel_status": receipt.get("cancel_status"),
        "idempotent":    False,
    }


@app.get("/saga/log/{order_id}")
async def saga_log(order_id: str, x_api_token: str = Header(default="")) -> dict:
    _require_api_token(x_api_token)
    hold_ctx = _get_hold_by_order(order_id)
    if hold_ctx is not None:
        return hold_ctx
    cancel = _get_cancel_receipt(order_id)
    if cancel is not None:
        return {"order_id": order_id, "receipt": cancel}
    raise HTTPException(status_code=404, detail=f"No saga log for {order_id}")

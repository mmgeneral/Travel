"""
FastAPI wrapper for agent.py — exposes the LangGraph agent over HTTP/SSE.

Why SSE instead of WebSocket?
  - One-way streaming (agent → browser) matches our use case
  - Works through any HTTP proxy without special config
  - Browser's EventSource API is built-in, no library needed

Routes:
  GET  /healthz       — liveness probe for Docker / browser offline detection
  POST /agent/query   — fire a query, receive SSE stream of agent events
  POST /agent/rollback — rollback to a snapshot_idx
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import sqlite3
from datetime import datetime
from typing import AsyncIterator
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import os
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from agent import build_graph, make_initial_state
from saga import SagaEngine
from duffel import DuffelService
from acl import ActionOutcome, SagaActionResult
from observability import init_tracing, traced

app = FastAPI(title="Travel Agent API")
logger = logging.getLogger(__name__)

init_tracing("travel-agent")

@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

_ALLOWED_ORIGINS = [x.strip() for x in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if x.strip()]
_API_TOKEN = os.getenv("TRAVEL_AGENT_API_TOKEN", "")
_DEV_MODE = os.getenv("DEV_MODE", "0") == "1"
_ENV = os.getenv("ENV", "development").lower()
_APP_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "Asia/Taipei"))


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

# CORS defaults to localhost origins; configure ALLOWED_ORIGINS for deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    day:   int = 1
    dietary_profile: dict | None = None
    advanced_mode: bool = False
    user_locale: str | None = None
    user_lat: float | None = None
    user_lng: float | None = None


class RollbackRequest(BaseModel):
    agent_run_id: str
    snapshot_idx: int


class DietaryProfileRequest(BaseModel):
    ethics: str = "omnivore"
    allergens: list[str] = []
    religious: str = "none"
    medical: list[str] = []


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz() -> dict:
    checks: dict[str, str] = {}
    try:
        _SAGA_DIR.mkdir(parents=True, exist_ok=True)
        probe = _SAGA_DIR / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks["saga_dir_writable"] = "ok"
    except Exception as exc:
        checks["saga_dir_writable"] = f"error:{exc}"
    checks["indexes_loaded"] = "ok" if (_PENDING_FILE.exists() or _CANCELLED_FILE.exists()) else "missing"
    try:
        token = os.getenv("DUFFEL_ACCESS_TOKEN", "")
        if not token:
            checks["duffel_connectivity"] = "skipped:no_token"
        else:
            resp = requests.get(
                "https://api.duffel.com/air/airports?limit=1",
                headers={"Authorization": f"Bearer {token}", "Duffel-Version": "v2", "Accept": "application/json"},
                timeout=3,
            )
            checks["duffel_connectivity"] = "ok" if resp.status_code < 500 else f"error:{resp.status_code}"
    except Exception as exc:
        checks["duffel_connectivity"] = f"error:{exc.__class__.__name__}"
    ready = checks["saga_dir_writable"] == "ok"
    return {"status": "ready" if ready else "degraded", "checks": checks}


async def _stream_agent(
    query: str,
    day: int,
    user_id: str,
    dietary_profile: dict | None = None,
    advanced_mode: bool = False,
    user_locale: str | None = None,
    user_lat: float | None = None,
    user_lng: float | None = None,
) -> AsyncIterator[str]:
    """
    Run the graph and emit SSE events as each node completes.
    Format: `event: <name>\\ndata: <json>\\n\\n`
    """
    agent_run_id = uuid.uuid4().hex
    graph = build_graph()
    graph_cfg = {"configurable": {"thread_id": agent_run_id}}
    initial_state = make_initial_state(
        query=query,
        dietary_profile=dietary_profile,
        agent_run_id=agent_run_id,
        advanced_mode=advanced_mode,
        user_locale=user_locale,
        user_lat=user_lat,
        user_lng=user_lng,
    )

    yield f"event: start\ndata: {json.dumps({'day': day, 'query': query, 'agent_run_id': agent_run_id})}\n\n"

    # Use background producer + queue to avoid cancelling __anext__()
    # when heartbeat timeout fires.
    final_state = None
    q: asyncio.Queue[dict | None] = asyncio.Queue()
    producer_error: Exception | None = None

    async def _produce_steps() -> None:
        nonlocal producer_error
        try:
            async for step in graph.astream(initial_state, config=graph_cfg):
                await q.put(step)
        except Exception as e:  # pragma: no cover - surfaced in parent
            producer_error = e
        finally:
            await q.put(None)

    producer_task = asyncio.create_task(_produce_steps())
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
                final_state = node_state
                payload = {
                    "node":   node_name,
                    "agent_run_id": agent_run_id,
                    "state":  {k: v for k, v in node_state.items()
                               if k in ("research_log", "transit_audit",
                                        "rollback_occurred", "saga_snapshot_idx", "agent_run_id")},
                }
                yield f"event: node\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)  # flush
        await producer_task
        if producer_error is not None:
            raise producer_error
    except Exception:
        producer_task.cancel()
        with contextlib.suppress(Exception):
            await producer_task
        logger.exception("Agent SSE stream failed, run_id=%s", agent_run_id)
        yield (
            "event: error\n"
            f"data: {json.dumps({'error': 'stream_failed', 'agent_run_id': agent_run_id, 'user_message': '網路或服務暫時不穩，請稍後重試。'}, ensure_ascii=False)}\n\n"
        )
        return

    if final_state is None:
        final_state = initial_state
    done_payload = json.dumps(
        {
            "itinerary":    final_state.get("final_itinerary", ""),
            "ui_cards": final_state.get("ui_cards", []),
            "snapshot_idx": final_state.get("saga_snapshot_idx", -1),
            "outcome": "SUCCESS",
            "semantic_status": "RUN_COMPLETED",
            "agent_run_id": agent_run_id,
        },
        ensure_ascii=False,
    )
    _save_itinerary(
        user_id=user_id,
        day=day,
        itinerary_text=final_state.get("final_itinerary", ""),
        semantic_status="RUN_COMPLETED",
    )
    yield f"event: done\ndata: {done_payload}\n\n"


@app.post("/agent/query")
@traced
async def agent_query(
    req: QueryRequest,
    x_user_id: str = Header(default=""),
    x_api_token: str = Header(default=""),
) -> StreamingResponse:
    _require_api_token(x_api_token)
    user_id = x_user_id.strip() or "anonymous"
    return StreamingResponse(
        _stream_agent(
            req.query,
            req.day,
            user_id,
            req.dietary_profile,
            req.advanced_mode,
            req.user_locale,
            req.user_lat,
            req.user_lng,
        ),
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


@app.post("/agent/rollback")
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
                PRIMARY KEY(user_id, day)
            )
            """
        )


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


def _save_itinerary(user_id: str, day: int, itinerary_text: str, semantic_status: str = "RUN_COMPLETED") -> None:
    now = _now_iso()
    with _db_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            INSERT INTO itineraries(user_id, day, updated_at, itinerary_text, semantic_status)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(user_id, day) DO UPDATE SET
                updated_at=excluded.updated_at,
                itinerary_text=excluded.itinerary_text,
                semantic_status=excluded.semantic_status
            """,
            (user_id, int(day), now, itinerary_text, semantic_status),
        )
        conn.execute("COMMIT")


def _get_itinerary(user_id: str, day: int) -> dict | None:
    with _db_conn() as conn:
        row = conn.execute(
            "SELECT user_id, day, updated_at, itinerary_text, semantic_status FROM itineraries WHERE user_id=? AND day=?",
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


@app.post("/duffel/search")
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


@app.post("/duffel/hold")
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


@app.delete("/duffel/orders/{order_id}")
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

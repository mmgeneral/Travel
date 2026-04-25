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
import json
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import build_graph, _conv_saga

app = FastAPI(title="Travel Agent API")

# CORS: allow any origin for dev; tighten in production to the frontend's domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    day:   int = 1


class RollbackRequest(BaseModel):
    snapshot_idx: int


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


async def _stream_agent(query: str, day: int) -> AsyncIterator[str]:
    """
    Run the graph and emit SSE events as each node completes.
    Format: `event: <name>\\ndata: <json>\\n\\n`
    """
    graph = build_graph()

    initial_state = {
        "query":             query,
        "research_log":      [],
        "transit_audit":     [],
        "final_itinerary":   "",
        "rollback_occurred": False,
        "saga_snapshot_idx": -1,
    }

    yield f"event: start\ndata: {json.dumps({'day': day, 'query': query})}\n\n"

    # astream yields state deltas as each node finishes — perfect for SSE
    try:
        async for step in graph.astream(initial_state):
            for node_name, node_state in step.items():
                payload = {
                    "node":   node_name,
                    "state":  {k: v for k, v in node_state.items()
                               if k in ("research_log", "transit_audit",
                                        "rollback_occurred", "saga_snapshot_idx")},
                }
                yield f"event: node\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)  # flush
    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        return

    # Final: fetch the finished state by invoking once more synchronously.
    # (astream gives deltas; we need the full final state for the UI.)
    final = graph.invoke(initial_state)
    done_payload = json.dumps(
        {
            "itinerary":    final["final_itinerary"],
            "snapshot_idx": final["saga_snapshot_idx"],
        },
        ensure_ascii=False,
    )
    yield f"event: done\ndata: {done_payload}\n\n"


@app.post("/agent/query")
async def agent_query(req: QueryRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_agent(req.query, req.day),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/agent/rollback")
async def agent_rollback(req: RollbackRequest) -> dict:
    try:
        restored = _conv_saga.rollback_to(req.snapshot_idx)
        return {"ok": True, "restored": restored}
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))

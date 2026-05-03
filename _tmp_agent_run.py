"""One-shot LangGraph CLI debug — ``ainvoke`` without SSE / FastAPI.

Use this to see whether ``plan`` filled ``final_itinerary`` and whether
``synthesis_history`` changed (``node_synthesizer`` only runs on pause in API).

Example::

    python _tmp_agent_run.py

Depends on the same env as the production agent (Ollama / Gemini / …); optional OTLP::

    PYTHONPATH=. python _tmp_agent_run.py
"""

from __future__ import annotations

import asyncio
import json
import pprint
import uuid

from observability import init_tracing
from tracing import agent_run, ensure_langfuse_env, flush, init_langfuse


def _dump_diag(state: dict) -> None:
    """Short summary after printing the full mapping (rich/json path)."""
    keys = sorted(state.keys())
    print("\n=== keys:", ", ".join(keys))
    ft = state.get("final_itinerary")
    syn = state.get("synthesis_history") or []
    err = state.get("error")
    print("=== diagnostic")
    print("  error:", json.dumps(err, ensure_ascii=False, default=str) if err else None)
    print("  final_itinerary length:", len(str(ft or "")))
    print("  synthesis_history entries:", len(syn) if isinstance(syn, list) else "n/a")
    print("\n=== final_itinerary excerpt (first 2000 chars)\n")
    print(str(ft or "")[:2000])
    if syn:
        print("\n=== synthesis_history (preview up to 2 entries)")
        pprint.pprint((syn[:2] if isinstance(syn, list) else syn))


async def _main_async() -> None:
    ensure_langfuse_env()
    init_langfuse()
    init_tracing("travel-agent-tmp-cli")

    from agent import build_graph, make_initial_state

    tid = uuid.uuid4().hex
    graph = build_graph()
    config = {"configurable": {"thread_id": tid}}
    initial = make_initial_state(query="京都美食行程", agent_run_id=tid, checkpoint_thread_id=tid)

    out = None
    try:
        with agent_run(
            tid,
            trace_input={"query": initial.get("query"), "thread_id": tid},
            metadata={"source": "_tmp_agent_run"},
        ):
            out = await graph.ainvoke(initial, config)
    finally:
        flush()

    if isinstance(out, dict):
        try:
            from rich.console import Console
            from rich.pretty import Pretty

            Console().print(Pretty(out))
            _dump_diag(out)
        except ImportError:
            pprint.pprint(out)
            _dump_diag(out)
        return

    pprint.pprint(out)


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()

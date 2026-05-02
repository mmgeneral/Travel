# Architecture

## Two Independent Agent Applications

This repo hosts two multi-agent applications that share infrastructure but are
otherwise isolated ¡X they don't import from each other and don't share state.

### 1. Food Discovery Agent (main app)

| | |
|---|---|
| **Entry point** | `uvicorn api:app` |
| **Core files** | `agent.py`, `api.py`, `decision_engine.py`, `shop_planning.py` |
| **State store** | LangGraph `SqliteSaver` (thread-level checkpoints) |
| **Frontend** | `index.html` (SSE streaming, pause/resume UI) |

Agent graph: `route_intent ¡÷ retriever ¡÷ researcher ¡÷ critic ¡÷ plan`

The `critic` node loops back to `researcher` (up to 3 rounds) when it issues
`verdict=request_more`.  Users can pause at any `interrupt_after` point and
trigger a `SynthesizerAgent` summary via `POST /agent/pause/{thread_id}`.

### 2. Code Review Agent (dev tool)

| | |
|---|---|
| **Entry point** | `python -m tools.code_review_chat <dir>` or `make code-review` |
| **Core files** | `tools/` package (completely isolated) |
| **State store** | In-memory `Transcript` + JSONL on disk |
| **UI** | Terminal / CLI only |

Two LLM agents (`Student` + `Professor`) discuss the codebase in an alternating
turn loop.  A third `Synthesizer` agent produces on-demand three-part summaries
(consensus / unresolved / frontier) when the user presses Enter.

---

## Shared Infrastructure

### `llm_router.py` ¡X Single point of LLM access

All LLM calls in both applications go through `LLMRouter.complete(task, messages)`.

**Priority chain (local-first, cloud-fallback):**

| Task | Backend 1 | Backend 2 | Backend 3 |
|---|---|---|---|
| `INTENT_PARSING` | Ollama (1080 Ti) | Gemini Flash | ¡X |
| `RETRIEVAL_REASONING` | Ollama (1080 Ti) | Gemini Flash | ¡X |
| `CRITIQUE` | vLLM (4090) | Gemini Flash | Ollama (1080 Ti) |
| `SYNTHESIS` | Ollama (1080 Ti) | Gemini Flash | ¡X |
| `EMBEDDING` | Ollama (1080 Ti) | Gemini Flash | ¡X |

Availability checks are cached 30 s.  Ollama and vLLM use HTTP health probes
(`/api/tags` and `/v1/models`).  Gemini/Claude check env keys only (avoids
burning free-tier rate limits on health calls).

When a backend has an open circuit breaker or fails transiently, `LLMRouter`
silently falls back to the next in chain and records `llm.backend.attempted`
in the OTEL span so you can see the fallback path in Langfuse.

### `observability.py` ¡X OpenTelemetry tracing

Every LLM call produces a span with:

```
llm.task_type          CRITIQUE
llm.backend.attempted  ["VLLMBackend:unavailable", "GeminiBackend:ok"]
llm.backend.selected   GeminiBackend
llm.model              gemini-2.0-flash
llm.tokens_in          312
llm.tokens_out         198
llm.cost_usd           0.0
llm.latency_ms         1840
```

Traces export to local Langfuse (`make langfuse` ¡÷ http://localhost:3000).

### `acl.py` ¡X Action outcome taxonomy

`ActionOutcome` enum (`SUCCESS / FAILED / RETRYABLE`) was originally designed
for Saga compensation decisions.  It's now also used by the LLM retry layer to
classify transient vs. permanent failures.

---

## Saga / Duffel / Polling Worker ¡X Why Still Here

`saga.py`, `duffel.py`, `polling_worker.py` are **not in the main agent flow**.
They were removed in an ADR (see README) but kept as engineering references for:

- **Saga pattern**: 2-phase distributed transaction with compensation
- **Idempotency**: how to make external API calls safe to retry
- **Polling worker**: decoupled async job completion (Duffel order status)

The `tests/legacy/` directory contains tests for these patterns.  Run them with
`pytest tests/legacy/ -v` when studying the patterns.

---

## Directory Layout

```
.
¢u¢w¢w agent.py              # LangGraph graph + all node functions
¢u¢w¢w api.py                # FastAPI endpoints + SSE streaming
¢u¢w¢w decision_engine.py    # ScoringEngine, RankingEngine (core IP)
¢u¢w¢w shop_planning.py      # ShopProfile, time-slot planning
¢u¢w¢w shop_catalog_io.py    # JSON catalog loader
¢u¢w¢w llm_router.py         # LLM backend abstraction + fallback chain
¢u¢w¢w intent_parser.py      # Hybrid rule + LLM intent parsing
¢u¢w¢w observability.py      # OpenTelemetry helpers
¢u¢w¢w acl.py                # ActionOutcome taxonomy
¢x
¢u¢w¢w agents/               # Multi-agent nodes (imported by agent.py)
¢x   ¢u¢w¢w retriever.py      # RetrieverAgent
¢x   ¢u¢w¢w critic.py         # CriticAgent
¢x   ¢|¢w¢w synthesizer.py    # SynthesizerAgent
¢x
¢u¢w¢w data/shop_catalogs/   # Seed shop data per region (JSON)
¢x   ¢u¢w¢w taipei.json
¢x   ¢u¢w¢w kyoto.json
¢x   ¢|¢w¢w tokyo.json
¢x
¢u¢w¢w tools/                # Standalone dev tools (no cross-imports to/from main app)
¢x   ¢u¢w¢w code_review_chat.py
¢x   ¢u¢w¢w personas.py
¢x   ¢u¢w¢w transcript.py
¢x   ¢|¢w¢w transcripts/      # Auto-saved JSONL sessions
¢x
¢u¢w¢w tests/legacy/         # Saga / idempotency pattern examples (not in CI)
¢x
¢u¢w¢w saga.py               # [DEPRECATED from main flow] distributed tx reference
¢u¢w¢w duffel.py             # [DEPRECATED from main flow] flight booking reference
¢|¢w¢w polling_worker.py     # [DEPRECATED from main flow] async job polling reference
```

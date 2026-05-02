# Travel Agent

A LangGraph-powered multi-agent travel itinerary system with LLM routing, shop catalog management, and geolocation support.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in API keys
uvicorn api:app --reload
```

Open `http://localhost:8000` in your browser.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `TRAVEL_AGENT_API_TOKEN` | *(empty)* | Bearer token; optional in `DEV_MODE=1` |
| `DEV_MODE` | `0` | Set `1` to skip auth in development |
| `DEFAULT_LOCALE_CITY` | `kyoto` | Fallback city when query has no location |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `ANTHROPIC_API_KEY` | — | Anthropic Claude API key |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Local Ollama endpoint |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse OTLP host |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | — | Langfuse project secret key |

## Architecture

```
Browser → FastAPI (api.py)
            ↓
        LangGraph (agent.py)
          route_intent
               ↓
           retriever (RetrieverAgent + Gemini)
               ↓
           researcher
               ↓
             critic ←──────────────────────────────┐
               ↓ verdict=satisfied / max-iters      │ verdict=request_more
         collect_feedback                      (back to researcher)
               ↓
             plan  →  final itinerary SSE stream

          [synthesizer]  ← on-demand only (POST /agent/pause/{thread_id})
```

### LLM Backend Priority (`llm_router.py`)

Backends are tried in order; the first **available** one wins.  
`is_available()` results are cached 30 s (no repeated health probes).

| Priority | Backend | Hardware | Env var | TaskTypes |
|---|---|---|---|---|
| 1 | `OllamaBackend` | 1080 Ti (always-on) | `OLLAMA_URL` | All (primary) |
| 2 | `VLLMBackend` | Lab 4090 (on-demand) | `VLLM_URL` | `CRITIQUE` (strong reasoning) |
| 3 | `GeminiBackend` | Cloud free-tier | `GEMINI_API_KEY` | All (cloud fallback) |
| 4 | `ClaudeBackend` | Cloud paid | `ANTHROPIC_API_KEY` | optional |

**Fallback chains by task:**

| `TaskType` | Chain |
|---|---|
| `INTENT_PARSING` | Ollama → Gemini |
| `CRITIQUE` | vLLM → Gemini → Ollama |
| `RETRIEVAL_REASONING`, `SYNTHESIS`, `EMBEDDING` | Ollama → Gemini |

**Availability health checks:**

| Backend | Probe |
|---|---|
| `OllamaBackend` | `GET {OLLAMA_URL}/api/tags` (HTTP 200, timeout 2 s) |
| `VLLMBackend` | `GET {VLLM_URL}/v1/models` (HTTP 200, timeout 2 s); skip if `VLLM_URL` is empty |
| `GeminiBackend` | `GEMINI_API_KEY` env must be non-empty (no HTTP probe) |
| `ClaudeBackend` | `ANTHROPIC_API_KEY` env must be non-empty (no HTTP probe) |

**Quick setup:**

```bash
cp .env.example .env
# Fill in OLLAMA_URL (default: http://localhost:11434) + optionally GEMINI_API_KEY

# Verify routing
python -c "
from llm_router import LLMRouter, TaskType
r = LLMRouter()
print(r.complete(TaskType.INTENT_PARSING, [{'role':'user','content':'hi'}]).model_used)
"
```

`LocalQwenBackend` is a backward-compat alias for `OllamaBackend`.
Each backend has configurable `timeout`, `max_retries`, and a `tenacity`-based circuit breaker.

## 如何本地跑 Langfuse

Langfuse 提供完整的 LLM observability dashboard，支援 OpenTelemetry 協定收集 traces。

### 1. 啟動 Langfuse（Docker）

```bash
docker run --rm -p 3000:3000 \
  -e NEXTAUTH_SECRET=local-secret \
  -e SALT=local-salt \
  -e DATABASE_URL=file:./langfuse.db \
  langfuse/langfuse:latest
```

或使用 Docker Compose（推薦，含 PostgreSQL）：

```yaml
# docker-compose.langfuse.yml
version: "3.8"
services:
  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://langfuse:langfuse@db:5432/langfuse
      - NEXTAUTH_SECRET=supersecret
      - SALT=supersalt
      - NEXTAUTH_URL=http://localhost:3000
    depends_on:
      - db
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    volumes:
      - langfuse_pg:/var/lib/postgresql/data

volumes:
  langfuse_pg:
```

```bash
docker compose -f docker-compose.langfuse.yml up -d
```

### 2. 建立 Langfuse 專案並取得 API 金鑰

1. 打開 `http://localhost:3000`，完成帳號設定
2. 建立新 Project（例如 `travel-agent`）
3. 在 **Settings → API Keys** 產生 Public Key 和 Secret Key

### 3. 設定環境變數

```bash
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxx
export LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxx
```

### 4. 啟動 Travel Agent

```bash
uvicorn api:app --reload
```

### 5. 發送測試請求並觀看 Dashboard

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -H "X-Api-Token: $TRAVEL_AGENT_API_TOKEN" \
  -d '{"query": "推薦台北早餐", "day": 1}'
```

打開 `http://localhost:3000` → **Traces** 頁面，即可看到：
- 整個 `/agent/query` request 的 root span（wall-clock time）
- 每個 LLM call 的子 span，包含 `llm.model`、`llm.tokens_in`、`llm.tokens_out`、`llm.cost_usd`、`llm.latency_ms`

## Running Tests

```bash
# All active tests (excludes tests/legacy/)
pytest

# Core agent stack
pytest test_llm_router.py test_observability.py
pytest test_intent_parser.py
pytest test_retriever_agent.py test_critic_agent.py

# Data & catalog
pytest test_agent_catalog.py test_agent_locale_fallback.py
pytest test_region_and_dietary.py test_geo_location_routing.py

# Legacy distributed-tx reference tests (not in default run)
pytest tests/legacy/ -v
```

---

## Architecture Decision Record (ADR): Duffel / Saga / Polling Worker removed from main flow

**Decision date:** 2026-05-01  
**Status:** Accepted

### Context

The original agent included a full distributed-transaction stack:

| Component | Purpose |
|-----------|---------|
| `node_flight_search` | Call Duffel API to fetch flight offers (TPE → NRT → SFO) |
| `node_audit` | 2-phase Saga commit: reserve leg-1, reserve leg-2, rollback on failure |
| `duffel.py` / `acl.py` | Duffel HTTP client + `ActionOutcome` error taxonomy |
| `saga.py` | Generic Saga engine with step-level compensation |
| `polling_worker.py` | Standalone batch worker for polling flight status |
| `AgentState` fields | `leg1_offer`, `leg2_offer`, `rollback_occurred`, `saga_snapshot_idx`, `conv_saga_path` |

The main `/agent/query` graph was: `route_intent → flight_search → retriever → researcher → auditor → audit → plan`.

### Problem

1. **Noise in the food-planning path.** Every non-flight query still ran `node_flight_search` (skipped only by a conditional edge), and `node_audit` was always wired in. This inflated latency, injected Duffel `transit_audit` entries into food-planning responses, and confused the LLM agents with unrelated state.
2. **State pollution.** `leg1_offer`, `leg2_offer`, `rollback_occurred` sat in `AgentState` for every request, even pure "show me ramen in Kyoto" queries.
3. **Scope creep.** The multi-agent foodie loop (`Retriever → Researcher → Critic → Synthesizer`) is the core product. Flight booking is a separate bounded context that deserves its own service/graph, not a bolted-on node in a food-recommendation graph.

### Decision

Remove `node_flight_search`, `node_audit`, and the 5 legacy `AgentState` fields from the **main graph** in `build_graph()`. The new graph is:

```
route_intent → retriever → researcher → critic ⟲ (up to 3 iterations)
                                          ↓
                                  collect_feedback → plan
```

**We deliberately keep all files** (`duffel.py`, `saga.py`, `acl.py`, `polling_worker.py`, `node_flight_search`, `node_audit`, `tests/legacy/`) because they demonstrate:

- **Idempotent 2-phase commit** with automatic Saga compensation (backward recovery).
- **`ActionOutcome` error taxonomy** (`SUCCESS | RETRYABLE | TERMINAL`) used by `LLMRouter`'s retry logic.
- **Polling worker** architecture for async, out-of-band status checks.
- **Duffel API integration** pattern for real flight booking systems.

### Consequences

- `pytest` (default) runs the food-agent stack only. `pytest tests/legacy/ -v` runs the distributed-tx reference tests.
- `/duffel/*` and `/agent/rollback` API endpoints are marked `deprecated=True` in FastAPI (visible in `/docs`).
- `wc -l agent.py` is ~700 lines shorter than the pre-Task-7 baseline.
- Any team member can re-enable flight booking by adding `node_flight_search` and `node_audit` back to `build_graph()` and wiring `route_intent → flight_search → retriever` — the code is all there.
- `polling_worker.py` is a **standalone batch job** — it is not invoked by the main agent. Run it directly: `python polling_worker.py --shop "燃えよ麺助"`.

### References

- `acl.py` — `ActionOutcome` enum (must not be deleted; used by `LLMRouter` retry logic).
- `saga.py` — Saga engine with step-level compensation and SQLite persistence.
- `duffel.py` — Duffel HTTP client with idempotency-key support.
- `tests/legacy/test_idempotency.py` — idempotency replay tests.
- `tests/legacy/test_saga_compensation.py` — Saga rollback tests.
- `tests/legacy/test_resilience.py` — circuit-breaker + retry resilience tests.

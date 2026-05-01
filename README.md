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
          ├── flight_search  →  Duffel API
          └── food_search    →  Google Places
               ↓
           researcher  ←──┐
               ↓          │ (rejected, < 3 iterations)
            auditor  ──────┘
               ↓
            plan  →  final itinerary SSE stream
```

### LLM Routing (`llm_router.py`)

| `TaskType` | Backend | Model |
|---|---|---|
| `INTENT_PARSING` | LocalQwenBackend | Ollama `qwen2.5:7b` |
| `CRITIQUE` | ClaudeBackend | `claude-3-5-sonnet-latest` |
| `RETRIEVAL_REASONING`, `SYNTHESIS`, `EMBEDDING` | GeminiBackend | `gemini-2.0-flash` |

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
# All tests
pytest

# Specific test suites
pytest test_llm_router.py
pytest test_observability.py
pytest test_agent_catalog.py
pytest test_agent_locale_fallback.py
pytest test_region_and_dietary.py
pytest test_geo_location_routing.py
pytest test_multi_agent_feedback_loop.py
```

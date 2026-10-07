# Chapter 17.5: Docker + Docker Compose Deployment

> **Phase 17 — Production & Deployment** | [← Previous: Cost & Caching](chapter-77-cost-caching.md) | [Next: Enterprise Capstone →](../phase-18-capstone/chapter-79-enterprise-ai-platform.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Containerize a FastAPI + LangChain/LangGraph service with a production-ready **Dockerfile**
- ✅ Orchestrate **API, Redis, PostgreSQL, and Chroma** with Docker Compose
- ✅ Wire **LiteLLM proxy** and **LangSmith** via environment variables
- ✅ Apply health checks, non-root users, and multi-stage builds
- ✅ Run migrations, ingest jobs, and the API as separate compose services
- ✅ Prepare a deployment checklist for staging vs production

| | |
|---|---|
| **Prerequisites** | Chapters 17.1–17.4 |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction — Same Machine, Every Time

```
"Works on my laptop"  →  demo succeeds, production fails
Docker                →  same Python, same deps, same env layout everywhere
Docker Compose        →  one command brings up API + Redis + DB + vector store
```

This chapter targets a **realistic enterprise stack** aligned with the capstone:

- **FastAPI** API layer  
- **LangChain + LangGraph** (agents, RAG)  
- **PostgreSQL** (users, metadata, LangGraph checkpoints optional)  
- **ChromaDB** (persisted volume) or FAISS baked into image for read-only indexes  
- **Redis** (LLM cache, rate limits, sessions)  
- **LiteLLM proxy** (external or sidecar URL)  
- **LangSmith** tracing  

---

## Part 1: Project Layout for Containers

```
enterprise-ai-platform/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── chains/
│   └── graph/
├── scripts/
│   └── ingest.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env.example
└── requirements.txt
```

Keep secrets in `.env` (gitignored). Ship `.env.example` with placeholders only.

---

## Part 2: Multi-Stage Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY app ./app
COPY scripts ./scripts

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why multi-stage:** smaller image, no build tools in production, faster deploys.

For **development**, mount source with compose override:

```yaml
# docker-compose.override.yml (local only)
services:
  api:
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    volumes:
      - ./app:/app/app
```

---

## Part 3: Docker Compose — Full Stack

```yaml
# docker-compose.yml
name: enterprise-ai

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      DATABASE_URL: postgresql+psycopg://ai:ai@postgres:5432/ai_platform
      CHROMA_HOST: chroma
      CHROMA_PORT: "8000"
      # LiteLLM — point to your proxy (hosted or separate compose project)
      LITELLM_PROXY_API_BASE: ${LITELLM_PROXY_API_BASE}
      LITELLM_PROXY_API_KEY: ${LITELLM_PROXY_API_KEY}
      LANGCHAIN_TRACING_V2: "true"
      LANGCHAIN_API_KEY: ${LANGCHAIN_API_KEY}
      LANGCHAIN_PROJECT: enterprise-ai-platform
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      chroma:
        condition: service_started
    volumes:
      - chroma_data:/app/chroma_persist
    restart: unless-stopped

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ai
      POSTGRES_PASSWORD: ai
      POSTGRES_DB: ai_platform
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ai -d ai_platform"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  chroma:
    image: chromadb/chroma:0.5.23
    environment:
      IS_PERSISTENT: "TRUE"
      PERSIST_DIRECTORY: /chroma/chroma
      ANONYMIZED_TELEMETRY: "FALSE"
    volumes:
      - chroma_data:/chroma/chroma
    ports:
      - "8001:8000"

  ingest:
    build: .
    profiles: ["jobs"]
    env_file:
      - .env
    environment:
      DATABASE_URL: postgresql+psycopg://ai:ai@postgres:5432/ai_platform
      CHROMA_HOST: chroma
    command: python scripts/ingest.py
    depends_on:
      - postgres
      - chroma

volumes:
  pg_data:
  redis_data:
  chroma_data:
```

### `.env.example`

```bash
# LLM via LiteLLM proxy (required)
LITELLM_PROXY_API_BASE=https://litellm.example.com/v1
LITELLM_PROXY_API_KEY=sk-proxy-replace-me

# LangSmith (recommended in staging/prod)
LANGCHAIN_API_KEY=lsv2_pt_replace_me
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# App
JWT_SECRET=replace-with-long-random-string
KB_VERSION=1
LOG_LEVEL=INFO
```

### Bring It Up

```bash
cp .env.example .env
# edit .env with real keys

docker compose build
docker compose up -d postgres redis chroma
docker compose run --rm ingest   # profile jobs — or: docker compose --profile jobs run --rm ingest
docker compose up -d api

curl http://localhost:8000/health
```

---

## Part 4: App Configuration Inside Containers

```python
# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    litellm_proxy_api_base: str
    litellm_proxy_api_key: str

    redis_host: str = "localhost"
    redis_port: int = 6379
    database_url: str = "postgresql+psycopg://ai:ai@localhost:5432/ai_platform"

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_dir: str = "./chroma_persist"

    jwt_secret: str
    kb_version: str = "1"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

Map LiteLLM env vars to LangChain clients (same pattern as earlier chapters):

```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=settings.litellm_proxy_api_key,
    openai_api_base=settings.litellm_proxy_api_base,
)
```

---

## Part 5: Production Compose Overlay

```yaml
# docker-compose.prod.yml
services:
  api:
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
    read_only: true
    tmpfs:
      - /tmp

  postgres:
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
```

Run:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

On real clusters you may move to Kubernetes or managed RDS/ElastiCache — **keep the same env var names** so the app stays portable.

---

## Part 6: CI Smoke Test (GitHub Actions Sketch)

```yaml
# .github/workflows/ci.yml
name: ci
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest -q

  compose-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cp .env.example .env
      - run: docker compose build api
      - run: docker compose up -d postgres redis
      - run: docker compose run --rm api pytest -q
```

Capstone expands this with full integration tests against Chroma and auth.

---

## Part 7: Deployment Checklist

| Item | Staging | Production |
|------|---------|------------|
| Secrets in vault / CI secrets, not image | ✅ | ✅ |
| Non-root container user | ✅ | ✅ |
| Health check on `/health` | ✅ | ✅ |
| Redis persistence for cache | optional | ✅ |
| Postgres backups | optional | ✅ |
| LangSmith project per environment | ✅ | ✅ |
| Resource limits | ✅ | ✅ |
| CORS locked to known origins | loose | ✅ |
| TLS termination at reverse proxy | optional | ✅ |

### Reverse Proxy (nginx) for SSE

```nginx
location /v1/ {
    proxy_pass http://api:8000;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;
    proxy_read_timeout 300s;
}
```

---

## Common Mistakes

### Mistake 1: Baking `.env` into the image
```dockerfile
# ❌ Secrets in layers — anyone with the image can extract them
COPY .env .

# ✅ env_file / orchestrator secrets at runtime
```

### Mistake 2: No health checks
```yaml
# ❌ API starts before Postgres is ready → crash loop

# ✅ depends_on + condition: service_healthy
```

### Mistake 3: Writable root filesystem
```yaml
# ❌ Container compromise = easy persistence

# ✅ read_only: true + tmpfs /tmp (where applicable)
```

### Mistake 4: Single monolithic container for DB + API
```yaml
# ❌ Can't scale API independently

# ✅ One service per concern (12-factor)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Multi-stage Dockerfile | Smaller attack surface and faster pulls |
| Named volumes for Chroma/Postgres/Redis | Data survives container recreation |
| Separate `ingest` job service | Indexing doesn't block API deploy |
| Same env var contract local and prod | Fewer "works in compose only" bugs |
| `--profile jobs` for one-off tasks | Clean operational model |
| Pin base image digests in prod | Reproducible security patches |
| Log to stdout (JSON) | Container platforms aggregate logs |

---

## Interview Preparation

### Easy
**Q: Why use Docker Compose for an LLM application?**

> Compose defines **multi-service topology** (API, cache, DB, vector store) as code, wires networking and health checks, and gives developers a **one-command** environment identical to staging. It reduces configuration drift and documents dependencies explicitly.

### Medium
**Q: How do you handle secrets in Docker-based LLM deployments?**

> Never COPY `.env` into images. Inject secrets at runtime via orchestrator secrets, Compose `env_file` on the host, or secret managers. Rotate LiteLLM and LangSmith keys independently. Use read-only filesystems where possible and separate keys per environment.

### Hard
**Q: Describe a zero-downtime deploy for a RAG API with Chroma and Redis cache.**

> Run blue/green or rolling updates on **stateless API** containers only; keep Chroma and Redis on persistent volumes. Bump `KB_VERSION` and run `ingest` job before switching traffic if schema changed. Warm Redis optional; expect cache miss spike. Use health checks that verify DB + Redis + Chroma connectivity. Drain connections on old API tasks; ensure SSE/WebSocket clients reconnect. Monitor LangSmith error rate and p95 latency during cutover.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Multi-stage build** | Slim runtime image without build tools |
| **Compose services** | API, Postgres, Redis, Chroma as separate containers |
| **Health checks** | Start API only when dependencies are ready |
| **Profiles** | On-demand ingest / batch jobs |
| **Env-driven config** | LiteLLM, LangSmith, JWT via environment |
| **Prod overlay** | Limits, logging, hardening without duplicating base file |

---

## Exercises

1. **Health depth:** Extend `/health` to report Redis ping, Postgres `SELECT 1`, and Chroma heartbeat.
2. **Ingest profile:** Run `docker compose --profile jobs run --rm ingest` after adding PDFs to `./data`.
3. **Prod overlay:** Add `read_only: true` to API and fix any write paths to `/tmp` or volumes.
4. **CI:** Add a compose-smoke job that curls `/health` after `docker compose up -d`.

---

## What's Next

You can run the stack locally and in staging. The **Enterprise AI Platform capstone** combines everything — multi-agent supervisor, hybrid RAG, JWT auth, SSE streaming, LangSmith, Pytest, and GitHub Actions — into one portfolio-grade project.

---

> [← Previous: Cost & Caching](chapter-77-cost-caching.md) | [Next: Enterprise Capstone →](../phase-18-capstone/chapter-79-enterprise-ai-platform.md)

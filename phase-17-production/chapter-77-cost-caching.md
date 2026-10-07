# Chapter 17.4: Cost Optimization & Caching

> **Phase 17 — Production & Deployment** | [← Previous: Observability](chapter-76-observability.md) | [Next: Docker Deploy →](chapter-78-docker-deploy.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Break down LLM cost drivers (model, tokens, retries, agents)
- ✅ Implement **Redis-backed caching** for LLM and embedding calls
- ✅ Use LangChain **LLM cache** and **semantic cache** patterns
- ✅ Cache **RAG responses** safely with TTL and cache keys
- ✅ Enforce budgets, quotas, and model routing via LiteLLM proxy settings
- ✅ Measure cache hit rate and cost savings in observability tools

| | |
|---|---|
| **Prerequisites** | Chapter 17.3 (Observability) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction — The Cost Iceberg

API bills are not just "input + output tokens." In production:

```
VISIBLE COST:
  └── Chat completions ($/1M tokens)

HIDDEN COST:
  ├── Embedding re-indexing on every deploy mistake
  ├── Agent loops (5–15 LLM calls per user message)
  ├── Retries after timeouts (duplicate charges)
  ├── Evaluation runs (nightly jobs × 500 examples)
  └── Duplicate questions (support desk asks the same FAQ 40×/day)
```

**Caching** attacks duplicate work. **Routing** sends easy queries to cheaper models. **Budgets** stop runaway spend.

---

## Part 1: Cost Model & LiteLLM Proxy

Centralize models behind a proxy so keys, budgets, and routing live in one place:

```bash
# .env — consumed by LangChain OpenAI-compatible clients
LITELLM_PROXY_API_BASE=https://litellm.yourcompany.com/v1
LITELLM_PROXY_API_KEY=sk-proxy-...

# Optional: default cheap model for routing
DEFAULT_CHAT_MODEL=gpt-4o-mini
PREMIUM_CHAT_MODEL=gpt-4o
```

```python
import os
from langchain_openai import ChatOpenAI

def get_llm(tier: str = "default"):
    model = (
        os.getenv("PREMIUM_CHAT_MODEL", "gpt-4o")
        if tier == "premium"
        else os.getenv("DEFAULT_CHAT_MODEL", "gpt-4o-mini")
    )
    return ChatOpenAI(
        model=model,
        temperature=0,
        openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
        openai_api_base=os.getenv("LITELLM_PROXY_API_BASE"),
        max_retries=2,
        request_timeout=45,
    )
```

Track spend per request in callbacks (from Chapter 17.3) and aggregate by `user_id`, `route`, and `model`.

---

## Part 2: Exact-Match LLM Cache (Redis)

LangChain supports a global LLM cache. **Redis** survives restarts and works across multiple API replicas.

```python
# app/cache/setup.py
import os
from langchain.globals import set_llm_cache
from langchain_community.cache import RedisCache
import redis


def init_llm_cache() -> None:
    client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        password=os.getenv("REDIS_PASSWORD") or None,
        decode_responses=False,
    )
    set_llm_cache(RedisCache(redis_=client))
```

Call `init_llm_cache()` during FastAPI lifespan **before** creating LLM instances.

```python
# app/main.py — lifespan excerpt
from app.cache.setup import init_llm_cache

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_llm_cache()
    ...
    yield
```

**What gets cached:** identical prompt + model parameters → same completion. Great for FAQs, classification, and deterministic extraction.

**What does NOT cache well:** creative writing, high temperature, or prompts with timestamps.

### Cache Key Design for RAG

Include everything that affects the answer:

```python
import hashlib
import json

def rag_cache_key(question: str, collection: str, k: int, model: str) -> str:
    payload = {"q": question.strip().lower(), "c": collection, "k": k, "m": model}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"rag:v1:{digest}"
```

```python
# app/services/rag_cache.py
import json
import redis

class RagResponseCache:
    def __init__(self, client: redis.Redis, ttl_seconds: int = 3600):
        self.client = client
        self.ttl = ttl_seconds

    def get(self, key: str) -> dict | None:
        raw = self.client.get(key)
        return json.loads(raw) if raw else None

    def set(self, key: str, value: dict) -> None:
        self.client.setex(key, self.ttl, json.dumps(value))
```

Wire into the route:

```python
@app.post("/v1/ask")
async def ask(body: QuestionRequest):
    key = rag_cache_key(body.question, "knowledge_base", k=5, model="gpt-4o-mini")
    cached = rag_cache.get(key)
    if cached:
        cached["cache_hit"] = True
        return cached

    result = await rag_chain.ainvoke(body.question)
    response = {"answer": result, "cache_hit": False}
    rag_cache.set(key, response)
    return response
```

---

## Part 3: Semantic Cache (Near-Duplicate Questions)

Exact match misses paraphrases:

```
"What's our refund policy?"
"How do refunds work?"
"Can I get my money back?"
```

**Semantic cache:** embed the question, search Redis vector index (or store embeddings in Redis + brute force for small scale), return cached answer if cosine similarity > threshold.

```python
# Simplified pattern — production often uses Redis Stack vector search or a dedicated store
from langchain_openai import OpenAIEmbeddings
import numpy as np

embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE"),
)

async def semantic_lookup(question: str, store: list[dict], threshold: float = 0.92):
    q_vec = np.array(await embeddings.aembed_query(question))
    best = None
    best_score = -1.0
    for row in store:
        score = float(np.dot(q_vec, row["vector"]) / (np.linalg.norm(q_vec) * np.linalg.norm(row["vector"])))
        if score > best_score:
            best_score, best = score, row
    if best and best_score >= threshold:
        return best["answer"], best_score
    return None, best_score
```

**Guardrails:**

- Never semantic-cache **personalized** answers (user-specific data in context).
- Invalidate when the knowledge base version changes (`kb_version` in cache metadata).
- Log similarity score when serving from semantic cache for audit.

---

## Part 4: Embedding & Retrieval Caching

Embeddings are cheaper than chat but add up at index time:

| Layer | Cache what | TTL |
|-------|------------|-----|
| Query embedding | `hash(text)` → vector | 24h |
| Retriever results | `(query_hash, filter, k)` → doc IDs | 15–60m |
| Full RAG answer | `rag_cache_key` | 1–6h |

```python
async def embed_query_cached(text: str, redis_client) -> list[float]:
    key = f"emb:{hashlib.sha256(text.encode()).hexdigest()}"
    hit = redis_client.get(key)
    if hit:
        return json.loads(hit)
    vec = await embeddings.aembed_query(text)
    redis_client.setex(key, 86400, json.dumps(vec))
    return vec
```

When documents update, bump `collection_version` in keys — stale retrieval cache is worse than a cache miss.

---

## Part 5: Budgets, Quotas & Degradation

```python
from fastapi import HTTPException

class BudgetGuard:
    def __init__(self, redis_client, daily_limit_usd: float):
        self.redis = redis_client
        self.limit = daily_limit_usd

    def check_and_add(self, user_id: str, cost_usd: float) -> None:
        key = f"budget:{user_id}:{date.today().isoformat()}"
        pipe = self.redis.pipeline()
        pipe.incrbyfloat(key, cost_usd)
        pipe.expire(key, 86400 * 2)
        total, = pipe.execute()[0:1]
        if float(total) > self.limit:
            raise HTTPException(status_code=429, detail="Daily budget exceeded")

    @staticmethod
    def degrade_model(tier: str) -> str:
        return "gpt-4o-mini" if tier != "premium" else "gpt-4o"
```

**Degradation ladder under load or budget pressure:**

1. Serve from exact cache  
2. Serve from semantic cache (with disclaimer if needed)  
3. Switch to smaller model  
4. Return 429 with retry-after  

---

## Part 6: Observability for Cache & Cost

Emit metrics your dashboard can chart:

```python
# In ProductionCallbackHandler.on_llm_end
metrics = {
    "cache_hit": getattr(response, "cache_hit", False),
    "prompt_tokens": token_usage.get("prompt_tokens", 0),
    "completion_tokens": token_usage.get("completion_tokens", 0),
    "model": serialized.get("kwargs", {}).get("model_name"),
}
logger.info("llm_call", extra=metrics)
```

LangSmith: tag runs with `cache_hit:true|false` for A/B comparisons.

**Key KPIs:**

- Cache hit rate (exact + semantic)
- Cost per successful request
- Tokens per request (p50 / p95)
- Budget utilization per tenant

---

## Common Mistakes

### Mistake 1: Caching personalized RAG
```python
# ❌ Same key for all users — user A sees user B's data
key = hash(question)

# ✅ Include tenant_id, auth scope, and kb_version in the key
```

### Mistake 2: Infinite TTL on policy answers
```python
# ❌ Legal/compliance docs change; cache serves outdated policy

# ✅ Short TTL + invalidate on document publish webhook
```

### Mistake 3: Caching before validation
```python
# ❌ Cache toxic or oversized prompts after bypassing validators

# ✅ Validate → then cache only successful, moderated responses
```

### Mistake 4: Ignoring agent multi-call cost
```python
# ❌ Cache only the final message — agent still burns 8 LLM calls

# ✅ Cache tool results, sub-agent outputs, and retrieval separately
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Version cache keys (`v1`, `kb_version`) | Safe invalidation on deploy |
| Separate caches for chat, embed, RAG | Different TTLs and key shapes |
| Track hit rate and $ saved | Prove ROI; tune TTL/threshold |
| Use Redis for multi-instance APIs | Shared cache across containers |
| Route easy tasks to mini models | 10× cost difference adds up |
| Cap retries | Retries double spend on failures |
| Nightly cost reports by team/user | Catch anomalies early |

---

## Interview Preparation

### Easy
**Q: What is the difference between exact and semantic caching for LLMs?**

> **Exact cache** returns a stored response when the prompt and relevant parameters match byte-for-byte (after normalization). **Semantic cache** compares **embedding similarity** between queries and returns a prior answer when paraphrases exceed a similarity threshold. Exact is simpler and safer; semantic improves hit rate but needs stricter privacy and freshness controls.

### Medium
**Q: How do you invalidate RAG caches when documents change?**

> Include a **knowledge base version** or content hash in every cache key (embeddings, retrieval, final answer). On ingest or delete, bump the version or publish invalidation events that delete key prefixes (`rag:v1:{old_version}:*`). Use shorter TTLs for high-churn collections and webhook-driven purge on document updates.

### Hard
**Q: Design a multi-tenant cost control system for an LLM API.**

> Per-tenant Redis counters for daily spend and request rate; hard stops at budget with 429. LiteLLM proxy enforces model allowlists and per-key budgets. Callbacks record token usage to Postgres for billing. Cache layers (exact + semantic) keyed by `tenant_id`. Degrade path: cache → cheaper model → queue async job. Alert on anomaly detection (3× baseline spend in 1 hour). LangSmith dashboards per tenant for support escalations.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **LiteLLM proxy** | Central routing, keys, and model policy |
| **Redis LLM cache** | Exact-match completion cache across replicas |
| **RAG response cache** | Hash of question + retrieval config + model |
| **Semantic cache** | Embedding similarity for paraphrase hits |
| **Budget guard** | Per-user/tenant daily spend limits |
| **Cache invalidation** | Version keys when KB or prompts change |

---

## Exercises

1. **Hit rate dashboard:** Log `cache_hit` for 100 requests; compute hit rate and estimated $ saved.
2. **KB version bump:** Add `KB_VERSION` env var to all RAG cache keys; simulate a deploy that invalidates old entries.
3. **Semantic threshold:** Test thresholds 0.88–0.95 on 20 paraphrased FAQ questions; pick one with zero false positives.
4. **Budget 429:** Implement `BudgetGuard` with Redis `INCRBYFLOAT` and return structured error JSON.

---

## What's Next

Caching and cost controls run in **containers** alongside Redis, Postgres, and your vector store. Next you'll package the full stack with **Docker and Docker Compose** for reproducible deployments.

---

> [← Previous: Observability](chapter-76-observability.md) | [Next: Docker Deploy →](chapter-78-docker-deploy.md)

# Chapter 8.3: Chat Message History (In-memory, Redis, PostgreSQL)

> **Phase 8 — Memory Systems** | [← Previous: Memory Strategies](chapter-33-memory-strategies.md) | [Next: Memory in LCEL →](chapter-35-memory-in-lcel.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Use **`InMemoryChatMessageHistory`** for development
- ✅ Persist sessions with **`RedisChatMessageHistory`**
- ✅ Store transcripts in **PostgreSQL** via LangChain SQL history
- ✅ Swap backends by changing **`get_session_history`** only
- ✅ Apply production concerns: TTL, namespacing, encryption

| | |
|---|---|
| **Prerequisites** | Chapters 8.1–8.2, basic Redis/SQL familiarity |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

### The Problem

```python
session_store = {}  # ❌ dies on deploy, not shared across workers
```

When you run **two FastAPI workers**, user requests hit different processes — in-memory dicts diverge. Restart clears all chats.

### The Solution

**Chat message history** backends implement the same interface: `add_user_message`, `add_ai_message`, `messages`, `clear`. LangChain's **`RunnableWithMessageHistory`** (Chapter 8.4) accepts a factory `get_session_history(session_id)` — swap implementation without changing the chain.

```
                    get_session_history(session_id)
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
 InMemoryChatMessageHistory   RedisChatMessageHistory    PostgresChatMessageHistory
   (laptop dev)                  (fast, TTL)                 (durable, SQL)
```

---

## Part 1: In-Memory — Development & Tests

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]

_store: dict[str, InMemoryChatMessageHistory] = {}

hist = get_session_history("user-42")
hist.add_user_message("Hello")
hist.add_ai_message("Hi there!")
print(hist.messages)
```

Use for unit tests and local notebooks only.

---

## Part 2: Redis — Production Cache Pattern

Install: `pip install redis langchain-community`

Run Redis locally: `docker run -d -p 6379:6379 redis:7`

```python
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        key_prefix="chat:",  # namespace keys
        ttl=60 * 60 * 24 * 7,  # optional: 7 days — if supported by version
    )
```

```
FastAPI Worker 1 ──┐
                   ├──► Redis  chat:user-42  ──► [msg, msg, msg]
FastAPI Worker 2 ──┘
```

**Why Redis:** low latency, built-in TTL, horizontal scaling, pub/sub if you add live updates later.

### Environment

```env
REDIS_URL=redis://localhost:6379/0
```

---

## Part 3: PostgreSQL — Durable Audit Trail

For compliance, analytics, and joins with user tables:

`pip install psycopg2-binary` or `psycopg[binary]`

```python
from langchain_community.chat_message_histories import PostgresChatMessageHistory

CONNECTION = os.getenv(
    "CHAT_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/chatdb",
)

def get_postgres_history(session_id: str) -> PostgresChatMessageHistory:
    return PostgresChatMessageHistory(
        connection_string=CONNECTION,
        session_id=session_id,
        table_name="chat_message_history",
    )
```

LangChain creates/uses a table storing serialized messages. **Backup** this table with your normal DB policy.

### SQLite for small deployments

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_sqlite_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///chat_history.db",
    )
```

Good for single-node apps; not ideal for high concurrent writes.

---

## Part 4: Same Chain, Different Backend

Preview integration with LCEL (details in 8.4):

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

BACKEND = os.getenv("CHAT_HISTORY_BACKEND", "memory")

def get_session_history(session_id: str):
    if BACKEND == "redis":
        from langchain_community.chat_message_histories import RedisChatMessageHistory
        return RedisChatMessageHistory(session_id=session_id, url=os.getenv("REDIS_URL"))
    if BACKEND == "postgres":
        from langchain_community.chat_message_histories import PostgresChatMessageHistory
        return PostgresChatMessageHistory(
            session_id=session_id,
            connection_string=os.getenv("CHAT_DATABASE_URL"),
        )
    from langchain_core.chat_history import InMemoryChatMessageHistory
    return get_in_memory(session_id)

_in_mem: dict = {}

def get_in_memory(session_id: str):
    if session_id not in _in_mem:
        _in_mem[session_id] = InMemoryChatMessageHistory()
    return _in_mem[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

config = {"configurable": {"session_id": "demo-session"}}
print(chain_with_history.invoke({"input": "Remember: favorite color is teal."}, config=config))
print(chain_with_history.invoke({"input": "What is my favorite color?"}, config=config))
```

Only **`get_session_history`** and env vars change between dev and prod.

---

## Part 5: Session ID Design

| Pattern | Example | Notes |
|---------|---------|-------|
| Authenticated user | `user_{uuid}` | Stable across devices if logged in |
| Anonymous browser | `anon_{cookie_uuid}` | Rotate on logout |
| Multi-tenant SaaS | `{tenant_id}:{user_id}` | Prevents cross-tenant leakage |
| Support ticket | `ticket_{id}` | One history per case |

**Never** accept raw client `session_id` without authZ — attackers could guess IDs.

---

## Part 6: Operations Checklist

```
┌─────────────────────────────────────────────────────────┐
│ □ TTL or retention job for old sessions                 │
│ □ Encrypt REDIS/DB credentials via secrets manager      │
│ □ PII scrubbing before long-term storage                │
│ □ Rate limit messages per session                       │
│ □ Monitor memory/key size (Redis) and table growth (SQL)│
└─────────────────────────────────────────────────────────┘
```

Apply **trim/summary** (Chapter 8.2) on read or before invoke — persistent store can hold more than you send to the LLM.

---

## Common Mistakes

### Mistake 1: Using in-memory store behind load balancer

Users see "random amnesia" when requests hit different workers.

### Mistake 2: Reusing `session_id` across users

Always bind session to auth context.

### Mistake 3: Storing secrets in chat history

Users paste API keys — scan/redact before persistence if required by policy.

### Mistake 4: No migration plan for history schema

Version your message table or use LangChain defaults consistently across deploys.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Redis for hot sessions, Postgres for archive | Cost + durability |
| Namespace Redis keys | Multi-env on same cluster |
| Separate read replica for analytics | Don't slow chat path |
| Feature-flag backend via env | Local vs prod parity |
| Test failover (Redis down → graceful error) | User trust |

---

## Interview Preparation

### Easy
**Q: Why persist chat history outside the application process?**

> Multi-worker deployments and restarts require a **shared store** (Redis, SQL). In-memory dicts are neither shared nor durable.

### Medium
**Q: How do you swap memory backends in LangChain without rewriting the chain?**

> Implement `get_session_history(session_id)` returning a `BaseChatMessageHistory`. Pass it to `RunnableWithMessageHistory`. The LCEL chain (`prompt | llm | parser`) stays unchanged.

### Hard
**Q: Redis vs PostgreSQL for chat history?**

> **Redis:** sub-ms reads, TTL, ideal for active sessions, limited querying. **PostgreSQL:** durable, SQL analytics, joins with users, slower than Redis. Many systems write **hot path to Redis** and **async archive to Postgres**.

### Senior
**Q: GDPR delete request for one user — what do you touch?**

> Delete or anonymize rows/keys for all `session_id`s mapped to that user across Redis, Postgres, object storage (attachments), and log aggregators. Maintain mapping table `user_id → session_ids`. Propagate deletion to vector stores if transcripts were embedded. Document latency SLA for erasure.

---

## Summary

| Backend | Durability | Speed | Typical use |
|---------|------------|-------|-------------|
| In-memory | None | Fastest | Tests, local dev |
| Redis | Configurable (TTL) | Very fast | Active sessions |
| PostgreSQL | Full | Moderate | Audit, compliance |
| SQLite | File | Moderate | Single-node apps |

---

## Hands-on Exercise

Run Redis in Docker. Create two Python processes (two shells) calling the same `session_id` through `RedisChatMessageHistory` — verify process B sees messages added by process A.

---

## Challenge Project

Add a **`CHAT_HISTORY_BACKEND`** env switch to a small FastAPI `/chat` endpoint using `RunnableWithMessageHistory`. Document required env vars for `memory`, `redis`, and `postgres` modes.

---

## What's Next

Chapter 8.4 wires everything together: **`RunnableWithMessageHistory`**, `MessagesPlaceholder`, multi-session configs, and token trimming in LCEL.

> [← Previous: Memory Strategies](chapter-33-memory-strategies.md) | [Next: Memory in LCEL →](chapter-35-memory-in-lcel.md)

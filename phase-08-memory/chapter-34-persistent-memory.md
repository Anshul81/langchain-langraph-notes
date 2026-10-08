# Chapter 8.3: Chat Message History (In-memory, Redis, PostgreSQL)

> **Phase 8 — Memory Systems** | [← Previous: Memory Strategies](chapter-33-memory-strategies.md) | [Next: Memory in LCEL →](chapter-35-memory-in-lcel.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Explain **why a process-local dict fails** in any real deployment (multi-worker, restarts, autoscaling)
- ✅ Use the `BaseChatMessageHistory` contract and know exactly what every backend must implement
- ✅ Persist chats with **Redis** (TTL, `key_prefix`), **SQL** (SQLite/PostgreSQL) and **MongoDB**
- ✅ Switch backends with **one env var** (`CHAT_HISTORY_BACKEND`) and a single `get_session_history`
- ✅ Design **safe session IDs** (auth-bound, multi-tenant, ticket-based)
- ✅ Write a **custom history class** (DIY SQLite) and wrap histories with decorators (redaction, tiering)
- ✅ Apply ops hygiene: TTL, encryption, GDPR erasure, secret redaction, hot (Redis) + cold (Postgres) storage

| | |
|---|---|
| **Prerequisites** | Chapters 8.1–8.2 (why memory, buffer/window/summary), basic Redis/SQL familiarity, Docker installed |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

> **Scope note:** 8.2 taught *what to send to the LLM* (trim/summarize). This chapter is about *where the full transcript lives*. Chapter 8.4 wires either of them into LCEL. Keep the two ideas separate: **storage ≠ context window**.

---

## Introduction

### The Problem

Every tutorial chatbot starts like this:

```python
session_store = {}   # ❌ module-level dict: "memory" for the demo, amnesia in production
```

It works on your laptop with one process. In production, three things break it at once:

```
                    ┌────────────────────────────────────────────┐
   User turn 1 ───► │  Load Balancer                             │
   User turn 2 ───► │  (round-robin)                             │
                    └───────┬────────────────────┬───────────────┘
                            │                    │
                            ▼                    ▼
                  ┌──────────────────┐  ┌──────────────────┐
                  │ Worker A (pid 11)│  │ Worker B (pid 12)│
                  │ store = {        │  │ store = {        │
                  │  "u1": [t1]      │  │  }   ← EMPTY!    │
                  │ }                │  │                  │
                  └──────────────────┘  └──────────────────┘

   Turn 2 hits Worker B → "Sorry, what's your name again?"   (random amnesia)

   Deploy / crash / autoscale-down → Worker A dies → store gone forever
   Memory leak: dict grows with every session, never evicted → OOM
```

| Failure | Cause | Symptom users see |
|---------|-------|-------------------|
| **Multi-worker divergence** | Each process owns its own dict | Bot "forgets" randomly |
| **Restart / deploy loss** | RAM is not durable | All conversations reset after release |
| **Unbounded growth** | No TTL or eviction | Slow death by OOM |
| **No audit / analytics** | Nothing queryable | Compliance asks "what did the bot say?" — you can't answer |

### The Solution

LangChain defines **one small interface** — `BaseChatMessageHistory` — and many backends implement it. Your chain never knows which one it talks to:

```
                      get_session_history(session_id)
                                   │
        ┌───────────┬──────────────┼───────────────┬─────────────┐
        ▼           ▼              ▼               ▼             ▼
   InMemory      Redis        SQL (SQLite/     MongoDB       Custom
   (tests)    (hot, TTL)      Postgres)       (documents)   (your class)
        │           │              │               │             │
        └───────────┴──────┬───────┴───────────────┴─────────────┘
                           ▼
               BaseChatMessageHistory
        .messages   .add_messages()   .clear()
```

### History

Early LangChain bundled memory *inside* chain classes (`ConversationChain` + `ConversationBufferMemory`), tying storage to a specific chain. Storage and logic were fused, so every new database meant a new memory class. The v0.1–0.2 refactor split them: **chat message history** (pure storage, this chapter) and **`RunnableWithMessageHistory`** (wiring, next chapter). Backends moved to `langchain-community` and partner packages (`langchain-redis`, `langchain-postgres`, `langchain-mongodb`).

### Industry Usage

- **Support copilots** keep active tickets in Redis (fast, expiring) and archive closed ones to Postgres for QA and compliance.
- **Enterprise assistants** use SQL because auditors need `SELECT ... WHERE user_id = ...`.
- **Consumer apps** use Redis/Mongo with aggressive TTLs to limit privacy exposure and cost.
- **Regulated industries** add encryption at rest, redaction, and per-user erasure pipelines (GDPR/DPDP).

### Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "Persisting history means the LLM sees all of it" | Storage and prompt are separate. You store everything, but **send** a trimmed view (Ch 8.2/8.4) |
| "Redis is just a cache, so it's unsafe for chats" | It's fine as the **hot path**; durability depends on config (AOF/RDB) — and TTL *is* a feature, not a bug |
| "Postgres history = automatically GDPR compliant" | Compliance is a *process* (mapping, erasure, retention), not a column type |
| "`session_id` is just a string, any value is OK" | It's your **tenancy boundary**. A guessable or client-supplied ID is a data leak |
| "Swapping backends needs chain changes" | Only `get_session_history` changes — that is the whole point of the abstraction |

---

## Mental Model

### Analogy: The Hospital Records Room

- **Patient chart on the nurse's clipboard** = **Redis**: instant access, only for current patients, shredded after discharge (TTL).
- **Central records archive** = **PostgreSQL**: slower to fetch, but permanent, indexed, auditable.
- **Sticky note in one nurse's pocket** = **in-process dict**: the nurse goes home, the note goes with her.
- **Chart ID (MRN)** = **`session_id`**: mix two up and you've leaked a patient's records.

### The Contract Every Backend Honors

```
┌───────────────────────────────────────────────────────────────┐
│  BaseChatMessageHistory  (langchain_core.chat_history)        │
│                                                               │
│   messages  (property) → list[BaseMessage]   # READ  all      │
│   add_messages(msgs)   → None                # APPEND batch   │
│   clear()              → None                # DELETE session │
│                                                               │
│  Helpers (free):  add_user_message(str)  add_ai_message(str)  │
│                   add_message(msg)                            │
│  Async (free):    aget_messages()  aadd_messages()  aclear()  │
│                   (default = thread-pool wrapper over sync)   │
└───────────────────────────────────────────────────────────────┘
```

Three methods. If you can read, append, and delete a list of messages for a key, you can build a backend.

---

## Theory

### Part 1: `InMemoryChatMessageHistory` — Tests and Notebooks Only

```python
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

_MEMORY: dict[str, InMemoryChatMessageHistory] = {}

def get_memory_history(session_id: str) -> BaseChatMessageHistory:
    # The dict holds *history objects*, not messages. The factory is
    # called on EVERY request, so it must return the SAME object per session.
    if session_id not in _MEMORY:
        _MEMORY[session_id] = InMemoryChatMessageHistory()
    return _MEMORY[session_id]

h = get_memory_history("user-42")
h.add_user_message("Hello")
h.add_ai_message("Hi there!")
print([(m.type, m.content) for m in h.messages])
# [('human', 'Hello'), ('ai', 'Hi there!')]
```

**Use it for:** unit tests (zero infra), notebooks, CI, and as a **fallback** when Redis is down in non-critical contexts.
**Never use it for:** anything behind more than one process, or anything that must survive a deploy.

> Why the `dict` is still here: `InMemoryChatMessageHistory()` constructed fresh each call would be *empty each call*. The factory must memoize. Persistent backends memoize implicitly — the *database* is the memo.

### Part 2: Redis — The Hot Path

```bash
# Bash / PowerShell — one-liner Redis for development
docker run -d --name chat-redis -p 6379:6379 redis:7
docker exec -it chat-redis redis-cli PING     # → PONG
```

```bash
pip install redis langchain-community
```

```python
import os
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        key_prefix="chat:",          # final key = "chat:" + session_id
        ttl=60 * 60 * 24 * 7,        # 7 days; refreshed on each write (sliding window)
    )
```

| Parameter | Meaning | Gotcha |
|-----------|---------|--------|
| `session_id` | Suffix of the Redis key | Must be unique **and** authorization-bound |
| `url` | `redis://`, `rediss://` (TLS), with `user:pass@` | Use `rediss://` over any network you don't own |
| `key_prefix` | Namespace (`chat:`, `prod:chat:`) | Default is `message_store:` — set your own so envs/tenants don't collide |
| `ttl` | Seconds until key expires; `None` = forever | Sliding: each write resets the clock |

Messages are stored as JSON strings in a Redis **list** under one key per session:

```
 Redis
 ┌──────────────────────────────────────────────────────────────┐
 │ key: "chat:acme:user-42:c1"        type: list     TTL: 604800│
 │   [ {"type":"human","data":{"content":"Hi"...}},             │
 │     {"type":"ai","data":{"content":"Hello!"...}}, ... ]     │
 └──────────────────────────────────────────────────────────────┘
        ▲                                  ▲
        │ Worker A (add)                   │ Worker B (read)
```

**Multi-worker demo concept:** run `uvicorn app:app --workers 4`. Four OS processes, four separate Python heaps — yet every worker's `get_redis_history("u1")` reads the **same key**. That is the entire reason Redis fixes the load-balancer problem. You'll prove it in the hands-on section with two shells.

> Newer option: the `langchain-redis` partner package ships its own `RedisChatMessageHistory` (parameter `redis_url=` instead of `url=`). The contract is identical; check which package your version of the course environment installs.

**Durability note:** Redis is RAM-first. Enable AOF (`--appendonly yes`) if you cannot lose active chats on a Redis restart, or accept the loss and rely on the cold archive (Part 9).

### Part 3: SQL — `SQLChatMessageHistory` (SQLite and PostgreSQL)

One class, any SQLAlchemy URL. This is the most portable "durable" option.

```bash
pip install sqlalchemy langchain-community
pip install psycopg2-binary            # for Postgres only
docker run -d --name chat-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=chatdb \
           -p 5432:5432 postgres:16
```

```python
import os
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_sql_history(session_id: str) -> SQLChatMessageHistory:
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=os.getenv("CHAT_DATABASE_URL", "sqlite:///chat_history.db"),
        table_name="chat_message_history",
    )

# SQLite : sqlite:///chat_history.db
# Postgres: postgresql+psycopg2://postgres:postgres@localhost:5432/chatdb
```

> Older versions of `langchain-community` name this argument `connection_string=`. If you see a deprecation warning or `TypeError`, switch the keyword — the URL value is the same.

The table is created automatically. Conceptually:

```
 message_store / chat_message_history
 ┌────┬─────────────────────┬──────────────────────────────────────────┐
 │ id │ session_id (indexed)│ message (JSON text)                      │
 ├────┼─────────────────────┼──────────────────────────────────────────┤
 │ 1  │ acme:user-42:c1     │ {"type":"human","data":{"content":"Hi"}} │
 │ 2  │ acme:user-42:c1     │ {"type":"ai","data":{"content":"Hello!"}}│
 └────┴─────────────────────┴──────────────────────────────────────────┘
```

Because `session_id` is a real column you can run: `DELETE FROM chat_message_history WHERE session_id LIKE 'acme:user-42:%'` — GDPR erasure becomes one statement. That is the real selling point over Redis.

**Async mode** (for FastAPI): pass `async_mode=True` with an async driver, then use `await h.aget_messages()` / `await h.aadd_messages(...)`:

```python
SQLChatMessageHistory(
    session_id=sid,
    connection="postgresql+asyncpg://postgres:postgres@localhost:5432/chatdb",
    async_mode=True,
)
```

**SQLite caveat:** a single writer lock. Great for a single-node app or an internal tool; a poor choice for high concurrent write traffic.

#### The native Postgres package: `langchain-postgres`

```python
import uuid, psycopg
from langchain_postgres import PostgresChatMessageHistory

conn = psycopg.connect("postgresql://postgres:postgres@localhost:5432/chatdb")
PostgresChatMessageHistory.create_tables(conn, "chat_history")   # run once (migration step)

# ⚠️ This implementation REQUIRES a UUID session_id
sid = str(uuid.uuid5(uuid.NAMESPACE_URL, "acme:user-42:c1"))     # deterministic from your own key
history = PostgresChatMessageHistory("chat_history", sid, sync_connection=conn)
```

| Option | Pick when |
|--------|-----------|
| `SQLChatMessageHistory` | You want one class for SQLite (dev) → Postgres (prod), free-form string session IDs |
| `langchain_postgres.PostgresChatMessageHistory` | You are already on `psycopg` 3 / pgvector and want an explicit `create_tables` migration step |

The `uuid5` trick above gives you a **stable UUID derived from your human-readable key**, so the same user/conversation always maps to the same rows.

### Part 4: MongoDB — Document Store Pattern

Useful when chats carry extra metadata (feedback, attachments, per-message tags) and your stack is already Mongo.

```bash
pip install pymongo langchain-community
docker run -d --name chat-mongo -p 27017:27017 mongo:7
```

```python
import os
from langchain_community.chat_message_histories import MongoDBChatMessageHistory

def get_mongo_history(session_id: str) -> MongoDBChatMessageHistory:
    return MongoDBChatMessageHistory(
        connection_string=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        session_id=session_id,
        database_name="chatdb",
        collection_name="chat_histories",
    )
```

Each message becomes a document `{SessionId: "...", History: "<json>"}`. Add an index on `SessionId` (`db.chat_histories.createIndex({SessionId: 1})`) or reads become full collection scans. The partner package `langchain-mongodb` offers the same class with Atlas-oriented options.

### Part 5: One Switch — Backend Selection by Environment

This is the pattern the chapter-end project uses. **Only this file knows about backends.**

```python
# history_backends.py
import os
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

BACKEND = os.getenv("CHAT_HISTORY_BACKEND", "memory").lower()
if BACKEND not in {"memory", "redis", "sqlite", "postgres", "mongo"}:
    raise ValueError(f"Unknown CHAT_HISTORY_BACKEND={BACKEND!r}")      # fail fast at boot

_MEMORY: dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if BACKEND == "redis":
        from langchain_community.chat_message_histories import RedisChatMessageHistory
        return RedisChatMessageHistory(
            session_id, url=os.environ["REDIS_URL"], key_prefix="chat:", ttl=7 * 24 * 3600
        )
    if BACKEND in {"sqlite", "postgres"}:
        from langchain_community.chat_message_histories import SQLChatMessageHistory
        default = "sqlite:///chat_history.db"
        url = os.getenv("CHAT_DATABASE_URL", default if BACKEND == "sqlite" else "")
        if not url:
            raise RuntimeError("CHAT_DATABASE_URL is required for the postgres backend")
        return SQLChatMessageHistory(session_id=session_id, connection=url, table_name="chat_message_history")
    if BACKEND == "mongo":
        from langchain_community.chat_message_histories import MongoDBChatMessageHistory
        return MongoDBChatMessageHistory(
            connection_string=os.environ["MONGODB_URI"], session_id=session_id,
            database_name="chatdb", collection_name="chat_histories",
        )
    return _MEMORY.setdefault(session_id, InMemoryChatMessageHistory())
```

Lazy imports keep `pip install` surface small: a dev with `memory` mode never needs `redis` or `psycopg2`.

```env
# .env — pick ONE block
CHAT_HISTORY_BACKEND=memory

# CHAT_HISTORY_BACKEND=redis
# REDIS_URL=redis://localhost:6379/0

# CHAT_HISTORY_BACKEND=postgres
# CHAT_DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/chatdb
```

### Part 6: Session ID Design — Your Tenancy Boundary

| Pattern | Example | Notes |
|---------|---------|-------|
| Authenticated user + conversation | `user_{uid}:{conv_uuid}` | Multiple threads per user (like ChatGPT's sidebar) |
| Anonymous browser | `anon_{signed_cookie_uuid}` | Rotate on login; merge or discard |
| Multi-tenant SaaS | `{tenant}:{user}:{conv}` | Prefix scan = per-tenant export/erase |
| Support ticket | `ticket_{id}` | History belongs to the **case**, agents come and go |
| Per-channel bot | `slack_{team}_{channel}_{thread_ts}` | Thread-scoped context |

**Rules:**

1. **Derive the ID on the server** from the authenticated principal. Accept only the *conversation* part from the client.
2. **Validate** components (no `:`/`*`/whitespace) — otherwise `conv="x:*"` becomes a key-injection or wildcard-delete bug.
3. **Check ownership** before reading a pre-existing conversation ID.
4. Use **UUIDs/ULIDs**, never sequential integers for the unguessable part.

```python
import re, uuid

_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

def build_session_id(tenant: str, user: str, conversation: str | None = None) -> str:
    for part in (tenant, user, *(c for c in [conversation] if c)):
        if not _SAFE.match(part):
            raise ValueError("invalid session id component")
    return f"{tenant}:{user}:{conversation or uuid.uuid4().hex}"
```

### Part 7: Custom History — Build Your Own (DIY SQLite)

When your storage is exotic (DynamoDB, S3, an internal service), subclass `BaseChatMessageHistory`. Implement the three members and serialize with `messages_to_dict` / `messages_from_dict`.

```python
import json, sqlite3
from contextlib import closing
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


class SQLiteChatMessageHistory(BaseChatMessageHistory):
    """Tiny DIY backend: one table, JSON per message."""

    def __init__(self, session_id: str, db_path: str = "diy_chat.db"):
        self.session_id, self.db_path = session_id, db_path
        with closing(sqlite3.connect(db_path)) as c, c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS msgs ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " session_id TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            c.execute("CREATE INDEX IF NOT EXISTS ix_msgs_sid ON msgs(session_id)")

    @property
    def messages(self) -> list[BaseMessage]:                     # READ
        with closing(sqlite3.connect(self.db_path)) as c:
            rows = c.execute(
                "SELECT payload FROM msgs WHERE session_id=? ORDER BY id", (self.session_id,)
            ).fetchall()
        return messages_from_dict([json.loads(r[0]) for r in rows])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:   # APPEND
        with closing(sqlite3.connect(self.db_path)) as c, c:         # `c` as ctx = transaction
            c.executemany(
                "INSERT INTO msgs (session_id, payload) VALUES (?, ?)",
                [(self.session_id, json.dumps(message_to_dict(m))) for m in messages],
            )

    def clear(self) -> None:                                      # DELETE
        with closing(sqlite3.connect(self.db_path)) as c, c:
            c.execute("DELETE FROM msgs WHERE session_id=?", (self.session_id,))
```

Test it with no LLM at all — this is how you unit-test any backend:

```python
h = SQLiteChatMessageHistory("t-1")
h.clear(); h.add_user_message("ping"); h.add_ai_message("pong")
assert [m.content for m in SQLiteChatMessageHistory("t-1").messages] == ["ping", "pong"]  # new instance, same data
```

### Part 8: Decorator Pattern — Redaction and Encryption

Users paste API keys, passwords, card numbers. If your policy says "never persist secrets", enforce it at the **storage boundary**, not in the prompt:

```python
import re
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9_-]{16,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"), "Bearer [REDACTED]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[REDACTED_CARD]"),
]

def redact(text: str) -> str:
    for rx, repl in _PATTERNS:
        text = rx.sub(repl, text)
    return text


class RedactingHistory(BaseChatMessageHistory):
    """Wrap ANY backend; scrub string content before it touches disk."""

    def __init__(self, inner: BaseChatMessageHistory):
        self.inner = inner

    @property
    def messages(self) -> list[BaseMessage]:
        return self.inner.messages

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        clean = [
            m.model_copy(update={"content": redact(m.content)}) if isinstance(m.content, str) else m
            for m in messages
        ]
        self.inner.add_messages(clean)

    def clear(self) -> None:
        self.inner.clear()
```

> Redaction at write time also protects your **future prompts**: the secret never re-enters the context window on later turns. (The *current* turn already reached the LLM provider — redact upstream with a guardrail if that matters too.)

**Encryption** layers:

| Layer | How |
|-------|-----|
| In transit | `rediss://`, `sslmode=require` in Postgres URL, TLS to Mongo |
| At rest | Managed-DB encryption / disk encryption (KMS) |
| Field-level | A decorator like above that Fernet-encrypts `content` on write and decrypts on read — needed if DBAs must not read chats |
| Secrets | Connection strings from a secrets manager, never committed |

### Part 9: Redis Hot Path + Postgres Archive

Active conversations need sub-millisecond reads; compliance needs permanence. Compose both behind **one** `BaseChatMessageHistory`:

```
  chain ──► TieredChatMessageHistory
              │
   read ──►   ├─ 1. Redis hit?  ──yes──► return (fast path)
              │        │no
              │        ▼
              │   2. Postgres read ──► rehydrate Redis ──► return
              │
   write ─►   ├─ Redis  (hot, TTL 24h)
              └─ Postgres (cold, forever / retention policy)
```

```python
from typing import Sequence
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class TieredChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, hot: BaseChatMessageHistory, cold: BaseChatMessageHistory):
        self.hot, self.cold = hot, cold

    @property
    def messages(self) -> list[BaseMessage]:
        msgs = self.hot.messages
        if not msgs:                          # cache miss or TTL expired
            msgs = self.cold.messages
            if msgs:
                self.hot.add_messages(msgs)   # rehydrate for the next turn
        return msgs

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self.hot.add_messages(messages)
        self.cold.add_messages(messages)      # write-through; see note

    def clear(self) -> None:
        self.hot.clear(); self.cold.clear()   # GDPR: BOTH tiers
```

> **Note:** synchronous write-through doubles write latency. In high-traffic systems push the cold write to a queue/background task (Celery, SQS, `asyncio.create_task`) and accept eventual consistency — Redis stays the source of truth for *active* sessions.

---

## Architecture

### Reference Production Layout

```
┌────────┐   auth token    ┌──────────────┐  session_id = tenant:user:conv
│ Client │ ───────────────►│  API Gateway │ ─────────────────────────────┐
└────────┘                 └──────────────┘                              ▼
                                                          ┌──────────────────────────┐
                                                          │ FastAPI workers (N)      │
                                                          │  RedactingHistory(       │
                                                          │   TieredChatMessage...)  │
                                                          └───────┬──────────┬───────┘
                                                      hot         │          │ cold (async)
                                                                  ▼          ▼
                                                        ┌───────────────┐ ┌────────────────┐
                                                        │ Redis (TTL)   │ │ PostgreSQL     │
                                                        │ chat:*        │ │ audit / export │
                                                        └───────────────┘ └────────────────┘
                                                                              ▲
                                                           GDPR job ──────────┘ (DELETE by user prefix)
```

---

## Step-by-Step Explanation

### Execution Walkthrough: Turn 2 With Redis (Two Workers)

```
Turn 1 → Worker A
  1. factory("acme:u1:c9") → RedisChatMessageHistory(key "chat:acme:u1:c9")
  2. messages → []                           (key doesn't exist yet)
  3. LLM answers
  4. add_messages([human, ai])  → RPUSH/LPUSH JSON; EXPIRE key 604800

Turn 2 → Worker B   (different process!)
  1. factory("acme:u1:c9") → new object, SAME key
  2. messages → [human, ai]                  (read from Redis, not from RAM)
  3. Prompt = system + [human, ai] + new input
  4. LLM answers with context ✅
  5. add_messages([human2, ai2]); EXPIRE reset
```

The history *object* is cheap and disposable. The **key** is the identity. This is why the factory can create a new object every request.

### Preview: Plugging Any Backend Into a Chain

Full theory is in Chapter 8.4; this proves the swap works:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from history_backends import get_session_history          # ← the only backend-aware import

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chat = RunnableWithMessageHistory(
    prompt | llm | StrOutputParser(),
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

cfg = {"configurable": {"session_id": "acme:demo:c1"}}
print(chat.invoke({"input": "Remember: my favorite color is teal."}, cfg))
print(chat.invoke({"input": "What is my favorite color?"}, cfg))
```

Run it once with `CHAT_HISTORY_BACKEND=memory`, once with `redis`. Kill the script between the two turns: only Redis answers "teal" on the second run.

---

## Operations Playbook

### TTL and Retention

| Data | Suggested policy |
|------|------------------|
| Redis hot sessions | TTL 1–7 days (sliding) |
| Postgres archive | Retention job: delete/anonymize after N months per policy |
| Anonymous sessions | Short TTL (hours) |

```sql
-- Nightly retention (Postgres)
DELETE FROM chat_message_history WHERE id IN (
  SELECT id FROM chat_message_history WHERE created_at < now() - interval '180 days' LIMIT 10000);
-- (requires a created_at column — add one via a custom converter or migration)
```

### GDPR / DPDP Erasure

```python
import redis

def erase_user(r: redis.Redis, engine, tenant: str, user: str) -> dict:
    pattern = f"chat:{tenant}:{user}:*"
    deleted_redis = 0
    for key in r.scan_iter(match=pattern, count=500):      # SCAN, never KEYS in prod
        deleted_redis += r.delete(key)
    with engine.begin() as conn:
        res = conn.exec_driver_sql(
            "DELETE FROM chat_message_history WHERE session_id LIKE %s", (f"{tenant}:{user}:%",)
        )
    return {"redis_keys": deleted_redis, "sql_rows": res.rowcount}
```

Also remember: backups, log aggregators, analytics exports, **vector stores** if transcripts were embedded, and LLM-provider retention settings. Keep a `user → session_ids` mapping if your IDs aren't prefix-scannable.

### Failure Modes

| Failure | Mitigation |
|---------|-----------|
| Redis down | Fail **closed** with a clear 503, or degrade to stateless mode with a banner — never silently fall back to in-memory per-worker (random amnesia returns) |
| Postgres slow | Cold write async; circuit-breaker |
| Huge transcripts | Trim what you *send* (8.2); cap messages per session at write |
| Schema drift between deploys | Pin `langchain-community`; test deserialization of old rows in CI |

---

## Common Mistakes

### Mistake 1: In-memory behind a load balancer

```python
# ❌ Works in dev, random amnesia with 2+ workers
_store = {}
def get_session_history(sid): return _store.setdefault(sid, InMemoryChatMessageHistory())

# ✅ Shared store
def get_session_history(sid): return RedisChatMessageHistory(sid, url=os.environ["REDIS_URL"], key_prefix="chat:")
```

### Mistake 2: Trusting a client-supplied `session_id`

```python
# ❌ Anyone can read anyone's chat by guessing "user_43"
sid = request.headers["X-Session-Id"]

# ✅ Server derives identity; client only picks the conversation slug
sid = build_session_id(auth.tenant, auth.user_id, request.headers.get("X-Conversation"))
```

### Mistake 3: Forgetting `key_prefix` / using defaults across environments

```python
# ❌ staging and prod share a Redis → "message_store:alice" collides
RedisChatMessageHistory("alice", url=URL)

# ✅
RedisChatMessageHistory("alice", url=URL, key_prefix=f"{ENV}:chat:")
```

### Mistake 4: Passing a non-UUID to `langchain_postgres`

```python
# ❌ ValueError: session_id must be a valid UUID
PostgresChatMessageHistory("chat_history", "user-42", sync_connection=conn)
# ✅ uuid5 from your key
PostgresChatMessageHistory("chat_history", str(uuid.uuid5(uuid.NAMESPACE_URL, "user-42")), sync_connection=conn)
```

### Mistake 5: Creating the in-memory object inside the factory every time

```python
# ❌ Always empty — "persistence" bug that is really a factory bug
def get_session_history(sid): return InMemoryChatMessageHistory()
```

### Mistake 6: Storing secrets and PII in plain text, forever

Add `RedactingHistory`, TTL, and a retention job. "We'll clean it up later" becomes a breach report.

---

## Debugging Guide

| Symptom | Likely cause | Check |
|---------|--------------|-------|
| Bot forgets across requests but works in a single script | In-memory + multiple workers/reload | `uvicorn --reload` restarts the process on file save → dict wiped |
| History exists in Redis but chain ignores it | Wrong key between writer and reader | `redis-cli KEYS "chat:*"` (dev only), compare `key_prefix` + `session_id` |
| History disappears after a few minutes | TTL too small / sliding not refreshed | `redis-cli TTL chat:<sid>` |
| `ModuleNotFoundError: redis` / `psycopg2` | Optional deps | Install per backend; use lazy imports |
| `ValueError: ... UUID` | `langchain_postgres` requires UUIDs | Use `uuid5` |
| `TypeError: unexpected keyword 'connection'` | Older `langchain-community` | Use `connection_string=` |
| Two users see each other's chats | Shared/guessable `session_id` | Audit ID derivation |

Quick inspection helper:

```python
def dump(session_id: str) -> None:
    for i, m in enumerate(get_session_history(session_id).messages):
        print(f"{i:02d} {m.type:<6} {str(m.content)[:70]!r}")
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Keep **all** backend knowledge in one module | Dev/prod parity, easy tests |
| Fail fast on bad `CHAT_HISTORY_BACKEND` at startup | Misconfig found at deploy, not by users |
| Redis for hot, SQL for archive/audit | Cost, latency, and compliance together |
| Namespace keys by env + tenant | No cross-env/tenant collisions |
| Server-derived, validated, unguessable IDs | Tenancy and security |
| Redact at the storage boundary | Secrets never replay into prompts |
| TTL + retention job + erasure endpoint | Privacy by design |
| Test backends without an LLM (add/read/clear) | Fast, deterministic CI |
| Share connection pools in production | The community Redis class opens a client per call; wrap or cache for heavy traffic |
| Store everything, *send* a trimmed view | Persistence ≠ context window |

---

## Interview Preparation

### Easy

**Q: Why persist chat history outside the application process?**
> Multi-worker deployments, restarts, and autoscaling all destroy process-local state. A shared store (Redis/SQL) is both visible to every worker and durable across restarts.

**Q: What three things must a `BaseChatMessageHistory` implement?**
> `messages` (read), `add_messages` (append), `clear` (delete). Everything else — `add_user_message`, async variants — derives from those.

### Medium

**Q: How do you swap memory backends without rewriting the chain?**
> The chain depends only on `get_session_history(session_id) -> BaseChatMessageHistory`. Change the factory (driven by an env var); `prompt | llm | parser` and the `RunnableWithMessageHistory` wrapper remain untouched.

**Q: Why does `get_session_history` return a new object per call for Redis but must cache for in-memory?**
> Redis/SQL objects are thin handles on external state, identified by key. In-memory objects *are* the state — recreating them erases it.

### Hard

**Q: Redis vs PostgreSQL for chat history?**
> Redis: sub-ms, native TTL, ideal for active sessions; limited querying, RAM cost, durability by config. Postgres: durable, queryable, joins to user tables, easy per-user erasure; slower writes. Production pattern: Redis hot path + Postgres archive with read-through rehydration.

**Q: A user's conversation randomly "forgets" only in production. Diagnose.**
> Compare worker count: dev has 1, prod N. A process-local store diverges across workers. Confirm by logging `os.getpid()` with the session ID; fix with a shared backend. Also check `--reload`, container restarts, and TTL.

### Senior

**Q: Design GDPR erasure for chat across Redis, Postgres, and embeddings.**
> Maintain a `user → session_ids` mapping or prefix-structured IDs. An erasure job: SCAN/DELETE Redis keys, `DELETE` SQL rows, delete vector-store entries by metadata filter, purge object-storage attachments, request provider-side deletion if retained, and expire backups by policy. Log the erasure (without content), define an SLA, and make the job idempotent.

**Q: How would you build multi-tenant isolation for chat history?**
> Server-side ID derivation `tenant:user:conv` from verified claims; validated components; per-tenant key prefixes (and optionally per-tenant DB schemas/encryption keys); ownership check on conversation reads; row-level security in Postgres; tests that attempt cross-tenant reads.

**Q: Redis is down mid-conversation. What does your system do?**
> Prefer explicit degradation: return 503/"temporarily unavailable" or serve stateless replies flagged to the user. Silent per-worker fallback creates inconsistent partial histories that corrupt later turns. Use health checks, retries with backoff, and a circuit breaker; replay from the Postgres archive when Redis returns.

---

## Summary

| Backend | Durability | Speed | Queryable | Typical use |
|---------|-----------|-------|-----------|-------------|
| `InMemoryChatMessageHistory` | None | Fastest | No | Tests, notebooks |
| `RedisChatMessageHistory` | Config-dependent + TTL | Very fast | Key-only | Active sessions |
| `SQLChatMessageHistory` (SQLite) | File | Good | SQL | Single-node apps |
| `SQLChatMessageHistory` / `langchain_postgres` | Full | Moderate | SQL, joins | Audit, compliance |
| `MongoDBChatMessageHistory` | Full | Good | Documents | Metadata-rich chats |
| Custom subclass | Yours | Yours | Yours | Exotic stores, decorators |

Key ideas: **identity = session_id**, **storage ≠ context window**, **decorate at the boundary** (redaction, tiering), **one factory, one env switch**.

---

## Cheat Sheet

```python
# Contract
class MyHistory(BaseChatMessageHistory):
    @property
    def messages(self): ...
    def add_messages(self, messages): ...
    def clear(self): ...

# Backends
InMemoryChatMessageHistory()
RedisChatMessageHistory(sid, url=REDIS_URL, key_prefix="chat:", ttl=604800)
SQLChatMessageHistory(session_id=sid, connection="postgresql+psycopg2://...")
MongoDBChatMessageHistory(connection_string=URI, session_id=sid,
                          database_name="chatdb", collection_name="chat_histories")

# Ops
r.scan_iter(match="chat:acme:u1:*")        # find user keys   (never KEYS in prod)
redis-cli TTL chat:<sid>                   # check expiry
DELETE FROM chat_message_history WHERE session_id LIKE 'acme:u1:%';

# Env switch
CHAT_HISTORY_BACKEND = memory | redis | sqlite | postgres | mongo
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| Why does a module-level dict fail in prod? | Per-process, non-durable, unbounded — breaks with multiple workers/restarts |
| Three members of `BaseChatMessageHistory`? | `messages`, `add_messages`, `clear` |
| What does Redis `ttl` do in `RedisChatMessageHistory`? | Expires the session key after N seconds, refreshed on writes |
| What is `key_prefix` for? | Namespacing keys (env/tenant), avoiding collisions |
| Which Postgres class needs UUID session IDs? | `langchain_postgres.PostgresChatMessageHistory` |
| How do you get a stable UUID from a string key? | `uuid.uuid5(NAMESPACE_URL, key)` |
| Who should create `session_id`? | The server, from authenticated identity |
| Where do you redact secrets? | At the storage boundary (a wrapper history) |
| Why `SCAN` not `KEYS`? | `KEYS` blocks Redis on large keyspaces |
| What is the hot/cold pattern? | Redis for active sessions, Postgres for archive, read-through rehydration |

---

## Hands-on Exercises

### Exercise 1: Two-Process Redis Verification

1. Start Redis: `docker run -d --name chat-redis -p 6379:6379 redis:7`
2. `pip install redis langchain-community`
3. **Shell A — `writer.py`:**

```python
import os
from langchain_community.chat_message_histories import RedisChatMessageHistory

h = RedisChatMessageHistory("demo-42", url="redis://localhost:6379/0", key_prefix="chat:", ttl=600)
h.add_user_message("My name is Rahul.")
h.add_ai_message("Nice to meet you, Rahul!")
print(f"writer pid={os.getpid()} wrote {len(h.messages)} messages")
```

4. **Shell B — `reader.py`** (run *after* `writer.py` exits):

```python
import os
from langchain_community.chat_message_histories import RedisChatMessageHistory

h = RedisChatMessageHistory("demo-42", url="redis://localhost:6379/0", key_prefix="chat:")
print(f"reader pid={os.getpid()} sees:")
for m in h.messages:
    print(" ", m.type, "→", m.content)
```

5. Inspect: `docker exec -it chat-redis redis-cli TTL chat:demo-42` and `LRANGE chat:demo-42 0 -1`.
6. Wait for TTL expiry (set `ttl=20`) and re-run the reader. What changed?

**Expected:** different PIDs, identical messages; TTL counts down; after expiry the reader sees nothing.

### Exercise 2: Backend-Agnostic Contract Test

Write `test_backends.py` with a function `check_contract(factory)` asserting: empty at start → add two messages → a **new** instance for the same ID sees them in order → `clear()` empties → another session ID is unaffected. Run it against `InMemory` (via the memoizing factory), your DIY `SQLiteChatMessageHistory`, `SQLChatMessageHistory("sqlite:///t.db")`, and Redis.

### Exercise 3: Redaction

Wrap `InMemoryChatMessageHistory` in `RedactingHistory`. Add a message containing `sk-abcdefghijklmnopqrstuvwxyz` and a fake card number. Assert the stored content contains no secret. Then add one false-positive case (e.g., a 16-digit order number) and discuss how you'd reduce false positives.

---

## Challenge Project

**FastAPI `/chat` with a backend switch (`memory | redis | postgres`).**

Requirements:

1. `history_backends.py` from Part 5 (extend with `RedactingHistory` wrapping).
2. `POST /chat` body `{"message": "..."}` with header `X-User-Id` (pretend-auth) and optional `X-Conversation`; session ID built by `build_session_id`.
3. Use `RunnableWithMessageHistory` (see preview) with the LiteLLM `ChatOpenAI` pattern.
4. `DELETE /chat/{conversation}` clears that conversation.
5. A `README` section documenting env vars for each mode:

| Mode | Required env |
|------|--------------|
| `memory` | none |
| `redis` | `REDIS_URL` |
| `postgres` | `CHAT_DATABASE_URL` |

6. **Proof:** run `uvicorn app:app --workers 2` in `memory` mode and show the amnesia (log `os.getpid()` in the response); switch to `redis` and show it disappear.

Bonus: implement `TieredChatMessageHistory` as a fourth mode `tiered`.

---

## Homework

1. **Reading:** Skim the source of `RedisChatMessageHistory` in `langchain_community` — find how `ttl` is applied and how messages are serialized.
2. **Coding:** Add a `created_at` timestamp to the DIY SQLite history and implement `purge_older_than(days)`.
3. **Design:** Write a one-page retention policy for a support chatbot: TTLs per tier, erasure SLA, what is redacted, and who may read transcripts.
4. **Security:** List three ways an attacker could read another user's chat in your design and the control that blocks each.

---

## Additional Resources

- [LangChain — Chat message history concept](https://python.langchain.com/docs/concepts/chat_history/)
- [LangChain Community — chat message histories](https://python.langchain.com/docs/integrations/memory/)
- [`langchain-postgres` on GitHub](https://github.com/langchain-ai/langchain-postgres)
- [Redis — EXPIRE and key eviction](https://redis.io/docs/latest/commands/expire/)
- [OWASP — Insecure Direct Object Reference](https://owasp.org/www-community/attacks/Insecure_Direct_Object_Reference)

---

## What's Next

You can now store conversations anywhere, safely. In **Chapter 8.4** we go deep on the other half: plugging these histories into LCEL with `RunnableWithMessageHistory` — key matching, `MessagesPlaceholder` placement, trimming inside the pipeline, streaming, async, and a complete FastAPI chatbot with `/chat` and `/chat/stream`.

> [← Previous: Memory Strategies](chapter-33-memory-strategies.md) | [Next: Memory in LCEL →](chapter-35-memory-in-lcel.md)

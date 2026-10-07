# Chapter 15.1: Checkpointing (SQLite, PostgreSQL)

> **Phase 15 — LangGraph Persistence & Memory** | [← Previous: Error Recovery](../phase-14-langgraph-routing/chapter-63-error-recovery.md) | [Next: Interrupt & Resume →](chapter-65-interrupt-resume.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Explain **checkpoints** and **thread IDs** in LangGraph
- ✅ Use `MemorySaver` for development
- ✅ Persist threads with **SQLite** and **PostgreSQL** checkpointers
- ✅ Resume conversations with `invoke(..., config={"configurable": {"thread_id": ...}})`
- ✅ Inspect checkpoint history for debugging

| | |
|---|---|
| **Prerequisites** | Phase 14 (Routing & Control Flow) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 55 minutes |

---

## Introduction — The Problem

In-memory graphs forget everything when the process restarts:

```
User (Monday):  "My ticket is #8821"
User (Tuesday): "Any update?"        → Bot: "What's your ticket?" ❌
```

Serverless scale-out kills RAM-only state. You need **durable checkpoints** keyed by **thread_id**.

```
RUN 1:  invoke(..., thread_id="abc") ──► checkpoint saved
RUN 2:  invoke(..., thread_id="abc") ──► loads prior state ✅
```

### The Solution — Checkpointer + Compile

Pass a checkpointer to `graph.compile(checkpointer=...)`. Every super-step can persist state snapshots.

---

## Part 1: Core Concepts

| Term | Meaning |
|------|---------|
| **Thread** | Logical conversation / workflow instance |
| **thread_id** | Config key identifying the thread |
| **Checkpoint** | Serialized state + metadata at a step |
| **checkpoint_id** | Pointer for time-travel / fork (advanced) |

```
config = {"configurable": {"thread_id": "user-42-session-9"}}
app.invoke(input_state, config=config)
```

---

## Part 2: MemorySaver (Dev)

```python
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def chat_node(state: ChatState) -> dict:
    reply = llm.invoke(state["messages"])
    return {"messages": [reply]}


graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "demo-thread-1"}}

app.invoke({"messages": [HumanMessage(content="My name is Priya.")]}, config=config)
result = app.invoke({"messages": [HumanMessage(content="What's my name?")]}, config=config)
print(result["messages"][-1].content)
```

MemorySaver is **not** for production persistence across deploys — use SQL backends.

---

## Part 3: SQLite Checkpointer

```python
# pip install langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite import SqliteSaver

# Context manager opens DB file
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    sql_app = graph.compile(checkpointer=checkpointer)
    cfg = {"configurable": {"thread_id": "sqlite-user-7"}}
    sql_app.invoke(
        {"messages": [HumanMessage(content="Remember: project codename is NEBULA.")]},
        config=cfg,
    )
    out = sql_app.invoke(
        {"messages": [HumanMessage(content="What's the codename?")]},
        config=cfg,
    )
    print(out["messages"][-1].content)
```

```
checkpoints.db
├── thread sqlite-user-7
│   ├── step 0 state blob
│   └── step 1 state blob
```

Good for **local dev**, single-node apps, and integration tests.

---

## Part 4: PostgreSQL Checkpointer

```python
# pip install langgraph-checkpoint-postgres psycopg[binary]
# from langgraph.checkpoint.postgres import PostgresSaver

# Typical pattern (see package docs for exact API version):
# DB_URI = "postgresql://user:pass@localhost:5432/langgraph"
# with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
#     prod_app = graph.compile(checkpointer=checkpointer)
```

Production checklist:

```
PostgreSQL checkpointer:
├── Managed RDS / Cloud SQL
├── Connection pooling (PgBouncer)
├── Migrations handled by library setup()
├── Backup + retention policy
└── Separate DB from OLTP if possible
```

---

## Part 5: Multi-Turn Without Rewriting Graph

Same graph, new user message each HTTP request:

```python
def handle_chat(thread_id: str, text: str) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"messages": [HumanMessage(content=text)]},
        config=config,
    )
    return result["messages"][-1].content
```

```
FastAPI request ──► thread_id from session cookie
                 ──► invoke incremental input
                 ──► checkpoint append
```

---

## Part 6: Reading State History

```python
# List checkpoints for debugging (API varies slightly by version)
for snap in app.get_state_history(config):
    print(snap.config, snap.metadata)
```

Useful for audit: which node ran last before failure?

---

## Part 7: Checkpointing + Routing

Checkpointing is **orthogonal** to conditional edges — every node transition can persist. Combine with error recovery (Phase 14) to resume mid-workflow after crash.

---

## Part 8: Thread ID Strategies

| Strategy | Example | Notes |
|----------|---------|-------|
| Per browser session | `uuid4()` in cookie | New thread each visit |
| Per authenticated user + project | `f"{user_id}:{project_id}"` | Long-lived work |
| Per Slack thread | `channel_id:message_ts` | Natural mapping |
| Ephemeral eval | fixed `"eval-1"` | Tests only |

Never embed PII raw in thread_id if logs are exported — hash if needed.

### Part 9: Compaction (Conceptual)

Long threads blow context limits. Patterns:

```
1. Summarize messages node before LLM call (state still full in checkpoint)
2. Fork new thread_id with summary seed
3. Store raw history in cold storage; checkpoint holds summary only
```

Checkpoint retention ≠ context window management — plan both.

### Part 10: Local Dev SQLite File Locking

On Windows, ensure only one process opens `checkpoints.db` writer. FastAPI `--reload` can spawn multiple processes — use Postgres in team dev or separate DB files per developer.

---

## Common Mistakes

### Mistake 1: Forgetting `thread_id`

Each invoke without config creates implicit ephemeral threads — looks "broken" in multi-turn tests.

### Mistake 2: Same thread_id for all users

**Privacy bug** — one user sees another's messages.

### Mistake 3: Storing secrets in state

Checkpoints serialize state — scrub tokens before persist.

### Mistake 4: No migration plan on PostgreSQL

Coordinate library upgrades with schema setup scripts.

---

## Best Practices

| Practice | Why |
|----------|-----|
| UUID thread_ids per user session | Isolation |
| MemorySaver in unit tests | Fast, isolated |
| SQLite for local staging | Parity with SQL semantics |
| Postgres in production | Durability + scale |
| TTL / cleanup job | Control storage cost |
| Encrypt DB at rest | Compliance |

---

## Interview Preparation

### Easy
**Q: What is a LangGraph checkpointer?**

> A persistence layer that saves graph state snapshots per thread so runs can resume across calls and process restarts.

### Medium
**Q: MemorySaver vs PostgreSQL?**

> MemorySaver is in-process RAM for dev/tests. PostgreSQL (via PostgresSaver) provides durable, multi-instance persistence suitable for production APIs.

### Hard
**Q: How do thread_id and checkpoint interact with horizontal scaling?**

> All app replicas must share the same checkpointer backend (Postgres). Requests for a thread_id load latest checkpoint from DB; sticky sessions optional if checkpointer is central.

---

## Summary

| Component | Role |
|-----------|------|
| **checkpointer** | Storage adapter |
| **thread_id** | Conversation key |
| **compile(checkpointer=)** | Enables persistence |
| **configurable** | Passes thread context |
| **SQLite / Postgres** | Dev vs prod stores |

---

## Exercises

1. Run two threads (`thread-a`, `thread-b`) and verify names don't leak.

2. Delete `checkpoints.db` mid-test — explain when MemorySaver vs SQLite differs.

3. Wrap `handle_chat` in a FastAPI stub (pseudo-code OK) showing thread_id from header.

4. Sketch retention policy: delete checkpoints older than 90 days — what breaks?

---

## What's Next

[Chapter 15.2 — Interrupt & Resume](chapter-65-interrupt-resume.md) pauses graphs for human approval and continues from the same checkpoint.

---

> [← Previous: Error Recovery](../phase-14-langgraph-routing/chapter-63-error-recovery.md) | [Next: Interrupt & Resume →](chapter-65-interrupt-resume.md)

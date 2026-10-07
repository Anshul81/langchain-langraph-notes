# Chapter 15.3: Long-Term Memory (Cross-Session)

> **Phase 15 — LangGraph Persistence & Memory** | [← Previous: Interrupt & Resume](chapter-65-interrupt-resume.md) | [Next: Streaming →](chapter-67-streaming-langgraph.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Distinguish **thread checkpoint memory** vs **cross-session long-term memory**
- ✅ Store user facts in an external **store** (vector or KV)
- ✅ Load and inject memories at graph **start**
- ✅ Update memory after successful runs
- ✅ Design privacy-aware retention for user profiles

| | |
|---|---|
| **Prerequisites** | Chapters 15.1–15.2 |
| **Estimated Reading Time** | 28 minutes |
| **Estimated Coding Time** | 55 minutes |

---

## Introduction — The Problem

Checkpoints tie memory to **one thread**. Users expect:

```
Session last week:  "I'm vegetarian"
Session today:      "Suggest dinner near my office"
                    → should remember diet + office city
```

Thread checkpoints alone don't generalize across **new thread_ids** (new browser sessions).

```
SHORT-TERM (checkpoint):   same thread_id history
LONG-TERM (store):         user_id → facts/preferences
```

### The Solution — Store + Memory Nodes

Use LangGraph **Store** (or your DB) keyed by `user_id`. Nodes **read** memories at entry and **write** after extraction.

---

## Part 1: Two Layers of Memory

| Layer | Scope | Mechanism |
|-------|-------|-----------|
| **Checkpoint** | One thread | SqliteSaver / PostgresSaver |
| **Long-term** | User across threads | Store, Redis, Postgres profile table |
| **Working** | Current run | State fields |

```
invoke thread "t-9" ──► checkpoint t-9
user_id "u-1"     ──► store namespace ("memories", u-1)
```

---

## Part 2: In-Memory Store Pattern (Concept)

```python
import os
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)

# Simple stand-in for LangGraph BaseStore — production use langgraph.store
USER_MEMORIES: dict[str, list[str]] = {}


class MemoryState(TypedDict):
    user_id: str
    messages: Annotated[list, add_messages]
    injected_context: str


def load_memories(state: MemoryState) -> dict:
    uid = state["user_id"]
    facts = USER_MEMORIES.get(uid, [])
    ctx = "\n".join(f"- {f}" for f in facts) if facts else "(no prior facts)"
    return {"injected_context": ctx}


def chat(state: MemoryState) -> dict:
    prompt = [
        SystemMessage(content=f"User facts:\n{state['injected_context']}"),
        *state["messages"],
    ]
    reply = llm.invoke(prompt)
    return {"messages": [reply]}


def extract_memories(state: MemoryState) -> dict:
    """Parse last user message for durable facts (demo heuristic)."""
    uid = state["user_id"]
    USER_MEMORIES.setdefault(uid, [])
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            text = msg.content.lower()
            if "i am vegetarian" in text and "vegetarian" not in str(USER_MEMORIES[uid]):
                USER_MEMORIES[uid].append("User is vegetarian.")
            if "office in" in text:
                USER_MEMORIES[uid].append(msg.content.strip())
            break
    return {}


graph = StateGraph(MemoryState)
graph.add_node("load_memories", load_memories)
graph.add_node("chat", chat)
graph.add_node("extract_memories", extract_memories)

graph.add_edge(START, "load_memories")
graph.add_edge("load_memories", "chat")
graph.add_edge("chat", "extract_memories")
graph.add_edge("extract_memories", END)

app = graph.compile(checkpointer=MemorySaver())
```

---

## Part 3: Cross-Session Demo

```python
if __name__ == "__main__":
    user = "user-priya"
    t1 = {"configurable": {"thread_id": "session-monday"}}
    t2 = {"configurable": {"thread_id": "session-wednesday"}}

    app.invoke(
        {"user_id": user, "messages": [HumanMessage(content="I am vegetarian.")]},
        config=t1,
    )
    out = app.invoke(
        {
            "user_id": user,
            "messages": [HumanMessage(content="Suggest a lunch option.")],
        },
        config=t2,  # NEW thread — still sees store facts via load_memories
    )
    print(out["messages"][-1].content)
    print("Store:", USER_MEMORIES[user])
```

---

## Part 4: LangGraph Store API (Production)

LangGraph provides a **Store** interface for namespaced documents:

```
namespace: ("users", user_id)
key: "preferences"
value: {"diet": "vegetarian", "city": "Berlin"}
```

Compile with store:

```python
# from langgraph.store.memory import InMemoryStore
# store = InMemoryStore()
# app = graph.compile(checkpointer=checkpointer, store=store)
```

Nodes receive `store` via runtime config / injectable context depending on version — consult current docs when wiring `get_store()`.

---

## Part 5: Vector Long-Term Memory

For unstructured recall (past tickets, notes):

```
User message ──► embed ──► search user_id filter ──► top-k memories ──► prompt
```

Use the same vector DB patterns from Phase 11–12 with metadata filter `user_id`.

---

## Part 6: Privacy & Governance

```
GDPR / enterprise:
├── Export memories on request
├── Delete user namespace on account delete
├── Don't store raw PCI/PHI in LLM-extracted facts
├── Audit who wrote each memory (node + timestamp)
└── Separate prod/staging store namespaces
```

---

## Part 7: Memory Write Policies

| Policy | When |
|--------|------|
| Write every turn | Demos |
| Write on explicit "remember this" | Consumer apps |
| Write after validator pass | Enterprise |
| Human approve write | Regulated data |

```python
def maybe_extract(state: MemoryState) -> dict:
    if "remember" not in state["messages"][-1].content.lower():
        return {}
    return extract_memories(state)
```

### Part 8: Conflict Resolution

Two facts same key — latest wins:

```python
def merge_facts(old: dict, new: dict) -> dict:
    return {**old, **new}
```

For list-style memories, store `(timestamp, fact)` tuples and sort on load.

### Part 9: Evaluating Recall

```python
# Golden test: after session A, session B should mention vegetarian
assert "vegetarian" in out["messages"][-1].content.lower()
```

Automate cross-thread tests in CI with mocked LLM for deterministic extracts.

---

## Common Mistakes

### Mistake 1: Confusing checkpoint with profile

New thread without load node → "forgetful" bot despite DB full of facts.

### Mistake 2: Unbounded memory list

Summarize or embed — don't append forever.

### Mistake 3: Storing hallucinated facts

Validate extractions (user confirm or structured schema).

### Mistake 4: Same user_id in dev and prod stores

Namespace by environment.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Explicit load / save nodes | Clear data flow |
| user_id from auth, not client-only | Prevent spoofing |
| Summarize old memories | Token limits |
| Version memory schema | Migrations |
| Test cross-thread recall | Core acceptance criteria |

---

## Interview Preparation

### Easy
**Q: Difference between checkpoint and long-term memory?**

> Checkpoints persist a single thread's graph state across steps. Long-term memory persists user-specific facts across new threads/sessions.

### Medium
**Q: Where would you write memories in the graph?**

> After successful completion or dedicated extraction node post-response, writing to a store keyed by user_id; load node at START injects into prompt context.

### Hard
**Q: How avoid memory injection attacks?**

> Trust authenticated user_id, sanitize stored content, don't execute retrieved text as instructions (use delimited system blocks), rate-limit writes, audit trail.

---

## Summary

| Concept | Role |
|---------|------|
| **thread_id** | Session checkpoint |
| **user_id** | Long-term key |
| **load node** | Inject facts |
| **extract node** | Persist new facts |
| **Store / Vector** | Durable backend |

---

## Exercises

1. Replace heuristic extractor with LLM structured output (`facts: list[str]`).

2. Add max 20 facts — drop oldest when exceeded.

3. Run two users and verify isolation in `USER_MEMORIES`.

4. Design JSON schema for preferences store — document fields.

---

## What's Next

[Chapter 15.4 — Streaming in LangGraph](chapter-67-streaming-langgraph.md) streams tokens and node updates to clients.

---

> [← Previous: Interrupt & Resume](chapter-65-interrupt-resume.md) | [Next: Streaming →](chapter-67-streaming-langgraph.md)

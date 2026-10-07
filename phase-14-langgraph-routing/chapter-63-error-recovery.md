# Chapter 14.4: Error Handling, Retries & Recovery

> **Phase 14 — LangGraph Routing & Control Flow** | [← Previous: Parallel Branches](chapter-62-parallel-branches.md) | [Next: Checkpointing →](../phase-15-langgraph-persistence/chapter-64-checkpointing.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Handle tool and LLM failures inside nodes without crashing the graph
- ✅ Implement **retry loops** with backoff counters in state
- ✅ Route to **fallback** nodes on persistent errors
- ✅ Use LangChain **retries** on runnables where appropriate
- ✅ Design **graceful degradation** for production agents

| | |
|---|---|
| **Prerequisites** | Chapters 14.2–14.3 |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction — The Problem

Production graphs hit:

```
⚠️ REAL FAILURES:
├── LLM 429 rate limit
├── Tool timeout (payment API)
├── Malformed JSON from model
├── Vector DB connection reset
└── User input triggers exception in parser
```

A naked `raise` loses partial progress and breaks UX.

```
FAIL FAST:     crash ──► 500 to user

RECOVERY:      try ──► retry ──► fallback ──► partial answer ✅
```

### The Solution — Errors as State + Routing

Catch exceptions in nodes, write `error` / `retry_count` to state, and **route** to retry or fallback paths.

---

## Part 1: Errors in Nodes (Try / Except)

```python
from typing import TypedDict, Annotated, Literal
from operator import add


class ResilientState(TypedDict):
    query: str
    result: str
    error: str
    retry_count: int
    log: Annotated[list[str], add]


def flaky_tool(query: str) -> str:
    if "fail" in query.lower():
        raise ConnectionError("upstream unavailable")
    return f"Data for {query}"


def call_tool_node(state: ResilientState) -> dict:
    try:
        data = flaky_tool(state["query"])
        return {"result": data, "error": "", "log": ["tool:success"]}
    except Exception as exc:
        return {
            "error": str(exc),
            "log": [f"tool:error:{exc}"],
        }
```

Nodes should **return** errors, not always raise — keeps checkpoint threads resumable.

---

## Part 2: Retry Router

```python
from langgraph.graph import StateGraph, START, END

MAX_RETRIES = 3


def route_after_tool(state: ResilientState) -> Literal["call_tool", "fallback", END]:
    if not state.get("error"):
        return END
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "call_tool"
    return "fallback"


def increment_retry(state: ResilientState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def fallback_node(state: ResilientState) -> dict:
    return {
        "result": "Sorry — live lookup failed. Here are general troubleshooting steps.",
        "log": ["fallback:generic"],
    }
```

Graph shape:

```
START ──► call_tool ──► route ──► END (success)
                ▲          │
                │          ├──► increment_retry ──► (loop)
                │          └──► fallback ──► END
```

---

## Part 3: Full Recovery Graph

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)


graph = StateGraph(ResilientState)
graph.add_node("call_tool", call_tool_node)
graph.add_node("increment_retry", increment_retry)
graph.add_node("fallback", fallback_node)

graph.add_edge(START, "call_tool")
graph.add_conditional_edges(
    "call_tool",
    route_after_tool,
    {
        "call_tool": "increment_retry",
        "fallback": "fallback",
        END: END,
    },
)
graph.add_edge("increment_retry", "call_tool")
graph.add_edge("fallback", END)

recovery_app = graph.compile()

if __name__ == "__main__":
    ok = recovery_app.invoke({"query": "status page"})
    fail = recovery_app.invoke({"query": "please fail now"})
    print(ok["result"])
    print(fail["result"], fail["log"])
```

---

## Part 4: Runnable-Level Retries (LLM)

For transient LLM errors, wrap the model:

```python
from langchain_core.runnables import RunnableRetry

robust_llm = RunnableRetry(
    bound=llm,
    max_attempt_number=3,
    retry_if_exception_type=(Exception,),
)
```

Use **inside** nodes for LLM-only failures; use **graph routing** when you need visibility in state/logs.

---

## Part 5: Fallback Model Pattern

```python
cheap = ChatOpenAI(model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"), ...)
strong = ChatOpenAI(model="gpt-4o", ...)


def llm_with_fallback(prompt_messages):
    try:
        return cheap.invoke(prompt_messages)
    except Exception:
        return strong.invoke(prompt_messages)
```

Combine with state `model_used` for observability.

---

## Part 6: Idempotency & Side Effects

```
RETRY SAFE:     read-only search, idempotent GET
RETRY UNSAFE:   charge card, send email without idempotency key
```

For unsafe tools, gate retries behind human approval (Phase 15).

---

## Part 7: Error Taxonomy in State

```python
class ResilientState(TypedDict):
    query: str
    result: str
    error: str
    error_kind: str  # transient | auth | validation | unknown
    retry_count: int
    log: Annotated[list[str], add]


def classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "401" in text or "403" in text:
        return "auth"
    if "timeout" in text or "429" in text:
        return "transient"
    if "invalid" in text or "parse" in text:
        return "validation"
    return "unknown"
```

Route `auth` and `validation` directly to fallback or END — never retry.

### Part 8: Dead-Letter Node

```python
def dead_letter(state: ResilientState) -> dict:
    return {
        "result": "",
        "log": [f"dead_letter:{state.get('error_kind')}"],
    }
```

Wire after max retries for ops alerting (webhook tool in production).

### Part 9: Checkpoint + Retry Interaction

When using a checkpointer, retried nodes re-emit updates — ensure tool side effects are idempotent or guard with `request_id` in state so retry does not double-charge.

---

## Common Mistakes

### Mistake 1: Infinite retry loop

Always cap `retry_count` and route to fallback.

### Mistake 2: Raising in node after partial state update

Prefer single return dict; avoid half-applied side effects before raise.

### Mistake 3: Retrying non-transient errors

Validation errors won't fix themselves — route to clarification, not retry.

### Mistake 4: Hidden retries only in SDK

Surface retry events to logs/UI — users notice latency spikes.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Error fields in state | Routable, checkpointed |
| Exponential backoff (sleep in node or task queue) | Respect 429 |
| Classify errors | retry vs fallback vs escalate |
| Test fallback path | Most neglected branch |
| Circuit breaker at tool layer | Protect dependencies |
| LangSmith traces | See retry spans |

---

## Interview Preparation

### Easy
**Q: How should LangGraph nodes handle exceptions?**

> Catch expected failures, return partial state updates with error metadata, and use conditional edges for retry or fallback instead of crashing the run when possible.

### Medium
**Q: Graph retry vs RunnableRetry?**

> RunnableRetry handles transparent LLM/API retries inside a node. Graph-level retry loops expose count, logging, and alternate paths (fallback model, degraded answer) in state.

### Hard
**Q: Design error handling for a payment tool in an agent.**

> No blind retries on charge. interrupt_before tool, idempotency keys, on failure route to support fallback with error code in state, never double-charge; log thread id for reconciliation.

---

## Summary

| Pattern | Use |
|---------|-----|
| try/except in node | Controlled failures |
| retry_count + route | Bounded retries |
| fallback node | User-visible degradation |
| RunnableRetry | LLM transient errors |
| Idempotency | Safe side effects |

---

## Exercises

1. Add exponential backoff: `import time; time.sleep(2 ** retry_count)` in increment node (mock timing).

2. Classify errors: if `"401"` in error, route to END with auth message — no retry.

3. Log all attempts with reducer; print log for failing query.

4. Wrap LLM node with RunnableRetry and compare behavior when API key is invalid.

---

## What's Next

[Chapter 15.1 — Checkpointing](../phase-15-langgraph-persistence/chapter-64-checkpointing.md) persists graph state to SQLite or PostgreSQL for durable threads.

---

> [← Previous: Parallel Branches](chapter-62-parallel-branches.md) | [Next: Checkpointing →](../phase-15-langgraph-persistence/chapter-64-checkpointing.md)

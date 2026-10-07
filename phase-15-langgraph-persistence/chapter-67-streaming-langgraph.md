# Chapter 15.4: Streaming in LangGraph

> **Phase 15 — LangGraph Persistence & Memory** | [← Previous: Long-term Memory](chapter-66-long-term-memory.md) | [Next: Supervisor Architecture →](../phase-16-multi-agent/chapter-68-supervisor.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Stream graph runs with `stream()` and `astream()`
- ✅ Use **stream modes** (`values`, `updates`, `messages`, `debug`)
- ✅ Show **node-level progress** in UIs
- ✅ Combine streaming with **checkpointers**
- ✅ Bridge LangGraph streams to **SSE** APIs (conceptual)

| | |
|---|---|
| **Prerequisites** | Chapters 15.1–15.3 |
| **Estimated Reading Time** | 28 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction — The Problem

Users stare at spinners for 30s while a graph runs five nodes:

```
Silent invoke:  [==========] 30s ──► full JSON dump
```

LLM apps need **token streaming** and **step visibility** — especially with HITL and multi-agent flows.

```
STREAM:
  node started: retrieve
  token: "The"
  token: " refund"
  node finished: retrieve
  node started: draft
  ...
```

### The Solution — `stream()` / `astream()`

LangGraph exposes incremental events as the graph executes — pair with FastAPI SSE in Phase 17.

---

## Part 1: Basic `stream()` with `values`

```python
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    streaming=True,
)


class StreamState(TypedDict):
    topic: str
    outline: str
    article: str


def outline_node(state: StreamState) -> dict:
    r = llm.invoke([HumanMessage(content=f"3-bullet outline for: {state['topic']}")])
    return {"outline": r.content}


def write_node(state: StreamState) -> dict:
    r = llm.invoke([
        HumanMessage(content=f"Expand outline into 2 paragraphs:\n{state['outline']}"),
    ])
    return {"article": r.content}


graph = StateGraph(StreamState)
graph.add_node("outline", outline_node)
graph.add_node("write", write_node)
graph.add_edge(START, "outline")
graph.add_edge("outline", "write")
graph.add_edge("write", END)

app = graph.compile()

for event in app.stream({"topic": "LangGraph streaming"}, stream_mode="values"):
    print("--- state snapshot keys:", event.keys())
```

`values` emits **full state** after each super-step.

---

## Part 2: `updates` Mode — Per-Node Deltas

```python
for chunk in app.stream({"topic": "SSE APIs"}, stream_mode="updates"):
    for node_name, update in chunk.items():
        print(f"Node {node_name} wrote:", list(update.keys()))
```

Ideal for progress bars: "Step 2/5: write".

---

## Part 3: Message / Token Streaming

When nodes call streaming LLMs, use **`stream_mode="messages"`** (LangGraph 0.2+) to surface token chunks from nested runnables:

```python
for msg_chunk, metadata in app.stream(
    {"topic": "Checkpointing"},
    stream_mode="messages",
):
    if msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)
```

Metadata often includes `langgraph_node` — label tokens by node in UI.

---

## Part 4: Async Streaming

```python
import asyncio


async def run_async():
    async for chunk in app.astream({"topic": "asyncio"}, stream_mode="updates"):
        print(chunk)


asyncio.run(run_async())
```

Use in FastAPI `StreamingResponse` with async generators.

---

## Part 5: Streaming + Checkpointing

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app_cp = graph.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "stream-thread-1"}}

for _ in app_cp.stream(
    {"topic": "persistent stream"},
    config=config,
    stream_mode="updates",
):
    pass

# Second call continues thread while still streamable
```

Stream doesn't bypass checkpoints — state still persists when configured.

---

## Part 6: Subgraphs & Multi-Agent (Preview)

Multi-agent graphs stream **nested** updates — filter by node name prefix in UI:

```
supervisor → researcher → synthesizer
     │            │              │
     └────────────┴──────────────┴── separate progress lanes
```

Phase 16 builds these patterns; streaming API stays the same.

---

## Part 7: SSE Bridge (Sketch)

```python
# FastAPI-style pseudocode
# async def chat_sse(topic: str):
#     async for chunk in app.astream(..., stream_mode="messages"):
#         yield f"data: {json.dumps(chunk)}\n\n"
```

Phase 17 covers production SSE/WebSocket patterns.

---

## Part 8: Multiplexed UI Event Model

```python
# Normalized event for front-end
def normalize(event, mode: str) -> dict:
    return {"mode": mode, "payload": event, "ts": time.time()}
```

Consumers switch on `mode` to update chat tokens vs step checklist.

### Part 9: Backpressure

If client reads SSE slowly, buffer or drop token events — never block graph execution indefinitely. Prefer bounded queues between graph thread and HTTP writer.

### Part 10: `debug` Stream Mode

Use during development to inspect routing decisions:

```python
for chunk in app.stream(inputs, stream_mode="debug"):
    print(chunk)
```

Disable in production — verbose and may leak internal state.

---

## Common Mistakes

### Mistake 1: Forgetting `streaming=True` on ChatOpenAI

Graph streams steps but LLM returns one blob.

### Mistake 2: Parsing `values` as deltas

`values` is full state — large payloads; use `updates` for diffs.

### Mistake 3: Blocking event loop

Use `astream` in async servers, not sync `stream` in hot paths.

### Mistake 4: No flush in CLI demos

Use `flush=True` when printing tokens.

---

## Best Practices

| Practice | Why |
|----------|-----|
| `updates` for ops UI | Small payloads |
| `messages` for chat UI | Token-by-token |
| Tag events with node name | Debug multi-step |
| Heartbeat SSE comments | Proxies don't timeout |
| Cancel tasks on client disconnect | Save tokens |
| Log final state once | Analytics |

---

## Interview Preparation

### Easy
**Q: How do you stream a LangGraph run?**

> Call `graph.compile().stream(input, stream_mode=...)` or async `astream`, iterating chunks as nodes complete or messages tokenize.

### Medium
**Q: `values` vs `updates` stream mode?**

> `values` yields full state after steps; `updates` yields per-node partial updates — better for progress indicators.

### Hard
**Q: Stream multi-agent graph without confusing users?**

> Include node metadata in stream events, show active agent in UI, optionally filter internal supervisor tokens, finalize with single consolidated assistant message.

---

## Summary

| Mode | Emits |
|------|--------|
| **values** | Full state snapshots |
| **updates** | Node-level patches |
| **messages** | LLM tokens + metadata |
| **debug** | Verbose internals |

---

## Exercises

1. Print only `outline` key from each `values` event.

2. Count how many `updates` events fire for the 2-node graph.

3. Add a third `polish` node — observe stream order.

4. Sketch JSON shape for one SSE event carrying a token + node name.

---

## What's Next

[Chapter 16.1 — Supervisor Architecture](../phase-16-multi-agent/chapter-68-supervisor.md) coordinates specialist agents with a routing supervisor.

---

> [← Previous: Long-term Memory](chapter-66-long-term-memory.md) | [Next: Supervisor Architecture →](../phase-16-multi-agent/chapter-68-supervisor.md)

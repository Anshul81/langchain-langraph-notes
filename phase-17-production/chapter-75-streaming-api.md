# Chapter 17.2: Streaming Responses — SSE & WebSockets

> **Phase 17 — Production & Deployment** | [← Previous: FastAPI Integration](chapter-74-fastapi-integration.md) | [Next: Observability →](chapter-76-observability.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Explain why streaming matters for LLM UX and time-to-first-token
- ✅ Stream LangChain/LCEL output with `.stream()` and `.astream()`
- ✅ Expose **Server-Sent Events (SSE)** from FastAPI for browser clients
- ✅ Stream **LangGraph** runs with `stream_mode` and graph events
- ✅ Choose between SSE, WebSockets, and plain chunked HTTP
- ✅ Handle client disconnects, backpressure, and proxy timeouts safely

| | |
|---|---|
| **Prerequisites** | Chapter 17.1 (FastAPI + LangChain/LangGraph Integration) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction — Why Stream?

Without streaming, the user stares at a spinner until the **entire** completion finishes:

```
NON-STREAMING:
  User sends question ──► [........ 8 seconds ........] ──► Full answer appears

STREAMING:
  User sends question ──► "The" ──► " revenue" ──► " for" ──► " Q3" ──► ...
                          ↑
                    ~300ms to first token — feels instant
```

Production chat UIs (ChatGPT, Claude, Copilot) all stream. Your API should too.

**Three transport options:**

| Transport | Direction | Best for |
|-----------|-----------|----------|
| **SSE** | Server → client (one-way) | Chat tokens, progress events, simple browsers |
| **WebSocket** | Bidirectional | Live tool status, cancel mid-run, multi-turn over one socket |
| **Chunked HTTP** | Server → client | CLI clients, `curl`, minimal setup |

For most RAG/chat APIs, **SSE + FastAPI `StreamingResponse`** is the sweet spot.

---

## Part 1: LangChain Streaming Basics

### `.stream()` vs `.astream()`

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE"),
)

prompt = ChatPromptTemplate.from_template("Explain {topic} in two sentences.")
chain = prompt | llm | StrOutputParser()

# Sync — scripts, notebooks
for chunk in chain.stream({"topic": "vector databases"}):
    print(chunk, end="", flush=True)

# Async — FastAPI, asyncio
async def print_async():
    async for chunk in chain.astream({"topic": "embeddings"}):
        print(chunk, end="", flush=True)
```

### `.astream_events()` — Step-Level Visibility

Use when you need **retrieval finished**, **LLM started**, etc. (great for RAG progress bars):

```python
async def debug_chain(question: str):
    rag_chain = ...  # your LCEL RAG chain

    async for event in rag_chain.astream_events(
        {"question": question},
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            token = event["data"]["chunk"].content
            if token:
                yield token
        elif kind == "on_retriever_end":
            yield f"\n[retrieved {len(event['data']['output'])} docs]\n"
```

---

## Part 2: SSE with FastAPI

### SSE Format

Each event is text framed as:

```
data: {"type":"token","content":"Hello"}\n\n
```

Browsers use `EventSource`; mobile apps often use SSE libraries or raw HTTP read loops.

### Minimal SSE Endpoint

```python
# app/routers/stream.py
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.chains.chat_chain import get_chat_chain

router = APIRouter(prefix="/v1", tags=["stream"])


class StreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)


def sse_pack(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def chat_stream(body: StreamRequest):
    chain = get_chat_chain()

    async def event_generator():
        yield sse_pack({"type": "start", "message": body.message})
        try:
            async for chunk in chain.astream({"question": body.message}):
                if chunk:
                    yield sse_pack({"type": "token", "content": chunk})
            yield sse_pack({"type": "done"})
        except Exception as exc:
            yield sse_pack({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx: disable response buffering
        },
    )
```

### Register the Router

```python
# app/main.py
from app.routers.stream import router as stream_router

app.include_router(stream_router)
```

### Browser Client (JavaScript)

```javascript
async function streamChat(message) {
  const res = await fetch("/v1/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";

    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const payload = JSON.parse(line.slice(5).trim());
      if (payload.type === "token") process.stdout.write(payload.content);
    }
  }
}
```

> **Note:** Native `EventSource` only supports **GET**. For POST bodies (typical for chat), use `fetch` + stream reader as above, or expose GET with query params for demos only.

---

## Part 3: Streaming LangGraph

LangGraph exposes multiple **stream modes**. For APIs, `stream_mode="messages"` or `"updates"` is common:

```python
# app/graph/agent.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE"),
)

# ... build graph: agent node, tool node, etc.
# graph = builder.compile(checkpointer=MemorySaver())


async def stream_agent(graph, user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    async for mode, chunk in graph.astream(
        {"messages": [("user", user_input)]},
        config=config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, metadata = chunk
            if msg.content:
                yield {"type": "token", "node": metadata.get("langgraph_node"), "content": msg.content}
        elif mode == "updates":
            yield {"type": "state", "update": chunk}
```

### FastAPI SSE Wrapper for LangGraph

```python
@router.post("/agent/stream")
async def agent_stream(body: StreamRequest, thread_id: str = "default"):
    from app.graph.agent import graph, stream_agent

    async def gen():
        async for event in stream_agent(graph, body.message, thread_id):
            yield sse_pack(event)

    return StreamingResponse(gen(), media_type="text/event-stream")
```

---

## Part 4: WebSockets — When You Need Bidirectional Control

Use WebSockets when the client must **cancel**, **send follow-ups without new HTTP requests**, or receive **tool-call progress** while sending heartbeats.

```python
# app/routers/ws.py
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("action") == "stop":
                break  # cooperative cancel flag in shared task registry

            chain = get_chat_chain()
            async for chunk in chain.astream({"question": msg["text"]}):
                await ws.send_json({"type": "token", "content": chunk})
            await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
```

**Production tips:**

- Authenticate on connect (JWT query param or first message).
- Cap message size and rate per connection.
- Run the graph in a **background task** so `stop` can cancel via `asyncio.Task.cancel()`.

---

## Part 5: Production Hardening

### Client Disconnects

```python
from starlette.requests import Request

async def event_generator(request: Request, chain, question: str):
    try:
        async for chunk in chain.astream({"question": question}):
            if await request.is_disconnected():
                break
            yield sse_pack({"type": "token", "content": chunk})
    finally:
        yield sse_pack({"type": "done"})
```

Pass `request: Request` into the route and forward it to the generator.

### Reverse Proxy Timeouts

Nginx and load balancers often default to **60s** read timeouts. Long agent runs need:

```nginx
# nginx snippet
location /v1/ {
    proxy_read_timeout 300s;
    proxy_buffering off;
}
```

### Heartbeats for Idle Streams

During tool calls, tokens may pause for seconds. Send comments or ping events:

```python
async def gen_with_heartbeat(chain, question: str):
    import asyncio
    queue: asyncio.Queue = asyncio.Queue()

    async def producer():
        async for chunk in chain.astream({"question": question}):
            await queue.put(sse_pack({"type": "token", "content": chunk}))
        await queue.put(sse_pack({"type": "done"}))

    task = asyncio.create_task(producer())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=15.0)
                yield item
                if '"type": "done"' in item:
                    break
            except asyncio.TimeoutError:
                yield ": ping\n\n"  # SSE comment — keeps connection alive
    finally:
        task.cancel()
```

### Structured Event Schema

Define a stable contract for frontends:

```python
# types: start | token | tool_start | tool_end | retrieval | error | done
{"type": "tool_start", "name": "search_docs", "input": {"query": "..."}}
```

Version the schema (`v1`) in a header or first `start` event.

---

## Common Mistakes

### Mistake 1: Buffering the full response before sending
```python
# ❌ Defeats streaming — user waits for full completion
text = await chain.ainvoke({"question": q})
return {"answer": text}

# ✅ Use astream + StreamingResponse
```

### Mistake 2: Wrong `media_type`
```python
# ❌ Browsers won't parse SSE
return StreamingResponse(gen(), media_type="application/json")

# ✅
media_type="text/event-stream"
```

### Mistake 3: Forgetting proxy buffering
```python
# ❌ nginx buffers 8KB — user sees nothing until buffer fills

# ✅ X-Accel-Buffering: no + proxy_buffering off
```

### Mistake 4: Streaming without tracing
```python
# ❌ LangSmith shows one long span with no token visibility

# ✅ Pass config with callbacks / enable LANGCHAIN_TRACING_V2
config = {"callbacks": [ProductionCallbackHandler()]}
async for chunk in chain.astream(input, config=config):
    ...
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Prefer SSE for one-way token streams | Simple, HTTP-friendly, works through many corporate proxies |
| Use WebSockets for cancel + bidirectional agent UIs | Stop run, multi-turn on one connection |
| Send typed JSON events, not raw text only | Frontends can show retrieval vs generation phases |
| Handle disconnects and cancel upstream tasks | Avoid wasted LLM spend after user leaves |
| Tune proxy/load balancer timeouts | Agent + tool runs exceed 60s regularly |
| Heartbeat during silent tool execution | Prevents idle connection drops |
| Log time-to-first-token (TTFT) | Core SLA metric for chat APIs |
| Rate-limit streaming endpoints | Streaming doesn't bypass abuse controls |

---

## Interview Preparation

### Easy
**Q: Why do production LLM APIs use streaming?**

> Streaming improves **perceived latency** via time-to-first-token, keeps connections active during long generations, and matches user expectations from commercial chat products. It also allows partial rendering and early cancellation, which reduces wasted compute when users abandon a response.

### Medium
**Q: SSE vs WebSockets for a RAG chat API?**

> **SSE** is ideal when the server pushes tokens and metadata one-way over HTTP/1.1 or HTTP/2 — easy to implement in FastAPI, works with standard load balancers, and fits most chat UIs. **WebSockets** add bidirectional messaging for cancellation, typing indicators, or multiple in-flight tool updates without opening new requests. Choose SSE unless you need true bidirectional control or binary frames.

### Hard
**Q: How would you stream a multi-step LangGraph agent while showing tool progress?**

> Compile the graph with checkpointing, then use `astream` with `stream_mode=["messages", "updates"]` (or `astream_events` for fine-grained hooks). Map graph events to a **versioned SSE schema**: `retrieval`, `tool_start`, `tool_end`, `token`, `error`, `done`. Run the stream in a cancellable asyncio task; on client disconnect or `stop` message, cancel the task and optionally roll back partial UI state. Emit heartbeats during long tool calls, disable proxy buffering, and record TTFT plus total latency in LangSmith via run metadata.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **`.astream()`** | Async token/chunk stream from LCEL chains |
| **`astream_events(v2)`** | Stream events from every runnable step |
| **SSE** | One-way server push over HTTP (`text/event-stream`) |
| **WebSocket** | Bidirectional stream for cancel and rich agent UIs |
| **LangGraph `stream_mode`** | Control what the graph emits while running |
| **TTFT** | Time to first token — key UX/latency metric |

---

## Exercises

1. **SSE RAG stream:** Add `/v1/rag/stream` that emits `retrieval` then `token` events using `astream_events`.
2. **TTFT metric:** Log milliseconds from request start to first token; expose in response headers.
3. **Cancel:** WebSocket `stop` action that cancels the in-flight `astream` task.
4. **nginx config:** Write a snippet with `proxy_buffering off` and 300s read timeout for your API path.

---

## What's Next

Streaming makes the app feel fast; **observability** tells you whether it *is* fast (and correct) in production. Next you'll wire **LangSmith**, custom callbacks, and structured logging so every streamed run is traceable.

---

> [← Previous: FastAPI Integration](chapter-74-fastapi-integration.md) | [Next: Observability →](chapter-76-observability.md)

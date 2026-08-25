# Chapter 4.4: Streaming & Async Chains

> **Phase 4 — Chains & Runnables** | [← Previous: Retry, Fallback & Error Handling](chapter-20-retry-fallback-error-handling.md) | [Next: Phase 5 — Memory & Chat History →](../phase-05-memory/chapter-22-conversation-memory.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Stream LLM output token-by-token with `.stream()` and `.astream()`
- ✅ Stream intermediate chain events with `.astream_events()`
- ✅ Use async chains with `.ainvoke()` and `.abatch()`
- ✅ Process multiple inputs concurrently with `.batch()`
- ✅ Understand when to use sync vs async vs streaming

| | |
|---|---|
| **Prerequisites** | Chapter 4.3 (Retry & Fallback), Chapter 0.4 (Async Python) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 40 minutes |

---

## Introduction

Without streaming, the user stares at a blank screen for 5-10 seconds:

```
❌ Without streaming:
[User types question] ──→ [......5 seconds of nothing......] ──→ [Full answer appears]

✅ With streaming:
[User types question] ──→ [Tokens appear word-by-word instantly] ──→ [Done]
```

And without async, you process one request at a time:

```
❌ Sync (sequential):    User A [3s] → User B [3s] → User C [3s] = 9 seconds total
✅ Async (concurrent):   User A [3s] ─┐
                         User B [3s] ─┤ = 3 seconds total
                         User C [3s] ─┘
```

---

## Part 1: `.stream()` — Token-by-Token Output

Every Runnable in LangChain supports `.stream()`:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "standard"),
    temperature=0.7,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=300
)

# Basic streaming from LLM
chain = (
    ChatPromptTemplate.from_template("Tell me a joke about {topic}.")
    | llm
    | StrOutputParser()
)

# .stream() returns a generator — yields chunks as they arrive
for chunk in chain.stream({"topic": "programming"}):
    print(chunk, end="", flush=True)  # prints token by token
print()  # newline at end
```

### What Each Component Streams

```python
# LLM: streams AIMessageChunks (token by token)
for chunk in llm.stream("Tell me a joke"):
    print(chunk.content, end="")

# StrOutputParser: streams strings (passes through LLM chunks)
chain = llm | StrOutputParser()
for chunk in chain.stream("Tell me a joke"):
    print(chunk, end="")  # chunk is a string

# Prompt: does NOT stream — emits one complete value
# RunnableLambda: does NOT stream (by default) — emits one value
```

### Collecting Stream into a Full String

```python
# If you need the full response but want to stream too:
full_response = ""
for chunk in chain.stream({"topic": "cats"}):
    print(chunk, end="", flush=True)
    full_response += chunk

print(f"\n\nFull response ({len(full_response)} chars): {full_response}")
```

---

## Part 2: `.astream()` — Async Streaming

Same as `.stream()` but for async contexts (FastAPI, async scripts):

```python
import asyncio

async def stream_response(topic: str):
    chain = (
        ChatPromptTemplate.from_template("Explain {topic} in 3 sentences.")
        | llm
        | StrOutputParser()
    )

    # astream = async version of stream
    async for chunk in chain.astream({"topic": topic}):
        print(chunk, end="", flush=True)
    print()

# Run it
asyncio.run(stream_response("machine learning"))
```

### FastAPI Streaming Endpoint (Real-World)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/stream")
async def stream_answer(question: str):
    chain = (
        ChatPromptTemplate.from_template("Answer: {question}")
        | llm
        | StrOutputParser()
    )

    async def generate():
        async for chunk in chain.astream({"question": question}):
            yield chunk  # sends each token to the client

    return StreamingResponse(generate(), media_type="text/plain")
```

---

## Part 3: `.astream_events()` — Stream Everything (v2)

The most powerful streaming method — streams events from **every step** in the chain:

```python
import asyncio

async def stream_with_events():
    chain = (
        ChatPromptTemplate.from_template("Write a haiku about {topic}.")
        | llm
        | StrOutputParser()
    )

    async for event in chain.astream_events(
        {"topic": "rain"},
        version="v2"  # Always use v2
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # Token-by-token from LLM
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)

        elif kind == "on_chain_start":
            print(f"\n🔗 Chain started: {event['name']}")

        elif kind == "on_chain_end":
            print(f"\n✅ Chain ended: {event['name']}")

    print()

asyncio.run(stream_with_events())
```

### Key Event Types

| Event | When | Data |
|-------|------|------|
| `on_chain_start` | A chain/runnable begins | `{"input": ...}` |
| `on_chain_end` | A chain/runnable completes | `{"output": ...}` |
| `on_chat_model_start` | LLM call begins | `{"messages": [...]}` |
| `on_chat_model_stream` | Each LLM token arrives | `{"chunk": AIMessageChunk}` |
| `on_chat_model_end` | LLM call completes | `{"output": AIMessage}` |
| `on_parser_start` | Parser begins | `{"input": ...}` |
| `on_parser_end` | Parser completes | `{"output": ...}` |

### Filtering Events

```python
async for event in chain.astream_events(
    {"topic": "AI"},
    version="v2",
    include_types=["on_chat_model_stream"]  # Only LLM tokens
):
    print(event["data"]["chunk"].content, end="")
```

---

## Part 4: `.ainvoke()` and `.abatch()` — Async Execution

### `.ainvoke()` — Single async call

```python
import asyncio

async def main():
    chain = (
        ChatPromptTemplate.from_template("Explain {topic}.")
        | llm
        | StrOutputParser()
    )

    # Non-blocking — other async tasks can run while waiting
    result = await chain.ainvoke({"topic": "neural networks"})
    print(result)

asyncio.run(main())
```

### `.abatch()` — Multiple inputs concurrently

```python
import asyncio
import time

async def batch_demo():
    chain = (
        ChatPromptTemplate.from_template("Define {term} in one sentence.")
        | llm
        | StrOutputParser()
    )

    topics = [
        {"term": "machine learning"},
        {"term": "deep learning"},
        {"term": "natural language processing"},
        {"term": "computer vision"},
        {"term": "reinforcement learning"}
    ]

    start = time.time()
    # All 5 run concurrently!
    results = await chain.abatch(topics)
    elapsed = time.time() - start

    for topic, result in zip(topics, results):
        print(f"📖 {topic['term']}: {result[:80]}...")

    print(f"\n⏱️ {len(topics)} calls in {elapsed:.1f}s")

asyncio.run(batch_demo())
```

### `.batch()` — Sync version (uses thread pool)

```python
# Sync batch — uses threads internally for parallelism
results = chain.batch([
    {"term": "AI"},
    {"term": "ML"},
    {"term": "NLP"}
])
# Returns list of results
```

### `.batch()` with Max Concurrency

```python
# Limit concurrent calls (to avoid rate limits)
results = chain.batch(
    [{"term": t} for t in ["AI", "ML", "NLP", "CV", "RL"]],
    config={"max_concurrency": 2}  # Only 2 at a time
)
```

---

## Part 5: Making Custom Runnables Streamable

By default, `RunnableLambda` does NOT stream. Use `RunnableGenerator` for stream-aware transforms:

```python
from langchain_core.runnables import RunnableGenerator

def streaming_transform(chunks):
    """Transform that processes the stream chunk by chunk."""
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        # Yield complete sentences
        while ". " in buffer:
            sentence, buffer = buffer.split(". ", 1)
            yield sentence + ". "
    if buffer:
        yield buffer

chain = (
    ChatPromptTemplate.from_template("Tell a story about {topic}.")
    | llm
    | StrOutputParser()
    | RunnableGenerator(streaming_transform)
)

for chunk in chain.stream({"topic": "a wizard"}):
    print(chunk, end="", flush=True)
```

---

## The Methods Compared

| Method | Sync/Async | Returns | Use Case |
|--------|-----------|---------|----------|
| `.invoke(input)` | Sync | Single result | Simple scripts, one-off calls |
| `.ainvoke(input)` | Async | Single result | Web servers, concurrent code |
| `.stream(input)` | Sync | Generator of chunks | CLI apps, token-by-token display |
| `.astream(input)` | Async | AsyncGenerator of chunks | FastAPI streaming endpoints |
| `.batch(inputs)` | Sync | List of results | Process many inputs in parallel |
| `.abatch(inputs)` | Async | List of results | Async batch processing |
| `.astream_events(input)` | Async | Stream of all events | Debug, progress tracking, complex UIs |

---

## Common Mistakes

### Mistake 1: Forgetting `flush=True` when printing stream
```python
# ❌ Tokens buffer — appear in bursts
for chunk in chain.stream(input):
    print(chunk, end="")

# ✅ Flush forces immediate display
for chunk in chain.stream(input):
    print(chunk, end="", flush=True)
```

### Mistake 2: Using `.invoke()` in async context
```python
# ❌ Blocks the event loop — other requests wait
@app.get("/answer")
async def answer(q: str):
    return chain.invoke({"question": q})  # BLOCKING!

# ✅ Use ainvoke — non-blocking
@app.get("/answer")
async def answer(q: str):
    return await chain.ainvoke({"question": q})
```

### Mistake 3: Not limiting batch concurrency
```python
# ❌ 100 concurrent LLM calls — instant rate limit!
results = chain.batch([{"q": q} for q in questions_100])

# ✅ Limit concurrency
results = chain.batch(
    [{"q": q} for q in questions_100],
    config={"max_concurrency": 5}
)
```

### Mistake 4: Forgetting `version="v2"` in astream_events
```python
# ❌ Old v1 format — deprecated, different event structure
async for event in chain.astream_events(input):
    ...

# ✅ Always specify v2
async for event in chain.astream_events(input, version="v2"):
    ...
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use `.stream()` for user-facing apps | Instant feedback, better UX |
| Use `.ainvoke()` in web servers (FastAPI) | Non-blocking, concurrent requests |
| Use `.batch()` with `max_concurrency` for bulk processing | Prevents rate limits |
| Use `.astream_events(version="v2")` for complex chains | See what's happening at each step |
| Always `flush=True` when printing stream chunks | Immediate token display |
| Use `RunnableGenerator` for stream-aware transforms | Maintains streaming through the chain |

---

## Interview Preparation

### Easy
**Q: What's the difference between `.invoke()` and `.stream()`?**

> `.invoke()` waits for the complete result and returns it all at once. `.stream()` returns a generator that yields chunks (tokens) as they arrive from the LLM. Streaming gives users immediate feedback instead of waiting for the full response.

### Medium
**Q: When would you use `.batch()` vs `.abatch()`?**

> Use `.batch()` in synchronous code — it internally uses a thread pool for parallelism. Use `.abatch()` in async code (FastAPI, asyncio scripts) — it uses `asyncio.gather()` for true async concurrency. Both process multiple inputs in parallel, but `.abatch()` is more efficient in async contexts because it doesn't create extra threads.

### Hard
**Q: How does `.astream_events()` differ from `.astream()`?**

> `.astream()` only gives you the **final output chunks** from the last step. `.astream_events(version="v2")` gives you **every event from every step** — chain starts/ends, LLM token chunks, parser outputs, etc. Use `astream_events` when you need progress tracking, debugging, or to show users which step is currently running in a multi-step chain.

### Senior
**Q: How would you implement streaming in a sequential chain where intermediate steps need the full output?**

> This is a fundamental tension. In a chain like `prompt | llm | parser | next_prompt | llm`, the first LLM can stream, but `next_prompt` needs the **full** parsed output before it can render. The solution: use `.astream_events()` — it lets you stream the first LLM's tokens to the UI while internally buffering for the next step. The user sees progress from both LLM calls. Alternatively, split the chain and stream each segment separately.

---

## Mini Assignment — Answer

Build an **Async Streaming Q&A System** with event tracking and batch processing:

```python
import os
import asyncio
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "standard"),
    temperature=0.7,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=200
)

chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer concisely."),
        ("human", "{question}")
    ])
    | llm
    | StrOutputParser()
)


# --- Part 1: Streaming with event tracking ---
async def stream_with_tracking(question: str):
    """Stream tokens and track LLM start/end events."""
    token_count = 0
    start = time.time()

    print(f"\n{'='*50}")
    print(f"📝 Question: {question}")
    print(f"{'='*50}")

    async for event in chain.astream_events(
        {"question": question},
        version="v2"
    ):
        kind = event["event"]

        if kind == "on_chat_model_start":
            print("⏳ Thinking...")

        elif kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
                token_count += 1

        elif kind == "on_chat_model_end":
            elapsed = time.time() - start
            print(f"\n✅ Done! ({token_count} tokens in {elapsed:.1f}s)")

    return token_count


# --- Part 2: Batch processing with timing ---
async def batch_with_timing(questions: list[str]):
    """Process multiple questions concurrently."""
    inputs = [{"question": q} for q in questions]

    start = time.time()
    results = await chain.abatch(inputs, config={"max_concurrency": 2})
    elapsed = time.time() - start

    print(f"\n{'='*50}")
    print(f"📦 Batch Results ({len(questions)} questions in {elapsed:.1f}s)")
    print(f"{'='*50}")

    for question, result in zip(questions, results):
        print(f"\n📝 Q: {question}")
        print(f"💬 A: {result[:100]}...")

    return results


# --- Run both ---
async def main():
    # Part 1: Stream a single question
    await stream_with_tracking("What is the difference between TCP and UDP?")

    # Part 2: Batch 3 questions
    await batch_with_timing([
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?"
    ])


asyncio.run(main())
```

### Expected Output:
```
==================================================
📝 Question: What is the difference between TCP and UDP?
==================================================
⏳ Thinking...
TCP is a connection-oriented protocol that ensures reliable...
✅ Done! (87 tokens in 2.3s)

==================================================
📦 Batch Results (3 questions in 1.8s)
==================================================

📝 Q: What is Python?
💬 A: Python is a high-level, interpreted programming language known for...

📝 Q: What is JavaScript?
💬 A: JavaScript is a dynamic programming language primarily used for...

📝 Q: What is Rust?
💬 A: Rust is a systems programming language focused on safety...
```

---

## Summary

| Component | What It Does |
|-----------|-------------|
| `.stream(input)` | Yields output chunks synchronously (token by token) |
| `.astream(input)` | Async version of stream (for FastAPI, asyncio) |
| `.astream_events(input, version="v2")` | Streams events from every step in the chain |
| `.ainvoke(input)` | Single async call (non-blocking) |
| `.batch(inputs, config)` | Parallel processing with thread pool |
| `.abatch(inputs, config)` | Async parallel processing |
| `max_concurrency` config | Limits concurrent calls to prevent rate limits |
| `RunnableGenerator` | Makes custom transforms stream-aware |

---

## 🎉 Phase 4 Complete!

You've mastered the **Chains & Runnables** toolkit:

| Chapter | What You Learned |
|---------|-----------------|
| 4.1 — Runnables Deep Dive | `RunnableLambda`, `RunnablePassthrough`, `RunnableParallel` |
| 4.2 — Sequential Chains | Data accumulation, multi-step workflows, debugging |
| 4.3 — Retry & Fallback | `.with_retry()`, `.with_fallbacks()`, `RunnableBranch` |
| 4.4 — Streaming & Async | `.stream()`, `.astream_events()`, `.batch()`, `.ainvoke()` |

**Next up: Phase 5 — Memory & Chat History** 🧠

---

> [← Previous: Retry, Fallback & Error Handling](chapter-20-retry-fallback-error-handling.md) | [Next: Phase 5 — Memory & Chat History →](../phase-05-memory/chapter-22-conversation-memory.md)

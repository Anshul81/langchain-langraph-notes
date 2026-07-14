# Chapter 0.5: Async Python — Concurrent Programming with `async`/`await`

> **Phase 0 — Python Power-Up** | [← Previous: Type Hints & Pydantic](chapter-03-type-hints-pydantic.md) | [Next: OOP Patterns →](chapter-05-oop-patterns.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand WHY async exists and what problem it solves
- ✅ Use `async def`, `await`, `asyncio.run()`, and `asyncio.gather()`
- ✅ Know the difference between sync and async code
- ✅ Write concurrent programs that don't waste time waiting
- ✅ Understand why LangChain has `ainvoke`, `astream`, `abatch`

| | |
|---|---|
| **Prerequisites** | Python functions, basic understanding of APIs |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

### The Problem

Imagine you need to call an LLM 5 times:

```python
# ❌ SYNCHRONOUS — Each call waits for the previous one
result1 = llm.invoke("Question 1")   # 2 seconds ⏳
result2 = llm.invoke("Question 2")   # 2 seconds ⏳
result3 = llm.invoke("Question 3")   # 2 seconds ⏳
result4 = llm.invoke("Question 4")   # 2 seconds ⏳
result5 = llm.invoke("Question 5")   # 2 seconds ⏳
# Total: 10 seconds! 😱
```

Each call sits idle for 2 seconds, waiting for the network response. During that time, your CPU does **nothing**.

```python
# ✅ ASYNC — All 5 calls happen simultaneously
results = await asyncio.gather(
    llm.ainvoke("Question 1"),
    llm.ainvoke("Question 2"),
    llm.ainvoke("Question 3"),
    llm.ainvoke("Question 4"),
    llm.ainvoke("Question 5"),
)
# Total: ~2 seconds! 🚀 (all run in parallel)
```

**5x faster** — same work, same computer.

### The Solution

Python's `asyncio` module enables **asynchronous programming** — a way to run multiple I/O-bound operations concurrently on a single thread.

### History

Async programming in Python evolved over many years:
- **Python 3.3** (2012): `yield from` for generator delegation
- **Python 3.4** (2014): `asyncio` module introduced
- **Python 3.5** (2015): `async`/`await` syntax added via [PEP 492](https://peps.python.org/pep-0492/)
- **Python 3.7** (2018): `asyncio.run()` added for easy entry point

### Industry Usage

- **LangChain** — `ainvoke()`, `astream()`, `abatch()` for concurrent LLM calls
- **FastAPI** — Async by default for handling many concurrent HTTP requests
- **Web scraping** — Fetching hundreds of pages concurrently
- **Database access** — Non-blocking queries with async drivers
- **Multi-agent systems** — Agents working concurrently

### Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| Async makes code faster | Async makes I/O-bound code more **efficient** — it doesn't speed up CPU computation |
| Async uses multiple threads | Async runs on a **single thread** using an event loop |
| You always need async | For simple scripts or CPU-bound work, sync is fine |
| `await` runs something in parallel | `await` **pauses** the current task; `asyncio.gather()` runs things concurrently |

---

## Mental Model

### The Coffee Shop Analogy

**Synchronous (1 cashier, 1 order at a time):**

```
Customer 1: Order → Wait for coffee → ☕ Get coffee → Leave
                                                        ↓
Customer 2:                                        Order → Wait → ☕ → Leave
                                                                          ↓
Customer 3:                                                          Order → Wait → ☕ → Leave

Timeline: ████████████████████████████████████  (30 minutes)
```

**Asynchronous (1 cashier, takes all orders, coffees brew in parallel):**

```
Customer 1: Order → ☕ Get coffee
Customer 2: Order → ☕ Get coffee
Customer 3: Order → ☕ Get coffee
                ↕ (all brewing simultaneously)

Timeline: ████████████  (10 minutes)
```

The cashier doesn't **wait** for each coffee to brew. They take the next order while the machine works. That's async.

### Key Insight

> Async is NOT about doing things faster. It's about **not wasting time waiting**.

Your CPU is the cashier. Network calls (LLM APIs, databases, file reads) are the coffee machine. While the machine works, the cashier should take more orders — not stand idle.

### Visual Diagram

```
SYNCHRONOUS (Sequential):
┌──────┐     ┌──────┐     ┌──────┐
│ LLM  │     │ LLM  │     │ LLM  │
│ Call  │────▶│ Call  │────▶│ Call  │
│  1   │     │  2   │     │  3   │
└──────┘     └──────┘     └──────┘
|--- 2s ---|--- 2s ---|--- 2s ---|
Total: 6 seconds

ASYNCHRONOUS (Concurrent):
┌──────┐
│ LLM  │
│ Call  │
│  1   │
├──────┤
│ LLM  │     All running
│ Call  │     at the same
│  2   │     time!
├──────┤
│ LLM  │
│ Call  │
│  3   │
└──────┘
|--- 2s ---|
Total: 2 seconds
```

---

## Theory

### Part 1: The Keywords

| Keyword | What It Does |
|---------|--------------|
| `async def` | Declares a function as a **coroutine** (async function) |
| `await` | Pauses THIS function until the awaited operation finishes, lets OTHER functions run meanwhile |
| `asyncio.run()` | Starts the async event loop and runs a coroutine |
| `asyncio.gather()` | Runs multiple coroutines **concurrently** |
| `asyncio.sleep()` | Non-blocking sleep (unlike `time.sleep()` which blocks) |

### Part 2: Sync vs Async — Side by Side

#### Synchronous Version

```python
import time

def fetch_data_sync(name, seconds):
    print(f"  {name}: Starting...")
    time.sleep(seconds)  # BLOCKS — nothing else can run
    print(f"  {name}: Done!")
    return f"{name} result"

def main_sync():
    print("SYNC START")
    result1 = fetch_data_sync("API-1", 2)
    result2 = fetch_data_sync("API-2", 2)
    result3 = fetch_data_sync("API-3", 2)
    print(f"Results: {result1}, {result2}, {result3}")

main_sync()
# Total time: ~6 seconds (2+2+2)
```

#### Asynchronous Version

```python
import asyncio

async def fetch_data_async(name, seconds):
    print(f"  {name}: Starting...")
    await asyncio.sleep(seconds)  # NON-BLOCKING — other tasks can run
    print(f"  {name}: Done!")
    return f"{name} result"

async def main_async():
    print("ASYNC START")
    results = await asyncio.gather(
        fetch_data_async("API-1", 2),
        fetch_data_async("API-2", 2),
        fetch_data_async("API-3", 2),
    )
    print(f"Results: {results}")

asyncio.run(main_async())
# Total time: ~2 seconds (all 3 run simultaneously!)
```

### Part 3: What `await` Actually Does

```python
async def make_llm_call():
    print("1. Sending request to OpenAI...")
    result = await call_openai_api()   # ← PAUSE here
    #                                       While paused, other tasks can run!
    print("2. Got response!")           # ← Resumes when API responds
    return result
```

```
Without await:    Task A ████████████████░░░░░░░░  (blocked, waiting)
                  Task B ░░░░░░░░░░░░░░░░████████  (can't start)

With await:       Task A ████░░░░░░░░████          (paused while waiting)
                  Task B ░░░░████████░░░░          (runs during A's wait)
                              ↑
                         A is awaiting, so B gets CPU time
```

### Part 4: Running Async Code

```python
import asyncio

async def say_hello():
    print("Hello!")
    await asyncio.sleep(1)
    print("World!")

# Method 1: asyncio.run() — Use this in scripts
asyncio.run(say_hello())

# Method 2: await — Use this inside another async function
async def main():
    await say_hello()
```

### Part 5: `asyncio.gather()` — Run Multiple Tasks Concurrently

```python
import asyncio

async def task(name, duration):
    print(f"{name} started")
    await asyncio.sleep(duration)
    print(f"{name} finished")
    return f"{name}: done"

async def main():
    # All 3 tasks run CONCURRENTLY
    results = await asyncio.gather(
        task("A", 2),
        task("B", 3),
        task("C", 1),
    )
    print(results)  # ['A: done', 'B: done', 'C: done']

asyncio.run(main())
# Total time: ~3 seconds (not 2+3+1=6)
# Because they run simultaneously, total = max(2,3,1) = 3
```

### Part 6: Sequential vs Concurrent Awaits

```python
# ❌ SEQUENTIAL — These run one after another (6 seconds)
async def slow():
    r1 = await task(2)  # Wait 2s
    r2 = await task(2)  # Then wait 2s
    r3 = await task(2)  # Then wait 2s
    # Total: 6 seconds

# ✅ CONCURRENT — These run simultaneously (2 seconds)
async def fast():
    r1, r2, r3 = await asyncio.gather(
        task(2),  # All start
        task(2),  # at the
        task(2),  # same time
    )
    # Total: 2 seconds
```

---

## Architecture

### The Event Loop

The event loop is the heart of asyncio. It's a single-threaded loop that:
1. Checks which coroutines are ready to run
2. Runs them until they hit an `await`
3. Switches to another ready coroutine
4. Repeats

```
┌─────────────────────────────────────────────────┐
│                  EVENT LOOP                      │
│                                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Task A  │  │ Task B  │  │ Task C  │        │
│  │(running)│  │(waiting)│  │(waiting)│        │
│  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │               │
│       ▼            │            │               │
│  Task A hits       │            │               │
│  'await' ──────────┘            │               │
│       │      Task B runs        │               │
│       │            │            │               │
│       │       Task B hits       │               │
│       │       'await' ──────────┘               │
│       │            │      Task C runs           │
│       │            │            │               │
│  Task A ready ◄────┘            │               │
│  (I/O complete)                 │               │
│       │                         │               │
│  Task A resumes                 │               │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## Practical Example

### Parallel API Calls

```python
import asyncio
import time

async def call_api(name, duration):
    """Simulate an API call with a given duration."""
    print(f"Starting {name}...")
    await asyncio.sleep(duration)
    print(f"Finished {name} in {duration}s")
    return f"{name} result"

async def parallel_api_calls():
    """Run multiple API calls concurrently and measure total time."""
    start = time.time()

    results = await asyncio.gather(
        call_api("API 1", 1),
        call_api("API 2", 2),
        call_api("API 3", 3)
    )

    elapsed = time.time() - start
    print("Results:", results)
    print(f"Total time: {elapsed:.1f}s")

async def main():
    await parallel_api_calls()

asyncio.run(main())
```

**Output:**
```
Starting API 1...
Starting API 2...
Starting API 3...
Finished API 1 in 1s
Finished API 2 in 2s
Finished API 3 in 3s
Results: ['API 1 result', 'API 2 result', 'API 3 result']
Total time: 3.0s
```

Notice all three start **immediately** — total time is `max(1, 2, 3) = 3` seconds, not `1 + 2 + 3 = 6`.

---

## Real Industry Example

### LangChain Async Methods

LangChain provides async versions of **every** major method:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# Sync versions
result = llm.invoke("Hello")
results = llm.batch(["Q1", "Q2", "Q3"])
for chunk in llm.stream("Hello"):
    print(chunk.content, end="")

# Async versions (prefix with 'a')
result = await llm.ainvoke("Hello")
results = await llm.abatch(["Q1", "Q2", "Q3"])
async for chunk in llm.astream("Hello"):
    print(chunk.content, end="")
```

| Sync Method | Async Method | Use When |
|-------------|-------------|----------|
| `invoke()` | `ainvoke()` | Single call |
| `batch()` | `abatch()` | Multiple inputs |
| `stream()` | `astream()` | Streaming output |

### FastAPI + Async LangChain

```python
from fastapi import FastAPI
from langchain_openai import ChatOpenAI

app = FastAPI()
llm = ChatOpenAI(model="gpt-4")

@app.get("/chat")
async def chat(question: str):
    # This doesn't block the server while waiting for OpenAI
    result = await llm.ainvoke(question)
    return {"answer": result.content}
```

**Why async matters here:** Without async, each HTTP request blocks the entire server while waiting for OpenAI's response (~2 seconds). With async, the server can handle hundreds of concurrent requests.

---

## Common Mistakes

### Mistake 1: Calling async function without `await`

```python
async def get_data():
    return "data"

# ❌ Returns a coroutine OBJECT, not the result!
result = get_data()
print(result)  # <coroutine object get_data at 0x...>

# ✅ Must await it
result = await get_data()
print(result)  # "data"
```

### Mistake 2: Using `await` outside an async function

```python
# ❌ SyntaxError — await can only be used inside async def
def main():
    result = await get_data()

# ✅ Must be inside async def
async def main():
    result = await get_data()
```

### Mistake 3: Using `time.sleep()` instead of `asyncio.sleep()`

```python
# ❌ time.sleep BLOCKS the entire event loop — no other tasks can run!
async def bad_task():
    time.sleep(2)  # Blocks everything for 2 seconds

# ✅ asyncio.sleep yields control to other tasks
async def good_task():
    await asyncio.sleep(2)  # Other tasks can run during this wait
```

**This is the #1 async mistake.** `time.sleep()` in async code defeats the entire purpose.

### Mistake 4: Sequential awaits when you want concurrent

```python
# ❌ These run one after another (6 seconds total)
async def slow():
    r1 = await task(2)
    r2 = await task(2)
    r3 = await task(2)

# ✅ These run concurrently (2 seconds total)
async def fast():
    r1, r2, r3 = await asyncio.gather(task(2), task(2), task(2))
```

### Mistake 5: Forgetting `asyncio.run()`

```python
# ❌ This doesn't execute the coroutine
main()  # Returns coroutine object, prints nothing

# ✅ Must use asyncio.run() to start the event loop
asyncio.run(main())
```

---

## Debugging Guide

### Error: `RuntimeWarning: coroutine 'func' was never awaited`

**Cause:** You called an async function without `await`.

**Fix:** Add `await` before the call, or use `asyncio.run()`.

### Error: `SyntaxError: 'await' outside function`

**Cause:** Using `await` in a regular (non-async) function or at module level.

**Fix:** Put the `await` inside an `async def` function.

### Error: `RuntimeError: This event loop is already running`

**Cause:** Calling `asyncio.run()` inside an already-running event loop (common in Jupyter notebooks).

**Fix:** In Jupyter, use `await main()` directly instead of `asyncio.run(main())`. Or use `nest_asyncio`:
```python
import nest_asyncio
nest_asyncio.apply()
```

### Code runs but is not faster than sync

**Cause:** Either using `time.sleep()` instead of `asyncio.sleep()`, or awaiting sequentially instead of using `gather()`.

**Fix:** Use `asyncio.sleep()` for waits, and `asyncio.gather()` for concurrent execution.

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Use `asyncio.gather()` for concurrent I/O tasks | Runs multiple operations simultaneously |
| Use `await asyncio.sleep()`, never `time.sleep()` | Non-blocking — lets other tasks run |
| Use `ainvoke()` / `astream()` in LangChain for production | Better throughput and responsiveness |
| Use `asyncio.run()` as the entry point in scripts | Proper event loop management |
| Don't use async for CPU-bound work | Async helps I/O-bound work only; use `multiprocessing` for CPU-bound |
| Use `async for` with async generators | Proper way to iterate over async streams |

---

## When to Use Sync vs Async

| Use Sync When... | Use Async When... |
|-------------------|-------------------|
| Simple scripts with one task | Multiple concurrent I/O operations |
| CPU-intensive computation | Waiting for APIs, databases, files |
| Quick prototyping | Production web servers (FastAPI) |
| No concurrent operations needed | Multi-agent systems |
| Learning/debugging (simpler to reason about) | Handling many user requests simultaneously |

---

## Interview Preparation

### Easy

**Q: What is the difference between synchronous and asynchronous code?**

**A:** Synchronous code runs one task at a time, blocking until each completes. Asynchronous code can start multiple tasks and switch between them while waiting for I/O, using CPU time more efficiently without wasting it on idle waits.

### Medium

**Q: What does `await` do?**

**A:** `await` pauses the current coroutine and gives control back to the event loop. While this coroutine is paused (waiting for I/O), other coroutines can run. When the awaited operation completes, the coroutine resumes from where it left off.

### Hard

**Q: Why does LangChain have both `invoke()` and `ainvoke()`?**

**A:** `invoke()` is synchronous — it blocks the calling thread until the LLM responds. `ainvoke()` is asynchronous — it pauses the coroutine (not the thread) while waiting, allowing other coroutines to run. In production environments (FastAPI, multi-agent systems), async methods are essential for handling multiple concurrent requests without blocking the server.

### Senior

**Q: What is the event loop and how does `asyncio.gather()` work?**

**A:** The event loop is the core scheduler of asyncio — it runs on a single thread and manages a queue of coroutines. It runs each coroutine until it hits an `await`, then switches to another ready coroutine. `asyncio.gather()` registers multiple coroutines with the event loop simultaneously. The event loop interleaves their execution at `await` points, achieving concurrency on a single thread. The total time is `max(durations)` instead of `sum(durations)`.

### System Design

**Q: You're building a RAG system that needs to embed a query, retrieve documents, and call an LLM. How would you use async to optimize this?**

**A:** Structure the pipeline so independent operations run concurrently:
```python
async def rag_pipeline(query):
    # Step 1: Embed query and retrieve in parallel (independent operations)
    embedding, cached_context = await asyncio.gather(
        embed_query(query),
        check_cache(query)
    )
    
    # Step 2: Retrieve (depends on embedding)
    if not cached_context:
        documents = await vector_store.asimilarity_search(embedding)
    
    # Step 3: Generate (depends on retrieval)
    answer = await llm.ainvoke(format_prompt(query, documents))
    return answer
```

---

## Summary

| Concept | What It Does |
|---------|--------------|
| `async def` | Declares an async function (coroutine) |
| `await` | Pauses current coroutine, lets others run |
| `asyncio.run()` | Entry point — starts the event loop |
| `asyncio.gather()` | Runs multiple coroutines **concurrently** |
| `asyncio.sleep()` | Non-blocking sleep (use instead of `time.sleep()`) |
| `ainvoke()` | LangChain's async version of `invoke()` |
| `astream()` | LangChain's async version of `stream()` |
| Event loop | Single-thread scheduler that switches between coroutines |

---

## Cheat Sheet

```python
import asyncio

# BASIC ASYNC FUNCTION
async def my_func():
    await asyncio.sleep(1)
    return "done"

# RUN FROM SCRIPT
asyncio.run(my_func())

# RUN CONCURRENTLY
async def main():
    results = await asyncio.gather(
        my_func(),
        my_func(),
        my_func(),
    )

# ASYNC FOR (streaming)
async for item in async_generator():
    print(item)

# LANGCHAIN ASYNC
result = await llm.ainvoke("question")
results = await llm.abatch(["q1", "q2"])
async for chunk in llm.astream("question"):
    print(chunk.content)

# TIMING
import time
start = time.time()
await asyncio.gather(task1(), task2())
print(f"Took {time.time() - start:.1f}s")
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| What does `async def` create? | A coroutine function |
| What does `await` do? | Pauses current coroutine, lets event loop run others |
| Does async use multiple threads? | No — single thread with event loop |
| `time.sleep()` vs `asyncio.sleep()`? | `time.sleep` blocks everything; `asyncio.sleep` is non-blocking |
| How to run async from a script? | `asyncio.run(main())` |
| How to run tasks concurrently? | `asyncio.gather(task1(), task2())` |
| Total time for `gather(2s, 3s, 1s)`? | ~3 seconds (max, not sum) |
| LangChain async invoke? | `await llm.ainvoke("question")` |
| When NOT to use async? | CPU-bound work, simple scripts, quick prototyping |

---

## Hands-on Exercise

### Exercise 1: Parallel API Calls

```python
import asyncio
import time

async def call_api(name, duration):
    print(f"Starting {name}...")
    await asyncio.sleep(duration)
    print(f"Finished {name} in {duration}s")
    return f"{name} result"

async def parallel_api_calls():
    start = time.time()
    results = await asyncio.gather(
        call_api("API 1", 1),
        call_api("API 2", 2),
        call_api("API 3", 3)
    )
    elapsed = time.time() - start
    print("Results:", results)
    print(f"Total time: {elapsed:.1f}s")

async def main():
    await parallel_api_calls()

asyncio.run(main())
```

### Exercise 2: Async Data Fetcher

Build an async function that:
1. Takes a list of URLs (simulated with names and durations)
2. "Fetches" all of them concurrently
3. Returns results sorted by completion time

---

## Challenge Project

Build an async `BatchLLMCaller` class that:
1. Accepts a list of prompts
2. Calls a simulated LLM for each prompt concurrently
3. Limits concurrency to N simultaneous calls (using `asyncio.Semaphore`)
4. Returns all results

```python
caller = BatchLLMCaller(max_concurrent=3)
results = await caller.process(["prompt1", "prompt2", ..., "prompt10"])
# Only 3 run at a time, but all 10 complete
```

---

## Homework

1. **Reading:** Read [Python asyncio docs](https://docs.python.org/3/library/asyncio.html)
2. **Coding:** Complete exercises 1 and 2
3. **Experiment:** Measure the time difference between sequential awaits and `gather()`
4. **Debugging:** Fix this code:
    ```python
    import asyncio, time
    
    async def fetch(name):
        time.sleep(2)  # Bug!
        return name
    
    async def main():
        results = await asyncio.gather(fetch("A"), fetch("B"), fetch("C"))
        print(results)
    
    asyncio.run(main())
    # Expected: ~2s, Actual: ~6s. Why?
    ```

---

## Additional Resources

- [Python Official Docs — asyncio](https://docs.python.org/3/library/asyncio.html)
- [PEP 492 — Coroutines with async and await syntax](https://peps.python.org/pep-0492/)
- [Real Python — Async IO in Python](https://realpython.com/async-io-python/)
- [LangChain Async Docs](https://python.langchain.com/docs/concepts/async/)

---

## What's Next

In the next chapter, we'll learn about **OOP Patterns for AI Applications** — how to structure classes for AI projects. You'll learn the class design patterns that LangChain uses internally: inheritance, composition, abstract classes, and the strategy pattern. This is the last chapter of Phase 0, and after completing it, you'll have all the Python skills needed to start building with LangChain.

> [← Previous: Type Hints & Pydantic](chapter-03-type-hints-pydantic.md) | [Next: OOP Patterns →](chapter-05-oop-patterns.md)

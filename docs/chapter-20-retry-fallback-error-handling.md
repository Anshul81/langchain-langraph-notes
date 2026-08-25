# Chapter 4.3: Retry, Fallback & Error Handling

> **Phase 4 — Chains & Runnables** | [← Previous: Sequential Chains](chapter-19-sequential-chains.md) | [Next: Streaming & Async Chains →](chapter-21-streaming-async-chains.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Use `.with_retry()` to automatically retry failed LLM calls
- ✅ Use `.with_fallbacks()` to switch to backup models/chains
- ✅ Combine retry + fallback for bulletproof pipelines
- ✅ Handle specific exceptions (rate limits, timeouts, bad output)
- ✅ Use `RunnableBranch` for conditional routing

| | |
|---|---|
| **Prerequisites** | Chapter 4.2 (Sequential Chains) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 40 minutes |

---

## Introduction

In production, LLM calls **fail**. A lot:

```
❌ Rate limit hit (429 Too Many Requests)
❌ Timeout (model took too long)
❌ Bad output (JSON parsing failed)
❌ Model down (OpenAI/Anthropic outage)
❌ Content filtered (safety filter triggered)
```

Your chain WILL crash if you don't handle these. LangChain gives you **three defenses**:

```
Request → [Retry] → [Fallback] → [Branch] → Response
            ↓           ↓            ↓
        "Try again"  "Use backup"  "Route by condition"
```

---

## Part 1: `.with_retry()` — Automatic Retry

Wraps any Runnable to retry on failure with configurable attempts and wait times:

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

# Basic retry — retries up to 2 times on ANY exception
reliable_llm = llm.with_retry(
    stop_after_attempt=3  # 1 initial + 2 retries = 3 total attempts
)

chain = (
    ChatPromptTemplate.from_template("Explain {topic} simply.")
    | reliable_llm
    | StrOutputParser()
)

# If the LLM fails, it automatically retries up to 3 times
result = chain.invoke({"topic": "quantum computing"})
```

### Retry with Exponential Backoff

```python
# Production-grade: wait longer between each retry
reliable_llm = llm.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True  # Wait 1s, 2s, 4s... with random jitter
)
```

### Retry Only on Specific Exceptions

```python
from openai import RateLimitError, APITimeoutError

# Only retry on rate limits and timeouts — NOT on invalid API key, etc.
reliable_llm = llm.with_retry(
    retry_if_exception_type=(RateLimitError, APITimeoutError),
    stop_after_attempt=5,
    wait_exponential_jitter=True
)
```

### `.with_retry()` Works on ANY Runnable

```python
from langchain_core.runnables import RunnableLambda

# Retry a custom function too!
@RunnableLambda
def flaky_api_call(data: dict) -> dict:
    """Simulates an unreliable external API."""
    import random
    if random.random() < 0.5:
        raise ConnectionError("API unavailable")
    return {"result": "success", **data}

reliable_api = flaky_api_call.with_retry(stop_after_attempt=3)
```

---

## Part 2: `.with_fallbacks()` — Backup Models/Chains

When retries aren't enough — switch to a completely different model or chain:

```python
# Primary: fast, cheap model
primary_llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=300,
    request_timeout=10  # Fail fast
)

# Fallback: different model/provider
fallback_llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=300,
    request_timeout=30  # More patient
)

# If primary fails → automatically tries fallback
resilient_llm = primary_llm.with_fallbacks([fallback_llm])

chain = (
    ChatPromptTemplate.from_template("Explain {topic}.")
    | resilient_llm
    | StrOutputParser()
)
```

### Fallback to a Completely Different Chain

```python
from langchain_core.runnables import RunnableLambda

# Primary: LLM-powered chain
primary_chain = (
    ChatPromptTemplate.from_template("Translate to Hindi: {text}")
    | llm
    | StrOutputParser()
)

# Fallback: hardcoded/cached response (no LLM needed!)
@RunnableLambda
def fallback_response(input_data: dict) -> str:
    return f"[Translation unavailable] Original: {input_data['text']}"

# If the LLM chain fails entirely → return a graceful message
safe_chain = primary_chain.with_fallbacks([fallback_response])

result = safe_chain.invoke({"text": "Hello, world!"})
```

### Multiple Fallbacks (Cascade)

```python
# Try Model A → Model B → Model C → hardcoded response
resilient_llm = model_a.with_fallbacks([model_b, model_c, hardcoded_fallback])
# Tries each in order until one succeeds
```

---

## Part 3: Combining Retry + Fallback (Production Pattern 🔥)

The **gold standard** — retry each model before falling back to the next:

```python
# Each model retries 3x before giving up
primary = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=300
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

fallback = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=300
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

# Retry primary 3x → if all fail → retry fallback 3x
# Total: up to 6 attempts before giving up
bulletproof_llm = primary.with_fallbacks([fallback])

chain = (
    ChatPromptTemplate.from_template("Explain {topic}.")
    | bulletproof_llm
    | StrOutputParser()
)
```

### Visualizing the Flow

```
Request
  │
  ├──→ Primary (attempt 1) ──→ ❌ fail
  ├──→ Primary (attempt 2) ──→ ❌ fail
  ├──→ Primary (attempt 3) ──→ ❌ fail (give up on primary)
  │
  ├──→ Fallback (attempt 1) ──→ ❌ fail
  ├──→ Fallback (attempt 2) ──→ ✅ success! → Response
  │
  └──→ (If all fail → raises exception)
```

---

## Part 4: `RunnableBranch` — Conditional Routing

Routes input to different chains based on conditions:

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Different chains for different tasks
math_chain = (
    ChatPromptTemplate.from_template("Solve this math problem: {question}")
    | llm | StrOutputParser()
)

science_chain = (
    ChatPromptTemplate.from_template("Answer this science question: {question}")
    | llm | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_template("Answer: {question}")
    | llm | StrOutputParser()
)

# Route based on input
branch = RunnableBranch(
    # (condition, chain) pairs — checked in order
    (lambda x: "math" in x["topic"].lower(), math_chain),
    (lambda x: "science" in x["topic"].lower(), science_chain),
    general_chain  # ← default (no condition) — MUST be last
)

result = branch.invoke({"topic": "math", "question": "What is 2+2?"})
# → Routes to math_chain
```

### Real-World: Route by Language

```python
from langchain_core.runnables import RunnableBranch

english_chain = ChatPromptTemplate.from_template(
    "Answer in English: {question}"
) | llm | StrOutputParser()

hindi_chain = ChatPromptTemplate.from_template(
    "Answer in Hindi: {question}"
) | llm | StrOutputParser()

# Classify language first, then route
@RunnableLambda
def detect_language(data: dict) -> dict:
    """Simple language detection (production: use a classifier)."""
    hindi_chars = set("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")
    if any(c in hindi_chars for c in data["question"]):
        data["language"] = "hindi"
    else:
        data["language"] = "english"
    return data

chain = detect_language | RunnableBranch(
    (lambda x: x["language"] == "hindi", hindi_chain),
    english_chain  # default
)
```

### RunnableBranch vs Python if/else

```python
# ❌ Python if/else — NOT part of the chain, can't stream/batch
def route(data):
    if "math" in data["topic"]:
        return math_chain.invoke(data)
    return general_chain.invoke(data)

# ✅ RunnableBranch — IS a Runnable, supports invoke/stream/batch/async
branch = RunnableBranch(
    (lambda x: "math" in x["topic"], math_chain),
    general_chain
)
# branch.invoke(), branch.stream(), branch.batch() all work!
```

---

## The Three Defenses Compared

| Defense | What It Does | When to Use |
|---------|-------------|-------------|
| `.with_retry(n)` | Retries same Runnable N times | Transient failures (rate limits, timeouts) |
| `.with_fallbacks([b])` | Switches to backup Runnable | Model outages, different providers |
| `RunnableBranch` | Routes to different chains by condition | Different logic paths based on input |

---

## Common Mistakes

### Mistake 1: Retry without backoff
```python
# ❌ Hammers the API immediately — makes rate limits WORSE
llm.with_retry(stop_after_attempt=10)

# ✅ Exponential backoff — waits progressively longer
llm.with_retry(stop_after_attempt=5, wait_exponential_jitter=True)
```

### Mistake 2: Forgetting the default branch
```python
# ❌ Crashes if no condition matches!
branch = RunnableBranch(
    (lambda x: "math" in x["topic"], math_chain),
    (lambda x: "science" in x["topic"], science_chain),
    # No default — will throw ValueError!
)

# ✅ Always include a default (last positional arg, no condition)
branch = RunnableBranch(
    (lambda x: "math" in x["topic"], math_chain),
    (lambda x: "science" in x["topic"], science_chain),
    general_chain  # ← default, always last
)
```

### Mistake 3: Fallback with incompatible output types
```python
# ❌ Primary returns structured Pydantic, fallback returns raw string
primary = prompt | llm.with_structured_output(MyModel)
fallback = prompt | llm | StrOutputParser()

safe = primary.with_fallbacks([fallback])
# Downstream code expects MyModel but might get str!

# ✅ Both should return the same type
fallback = prompt | fallback_llm.with_structured_output(MyModel)
safe = primary.with_fallbacks([fallback])
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Always use `wait_exponential_jitter=True` with retry | Prevents thundering herd on rate limits |
| Chain retry THEN fallback: `model.with_retry().with_fallbacks()` | Exhaust retries before switching |
| Match output types across fallbacks | Prevents downstream type errors |
| Use `RunnableBranch` for routing, not Python if/else | Keeps it inside the Runnable protocol (streaming, batching) |
| Always include a default branch in `RunnableBranch` | Prevents crashes on unexpected input |
| Log which fallback was used in production | Helps diagnose recurring primary failures |

---

## Interview Preparation

### Easy
**Q: What does `.with_retry()` do?**

> Wraps any Runnable to automatically retry on failure. You configure `stop_after_attempt` (max tries) and optionally `wait_exponential_jitter` (increasing delays between retries). It catches exceptions and re-invokes the Runnable.

### Medium
**Q: What's the difference between retry and fallback?**

> **Retry** tries the **same** Runnable again (same model, same chain) — good for transient errors like rate limits. **Fallback** switches to a **different** Runnable entirely (different model, different chain, or even a hardcoded response) — good for when the primary is completely unavailable.

### Hard
**Q: How would you build a production-resilient chain?**

> Layer the defenses: (1) Each model gets `.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)` for transient failures. (2) Wrap in `.with_fallbacks([backup_model])` to switch providers on total failure. (3) Use `RunnableBranch` or exception handling for graceful degradation. The pattern is: `primary.with_retry().with_fallbacks([fallback.with_retry()])`.

### Senior
**Q: When would you NOT use `.with_retry()`?**

> When the error is **deterministic** — retrying won't help. Examples: invalid API key (AuthenticationError), malformed prompt, model doesn't support a feature, content policy violation. Only retry on **transient** errors: rate limits, timeouts, temporary network issues. Use `retry_if_exception_type` to be selective.

---

## Mini Assignment — Answer

Build a **Resilient Q&A Chain** with retry, fallback, and conditional routing:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

# --- 1. Primary model with retry ---
primary_llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "standard"),
    temperature=0.7,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=200,
    request_timeout=10
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

# --- 2. Fallback model with retry ---
fallback_llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "standard"),
    temperature=0,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=100
).with_retry(stop_after_attempt=3, wait_exponential_jitter=True)

# --- 3. Last resort fallback ---
@RunnableLambda
def last_resort(data: dict) -> str:
    return "I'm sorry, I couldn't process your question. Please try again later."

# --- 4. Bulletproof LLM: retry primary → fallback → hardcoded ---
bulletproof_llm = primary_llm.with_fallbacks([fallback_llm, last_resort])

# --- 5. Category-specific prompts ---
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a senior software engineer. Answer coding questions with code examples."),
    ("human", "{question}")
])

general_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer clearly and concisely."),
    ("human", "{question}")
])

# --- 6. Build category-specific chains ---
code_chain = code_prompt | bulletproof_llm | StrOutputParser()
general_chain = general_prompt | bulletproof_llm | StrOutputParser()

# --- 7. RunnableBranch for conditional routing ---
qa_chain = RunnableBranch(
    (lambda x: x.get("category", "").lower() == "code", code_chain),
    general_chain  # default
)

# --- Test it! ---
# Code question
result1 = qa_chain.invoke({
    "category": "code",
    "question": "How do I read a CSV file in Python?"
})
print(f"Code answer:\n{result1}\n")

# General question
result2 = qa_chain.invoke({
    "category": "general",
    "question": "What is photosynthesis?"
})
print(f"General answer:\n{result2}")
```

### Architecture Diagram:

```
Input: {"category": "code/general", "question": "..."}
  │
  └──→ RunnableBranch
        ├── category == "code" → code_prompt → bulletproof_llm → StrOutputParser
        └── default            → general_prompt → bulletproof_llm → StrOutputParser
                                                        │
                                          ┌─────────────┼─────────────┐
                                          │        bulletproof_llm    │
                                          │  primary (retry 3x)      │
                                          │  → fallback (retry 3x)   │
                                          │  → last_resort (hardcoded)│
                                          └───────────────────────────┘
```

---

## Summary

| Component | What It Does |
|-----------|-------------|
| `.with_retry(stop_after_attempt=N)` | Retries the same Runnable N times on failure |
| `wait_exponential_jitter=True` | Adds increasing delays between retries |
| `retry_if_exception_type=(...)` | Only retries on specific exception types |
| `.with_fallbacks([backup])` | Switches to backup Runnable when primary fails |
| `primary.with_retry().with_fallbacks([...])` | Production pattern: retry then fallback |
| `RunnableBranch((cond, chain), ..., default)` | Routes input to different chains by condition |

---

> [← Previous: Sequential Chains](chapter-19-sequential-chains.md) | [Next: Streaming & Async Chains →](chapter-21-streaming-async-chains.md)

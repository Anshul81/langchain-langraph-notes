# Chapter 4.1: Runnables Deep Dive

> **Phase 4 — Chains & Runnables** | [← Previous: Structured Output](../phase-03-langchain-core/chapter-17-structured-output.md) | [Next: Sequential Chains →](chapter-19-sequential-chains.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Master `RunnableLambda` — wrap any Python function into a chain
- ✅ Master `RunnablePassthrough` — forward data unchanged
- ✅ Master `RunnableParallel` — run multiple branches simultaneously
- ✅ Combine them to build complex data pipelines
- ✅ Understand `itemgetter` for extracting dict keys

| | |
|---|---|
| **Prerequisites** | Chapter 3.2 (LCEL), Chapter 3.3 (ChatPromptTemplate) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

In Chapter 3.2, you learned `prompt | llm | parser`. That's a straight line. But real applications aren't straight lines:

```
                    ┌──→ Summarize ───┐
User Input ────────→├──→ Translate ───├──→ Combine ──→ Output
             │      └──→ Classify ───┘
             │
             └──→ Pass original input through
```

You need branching, merging, custom logic, and data routing. That's what these Runnables do.

---

## Part 1: `RunnableLambda` — Custom Functions in Chains

Wraps **any Python function** into a Runnable that works with `|`:

```python
from langchain_core.runnables import RunnableLambda

# Any function becomes a chain step
def uppercase(text: str) -> str:
    return text.upper()

def add_emoji(text: str) -> str:
    return f"🚀 {text} 🚀"

def word_count(text: str) -> dict:
    return {"text": text, "words": len(text.split())}

# Chain them!
chain = RunnableLambda(uppercase) | RunnableLambda(add_emoji) | RunnableLambda(word_count)

result = chain.invoke("hello world")
print(result)
# {"text": "🚀 HELLO WORLD 🚀", "words": 4}
```

### The `@RunnableLambda` Decorator

```python
from langchain_core.runnables import RunnableLambda

@RunnableLambda
def clean_text(text: str) -> str:
    """Remove extra whitespace and lowercase."""
    return " ".join(text.lower().split())

# Now it's a Runnable — use directly in chains
chain = clean_text | RunnableLambda(word_count)
```

### With LLM Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

# Custom post-processing after LLM
def format_response(text: str) -> str:
    return f"📝 Answer:\n{text}\n{'─' * 40}"

chain = prompt | llm | StrOutputParser() | RunnableLambda(format_response)
result = chain.invoke({"question": "What is LCEL?"})
```

---

## Part 2: `RunnablePassthrough` — Forward Data Unchanged

Passes input through **as-is** to the next step. Seems useless alone but is essential for parallel branches:

```python
from langchain_core.runnables import RunnablePassthrough

# Basic: passes input through unchanged
passthrough = RunnablePassthrough()
result = passthrough.invoke({"name": "John", "age": 30})
# {"name": "John", "age": 30}
```

### `RunnablePassthrough.assign()` — The Real Power

Keeps original input AND adds new computed fields:

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def get_length(data: dict) -> int:
    return len(data["text"].split())

chain = RunnablePassthrough.assign(
    word_count=RunnableLambda(lambda x: len(x["text"].split())),
    uppercased=RunnableLambda(lambda x: x["text"].upper())
)

result = chain.invoke({"text": "hello world"})
# {"text": "hello world", "word_count": 2, "uppercased": "HELLO WORLD"}
#  ↑ original kept         ↑ added           ↑ added
```

---

## Part 3: `RunnableParallel` — Run Branches Simultaneously

Runs multiple chains **in parallel** and collects results into a dict:

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    upper=RunnableLambda(lambda x: x.upper()),
    lower=RunnableLambda(lambda x: x.lower()),
    length=RunnableLambda(lambda x: len(x))
)

result = parallel.invoke("Hello World")
# {"upper": "HELLO WORLD", "lower": "hello world", "length": 11}
```

### Dict Shorthand (Equivalent)

```python
# This dict IS a RunnableParallel:
parallel = {
    "upper": RunnableLambda(lambda x: x.upper()),
    "lower": RunnableLambda(lambda x: x.lower()),
    "length": RunnableLambda(lambda x: len(x))
}
```

### With LLM Chains (Parallel LLM Calls)

```python
from operator import itemgetter

summary_chain = (
    ChatPromptTemplate.from_template("Summarize: {text}")
    | llm | StrOutputParser()
)

translation_chain = (
    ChatPromptTemplate.from_template("Translate to French: {text}")
    | llm | StrOutputParser()
)

# Three 1-second LLM calls → ~1 second total (not 3!)
parallel_chain = RunnableParallel(
    summary=summary_chain,
    translation=translation_chain,
    original=RunnablePassthrough()
)

result = parallel_chain.invoke({"text": "LangChain is amazing"})
```

---

## Part 4: `itemgetter` — Clean Dict Key Extraction

```python
from operator import itemgetter

# Instead of: RunnableLambda(lambda x: x["text"])
# Use:        itemgetter("text")

chain = (
    {
        "context": itemgetter("context"),
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## The Three Runnables Compared

| Runnable | Input | Output | Use Case |
|----------|-------|--------|----------|
| `RunnableLambda(fn)` | Anything | Anything | Custom Python logic in chains |
| `RunnablePassthrough()` | Dict | Same dict | Forward data unchanged |
| `RunnablePassthrough.assign(key=...)` | Dict | Dict + new keys | Add computed fields, keep original |
| `RunnableParallel(a=..., b=...)` | Anything | Dict `{a: ..., b: ...}` | Run branches in parallel |

---

## Common Mistakes

### Mistake 1: Forgetting `itemgetter` input type
```python
# ❌ summary_chain expects {"text": "..."} but gets the raw string
chain = itemgetter("text") | summary_chain

# ✅ Pipe extracted text into a chain that expects {text}
chain = RunnablePassthrough.assign(
    summary=itemgetter("text") | summary_chain
)
```

### Mistake 2: Using `RunnableParallel` when you want to keep original data
```python
# ❌ Loses original input
parallel = RunnableParallel(summary=summary_chain)

# ✅ Keeps original + adds summary
chain = RunnablePassthrough.assign(summary=summary_chain)
```

### Mistake 3: Complex logic in lambdas
```python
# ❌ Hard to debug
chain = RunnableLambda(lambda x: {k: v.strip().lower() for k, v in x.items() if isinstance(v, str)})

# ✅ Named function — readable, debuggable, testable
@RunnableLambda
def clean_dict(data: dict) -> dict:
    """Lowercase and strip all string values."""
    return {k: v.strip().lower() for k, v in data.items() if isinstance(v, str)}
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use `@RunnableLambda` decorator for named functions | Cleaner than wrapping |
| Use `RunnablePassthrough.assign()` to add fields | Preserves original data |
| Use `RunnableParallel` for independent LLM calls | Faster — parallel execution |
| Use `itemgetter` for dict key extraction | Cleaner than lambdas |
| Use dict shorthand `{}` for simple parallel | Less verbose |
| Keep lambdas simple; use named functions for complex logic | Readability and debugging |

---

## Interview Preparation

### Easy
**Q: What does `RunnableLambda` do?**

> Wraps any Python function (sync or async) into a Runnable that works with the `|` pipe operator, gaining `invoke()`, `stream()`, `batch()`, and async methods automatically.

### Medium
**Q: What is the difference between `RunnableParallel` and `RunnablePassthrough.assign()`?**

> `RunnableParallel` creates a new dict from scratch — each key gets a branch's output. `RunnablePassthrough.assign()` preserves the original input dict and adds new keys to it. Use `assign()` when you want to keep existing data; use `RunnableParallel` when you want a fresh structure.

### Hard
**Q: How does `RunnableParallel` handle async execution under the hood?**

> When you call `ainvoke()`, it uses `asyncio.gather()` to run all branches concurrently. For sync `invoke()`, it uses a thread pool executor to parallelize the branches. This is why three 1-second LLM calls in parallel take ~1 second total instead of 3.

---

## Mini Assignment — Answer

Build a text analysis pipeline:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "standard"),
    temperature=0.3,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=200
)

def compute_stats(text: str) -> dict:
    return {"word_count": len(text.split()), "char_count": len(text)}

summary_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Summarize in 2 sentences."),
        ("human", "{text}")
    ]) | llm | StrOutputParser()
)

translation_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Translate to Hindi."),
        ("human", "{text}")
    ]) | llm | StrOutputParser()
)

pipeline = RunnablePassthrough.assign(
    summary=itemgetter("text") | summary_chain,
    translation=itemgetter("text") | translation_chain,
    stats=itemgetter("text") | RunnableLambda(compute_stats)
)

result = pipeline.invoke({
    "text": "LangChain is a framework for building applications powered by language models."
})
# {"text": "...", "summary": "...", "translation": "...", "stats": {...}}
```

---

## Summary

| Component | What It Does |
|-----------|-------------|
| `RunnableLambda(fn)` | Wraps any function into a chainable Runnable |
| `@RunnableLambda` decorator | Cleaner way to create named Runnables |
| `RunnablePassthrough()` | Forwards input unchanged |
| `RunnablePassthrough.assign()` | Keeps input + adds computed keys |
| `RunnableParallel(a=..., b=...)` | Runs branches in parallel, returns dict |
| `{...}` dict shorthand | Equivalent to `RunnableParallel` |
| `itemgetter("key")` | Cleanly extracts a dict key |

---

> [← Previous: Structured Output](../phase-03-langchain-core/chapter-17-structured-output.md) | [Next: Sequential Chains →](chapter-19-sequential-chains.md)

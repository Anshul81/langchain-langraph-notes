# Chapter 6.2: RunnablePassthrough, RunnableLambda, RunnableParallel

> **Phase 6 — Chains & Runnables** | [← Previous: Runnables Deep Dive](chapter-25-runnables-deep-dive.md) | [Next: Sequential Chains →](chapter-27-sequential-chains.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Wrap Python functions with **`RunnableLambda`**
- ✅ Forward and enrich dict inputs with **`RunnablePassthrough`**
- ✅ Run independent branches with **`RunnableParallel`**
- ✅ Use **`itemgetter`** for clean dict key routing
- ✅ Build multi-branch LLM pipelines that preserve original input

| | |
|---|---|
| **Prerequisites** | Chapter 5.4 (LCEL), Chapter 5.3 (Chat models) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

Chapter 5.4 taught straight lines: `prompt | llm | parser`. Production pipelines branch:

```
                         ┌──► summarize ──┐
User payload (dict) ────►├──► translate ──├──► merge / respond
                         └──► pass-through ┘
                               (original text)
```

Three primitives cover most routing needs:

| Runnable | One-line purpose |
|----------|------------------|
| `RunnableLambda` | Custom Python logic inside LCEL |
| `RunnablePassthrough` | Keep input; optionally `.assign()` new fields |
| `RunnableParallel` | Run branches concurrently; output a dict |

---

## Part 1: `RunnableLambda` — Functions as Chain Steps

```python
from langchain_core.runnables import RunnableLambda

def uppercase(text: str) -> str:
    return text.upper()

def word_count(text: str) -> dict:
    return {"text": text, "words": len(text.split())}

chain = RunnableLambda(uppercase) | RunnableLambda(word_count)

print(chain.invoke("hello world"))
# {'text': 'HELLO WORLD', 'words': 2}
```

### Decorator Form

```python
@RunnableLambda
def clean_text(text: str) -> str:
    return " ".join(text.lower().split())

print(clean_text.invoke("  Hello   WORLD  "))
```

### After the LLM

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}"),
])

def format_answer(text: str) -> str:
    return f"📝 Answer:\n{text}\n{'─' * 40}"

chain = prompt | llm | StrOutputParser() | RunnableLambda(format_answer)
print(chain.invoke({"question": "What is LCEL?"}))
```

---

## Part 2: `RunnablePassthrough` — Forward & Enrich

### Plain passthrough

```python
from langchain_core.runnables import RunnablePassthrough

pt = RunnablePassthrough()
print(pt.invoke({"user_id": 42, "text": "hi"}))
# {'user_id': 42, 'text': 'hi'}
```

### `.assign()` — keep original keys, add computed ones

```python
chain = RunnablePassthrough.assign(
    word_count=RunnableLambda(lambda x: len(x["text"].split())),
    upper=RunnableLambda(lambda x: x["text"].upper()),
)

print(chain.invoke({"text": "hello world"}))
# {'text': 'hello world', 'word_count': 2, 'upper': 'HELLO WORLD'}
```

This pattern is ideal when downstream steps need **both** raw input and LLM outputs.

---

## Part 3: `RunnableParallel` — Concurrent Branches

```python
from langchain_core.runnables import RunnableParallel, RunnableLambda

parallel = RunnableParallel(
    upper=RunnableLambda(lambda x: x.upper()),
    lower=RunnableLambda(lambda x: x.lower()),
    length=RunnableLambda(lambda x: len(x)),
)

print(parallel.invoke("Hello World"))
# {'upper': 'HELLO WORLD', 'lower': 'hello world', 'length': 11}
```

### Dict literal shorthand

```python
parallel = {
    "upper": RunnableLambda(lambda x: x.upper()),
    "lower": RunnableLambda(lambda x: x.lower()),
}
```

A bare `{...}` dict of Runnables **is** a `RunnableParallel`.

### Parallel LLM calls

```python
from langchain_core.output_parsers import StrOutputParser

summary_chain = (
    ChatPromptTemplate.from_template("Summarize in one sentence: {text}")
    | llm
    | StrOutputParser()
)

translation_chain = (
    ChatPromptTemplate.from_template("Translate to French: {text}")
    | llm
    | StrOutputParser()
)

parallel_chain = RunnableParallel(
    summary=summary_chain,
    translation=translation_chain,
    original=RunnablePassthrough(),
)

out = parallel_chain.invoke({"text": "LangChain composes LLM apps with pipes."})
print(out.keys())  # dict_keys(['summary', 'translation', 'original'])
```

Independent LLM calls in parallel reduce wall-clock time versus sequential invokes.

---

## Part 4: `itemgetter` — Routing Dict Fields

```python
from operator import itemgetter

rag_step = (
    {
        "context": itemgetter("context"),
        "question": itemgetter("question"),
    }
    | ChatPromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    | llm
    | StrOutputParser()
)
```

Prefer `itemgetter("key")` over `RunnableLambda(lambda x: x["key"])` for readability.

### Combining assign + itemgetter

```python
pipeline = RunnablePassthrough.assign(
    summary=itemgetter("text") | summary_chain,
    translation=itemgetter("text") | translation_chain,
)
```

---

## Part 5: Passthrough vs Parallel — When to Use Which

| Scenario | Use |
|----------|-----|
| Need original dict + new fields | `RunnablePassthrough.assign()` |
| Need only branch outputs (fresh dict) | `RunnableParallel` |
| Single string through many steps | `RunnableLambda` chain with `\|` |
| RAG context + question reformatting | `{...}` dict mapping + prompt |

---

## Common Mistakes

### Mistake 1: Losing the original input

```python
# ❌ Only branch outputs remain
bad = RunnableParallel(summary=summary_chain)

# ✅ Keep input via passthrough or parallel branch
good = RunnablePassthrough.assign(summary=summary_chain)
```

### Mistake 2: Wrong input shape to a sub-chain

```python
# summary_chain expects {"text": "..."}
# ❌ piping raw string from itemgetter incorrectly
```

Always trace what each Runnable expects: `str` vs `dict` vs `Message`.

### Mistake 3: Heavy logic inside anonymous lambdas

Use named `@RunnableLambda` functions — easier to test and stack-trace.

---

## Best Practices

| Practice | Why |
|----------|-----|
| `@RunnableLambda` for non-trivial logic | Testability |
| `assign()` for enrichment | Preserves audit fields (ids, timestamps) |
| `RunnableParallel` for independent I/O | Latency wins |
| `itemgetter` for dict routing | Clear data flow |
| Type hints on lambda inputs | Catch shape bugs early |

---

## Interview Preparation

### Easy
**Q: What does `RunnableLambda` do?**

> It adapts a Python callable into a Runnable so it supports `invoke`, `stream`, `batch`, and composes with `|`.

### Medium
**Q: Difference between `RunnableParallel` and `RunnablePassthrough.assign()`?**

> `RunnableParallel` returns **only** branch outputs as a new dict. `assign()` **merges** new keys into the incoming dict, preserving original fields.

### Hard
**Q: How does parallel execution work for sync `invoke()`?**

> LangChain runs independent branches concurrently (thread pool for blocking work). Async `ainvoke()` uses `asyncio.gather()` for I/O-bound steps — important when branches call LLMs or HTTP.

### Senior
**Q: Design a chain that summarizes user text, detects language, and stores metadata without blocking the response path.**

> Use `RunnablePassthrough.assign()` with parallel sub-chains: one branch returns summary to the user, another writes language + stats via `RunnableLambda` to a queue/DB. Separate **critical path** (user-facing) from **async side effects** (logging) — do not put DB writes inside LLM branches without timeouts; consider LangGraph or a task queue for durable side effects.

---

## Summary

| Component | What It Does |
|-----------|-------------|
| `RunnableLambda(fn)` | Custom step in LCEL |
| `RunnablePassthrough()` | Identity step |
| `RunnablePassthrough.assign()` | Add keys, keep input |
| `RunnableParallel(...)` | Concurrent branches → dict |
| `{ "a": r1, "b": r2 }` | Parallel shorthand |
| `itemgetter("k")` | Extract dict field |

---

## Hands-on Exercise

Build `RunnablePassthrough.assign()` pipeline that accepts `{"text": ...}` and adds:

- `word_count` (int)
- `summary` (via LLM chain)
- `sentiment` (LLM: return one word: positive/negative/neutral)

Print the final dict.

---

## Challenge Project

Create a **meeting notes processor**: input `{"title": str, "transcript": str}` → parallel branches for summary, action items (bullets), and translated title (Spanish). Merge into one JSON-serializable dict for your API response.

---

## What's Next

Chapter 6.3 covers **sequential multi-step chains** — when steps depend on each other's outputs (not parallel), including router patterns and multi-prompt workflows.

> [← Previous: Runnables Deep Dive](chapter-25-runnables-deep-dive.md) | [Next: Sequential Chains →](chapter-27-sequential-chains.md)

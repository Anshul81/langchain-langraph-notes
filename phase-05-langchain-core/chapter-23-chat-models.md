# Chapter 5.3: Chat Models (OpenAI, Gemini, Ollama)

> **Phase 5 — LangChain Core** | [← Previous: Installation & Setup](chapter-22-langchain-setup.md) | [Next: LCEL & Runnable Protocol →](chapter-24-runnable-lcel.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Use **`ChatOpenAI`** with the LiteLLM proxy pattern used in this course
- ✅ Understand **message-based** chat models vs legacy completion LLMs
- ✅ Configure **temperature**, **max_tokens**, and **model** via env vars
- ✅ Invoke **Google Gemini** and **Ollama** through LangChain integrations
- ✅ Choose the right integration package for each provider

| | |
|---|---|
| **Prerequisites** | Chapter 5.2 (LangChain setup), Phase 1 (model parameters) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

### The Problem

Every provider names things differently:

```
OpenAI:   client.chat.completions.create(messages=[...])
Gemini:   model.generate_content(...)
Ollama:   POST /api/chat with {"messages": [...]}
```

Your product roadmap says: *"Start with GPT-4o-mini, try Gemini Flash for cost, run Ollama offline for dev."* Without a shared interface, you maintain three call styles.

### The Solution

LangChain **chat models** accept a list of **messages** (`SystemMessage`, `HumanMessage`, `AIMessage`) and return an **`AIMessage`**. The same calling code works across providers — swap the class and constructor args.

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   Messages   │ ──► │  BaseChatModel  │ ──► │    AIMessage     │
│ (typed objs) │     │  .invoke()      │     │  .content        │
└──────────────┘     └─────────────────┘     └──────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      ChatOpenAI    ChatGoogleGenerativeAI  ChatOllama
```

---

## Part 1: The Standard OpenAI / LiteLLM Setup

This course routes all cloud models through a **LiteLLM proxy** so one API key and base URL work for many backends:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

messages = [
    SystemMessage(content="You are a concise technical tutor."),
    HumanMessage(content="What is a chat model in LangChain?"),
]

response = llm.invoke(messages)
print(type(response))       # AIMessage
print(response.content)
```

### Callable Shortcuts

```python
# Single string → treated as a human turn
print(llm.invoke("Say hello in one sentence.").content)

# Batch multiple inputs (sync)
results = llm.batch([
    "Name one Python data structure.",
    "Name one JavaScript data structure.",
])
for r in results:
    print(r.content)
```

---

## Part 2: Parameters That Matter in Production

| Parameter | Effect | Typical prod value |
|-----------|--------|---------------------|
| `temperature` | Randomness (0 = deterministic) | `0` for extraction, `0.7` for chat |
| `max_tokens` | Cap on completion length | Set to control cost/latency |
| `model` | Which weights/API route | From env / LiteLLM config |
| `timeout` | Fail fast on hung requests | 30–120s |
| `max_retries` | Built-in retry on rate limits | Default often OK; tune for SLA |

```python
llm_creative = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0.8,
    max_tokens=150,
)

llm_strict = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
    max_tokens=500,
)
```

**Interview note:** `temperature=0` does not guarantee identical outputs across runs (sampling, system load, model updates), but it reduces variance for structured tasks.

---

## Part 3: Streaming Tokens

Chat models implement `.stream()` — essential for UX:

```python
for chunk in llm.stream("Explain embeddings in 3 bullet points."):
    print(chunk.content, end="", flush=True)
print()
```

Each `chunk` is an `AIMessageChunk`. In LCEL chains, streaming propagates through the sequence (Chapter 5.4 and 6.1).

---

## Part 4: Google Gemini via LangChain

Install: `pip install langchain-google-genai`

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

gemini = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

answer = gemini.invoke([HumanMessage(content="What is RAG in one sentence?")])
print(answer.content)
```

**Same message types, same `.invoke()`** — your prompts and parsers stay reusable.

### Gemini Through LiteLLM (Optional)

If your org only exposes LiteLLM, you can keep using `ChatOpenAI` with a Gemini model name in `LITE_LLM_MODEL` — the proxy translates requests. That is why the course standardizes on `ChatOpenAI` + proxy for labs.

---

## Part 5: Ollama — Local & Offline Development

Install Ollama, pull a model (`ollama pull llama3.2`), then:

`pip install langchain-ollama`

```python
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

local_llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

print(local_llm.invoke([HumanMessage(content="2+2=?")]).content)
```

```
Developer laptop                    Production cluster
┌─────────────────┐                ┌──────────────────────┐
│ ChatOllama      │                │ ChatOpenAI + proxy   │
│ localhost:11434 │                │ LiteLLM / OpenAI     │
└─────────────────┘                └──────────────────────┘
         │                                      │
         └──────── same HumanMessage / AIMessage ────────┘
```

Use a **factory function** so environments pick the backend:

```python
import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

def get_chat_model() -> BaseChatModel:
    backend = os.getenv("LLM_BACKEND", "proxy").lower()
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)
    return ChatOpenAI(
        model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("LITELLM_PROXY_API_KEY"),
        base_url=os.getenv("LITELLM_PROXY_API_BASE"),
        temperature=0,
    )
```

---

## Part 6: Chat Models in LCEL

Chat models expect **messages** (or message lists from prompts):

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer in at most 2 sentences."),
    ("human", "{question}"),
])

chain = prompt | get_chat_model() | StrOutputParser()
print(chain.invoke({"question": "What is cosine similarity?"}))
```

The prompt Runnable outputs `ChatPromptValue` → converted to messages → fed to the model.

---

## Common Mistakes

### Mistake 1: Using completion LLMs for chat apps

```python
# ❌ Legacy string-in/string-out LLM for multi-turn chat
# from langchain_community.llms import Ollama

# ✅ Chat model + messages
from langchain_ollama import ChatOllama
```

### Mistake 2: Forgetting `base_url` when using a proxy

```python
# ❌ Hits OpenAI directly; proxy key fails
ChatOpenAI(api_key=os.getenv("LITELLM_PROXY_API_KEY"))

# ✅ Include base_url from env
ChatOpenAI(
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
)
```

### Mistake 3: Passing a raw string when the chain expects a dict

```python
chain = prompt | llm | StrOutputParser()
# ❌ chain.invoke("Hello")  # KeyError on {question}
chain.invoke({"question": "Hello"})  # ✅
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Type hints with `BaseChatModel` | Swap providers in tests |
| Centralize model construction | One place for env and defaults |
| `temperature=0` for eval and parsing | Reproducible benchmarks |
| Stream in UI-facing endpoints | Better perceived latency |
| Log model name + token usage | Cost and debugging |
| Keep secrets in `.env` | Never commit API keys |

---

## Interview Preparation

### Easy
**Q: What is the difference between a chat model and a completion LLM in LangChain?**

> Chat models consume **message lists** (roles: system/human/ai) and return `AIMessage`. Completion LLMs consume a **single string** prompt. Modern apps almost always use chat models because they match provider APIs and multi-turn conversations.

### Medium
**Q: How do you switch from OpenAI to Ollama with minimal code changes?**

> Abstract creation behind a factory returning `BaseChatModel`. Prompts, parsers, and LCEL chains stay the same; only the concrete class (`ChatOpenAI` vs `ChatOllama`) and env vars change.

### Hard
**Q: What does `llm.bind()` do?**

> It returns a **new Runnable** with partial kwargs bound (e.g., `response_format`, tools, `temperature`). Useful for reusing the same model with different settings in parallel branches without duplicating instances.

### Senior
**Q: How would you implement model routing (cheap vs premium) in production?**

> Use a router chain or middleware: classify query complexity, route simple queries to a small model via LiteLLM, escalate to a large model on low confidence or tool failure. Enforce budgets with max_tokens, log per-route metrics, and feature-flag model names in config — not hard-coded in business logic.

---

## Summary

| Component | Role |
|-----------|------|
| `ChatOpenAI` | OpenAI-compatible chat (incl. LiteLLM proxy) |
| `ChatGoogleGenerativeAI` | Native Gemini integration |
| `ChatOllama` | Local models via Ollama server |
| `HumanMessage` / `SystemMessage` | Typed chat inputs |
| `AIMessage` | Model output with `.content` |
| `invoke` / `stream` / `batch` | Standard Runnable methods |

---

## Hands-on Exercise

1. Call the same prompt through **proxy** (`ChatOpenAI`) and **Ollama** (`ChatOllama`) and compare latency.
2. Set `temperature` to `0` vs `0.9` on a creative writing prompt — note output variance.

---

## Challenge Project

Build `get_chat_model()` plus a tiny CLI: `python ask.py "your question"` that loads `.env`, picks backend from `LLM_BACKEND`, and streams the answer to stdout.

---

## What's Next

Chapter 5.4 introduces **LCEL** — chaining prompts, chat models, and parsers with `|` so your application logic reads like a pipeline.

> [← Previous: Installation & Setup](chapter-22-langchain-setup.md) | [Next: LCEL & Runnable Protocol →](chapter-24-runnable-lcel.md)

# Chapter 3.1: LangChain Installation & Hello World

> **Phase 3 — LangChain Core** | [← Previous: Output Parsing](chapter-13-output-parsing.md) | [Next: Runnable Protocol & LCEL →](chapter-15-runnable-lcel.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Install LangChain and its ecosystem packages
- ✅ Understand the package architecture
- ✅ Make your first LangChain LLM call
- ✅ Use `invoke()`, `stream()`, and `batch()`
- ✅ See how LangChain wraps the raw OpenAI API

| | |
|---|---|
| **Prerequisites** | Phase 0-2 complete |
| **Estimated Reading Time** | 15 minutes |
| **Estimated Coding Time** | 30 minutes |

---

## Introduction

### Why LangChain?

You've been calling the OpenAI API directly. That works for simple cases, but real AI applications need:
- Chaining multiple LLM calls together
- Memory management across conversations
- Tool/function calling
- Retrieval-Augmented Generation (RAG)
- Structured output parsing
- Switching between providers (OpenAI, Anthropic, Ollama)

LangChain provides all of this through a **unified interface**.

### Package Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   LangChain Ecosystem                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  langchain-core     ← Base abstractions (Runnable,      │
│                       PromptTemplate, Messages)          │
│                       ALWAYS NEEDED                      │
│                                                         │
│  langchain          ← Chains, agents, memory             │
│                       The "framework" layer              │
│                                                         │
│  langchain-openai   ← OpenAI integrations               │
│                       (ChatOpenAI, OpenAIEmbeddings)     │
│                                                         │
│  langchain-community ← 600+ third-party integrations    │
│                        (Ollama, Pinecone, Wikipedia...)  │
│                                                         │
│  langgraph          ← Stateful agents (advanced)         │
│                                                         │
│  langserve          ← Deploy chains as REST APIs         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
pip install langchain langchain-core langchain-openai langchain-community python-dotenv
```

---

## Theory

### Raw OpenAI vs LangChain

| Raw OpenAI | LangChain |
|-----------|-----------|
| `OpenAI(api_key=...)` | `ChatOpenAI(api_key=...)` |
| `client.chat.completions.create()` | `llm.invoke()` |
| `{"role": "user", "content": "..."}` | `HumanMessage(content="...")` |
| `{"role": "system", "content": "..."}` | `SystemMessage(content="...")` |
| `response.choices[0].message.content` | `response.content` |

LangChain is a wrapper. Underneath, it calls the exact same API you already understand.

### Message Types

```python
from langchain_core.messages import (
    SystemMessage,    # = {"role": "system", ...}
    HumanMessage,     # = {"role": "user", ...}
    AIMessage,        # = {"role": "assistant", ...}
)
```

### The 6 Runnable Methods

Every LangChain component supports:

| Sync | Async | Purpose |
|------|-------|---------|
| `invoke()` | `ainvoke()` | Single input |
| `stream()` | `astream()` | Token-by-token |
| `batch()` | `abatch()` | Multiple inputs |

---

## Code Example

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    temperature=0.7,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=100
)

# 1. invoke() — Single call
response = llm.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LangChain?")
])
print(response.content)

# 2. stream() — Token by token
for chunk in llm.stream("What is Python?"):
    print(chunk.content, end="", flush=True)

# 3. batch() — Multiple inputs
responses = llm.batch(["What is Apple", "What is banana"])
for r in responses:
    print(r.content)
```

Note: Using `base_url` allows routing through a LiteLLM proxy or any OpenAI-compatible endpoint. The same `ChatOpenAI` class works with any provider that exposes an OpenAI-compatible API.

---

## How Phase 0-2 Connects

| Phase 0-2 Concept | LangChain Usage |
|-------------------|-----------------|
| `**kwargs` (Ch 0.1) | `ChatOpenAI(model=..., temperature=..., **kwargs)` |
| Context Managers (Ch 0.2) | Callbacks and tracing |
| Generators (Ch 0.3) | `llm.stream()` returns a generator |
| Pydantic (Ch 0.4) | `llm.with_structured_output(PydanticModel)` |
| Async (Ch 0.5) | `llm.ainvoke()`, `llm.astream()` |
| OOP/Polymorphism (Ch 0.6) | `ChatOpenAI`, `ChatAnthropic` — same interface |
| Messages (Ch 2.1) | `SystemMessage`, `HumanMessage`, `AIMessage` |
| Templates (Ch 2.3) | `PromptTemplate`, `ChatPromptTemplate` |
| Output Parsing (Ch 2.4) | `with_structured_output()` |

---

## Common Mistakes

### Mistake 1: Wrong import path
```python
# ❌ Deprecated
from langchain.chat_models import ChatOpenAI

# ✅ Correct
from langchain_openai import ChatOpenAI
```

### Mistake 2: Forgetting `.content`
```python
# ❌ Returns AIMessage object
print(llm.invoke("Hello"))

# ✅ Extract text
print(llm.invoke("Hello").content)
```

### Mistake 3: Using old call syntax
```python
# ❌ Deprecated
response = llm("Hello")

# ✅ Use invoke
response = llm.invoke("Hello")
```

---

## Interview Preparation

### Easy
**Q: What is the difference between `langchain-core` and `langchain-openai`?**
**A:** `langchain-core` has provider-agnostic abstractions (Runnable, PromptTemplate, Messages). `langchain-openai` has OpenAI-specific implementations (ChatOpenAI, OpenAIEmbeddings).

### Medium
**Q: What are the 6 methods every Runnable supports?**
**A:** `invoke()`, `stream()`, `batch()` (sync) and `ainvoke()`, `astream()`, `abatch()` (async). This uniform interface lets any component be used interchangeably.

---

## Summary

| Concept | What It Does |
|---------|-------------|
| `ChatOpenAI` | LangChain's OpenAI LLM wrapper |
| `HumanMessage` | Typed user message (replaces role dicts) |
| `SystemMessage` | Typed system instruction |
| `AIMessage` | Typed AI response |
| `invoke()` | Single synchronous call |
| `stream()` | Token-by-token generator |
| `batch()` | Multiple inputs at once |
| `base_url` | Route through proxy or alternative endpoint |

---

## What's Next

In the next chapter, you'll learn the **Runnable Protocol & LCEL (LangChain Expression Language)** — LangChain's pipe operator (`|`) that lets you chain components together like Unix commands. This is the core abstraction that makes LangChain powerful.

> [← Previous: Output Parsing](../phase-02-prompt-engineering/chapter-13-output-parsing.md) | [Next: Runnable Protocol & LCEL →](chapter-15-runnable-lcel.md)

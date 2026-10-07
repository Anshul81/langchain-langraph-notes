# Chapter 8.2: Buffer, Window, Summary Memory Strategies

> **Phase 8 — Memory Systems** | [← Previous: Why Memory?](chapter-32-why-memory.md) | [Next: Persistent Memory →](chapter-34-persistent-memory.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Implement **full buffer** memory with `InMemoryChatMessageHistory`
- ✅ Apply **window** memory (last *N* messages)
- ✅ Trim by **token count** with `trim_messages`
- ✅ Build a **summary** memory pattern with an LLM condenser
- ✅ Choose a strategy based on use case trade-offs

| | |
|---|---|
| **Prerequisites** | Chapter 8.1, Chapter 5.3 (chat models) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction

Full history works until it doesn't:

```
Messages:  ████████████████████████████████████████► context limit
Cost:      $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $ $
```

**Memory strategies** control what the model sees each turn while your app may still store the full audit log elsewhere.

```
        ┌─────────────── Full store (DB) ───────────────┐
        │  all messages for compliance / analytics       │
        └────────────────────┬──────────────────────────┘
                             │ select / trim / summarize
                             ▼
                    ┌─────────────────┐
                    │  Prompt window  │  ← what the LLM sees
                    └─────────────────┘
```

---

## Part 1: Buffer Memory — Keep Everything (In Context)

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

history = InMemoryChatMessageHistory()
history.add_user_message("I prefer dark mode UI.")
history.add_ai_message("Got it — I'll suggest dark-friendly themes.")
history.add_user_message("Recommend a Python IDE.")

print(len(history.messages))  # 3
for m in history.messages:
    print(m.type, m.content[:50])
```

**Best for:** short support chats, demos, < ~20 turns.  
**Fails when:** context window or budget is exceeded.

---

## Part 2: Window Memory — Last *N* Messages

Keep only the most recent messages in the **prompt window** (you can still append everything to DB).

```python
from langchain_core.messages import BaseMessage

def get_window_messages(
    messages: list[BaseMessage],
    window_size: int = 6,
) -> list[BaseMessage]:
    """Keep last N messages (typically even N to preserve user/ai pairs)."""
    return messages[-window_size:]

store = InMemoryChatMessageHistory()
for i in range(10):
    store.add_user_message(f"Message {i}")
    store.add_ai_message(f"Reply {i}")

windowed = get_window_messages(store.messages, window_size=4)
print([m.content for m in windowed])
```

### Window in `get_session_history`

```python
WINDOW = 10

def get_windowed_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    hist = session_store[session_id]
    if len(hist.messages) > WINDOW:
        trimmed = hist.messages[-WINDOW:]
        hist.clear()
        for msg in trimmed:
            hist.add_message(msg)
    return hist
```

**Best for:** customer support, quick Q&A where old context matters little.

---

## Part 3: Token-Based Trimming — `trim_messages`

Count tokens with the same model (or a cheap tokenizer) and drop oldest messages:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import trim_messages, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

long_thread = [
    HumanMessage(content="Hello"),
    AIMessage(content="Hi!"),
]
for i in range(30):
    long_thread.append(HumanMessage(content=f"Question {i} " + "word " * 20))
    long_thread.append(AIMessage(content=f"Answer {i} " + "word " * 20))

trimmed = trim_messages(
    long_thread,
    max_tokens=500,
    strategy="last",
    token_counter=llm,
    include_system=True,
    start_on="human",
)

print(f"Before: {len(long_thread)} messages, after: {len(trimmed)}")
```

| Parameter | Role |
|-----------|------|
| `max_tokens` | Budget for history slice |
| `strategy="last"` | Keep most recent content |
| `start_on="human"` | Ensure valid turn structure |
| `include_system` | Keep system message if present |

**Best for:** general chatbots tied to a specific model's context limit.

### Trimmer in LCEL

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

def inject_and_trim(inputs: dict) -> dict:
    raw_history = inputs.get("history", [])
    inputs["history"] = trim_messages(
        raw_history,
        max_tokens=800,
        strategy="last",
        token_counter=llm,
        start_on="human",
    )
    return inputs

chain = RunnableLambda(inject_and_trim) | prompt | llm | StrOutputParser()
```

---

## Part 4: Summary Memory — Compress the Past

Periodically replace old messages with one **summary** `SystemMessage` or `HumanMessage`:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the conversation so far. Keep names, decisions, and open tasks."),
    ("human", "{transcript}"),
])

summarizer = summarize_prompt | llm | StrOutputParser()

def transcript_from_messages(messages: list[BaseMessage]) -> str:
    lines = []
    for m in messages:
        lines.append(f"{m.type}: {m.content}")
    return "\n".join(lines)

def maybe_summarize(history: InMemoryChatMessageHistory, threshold: int = 12) -> None:
    if len(history.messages) <= threshold:
        return
    keep = history.messages[-4:]
    old = history.messages[:-4]
    summary = summarizer.invoke({"transcript": transcript_from_messages(old)})
    history.clear()
    history.add_ai_message(f"[Summary of earlier conversation]\n{summary}")
    for m in keep:
        history.add_message(m)
```

Production pattern:

1. Store **full log** in DB  
2. Maintain **`rolling_summary`** field per session  
3. Prompt = system + summary + last K verbatim messages  

**Best for:** long coaching sessions, tutoring, B2B copilots.

**Cost:** extra LLM call when summarizing; summary can be **lossy**.

---

## Part 5: Strategy Comparison

| Strategy | Recall | Token use | Complexity | Best for |
|----------|--------|-----------|------------|----------|
| **Full buffer** | Perfect (until limit) | High | Low | Short chats |
| **Window (N msgs)** | Recent only | Bounded | Low | Support bots |
| **Token trim** | Recent, model-aware | Bounded | Medium | Production chat |
| **Summary** | Long-range facts (lossy) | Low | High | Long sessions |
| **Hybrid** | Summary + window | Controlled | Highest | Enterprise assistants |

---

## Part 6: Legacy `ConversationBufferMemory` (Awareness)

Older LangChain code uses:

```python
# Legacy — know it for interviews, prefer RunnableWithMessageHistory in new code
# from langchain.memory import ConversationBufferMemory
```

Modern stack: **`ChatMessageHistory` + `MessagesPlaceholder` + `RunnableWithMessageHistory`**. Strategies above apply to **what you put in history**, not which class name you use.

---

## Common Mistakes

### Mistake 1: Window size of 1

A single message loses pairing — keep even counts or use `start_on="human"` when trimming.

### Mistake 2: Summarizing without keeping recent verbatim turns

Model needs fresh detail for the current task — always keep last K raw messages.

### Mistake 3: Trimming without counting system prompt tokens

Leave headroom for the model's reply (`max_tokens` on completion).

### Mistake 4: Same window for all users

Power users need summarization; casual users need small windows — consider dynamic policies.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Full log in DB, trimmed view in prompt | Compliance + performance |
| Re-summarize on schedule, not every turn | Control cost |
| Test with longest realistic transcripts | Avoid prod surprises |
| Include summary version in metadata | Debug drift |
| Monitor "forgot my name" reports | Signal strategy too aggressive |

---

## Interview Preparation

### Easy
**Q: What is window memory?**

> Only the last **N messages** (or tokens) are sent to the model. Older turns are dropped from the prompt, not necessarily from permanent storage.

### Medium
**Q: Compare token trimming vs fixed message window.**

> Message window is simple but ignores message length — one long paste can blow the context. Token trimming adapts to actual model token counts and aligns with provider limits.

### Hard
**Q: How does summary memory fail, and how do you mitigate?**

> Summaries drop nuance (numbers, negation, conditions). Mitigate: keep recent verbatim messages, store structured **facts** (user profile JSON) separately, re-extract facts with a validator, and allow users to view/edit stored summary.

### Senior
**Q: Design memory for a coding assistant with 50-turn debugging sessions.**

> Hybrid: (1) structured state — current file, error stack, git branch in system or tool state; (2) rolling summary of decisions; (3) last 10–15 verbatim turns; (4) retrieve relevant docs via RAG instead of stuffing full logs; (5) optional compaction when token monitor > 70% of window; (6) LangGraph checkpoint if multi-step tool loops — message-only memory is insufficient for agent state.

---

## Summary

| Strategy | Mechanism |
|----------|-----------|
| Buffer | All messages in prompt |
| Window | Last N messages |
| Token trim | `trim_messages` + token_counter |
| Summary | LLM condenses old turns |
| Hybrid | Summary + window + structured facts |

---

## Hands-on Exercise

Simulate 15 turns in a loop. After each turn, apply **window=6** and print what would be sent to the model. Repeat with `trim_messages(max_tokens=300)`.

---

## Challenge Project

Implement **hybrid memory**: when `len(messages) > 20`, summarize messages 1–16, keep 17–20 raw, store summary in a dict `session_meta["summary"]`. Wire into a minimal `chat(session_id, text)` CLI.

---

## What's Next

Chapter 8.3 moves history out of process memory into **Redis and PostgreSQL** so sessions survive restarts and scale across workers.

> [← Previous: Why Memory?](chapter-32-why-memory.md) | [Next: Persistent Memory →](chapter-34-persistent-memory.md)

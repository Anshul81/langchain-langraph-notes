# Chapter 8.4: Memory in LCEL Chains — Production Conversational Chain

> **Phase 8 — Memory Systems** | [← Previous: Persistent Memory](chapter-34-persistent-memory.md) | [Next: What Are Tools? →](../phase-09-tools-tool-calling/chapter-36-what-are-tools.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Explain **exactly** what `RunnableWithMessageHistory` does before, during, and after a run
- ✅ Match **key names** correctly between prompt, input, output, and wrapper (the #1 source of silent bugs)
- ✅ Place `MessagesPlaceholder` correctly and configure **multi-key sessions** (`user_id` + `conversation_id`)
- ✅ Put `trim_messages` / `filter_messages` **inside** the pipeline so storage stays complete and prompts stay small
- ✅ Stream tokens **and** still persist the AI reply; know what happens on client disconnect
- ✅ Use async (`ainvoke` / `astream`) correctly with sync factories and async-capable histories
- ✅ Build and run a complete **FastAPI persistent chatbot** (`/chat`, `/chat/stream` SSE, Redis with in-memory fallback, hybrid trim + summary hook)
- ✅ Choose between `RunnableWithMessageHistory` and **LangGraph `MemorySaver`**
- ✅ Debug "my memory doesn't work" systematically

| | |
|---|---|
| **Prerequisites** | Chapters 8.1–8.3, Ch 5.4 (LCEL), Ch 7.1 (ChatPromptTemplate), basic FastAPI + `async/await` |
| **Estimated Reading Time** | 35 minutes |
| **Estimated Coding Time** | 90 minutes |

---

## Recap (read in 30 seconds)

- LLMs are stateless → you resend context each call (8.1).
- You *store* everything but *send* a bounded view: buffer / window / token trim / summary (8.2).
- Storage lives behind `BaseChatMessageHistory`, selected by a `get_session_history(session_id)` factory; Redis/SQL survive restarts and workers (8.3).
- **This chapter:** the glue — how that factory + a trimmed view plug into an LCEL pipeline correctly, in production.

---

## Introduction

### The Problem

Most "memory" code that works in a notebook fails in production for reasons that have nothing to do with storage:

```python
# Looks right. Bot still forgets, or leaks, or blows the token limit.
chain_with_memory = RunnableWithMessageHistory(chain, get_history,
        input_messages_key="question", history_messages_key="history")
```

| Silent failure | Root cause |
|----------------|-----------|
| Bot never remembers, no error | `history_messages_key` ≠ the `MessagesPlaceholder` variable name |
| Every request is a new conversation | `session_id` missing / different per call |
| Token limit errors after 40 turns | Trimmer placed in the wrong spot, or storage truncated destructively |
| Streaming works, but next turn forgets | Stream aborted before the run ended → nothing saved |
| Event-loop stalls under load | Blocking history I/O inside async endpoints |
| Tool-call noise confuses the model | Raw `ToolMessage`s replayed into the prompt |

### The Solution

Understand the wrapper as a **small, predictable machine** with an *enter* hook and an *exit* hook, and put your own logic (filter → trim → summarize) in the **middle**, where it only affects the prompt view:

```
                 RunnableWithMessageHistory
 ┌────────────────────────────────────────────────────────────────┐
 │  ENTER: load history ──► inject as inputs["history"]           │
 │                                                                │
 │      ┌──────────── YOUR INNER PIPELINE ──────────────┐         │
 │      │ filter ► trim ► (summary) ► prompt ► llm ► parser │      │
 │      └────────────────────────────────────────────────┘         │
 │                                                                │
 │  EXIT: append [human input, ai output] to the STORE (full)     │
 └────────────────────────────────────────────────────────────────┘
        Store keeps everything.  Prompt sees a curated view.
```

### Industry Usage

- Customer-support copilots: per-ticket sessions, trimmed context, archive in SQL.
- Internal helpdesk bots: SSE streaming UIs with Redis-backed sessions.
- Sales/CRM assistants: entity profile (name, plan, preferences) injected next to a short rolling window.

### Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "`RunnableWithMessageHistory` trims for me" | It does **not**. It loads *all* messages. Trimming is your job (inside the chain) |
| "It saves the prompt I sent" | It saves only the **new human input** and the **AI output**, not the trimmed/enriched prompt |
| "`session_id` is a function argument" | It is read from `config["configurable"]` — it's *config*, not input |
| "The wrapper works with any chain" | Only if input/output shapes match what you declared via the `*_key` params |
| "Async factory is supported" | The factory is a **sync** callable; async I/O comes from the history object's `aget_messages`/`aadd_messages` |
| "Memory = chat history" | Chat history is *one* kind (episodic). User profiles/facts are another — see Homework |

---

## Mental Model

### Analogy: The Court Stenographer and the Judge's Briefing

- **Stenographer (store)** records *every* word, forever (the full transcript).
- **Clerk (wrapper)** hands the judge the transcript at the start of each hearing and logs the new exchange at the end.
- **Judge's briefing note (your pipeline)** is a *condensed* version prepared from the transcript — recent exchanges plus a summary of old ones. The judge never reads 4,000 pages.
- The clerk never *edits* the transcript to make the briefing shorter. Editing the source to fit the view is the classic bug (destructive trimming).

### Data Flow for One Turn

```
 client ─ POST /chat ─► { "input": "What did I just say?" }
                              │  config = {"configurable": {"session_id": "acme:u1"}}
                              ▼
        ┌───────── RunnableWithMessageHistory ─────────┐
 (1)    │ factory("acme:u1") → history object          │
 (2)    │ history.messages  → [H1, A1, H2, A2, ...]    │
 (3)    │ inputs = {"input": "...", "history": [...]}  │
        └───────────────────────┬──────────────────────┘
                                ▼
 (4)   filter_messages ─► trim_messages ─► (summary) ─► prompt ─► llm ─► parser
                                │                                          │
                                ▼                                          ▼
                       prompt sees ≤ N tokens                       "You said ..."
                                                                           │
        ┌───────────────────────┴──────────────────────────────────────────┘
 (5)    │ on success: history.add_messages([Human(input), AI(output)])
        └─► Redis / SQL now has FULL history + 2 new messages
```

---

## Theory

### Part 1: `RunnableWithMessageHistory` — Deep Dive

#### Constructor

```python
RunnableWithMessageHistory(
    runnable,                    # the chain to wrap (LCEL Runnable)
    get_session_history,         # callable(session_id: str) -> BaseChatMessageHistory
    *,
    input_messages_key=None,     # which input-dict key holds the NEW user message
    history_messages_key=None,   # which input-dict key receives the HISTORY
    output_messages_key=None,    # which output-dict key holds the AI reply (if output is a dict)
    history_factory_config=None, # list[ConfigurableFieldSpec] for multi-key sessions
)
```

#### Which keys do I need? (Input/Output shape table)

| Wrapped runnable takes… | Set `input_messages_key`? | Set `history_messages_key`? | Notes |
|---|---|---|---|
| `dict` (typical prompt chain) | ✅ yes (the user text key) | ✅ yes (matches `MessagesPlaceholder`) | Most common |
| `list[BaseMessage]` / single message | ❌ no | ❌ no | History is **prepended** to the message list |
| `dict` where one key is a message list | ✅ yes | ❌ no | History merged into that key |

| Wrapped runnable returns… | Set `output_messages_key`? |
|---|---|
| `str` (via `StrOutputParser`) | ❌ — saved as `AIMessage` |
| `AIMessage` / list of messages | ❌ |
| `dict` (e.g. `{"answer":..., "sources":...}`) | ✅ yes, name the reply key |

#### Key Name Matching — The Critical Rule

```
 prompt = ChatPromptTemplate.from_messages([
     ("system", "..."),
     MessagesPlaceholder("chat_log"),   ◄──────────────┐  must equal
     ("human", "{question}"),           ◄───────┐      │
 ])                                              │      │
 RunnableWithMessageHistory(chain, factory,      │      │
     input_messages_key="question",     ─────────┘      │
     history_messages_key="chat_log",   ────────────────┘
 )
 invoke({"question": "hi"}, config={"configurable": {"session_id": "s1"}})
          ▲ same name as input_messages_key
```

Three names, three places, one spelling. A mismatch **rarely raises**: the placeholder becomes empty or the user message isn't found, and the bot simply "has no memory".

#### X-Ray the Wrapper Without Any LLM

Learn the mechanics with a fake model that reports what it received. Zero API cost, fully deterministic:

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

def fake_llm(prompt_value):
    msgs = prompt_value.to_messages()
    return AIMessage(content=f"saw {len(msgs)} msgs; last={msgs[-1].content!r}")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
core = prompt | RunnableLambda(fake_llm) | StrOutputParser()

_store: dict[str, InMemoryChatMessageHistory] = {}
def get_history(sid: str) -> InMemoryChatMessageHistory:
    return _store.setdefault(sid, InMemoryChatMessageHistory())

chat = RunnableWithMessageHistory(core, get_history,
        input_messages_key="input", history_messages_key="history")

cfg = {"configurable": {"session_id": "demo"}}
print(chat.invoke({"input": "one"}, cfg))    # saw 2 msgs  (system + human)
print(chat.invoke({"input": "two"}, cfg))    # saw 4 msgs  (system + H1 + A1 + human)
print(chat.invoke({"input": "three"}, cfg))  # saw 6 msgs
print(len(_store["demo"].messages))          # 6  (3 turns × 2)
```

**What you just proved:** history grows by exactly 2 per successful turn; the prompt receives system + all prior + the new input.

#### Under the Hood (Conceptual Source)

```
RunnableWithMessageHistory ≈
    (  RunnablePassthrough.assign(history = load_history_from_store)   # ENTER
       | your_runnable )
    .with_listeners(on_end = save_new_messages_to_store)               # EXIT
```

| Phase | Action | Consequence |
|-------|--------|-------------|
| **Enter** | Reads `config["configurable"]`, calls your factory, calls `.messages` (or `aget_messages`) | Loads **all** stored messages. No trimming |
| **Run** | Your chain executes with `history` injected | Anything you do to `history` here is *view-only* |
| **Exit (success)** | `add_messages([new human msg(s), AI output])` | Human + AI saved **together**, after completion |
| **Exit (error)** | Nothing saved | No orphan human message after an LLM failure — a good property |

> **Race condition to remember:** two concurrent requests on the **same** session both load the same history and both append — turns interleave. Lock per session (Redis lock) or serialize in your client if this matters.

---

### Part 2: `MessagesPlaceholder` Placement Rules

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a support agent for ACME."),   # 1. rules FIRST (always apply)
    MessagesPlaceholder("history", optional=True),     # 2. past turns
    ("human", "{input}"),                               # 3. the NEW message LAST
])
```

| Rule | Why |
|------|-----|
| System message **before** history | Instructions must not be pushed out by old turns |
| Placeholder **before** the new human message | Chronological order; the latest question must be last |
| Exactly **one** placeholder per history stream | Two placeholders fed from one key = duplicated context |
| `optional=True` when the key may be absent (first turn in some setups, unit tests) | Avoids `KeyError`-style failures |
| `MessagesPlaceholder("history", n_messages=10)` | Built-in "last 10 messages" cap — quick, but **count-based**, not token-based |
| `("placeholder", "{history}")` | Tuple shorthand = *optional* `MessagesPlaceholder` |
| Other placeholders (e.g. `agent_scratchpad`) go **after** the human message | Tool-call scratch space is conceptually the latest step |

```
 ✅ system │ history… │ human(new)            ❌ history… │ system │ human
 ✅ system │ summary-as-system │ history… │ human   ❌ human(new) │ history…  (question first!)
```

**Never** string-format history into the prompt (`f"{history}"`) — you lose role structure and break tool-call/message semantics.

---

### Part 3: Multi-Session Configs

#### Simple: one key

```python
chat.invoke({"input": "I love Python."}, {"configurable": {"session_id": "alice"}})
chat.invoke({"input": "I hate mornings."}, {"configurable": {"session_id": "bob"}})
```

#### Realistic: user **and** conversation

A user has many threads (ChatGPT's sidebar). Declare two config fields; the factory receives both:

```python
from langchain_core.runnables import ConfigurableFieldSpec

def get_history_2(user_id: str, conversation_id: str):
    return _store.setdefault(f"{user_id}:{conversation_id}", InMemoryChatMessageHistory())

chat2 = RunnableWithMessageHistory(
    core, get_history_2,
    input_messages_key="input", history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(id="user_id", annotation=str, name="User ID",
                              description="Authenticated user", default="", is_shared=True),
        ConfigurableFieldSpec(id="conversation_id", annotation=str, name="Conversation ID",
                              description="Thread within a user", default="", is_shared=True),
    ],
)

cfg = {"configurable": {"user_id": "alice", "conversation_id": "trip-planning"}}
chat2.invoke({"input": "Plan Goa."}, cfg)
```

| Style | Use when |
|-------|----------|
| Single `session_id` | Prototypes; you pre-compose the ID (`tenant:user:conv`) in your API layer |
| Multi-key `ConfigurableFieldSpec` | You want typed, self-documenting config and LangServe/OpenAPI visibility |

In the production app below we pre-compose a single validated `session_id` — fewer moving parts, and the security logic lives in one function (see Ch 8.3 Part 6).

---

### Part 4: `trim_messages` INSIDE the Pipeline

#### Where to put it

The wrapper injects `history` *at its boundary*. So trimming must happen **inside** the wrapped chain, operating on `inputs["history"]`:

```
 ❌ A) trimmer | wrapper(...)           trimmer sees the user's dict, not the history
 ❌ B) trim inside get_session_history  DESTRUCTIVE: deletes from the store
 ✅ C) wrapper( assign(history = history | trimmer) | prompt | llm | parser )
       store: full ──► view: trimmed ──► prompt
```

```python
from operator import itemgetter
from langchain_core.messages import trim_messages
from langchain_core.runnables import RunnablePassthrough

def approx_tokens(msgs) -> int:          # offline & fast; swap for the model's counter in prod
    return sum(len(str(m.content)) // 4 + 4 for m in msgs)

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",             # keep the most recent
    token_counter=approx_tokens,
    include_system=False,        # system prompt isn't in history; it lives in the template
    start_on="human",            # never start the window with an orphan AI message
    allow_partial=False,         # don't cut a message in half
)

core = (
    RunnablePassthrough.assign(history=itemgetter("history") | trimmer)   # view-only
    | prompt | llm | StrOutputParser()
)
chat = RunnableWithMessageHistory(core, get_session_history,
                                  input_messages_key="input", history_messages_key="history")
```

`RunnablePassthrough.assign(history=...)` **overwrites** the `history` key with the trimmed version while passing everything else (e.g., `input`) through untouched.

**Proof the store stays whole** (extend the X-ray demo): after 30 turns, `len(store[sid].messages) == 60` while `fake_llm` reports a bounded message count.

| Choice | Guidance |
|--------|----------|
| Budget | Context window − system prompt − expected answer − safety margin (≈15%) |
| Counter | Provider/tokenizer-accurate in prod; `len` + `max_tokens=N` for "last N messages" |
| `start_on="human"` | Prevents a window beginning mid-exchange |
| Summary | Summarize what the trimmer **dropped** (see the production app's `aprepare`) |

---

### Part 5: Streaming + Memory

```python
async for chunk in chat.astream({"input": "Explain RAG briefly."}, cfg):
    print(chunk, end="", flush=True)           # tokens as they arrive
# After the loop ends cleanly → human + full AI text are persisted automatically.
```

```
 t0  enter: load history
 t1  tokens: "RAG" " is" " a" ...   ◄── client sees these live
 t2  stream exhausted  ──► EXIT hook: save [Human, AI(full text)]
```

| Situation | Persisted? |
|-----------|-----------|
| Stream consumed to completion | ✅ Human + complete AI message |
| Client disconnects mid-stream (generator closed/cancelled) | ⚠️ Typically **neither** — the run didn't finish, so the exit hook is not a success path. Verify with your versions and test it |
| LLM raises mid-stream | ❌ nothing saved |

Design implications:

1. Treat a disconnect as "turn didn't happen"; the user simply resends. This is simple and consistent.
2. If you must keep partial answers, accumulate tokens yourself and write them in `finally` — and then **don't** rely on the wrapper for that endpoint (otherwise you double-save).
3. Always send a terminal SSE event (`done` / `error`) so clients know whether to trust the turn.

`astream_events(version="v2")` is available when you also need token events plus tool/retriever events; for a plain chatbot `astream` is enough.

---

### Part 6: Async Patterns

```python
reply = await chat.ainvoke({"input": "hi"}, cfg)
```

| Layer | Async story |
|-------|-------------|
| `get_session_history` | A **sync** factory. Keep it *cheap* — construct a handle, do no I/O in it |
| History object | I/O happens in `aget_messages()` / `aadd_messages()` — the wrapper's async path calls these |
| Sync-only history (community Redis class) | Default async methods run the sync version in a thread pool → works, but occupies threads |
| Truly async | `SQLChatMessageHistory(..., async_mode=True)` + `postgresql+asyncpg://…`, or a custom class using `redis.asyncio` |

**Pattern: share clients, create handles per request**

```python
import json, redis, redis.asyncio as aredis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import message_to_dict, messages_from_dict

_SYNC  = redis.Redis.from_url(REDIS_URL)          # module-level pools: created ONCE
_ASYNC = aredis.Redis.from_url(REDIS_URL)

class RedisHistory(BaseChatMessageHistory):       # the handle is per-request & cheap
    def __init__(self, session_id: str, ttl: int = 604800):
        self.key, self.ttl = f"chat:{session_id}", ttl

    @property
    def messages(self):
        return messages_from_dict([json.loads(r) for r in _SYNC.lrange(self.key, 0, -1)])
    def add_messages(self, messages):
        p = _SYNC.pipeline(); p.rpush(self.key, *[json.dumps(message_to_dict(m)) for m in messages])
        p.expire(self.key, self.ttl); p.execute()
    def clear(self): _SYNC.delete(self.key)

    async def aget_messages(self):
        return messages_from_dict([json.loads(r) for r in await _ASYNC.lrange(self.key, 0, -1)])
    async def aadd_messages(self, messages):
        p = _ASYNC.pipeline(); p.rpush(self.key, *[json.dumps(message_to_dict(m)) for m in messages])
        p.expire(self.key, self.ttl); await p.execute()
    async def aclear(self): await _ASYNC.delete(self.key)
```

> This stores a plain Redis list in its **own** JSON format, so don't point it at keys written by `RedisChatMessageHistory` from Ch 8.3 (different layout). Pick one per environment.

**Rule of thumb:** in FastAPI, use `ainvoke`/`astream` end-to-end. One blocking sync call inside an `async def` endpoint stalls every other request on that worker.

---

### Part 7: Filtering History Before It Reaches the Prompt

Real histories contain noise: tool calls/results, blank messages, system-injected notes, "typing…" markers. Filter the **view**, keep the store intact.

```python
from langchain_core.messages import filter_messages
from langchain_core.runnables import RunnableLambda

drop_tools = filter_messages(exclude_types=["tool", "system"], exclude_tool_calls=True)

def drop_blank(msgs):
    return [m for m in msgs if str(m.content).strip()]

def drop_noise(msgs):                      # custom rule: messages flagged at write time
    return [m for m in msgs if not m.additional_kwargs.get("noise")]

history_view = drop_tools | RunnableLambda(drop_blank) | RunnableLambda(drop_noise) | trimmer
core = (RunnablePassthrough.assign(history=itemgetter("history") | history_view)
        | prompt | llm | StrOutputParser())
```

| Order | Reason |
|-------|--------|
| **Filter → trim**, not trim → filter | Otherwise filtered-out messages consume the token budget |
| Remove tool calls **and** their results together (`exclude_tool_calls=True`) | An orphan `ToolMessage` without its call is rejected by many providers |
| Keep it a pure function | Easy to unit-test: list in, list out |

---

## Architecture — Putting It Together

```
                           ┌──────────────────────────────────────────┐
  Client (browser/CLI) ──► │ FastAPI                                  │
   X-User-Id / token       │  resolve_session() → "user:conv"         │
   POST /chat              │  validate · rate-limit · error mapping   │
   POST /chat/stream (SSE) └───────────────┬──────────────────────────┘
                                           ▼
                        RunnableWithMessageHistory (get_session_history)
                                           │
              ┌────────────────────────────┴─────────────────────────────┐
              │ aprepare: filter ► trim ► [summary of dropped] ► inputs  │
              │ prompt(system + {summary_block} │ history │ human)       │
              │ ChatOpenAI via LiteLLM proxy ► StrOutputParser           │
              └────────────────────────────┬─────────────────────────────┘
                                           ▼
                       Redis (shared, TTL)   ⇄   InMemory fallback (dev)
```

---

## Part 8: Production Project — Persistent Conversational Chatbot

### Setup

```bash
pip install fastapi "uvicorn[standard]" redis httpx python-dotenv \
            langchain-core langchain-openai pydantic
docker run -d --name chat-redis -p 6379:6379 redis:7     # optional: omit to use the in-memory fallback
```

`.env`:

```env
LITELLM_PROXY_API_KEY=sk-...
LITELLM_PROXY_API_BASE=http://localhost:4000
LITE_LLM_MODEL=gpt-4o-mini
REDIS_URL=redis://localhost:6379/0
HISTORY_TTL_SECONDS=604800
MAX_HISTORY_TOKENS=1500
SUMMARY_ENABLED=false
```

### `app.py` (complete, copy-paste runnable)

```python
import asyncio, json, logging, os, re
from contextlib import asynccontextmanager
from operator import itemgetter

import redis
import redis.asyncio as aredis
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import (filter_messages, get_buffer_string, message_to_dict,
                                     messages_from_dict, trim_messages)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chatbot")

# ───────────────────────── config ─────────────────────────
REDIS_URL = os.getenv("REDIS_URL")
HISTORY_TTL = int(os.getenv("HISTORY_TTL_SECONDS", 7 * 24 * 3600))
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", 1500))
SUMMARY_ENABLED = os.getenv("SUMMARY_ENABLED", "false").lower() == "true"
MAX_INPUT_CHARS = 4000
REQUEST_TIMEOUT = 60

# ───────────────────────── history backends ─────────────────────────
_SYNC = redis.Redis.from_url(REDIS_URL) if REDIS_URL else None      # shared pools, created once
_ASYNC = aredis.Redis.from_url(REDIS_URL) if REDIS_URL else None
_MEMORY: dict[str, InMemoryChatMessageHistory] = {}
_redis_ok = False                                                     # set at startup


class RedisHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.key = f"chat:{session_id}"

    @staticmethod
    def _dump(msgs): return [json.dumps(message_to_dict(m)) for m in msgs]

    @property
    def messages(self):
        return messages_from_dict([json.loads(r) for r in _SYNC.lrange(self.key, 0, -1)])
    def add_messages(self, messages):
        p = _SYNC.pipeline(); p.rpush(self.key, *self._dump(messages)); p.expire(self.key, HISTORY_TTL); p.execute()
    def clear(self): _SYNC.delete(self.key)

    async def aget_messages(self):
        return messages_from_dict([json.loads(r) for r in await _ASYNC.lrange(self.key, 0, -1)])
    async def aadd_messages(self, messages):
        p = _ASYNC.pipeline(); p.rpush(self.key, *self._dump(messages)); p.expire(self.key, HISTORY_TTL); await p.execute()
    async def aclear(self): await _ASYNC.delete(self.key)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if _redis_ok:
        return RedisHistory(session_id)
    return _MEMORY.setdefault(session_id, InMemoryChatMessageHistory())   # per-process fallback


# ───────────────────────── LLM + pipeline ─────────────────────────
llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0.3,
)

SYSTEM = "You are a concise, friendly assistant. Use the conversation context when relevant."
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "{summary_block}"),
    MessagesPlaceholder("history", optional=True),
    ("human", "{input}"),
])

def approx_tokens(msgs) -> int:
    return sum(len(str(m.content)) // 4 + 4 for m in msgs)

clean = filter_messages(exclude_types=["tool", "system"], exclude_tool_calls=True)
trimmer = trim_messages(max_tokens=MAX_HISTORY_TOKENS, strategy="last", token_counter=approx_tokens,
                        include_system=False, start_on="human", allow_partial=False)

summarizer = (
    ChatPromptTemplate.from_template(
        "Summarize this conversation in <=80 words. Keep names, facts, preferences, open tasks.\n\n{transcript}")
    | llm | StrOutputParser()
)
_SUMMARIES: dict[tuple[str, int], str] = {}   # demo cache; use Redis + incremental summaries in prod


async def aprepare(inputs: dict, config: RunnableConfig) -> dict:
    """View-only transform: filter -> trim -> optional summary of what was dropped."""
    visible = [m for m in clean.invoke(inputs.get("history", [])) if str(m.content).strip()]
    recent = trimmer.invoke(visible)
    summary = ""
    dropped = visible[: len(visible) - len(recent)]
    if SUMMARY_ENABLED and dropped:
        key = (config["configurable"]["session_id"], len(dropped))
        if key not in _SUMMARIES:
            _SUMMARIES[key] = await summarizer.ainvoke({"transcript": get_buffer_string(dropped)})
        summary = _SUMMARIES[key]
    block = f"\n\nSummary of earlier conversation:\n{summary}" if summary else ""
    return {**inputs, "history": recent, "summary_block": block}


core = RunnableLambda(aprepare) | prompt | llm | StrOutputParser()     # async-only step: use ainvoke/astream
chatbot = RunnableWithMessageHistory(
    core, get_session_history, input_messages_key="input", history_messages_key="history",
)

# ───────────────────────── API ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _redis_ok
    if _ASYNC is not None:
        try:
            await _ASYNC.ping(); _redis_ok = True
            log.info("History backend: Redis")
        except redis.RedisError:
            log.warning("Redis unreachable -> IN-MEMORY fallback (NOT shared across workers!)")
    else:
        log.warning("REDIS_URL not set -> IN-MEMORY fallback")
    yield

app = FastAPI(title="Persistent Chatbot", lifespan=lifespan)

@app.exception_handler(redis.RedisError)
async def redis_error_handler(_: Request, exc: redis.RedisError):
    log.error("Redis error: %s", exc)
    return JSONResponse({"detail": "Conversation store unavailable, try again shortly."}, status_code=503)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    messages_stored: int


_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

def resolve_session(user_id: str, conversation: str) -> str:
    """Server-side identity. Replace X-User-Id with a verified JWT subject in production."""
    if not (_SAFE.match(user_id) and _SAFE.match(conversation)):
        raise HTTPException(400, "Invalid X-User-Id or X-Conversation (allowed: A-Z a-z 0-9 _ -, max 64)")
    return f"{user_id}:{conversation}"

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "redis" if _redis_ok else "memory"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, x_user_id: str = Header(...), x_conversation: str = Header("default")):
    sid = resolve_session(x_user_id, x_conversation)
    cfg = {"configurable": {"session_id": sid}}
    try:
        reply = await asyncio.wait_for(chatbot.ainvoke({"input": req.message}, cfg), REQUEST_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(504, "The model took too long. Please retry.")
    except redis.RedisError:
        raise                                                       # handled -> 503
    except Exception:
        log.exception("LLM pipeline failed (sid=%s)", sid)          # nothing was saved on failure
        raise HTTPException(502, "Upstream model error. Please retry.")
    stored = len(await get_session_history(sid).aget_messages())
    return ChatResponse(reply=reply, session_id=sid, messages_stored=stored)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, x_user_id: str = Header(...), x_conversation: str = Header("default")):
    sid = resolve_session(x_user_id, x_conversation)
    cfg = {"configurable": {"session_id": sid}}
    await get_session_history(sid).aget_messages()      # pre-flight: store errors -> proper HTTP 503, not mid-stream

    async def gen():
        try:
            async for chunk in chatbot.astream({"input": req.message}, cfg):
                yield sse("token", {"text": chunk})
            yield sse("done", {"session_id": sid})        # history persisted by the wrapper at this point
        except asyncio.CancelledError:
            log.info("Client disconnected (sid=%s) - turn not persisted", sid)
            raise
        except Exception:
            log.exception("stream failed (sid=%s)", sid)
            yield sse("error", {"detail": "stream failed; this turn was not saved"})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/chat/history")
async def history(x_user_id: str = Header(...), x_conversation: str = Header("default")):
    sid = resolve_session(x_user_id, x_conversation)
    msgs = await get_session_history(sid).aget_messages()
    return {"session_id": sid, "messages": [{"role": m.type, "content": m.content} for m in msgs]}


@app.delete("/chat")
async def reset(x_user_id: str = Header(...), x_conversation: str = Header("default")):
    sid = resolve_session(x_user_id, x_conversation)
    await get_session_history(sid).aclear()
    return {"cleared": sid}
```

### Run and Test

```bash
uvicorn app:app --port 8000 --workers 2        # 2 processes: proves Redis sharing (no --reload with --workers)
```

```python
# client.py
import httpx
H = {"X-User-Id": "alice", "X-Conversation": "demo"}
with httpx.Client(base_url="http://localhost:8000", timeout=60) as c:
    print(c.get("/health").json())
    print(c.post("/chat", headers=H, json={"message": "My name is Rahul and I like trekking."}).json())
    with c.stream("POST", "/chat/stream", headers=H, json={"message": "What do I like?"}) as r:
        for line in r.iter_lines():
            if line: print(line)
    print(c.get("/chat/history", headers=H).json())
```

Equivalent `curl` (bash): `curl -N -X POST localhost:8000/chat/stream -H "X-User-Id: alice" -H "Content-Type: application/json" -d '{"message":"hi"}'` (use `curl.exe` in PowerShell).

### Design Decisions Worth Defending

| Decision | Reason |
|----------|--------|
| Single `aprepare` step | One place for filter → trim → summary; trivially unit-testable |
| Store untouched, view trimmed | Audit completeness + bounded cost |
| Pre-flight read in `/chat/stream` | Store outage becomes a clean 503 instead of a broken stream |
| Fail on error, save nothing | No half-turns in history |
| Redis fallback → in-memory with loud log | Dev-friendly; in prod you'd refuse to start instead |
| Server-built `session_id` | Tenancy boundary (Ch 8.3) |
| `SUMMARY_ENABLED` off by default | Extra LLM call per turn once overflowing; enable deliberately |

---

## Part 9: `RunnableWithMessageHistory` vs LangGraph `MemorySaver`

```python
# Phase 15 preview — same idea, graph-level persistence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START

def call_model(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

g = StateGraph(MessagesState)
g.add_node("model", call_model); g.add_edge(START, "model")
graph = g.compile(checkpointer=MemorySaver())       # swap for a Postgres/Redis checkpointer in prod

graph.invoke({"messages": [("human", "I'm Rahul")]}, {"configurable": {"thread_id": "t1"}})
out = graph.invoke({"messages": [("human", "Who am I?")]}, {"configurable": {"thread_id": "t1"}})
```

| Dimension | `RunnableWithMessageHistory` | LangGraph checkpointer (`MemorySaver`, Postgres, …) |
|-----------|------------------------------|------------------------------------------------------|
| What is persisted | **Messages only** | **Entire graph state** (messages + any fields) |
| Identity key | `session_id` (you choose) | `thread_id` (+ checkpoint id) |
| Time travel / replay / fork | ❌ | ✅ |
| Human-in-the-loop interrupts | ❌ | ✅ |
| Multi-step agents, branching, retries | Awkward | Native |
| Simplicity for a plain chatbot | ✅ ~10 lines | More ceremony |
| Wraps existing LCEL chain unchanged | ✅ | Needs graph wrapper |
| Storage backends | `BaseChatMessageHistory` (Ch 8.3) | Checkpointer savers |
| Ecosystem direction | Stable, legacy-friendly | Recommended for new stateful/agent apps |

```
 Need only "remember the conversation" in a chain?          → RunnableWithMessageHistory
 Need tools loops, approvals, resume after crash, branching? → LangGraph checkpointer
 Both? Common: LCEL chain for a node, graph for orchestration, checkpointer for state.
```

Phase 15 revisits this with real agents. For *this* course project, the wrapper is the right size tool.

---

## Part 10: Debugging — "Memory Doesn't Work"

### Decision Flow

```
 Bot forgets between turns
        │
        ├─ Is the SAME session_id sent each time?  ──no──► fix client/config (print config)
        │ yes
        ├─ Is anything in the store after turn 1?   ──no──► exit hook not firing: errors? aborted stream?
        │ yes                                               wrong key in output? (output_messages_key)
        ├─ Does the prompt contain history?         ──no──► key mismatch / missing placeholder / filter removed all
        │ yes
        ├─ Works on 1 worker, fails on many?        ──yes─► process-local store (Ch 8.3)
        │ no
        └─ Forgets only OLD facts?                  ──yes─► trimmer budget too small → summary / profile memory
```

### Instrument the Pipeline (the "tap")

```python
def tap(label: str):
    def _show(x):
        view = x.to_messages() if hasattr(x, "to_messages") else x
        print(f"\n── {label} ──")
        for m in (view if isinstance(view, list) else [view]):
            print("  ", getattr(m, "type", "·"), "|", str(getattr(m, "content", m))[:80])
        return x
    return RunnableLambda(_show)

debug_chain = RunnableLambda(aprepare) | prompt | tap("PROMPT SENT TO LLM") | llm | StrOutputParser()
```

Also: `from langchain_core.globals import set_debug; set_debug(True)`, and `chain.get_graph().print_ascii()` to see structure.

### Symptom Table

| Symptom | Cause | Fix |
|---------|-------|-----|
| No error, no memory | `history_messages_key` ≠ placeholder name | Align all three names |
| `KeyError: 'input'` / missing variable | `input_messages_key` ≠ prompt variable | Align |
| `ValueError: Expected configurable session_id` style error | Missing `config["configurable"]` | Always pass config |
| Store grows by 1, not 2 | Output is a dict and `output_messages_key` unset/wrong | Set it to the reply key |
| History has 2 copies per turn | Wrapper **and** manual `add_*` calls | Pick one writer |
| Works first run, "forgets" after redeploy | In-memory fallback silently active | Check `/health` backend |
| Prompt grows unbounded | Trim placed outside / before wrapper | Move into the inner chain |
| Provider 400 on orphan tool message | Filter removed call but not result | `exclude_tool_calls=True` |
| Order scrambled across requests | Concurrent writes same session | Per-session lock/queue |
| Stream OK but next turn blank | Client disconnected early | Expected; handle with retry UX |

---

## Common Mistakes

### Mistake 1: Trimming by mutating the store

```python
# ❌ Data loss + side effects in a "getter"
def get_session_history(sid):
    h = _store[sid]; h.messages[:] = h.messages[-10:]; return h
# ✅ Keep the store whole; trim in the pipeline (Part 4)
```

### Mistake 2: Trimmer before the wrapper

```python
# ❌ the trimmer receives {"input": ...}; history isn't injected yet
chat = RunnableWithMessageHistory(trimmer | prompt | llm, ...)
# ✅ transform inputs["history"] inside the wrapped chain
core = RunnablePassthrough.assign(history=itemgetter("history") | trimmer) | prompt | llm
```

### Mistake 3: Putting `session_id` in the input dict

```python
# ❌
chat.invoke({"input": "hi", "session_id": "u1"})
# ✅
chat.invoke({"input": "hi"}, config={"configurable": {"session_id": "u1"}})
```

### Mistake 4: Blocking calls in async endpoints

```python
# ❌ stalls the event loop for ALL users on this worker
@app.post("/chat")
async def chat(...): return chatbot.invoke(...)
# ✅
async def chat(...): return await chatbot.ainvoke(...)
```

### Mistake 5: Saving history manually *and* via the wrapper

```python
chat.invoke(...); history.add_user_message(msg)   # ❌ duplicate human message
```

### Mistake 6: Using the same session for unrelated users

Shared `"default"` session ID = everyone shares one brain. Always derive from identity.

### Mistake 7: `n_messages` as a token guard

`MessagesPlaceholder(n_messages=10)` caps *count*, not tokens. Ten long messages can still overflow context.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Name keys once as constants (`HISTORY_KEY = "history"`) | Eliminates mismatch bugs |
| Store full, send curated | Audit + cost control |
| Order: filter → trim → summarize → prompt | Predictable budget |
| Use `ainvoke`/`astream` in servers | Throughput |
| Share Redis/DB clients; create cheap per-request handles | Connection hygiene |
| Pre-flight store check before opening a stream | Clean error semantics |
| Always emit terminal SSE events | Client correctness |
| Timeouts on LLM calls | Protect workers |
| Unit-test with a fake model (no network) | Fast, deterministic CI |
| Log `session_id`, backend, message counts — **not** message content | Privacy-safe observability |
| Plan the LangGraph migration path early | Agents will need checkpointers |

---

## Interview Preparation

### Easy

**Q: What does `RunnableWithMessageHistory` add to a chain?**
> It loads the session's stored messages into the chain's input (under `history_messages_key`) before the run, and appends the new human input plus the AI output to the store after a successful run. The inner chain itself is unaware of storage.

**Q: Where do you pass the `session_id`?**
> In `config={"configurable": {"session_id": ...}}`, not in the input dict.

### Medium

**Q: How do you keep prompts bounded without losing the transcript?**
> Insert `trim_messages` (preceded by `filter_messages`) *inside* the wrapped chain via `RunnablePassthrough.assign(history=itemgetter("history") | trimmer)`. The store receives full messages; only the prompt view is trimmed.

**Q: Why does the bot have no memory and throw no error?**
> A key mismatch: `history_messages_key` doesn't match the `MessagesPlaceholder` name (or `input_messages_key` doesn't match the human variable), so history is never merged. Verify by printing the prompt with a `tap` runnable.

### Hard

**Q: Explain what happens when a streamed response is cut by a client disconnect.**
> The wrapper persists in an exit hook that runs when the run completes successfully. A cancelled stream typically never reaches it, so neither the human nor the partial AI message is stored. Design for "turn didn't happen" semantics, or accumulate tokens and persist explicitly in `finally` while bypassing the wrapper for that route.

**Q: How do you do async properly with `RunnableWithMessageHistory`?**
> The factory stays sync and cheap; the wrapper's async path uses the history's `aget_messages`/`aadd_messages`. Use an async-native history (`async_mode=True` SQL, or `redis.asyncio`), share client pools, and call `ainvoke`/`astream` end-to-end. Sync-only histories work via thread-pool defaults but consume threads.

### Senior

**Q: Design concurrency control for two simultaneous requests on one session.**
> Both would read the same state and append interleaved turns. Options: client-side serialization, a per-session distributed lock (Redis `SET NX PX` with a fencing token) around the whole turn, or a per-session queue/actor. Return 409/429 when a turn is in flight. For agent workflows, LangGraph's thread-level checkpointing with versioned checkpoints is a cleaner fit.

**Q: When do you outgrow `RunnableWithMessageHistory`?**
> When state is no longer just messages: tool loops, approvals/interrupts, branching, resume-after-crash, time travel, multi-agent handoffs. Then use LangGraph checkpointers (`thread_id`) — persisting the whole state graph — while keeping LCEL chains as node implementations.

**Q: How do you combine rolling-window memory with long-term user facts?**
> Two stores: episodic (chat history, trimmed + summarized) and semantic (an entity profile extracted by structured output, stored per user, injected into the system message each turn). Profile updates run asynchronously after turns; conflicts resolved by recency/confidence.

---

## Summary

| Concept | Takeaway |
|---------|----------|
| Wrapper machine | Enter (load all) → run → Exit (save human + AI on success) |
| Key matching | `input_messages_key`, `history_messages_key`, placeholder name, prompt variable — one spelling |
| Placeholder | After system, before new human; one per stream |
| Multi-session | `session_id` or `ConfigurableFieldSpec` (`user_id` + `conversation_id`) |
| Trimming | Inside the chain, view-only; never mutate the store |
| Filtering | Before trimming; keep tool call/result pairs together |
| Streaming | Saved after completion; disconnect = turn not saved |
| Async | Sync factory, async history I/O, `ainvoke`/`astream` |
| Production | FastAPI + Redis + fallback + SSE + timeouts + error mapping |
| Graph future | LangGraph checkpointers when state ≠ messages |

---

## Cheat Sheet

```python
chat = RunnableWithMessageHistory(
    core, get_session_history,
    input_messages_key="input",          # new user text
    history_messages_key="history",      # == MessagesPlaceholder("history")
    # output_messages_key="answer",      # only if core returns a dict
)
cfg = {"configurable": {"session_id": "tenant:user:conv"}}

core = (RunnablePassthrough.assign(history=itemgetter("history") | filter_messages(exclude_types=["tool"]) | trimmer)
        | prompt | llm | StrOutputParser())

await chat.ainvoke({"input": "..."}, cfg)
async for tok in chat.astream({"input": "..."}, cfg): ...

prompt = [("system", ...), MessagesPlaceholder("history", optional=True), ("human", "{input}")]
```

```
SSE:  event: token\ndata: {"text": "..."}\n\n   ...   event: done\ndata: {...}\n\n
Debug: tap(prompt) · set_debug(True) · check store length · check /health backend
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| Where does `session_id` go? | `config["configurable"]` |
| What gets saved after a turn? | New human input + AI output (on success) |
| Does the wrapper trim history? | No — it loads everything |
| Where to place the trimmer? | Inside the wrapped chain, on `inputs["history"]` |
| What does `exclude_tool_calls=True` do? | Removes AI tool-call messages and their ToolMessages |
| Is `get_session_history` async-capable? | No — sync factory; async I/O via the history object |
| What if a stream is aborted? | Turn typically not persisted |
| `output_messages_key` needed when…? | The chain returns a dict |
| Multi-key sessions? | `history_factory_config=[ConfigurableFieldSpec(...)]` |
| LangGraph persistence key? | `thread_id` via a checkpointer |
| `n_messages` on `MessagesPlaceholder`? | Caps message count, not tokens |
| Why pre-flight a store read before SSE? | To return a real HTTP 503 instead of a broken stream |

---

## Hands-on Exercises

### Exercise 1: Key-Mismatch Lab (no LLM needed)

Using the X-ray fake model (Part 1):
1. Make it work with `input_messages_key="question"` and placeholder `"chat_log"`.
2. Deliberately break each of the three names one at a time. Record the symptom (error vs. silent) for each.
3. Add a `tap()` and show the empty placeholder in the broken case.

### Exercise 2: Trim Proof

Run 30 turns against the fake model with `trimmer` configured for ~500 approximate tokens. Assert: `len(store) == 60` while the message count reported to the fake model never exceeds a bound you calculate. Then switch `start_on` to `None` and observe a window starting with an AI message.

### Exercise 3: Multi-key Sessions

Implement `user_id` + `conversation_id` with `ConfigurableFieldSpec`. Show that `(alice, trip)` and `(alice, work)` are isolated, and that `(bob, trip)` can't see Alice's `trip`.

### Exercise 4: Filtering Tool Noise

Pre-seed a history with `HumanMessage`, `AIMessage(tool_calls=[...])`, `ToolMessage`, `AIMessage`. Show the view after `filter_messages(exclude_tool_calls=True)` and that the store is unchanged.

### Exercise 5: Disconnect Experiment

Run `app.py`, start `/chat/stream` with `httpx`, break out after 3 tokens, then call `/chat/history`. Record whether the turn was saved and explain why.

---

## Challenge Project

**Finish the FastAPI persistent chatbot.**

Starting from `app.py`:

1. **Two-worker proof:** run with `--workers 2` + Redis; add `os.getpid()` to a response header; show 10 turns alternate PIDs yet keep context.
2. **Fallback behavior:** stop Redis mid-session. Make `/chat` return a clear 503 (not a silent switch to memory). Add an env flag `ALLOW_MEMORY_FALLBACK` (default `false` in prod).
3. **Rate limit:** max 20 messages/minute/session via a Redis counter (`INCR` + `EXPIRE`) or an in-memory token bucket.
4. **Summary hook:** enable `SUMMARY_ENABLED`, lower `MAX_HISTORY_TOKENS` to 300, hold a 15-turn conversation, and show (via `tap`) that the prompt contains *summary + recent window* while `/chat/history` still returns all 30 messages. Replace the naive cache with a stored incremental summary.
5. **Tests:** `pytest` + `TestClient`, with `llm` replaced by a fake runnable, covering: memory across turns, session isolation, invalid headers (400), empty message (422), and SSE event order (`token…`, `done`).
6. **Docs:** README with env var table and a sequence diagram of one turn.

**Acceptance:** all tests pass; two workers share memory; Redis outage returns 503; trimmed prompt provably bounded; full transcript intact.

---

## Homework

1. **Entity profile extraction endpoint (required).** Add `POST /profile/extract`:
   - Define `UserProfile(name: str | None, location: str | None, interests: list[str], preferences: dict[str, str])`.
   - Read the user's conversation via `get_session_history(sid).aget_messages()`.
   - Call `llm.with_structured_output(UserProfile)` over a transcript (`get_buffer_string`).
   - Merge into a Redis hash `profile:{user_id}` (new non-null values win; lists union).
   - Inject the profile into the system prompt as `{profile_block}` on later turns.
   - Verify: tell the bot your name in conversation A, then confirm it knows you in a **new** conversation B.
2. **Reading:** LangChain docs — "How to add message history" and the LangGraph persistence concepts page; list three capabilities the checkpointer adds.
3. **Reflection:** Write 5 bullet points on privacy risks of a persistent profile and how you'd mitigate each (consent, TTL, erase endpoint, redaction, access control).
4. **Stretch:** Port `core` into a one-node LangGraph with a checkpointer and compare lines of code and capabilities.

---

## Additional Resources

- [LangChain — How to add message history](https://python.langchain.com/docs/how_to/message_history/)
- [LangChain — `trim_messages` / message utilities](https://python.langchain.com/docs/how_to/trim_messages/)
- [LangGraph — Persistence concepts](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [FastAPI — Custom Response / StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/)
- [MDN — Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)

---

## What's Next

Your chatbot can now *talk* and *remember*. In **Phase 9 — Tools & Tool Calling**, it learns to *act*: call functions, query APIs, and search the web. Tool calls add new message types (`AIMessage.tool_calls`, `ToolMessage`) to the history you just learned to filter — and they're the first step toward the agent loops and LangGraph persistence previewed here.

> [← Previous: Persistent Memory](chapter-34-persistent-memory.md) | [Next: What Are Tools? →](../phase-09-tools-tool-calling/chapter-36-what-are-tools.md)

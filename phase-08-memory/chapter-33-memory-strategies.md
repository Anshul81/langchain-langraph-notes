# Chapter 8.2: Buffer, Window, Summary Memory Strategies

> **Phase 8 — Memory Systems** | [← Previous: Why Memory?](chapter-32-why-memory.md) | [Next: Persistent Memory →](chapter-34-persistent-memory.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Implement **full buffer** memory with `InMemoryChatMessageHistory` — in a complete chatbot loop
- ✅ Implement **window memory** (last *N* messages) safely, without orphaning AI replies
- ✅ Use `trim_messages` and explain **every key parameter** (`max_tokens`, `token_counter`, `strategy`, `start_on`, `end_on`, `include_system`, `allow_partial`)
- ✅ Build a **summary memory** with a rolling condenser and a threshold
- ✅ Build **entity / profile memory** that extracts facts into a dict injected into the system prompt
- ✅ Combine everything into a **hybrid** memory: summary + last-K verbatim + structured facts
- ✅ Read and explain **legacy** `ConversationBufferMemory`, `ConversationBufferWindowMemory`, `ConversationSummaryMemory` — and migrate them
- ✅ Choose a strategy with a **decision matrix** instead of guessing

| | |
|---|---|
| **Prerequisites** | Chapter 8.1 (why memory), Chapter 5.3 (chat models), Chapter 7.1 (prompt templates), Chapter 7.3 (structured output) |
| **Estimated Reading Time** | 40 minutes |
| **Estimated Coding Time** | 90 minutes |

---

## Introduction

### The Problem

In Chapter 8.1 you learned the load → inject → save loop and the price of **sending everything**:

```
Turn:           1    10    20    50    100
Input tokens:  250  2,050 4,050 10,050 20,050     ← per turn
Cumulative:    250  11.5k  43k   257k   1.0M       ← per conversation
```

At some turn, one of three things happens: the **context window overflows**, the **bill explodes**, or the **model gets distracted** by 40 turns of irrelevant chatter. So you must decide, **on every turn**:

> *Out of everything that was ever said, what does the model get to see right now?*

That decision is called a **memory strategy**.

### The Solution

There are four building blocks and one combination:

| Building block | Idea in one line |
|----------------|------------------|
| **Buffer** | Show everything |
| **Window** | Show the last *N* messages |
| **Token trim** | Show as many recent messages as fit in a token budget |
| **Summary** | Replace old messages with an LLM-written digest |
| **Entity / profile** | Extract durable facts into structured data |
| **Hybrid** | Summary + last *K* verbatim + profile — the production default |

```
        ┌──────────────── FULL STORE (DB / archive) ────────────────┐
        │  every message, forever (or until TTL) — audit + analytics │
        └───────────────────────────┬────────────────────────────────┘
                                    │  memory strategy = a VIEW over the store
                                    ▼
                         ┌─────────────────────┐
                         │    PROMPT WINDOW     │  ← the only thing the LLM sees
                         │ system + profile +   │
                         │ summary + recent msgs│
                         └─────────────────────┘
```

**Critical idea:** a strategy changes *what the prompt contains*, **not** what you store. Store everything (subject to privacy policy); show selectively.

### History

- **2022–2023:** LangChain shipped `ConversationBufferMemory`, `ConversationBufferWindowMemory`, `ConversationSummaryMemory`, `ConversationSummaryBufferMemory`, `ConversationTokenBufferMemory`, `ConversationEntityMemory` — each a class bolted to a chain.
- **2024:** Those classes were deprecated. Their problems — hidden mutable state inside the chain, one memory per chain (not per user), awkward async/streaming — led to explicit `ChatMessageHistory` + `RunnableWithMessageHistory`, plus message utilities like `trim_messages`.
- **2025–2026:** The legacy classes moved out of the main package (into `langchain-classic`), and LangGraph checkpointers became the recommended persistence layer. The *strategies* did not change — only where they live.

### Industry Usage

- **Support bots:** window of last 6–10 messages + ticket facts
- **Tutoring / coaching apps:** rolling summary + learner profile ("struggles with recursion")
- **Coding copilots:** last 10–15 turns verbatim + task state + retrieved files; compaction when near the limit
- **Sales / CRM copilots:** entity memory (customer name, company, budget, objections)
- **ChatGPT-style "Memory" features:** explicit profile facts that persist across conversations

### Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "Windowing deletes old messages" | It only hides them from the **prompt**; your store can keep everything |
| "Summary memory is just truncation" | Summaries *compress* — they keep meaning from old turns at the cost of detail |
| "`trim_messages` counts tokens automatically" | You must supply a `token_counter` (a model, a function, or `len`) |
| "k=3 in `ConversationBufferWindowMemory` keeps 3 messages" | It keeps **3 exchanges** (= 6 messages) |
| "A bigger window is always safer" | Bigger windows cost more, slow down, and add distracting noise |
| "Summaries are accurate" | They are **lossy** and can hallucinate or drop numbers and negations |
| "Legacy memory classes still work like before" | They are deprecated/moved; learn them for interviews, build with modern primitives |
| "One strategy fits all apps" | Pick per use case — and often **combine** them |

---

## Mental Model

### Analogy: Four Ways to Remember a Long Meeting

Imagine a 3-hour meeting and a colleague who joins late, asking, *"Where are we?"*

| Strategy | Real-world equivalent | Strength | Weakness |
|----------|----------------------|----------|----------|
| **Buffer** | Hand over the **full stenographer's transcript** | Nothing lost | 200 pages to read; slow and costly |
| **Window** | "Here are the **last 5 minutes** of notes" | Fast, cheap | Forgets that the budget was agreed an hour ago |
| **Token trim** | Same, but "**as many pages as fit** on your desk" | Respects a hard limit | Still forgets old decisions |
| **Summary** | The **meeting minutes** | Captures the whole arc | Written by a person (LLM) who may miss or distort details |
| **Profile / entity** | The **contact card / CRM record** | Precise, durable facts | Only holds what you decided to extract |
| **Hybrid** | **Minutes + contact card + last 5 minutes of notes** | Best coverage per token | Most moving parts |

A good executive assistant hands over **all three** of the last row. That is what production memory does.

### Visual: How Each Strategy Shapes the Prompt (turn 20, 8 messages kept)

```
STORE (40 messages)  m1 m2 m3 ... m32 m33 m34 m35 m36 m37 m38 m39 m40

BUFFER     [SYS][m1 m2 m3 m4 ................................ m39 m40]     40 msgs
WINDOW(6)  [SYS]                                  [m35 m36 m37 m38 m39 m40]  6 msgs
TRIM(400t) [SYS]                              [m33 ... m40]                (whatever fits)
SUMMARY    [SYS][ "Earlier: user is Priya, planning Japan trip, veg..." ][m37 m38 m39 m40]
HYBRID     [SYS + PROFILE{name,diet,city}][ summary of m1..m34 ][m35 .... m40]
```

### Decision Intuition

```
 How long can a conversation get?
   ├─ Short (< ~15 turns) ────────────────────────────► BUFFER
   ├─ Long, but only recent context matters ──────────► WINDOW or TRIM
   ├─ Long, and early decisions/facts matter ─────────► SUMMARY
   ├─ Long, and specific facts must never be lost ────► + PROFILE (entity memory)
   └─ Long, high stakes, enterprise ──────────────────► HYBRID (all of the above)
```

---

## Theory

> **Shared setup.** Snippets in this chapter reuse the `llm` below. Some blocks repeat it so they can be copy-pasted and run on their own.

### Part 1: Full Buffer Memory — Keep Everything

#### The Building Block

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

history = InMemoryChatMessageHistory()
history.add_user_message("I prefer dark mode UI.")
history.add_ai_message("Got it — I'll suggest dark-friendly themes.")
history.add_user_message("Recommend a Python IDE.")

print(len(history.messages))          # 3
for m in history.messages:
    print(f"{m.type:<6} {m.content[:50]}")
```

```
3
human  I prefer dark mode UI.
ai     Got it — I'll suggest dark-friendly themes.
human  Recommend a Python IDE.
```

| Method / attribute | Purpose |
|--------------------|---------|
| `.messages` | The list of `BaseMessage` objects (read it; don't mutate in place) |
| `.add_user_message(text)` | Append a `HumanMessage` |
| `.add_ai_message(text)` | Append an `AIMessage` |
| `.add_message(msg)` / `.add_messages([...])` | Append arbitrary message objects |
| `.clear()` | Remove all messages |

#### A Complete Buffer Chatbot — Framework-Free Loop

This is the **whole** buffer memory pattern from Chapter 8.1, now with `InMemoryChatMessageHistory` and a real interactive loop.

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

SYSTEM = SystemMessage(content="You are a friendly assistant. Be concise (max 3 sentences).")
history = InMemoryChatMessageHistory()


def buffer_chat(user_text: str) -> str:
    messages = [SYSTEM, *history.messages, HumanMessage(content=user_text)]  # FULL buffer
    ai = llm.invoke(messages)
    history.add_user_message(user_text)       # save both sides
    history.add_message(ai)
    return ai.content


def main() -> None:
    print("Buffer chatbot — type 'quit' to exit, 'show' to inspect memory.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if text.lower() == "show":
            for i, m in enumerate(history.messages):
                print(f"  {i:>2} {m.type:<5} {m.content[:70]}")
            continue
        print("Bot:", buffer_chat(text))


if __name__ == "__main__":
    main()
```

**Sample session:**

```
You: I'm Priya, I prefer dark mode.
Bot: Nice to meet you, Priya! I'll keep dark-friendly options in mind.

You: Recommend a Python IDE.
Bot: VS Code or PyCharm both ship excellent dark themes — VS Code is lighter-weight.

You: What's my name again?
Bot: You're Priya.
```

#### The Same Buffer with `RunnableWithMessageHistory`

For multi-user apps, let LangChain do load/save keyed by `session_id`:

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant. Be concise."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

cfg = {"configurable": {"session_id": "priya-1"}}
print(chatbot.invoke({"input": "Hi, I'm Priya."}, config=cfg))
print(chatbot.invoke({"input": "What's my name?"}, config=cfg))       # "Priya"
print(chatbot.invoke({"input": "What's my name?"}, config={"configurable": {"session_id": "other"}}))
```

The third call uses another `session_id` and gets a **fresh** history — isolation in action.

> **Note:** `RunnableWithMessageHistory` is marked for eventual replacement by LangGraph persistence. The `chain`, prompt, and *strategy* code in this chapter carries over unchanged.

**Best for:** demos, short support chats, anything under ~15–20 turns.
**Fails when:** the context window or budget is exceeded.

---

### Part 2: Window Memory — Last *N* Messages

Keep appending to the store, but only **show** the last `N` messages.

#### Naive Version (and Why It's Subtly Wrong)

```python
def naive_window(messages, n=6):
    return messages[-n:]
```

If `n` is odd (say 5) or the list was edited, the slice may **start with an `AIMessage`**. Some providers reject conversations that don't start with a user turn, and the model sees an answer with no question. Fix it by snapping to a human turn.

#### Safe Version

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage


def window_messages(messages: list[BaseMessage], n: int = 6) -> list[BaseMessage]:
    """Return the last n non-system messages, always starting on a human turn."""
    body = [m for m in messages if m.type != "system"]
    tail = body[-n:] if n > 0 else []
    while tail and tail[0].type != "human":      # drop an orphaned AI reply
        tail = tail[1:]
    return tail


store_demo = InMemoryChatMessageHistory()
for i in range(10):
    store_demo.add_user_message(f"Message {i}")
    store_demo.add_ai_message(f"Reply {i}")

print([m.content for m in window_messages(store_demo.messages, n=4)])
# ['Message 8', 'Reply 8', 'Message 9', 'Reply 9']

print([m.content for m in window_messages(store_demo.messages, n=5)])
# n=5 would begin with 'Reply 7'; the helper drops it →
# ['Message 8', 'Reply 8', 'Message 9', 'Reply 9']
```

#### Windowed Chatbot (Complete)

```python
SYSTEM = SystemMessage(content="You are a friendly assistant. Be concise.")
full_log = InMemoryChatMessageHistory()          # store EVERYTHING
WINDOW = 6                                       # but SHOW only 6 messages


def window_chat(user_text: str) -> str:
    visible = window_messages(full_log.messages, n=WINDOW)
    prompt_messages = [SYSTEM, *visible, HumanMessage(content=user_text)]
    ai = llm.invoke(prompt_messages)
    full_log.add_user_message(user_text)
    full_log.add_message(ai)
    return ai.content


print(window_chat("My name is Priya."))
for i in range(5):
    window_chat(f"Tell me a fun fact about the number {i}.")
print(window_chat("What's my name?"))      # ← likely "I don't know": 'Priya' fell out of the window
print(f"Stored: {len(full_log.messages)} messages, shown: {WINDOW}")
```

**This failure is the lesson:** a plain window forgets *durable* facts (name, allergy) as soon as they scroll out. The fixes are **summary** (Part 4) and **profile memory** (Part 5).

#### Windowing Inside LCEL

Apply the window *inside* the chain so `RunnableWithMessageHistory` still stores everything while the model sees only the tail:

```python
from operator import itemgetter
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

windowed_chain = (
    RunnablePassthrough.assign(
        history=itemgetter("history") | RunnableLambda(lambda msgs: window_messages(msgs, 6))
    )
    | prompt
    | llm
    | StrOutputParser()
)

windowed_bot = RunnableWithMessageHistory(
    windowed_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

| Where to window | Effect |
|-----------------|--------|
| In `get_session_history` (deleting from the store) | **Destructive** — data is gone |
| Inside the chain (before the prompt) | **Non-destructive** — store keeps all, model sees the tail ✅ |

**Best for:** customer support, FAQ bots, quick Q&A where old context rarely matters.

---

### Part 3: Token-Based Trimming — `trim_messages`

Message counts ignore message *size*. One pasted log file can be 10,000 tokens; ten "ok" messages are 40. **Token budgets** are what the model actually enforces.

```python
from langchain_core.messages import trim_messages
```

#### Parameters Explained

| Parameter | Meaning | Typical value |
|-----------|---------|---------------|
| `max_tokens` | Budget for the **returned** messages (as measured by `token_counter`) | `1000`–`4000` for history slice |
| `token_counter` | How to measure: a **chat model** (uses its tokenizer), a **function** `list[BaseMessage] -> int`, or **`len`** (counts *messages*) | custom function / `llm` / `len` |
| `strategy` | `"last"` keeps the **most recent**; `"first"` keeps the **oldest** | `"last"` for chat |
| `include_system` | With `"last"`, keep the leading `SystemMessage` even if it's old | `True` |
| `start_on` | After trimming, the first (non-system) message must be this type — avoids starting on an AI reply | `"human"` |
| `end_on` | The last message must be one of these types | `("human", "tool")` |
| `allow_partial` | If a message doesn't fit, keep part of its content | `False` for chat (cut-off text is confusing) |
| `text_splitter` | Used with `allow_partial=True` to decide how to cut a message | rarely needed |

#### Three Ways to Count

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, trim_messages,
)

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

thread = [SystemMessage(content="You are helpful.")]
for i in range(30):
    thread.append(HumanMessage(content=f"Question {i} " + "word " * 20))
    thread.append(AIMessage(content=f"Answer {i} " + "word " * 20))


# 1) token_counter=len  → "tokens" are MESSAGES. Fast, zero dependencies.
by_count = trim_messages(
    thread, max_tokens=7, token_counter=len,
    strategy="last", include_system=True, start_on="human",
)
print(len(by_count), [m.type for m in by_count])
# 7 ['system', 'human', 'ai', 'human', 'ai', 'human', 'ai']  ← system + 3 exchanges


# 2) Custom approximate counter → offline, deterministic, good enough for budgeting
def approx_tokens(messages) -> int:
    return sum(len(m.content) // 4 + 4 for m in messages)

by_approx = trim_messages(
    thread, max_tokens=400, token_counter=approx_tokens,
    strategy="last", include_system=True, start_on="human",
)
print(len(by_approx), approx_tokens(by_approx))


# 3) Model-based counter → uses the model's tokenizer (accurate, may need tiktoken data)
by_model = trim_messages(
    thread, max_tokens=400, token_counter=llm,
    strategy="last", include_system=True, start_on="human",
)
print(len(by_model))
```

> **Proxy note:** With a LiteLLM proxy, `token_counter=llm` asks LangChain's OpenAI wrapper to count with `tiktoken`. For model names it doesn't recognize it falls back to a default encoding — fine for budgeting, but never treat it as billing-exact. Use `response.usage_metadata` for the real numbers.

#### `start_on` and `end_on` Matter

```python
msgs = [SystemMessage(content="sys")]
for i in range(6):
    msgs += [HumanMessage(content=f"q{i}"), AIMessage(content=f"a{i}")]

with_start = trim_messages(msgs, max_tokens=5, token_counter=len,
                           strategy="last", include_system=True, start_on="human")
print([m.content for m in with_start])
# ['sys', 'q4', 'a4', 'q5', 'a5']        ← clean pairs

no_start = trim_messages(msgs, max_tokens=5, token_counter=len,
                         strategy="last", include_system=True, end_on=("human",))
print([m.content for m in no_start])
# ['sys', 'a3', 'q4', 'a4', 'q5']        ← starts on an orphaned AI reply ✗
```

**Rule:** for chat, almost always set `strategy="last"`, `include_system=True`, `start_on="human"`.

#### The Trimmer as a Runnable

Calling `trim_messages` **without** a `messages` argument returns a reusable `Runnable` you can pipe:

```python
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

trimmer = trim_messages(
    max_tokens=800,                 # budget for HISTORY only (leave room for system + input + reply)
    token_counter=approx_tokens,
    strategy="last",
    include_system=False,           # the system prompt lives in the template, not in history
    start_on="human",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])

chain = (
    RunnablePassthrough.assign(history=itemgetter("history") | trimmer)
    | prompt
    | llm
    | StrOutputParser()
)

# Wire it into per-session memory exactly as before:
trimmed_bot = RunnableWithMessageHistory(
    chain, get_session_history,
    input_messages_key="input", history_messages_key="history",
)
```

#### Budgeting the Whole Prompt

```
context window  =  system  +  history  +  new input  +  reply reserve
   128,000          1,200      ≤ 6,000      ~500           1,000
                                  ▲
                         this is max_tokens for trim_messages
```

Never set `max_tokens` equal to the model's full window — you will starve the reply and the system prompt.

**Best for:** general production chatbots tied to a specific model's limit.

---

### Part 4: Summary Memory — Compress the Past

Windows and trims **throw information away**. Summary memory **compresses** it: when the conversation grows past a threshold, an LLM rewrites the old part into a short digest, and only the digest plus the most recent messages are kept in the prompt.

```
BEFORE (14 messages)                              AFTER condensing (threshold hit)
┌──────────────────────────────┐                  ┌───────────────────────────────────┐
│ m1  m2  m3  m4  m5  m6  m7   │   summarize →    │ SUMMARY: "User Priya plans Japan  │
│ m8  m9  m10 m11 m12          │   (m1..m10)      │ trip in April, vegetarian, ¥ budget│
│ m13 m14                      │                  │ 150k; decided on Tokyo+Kyoto."     │
└──────────────────────────────┘                  │ + m11 m12 m13 m14 (verbatim)       │
                                                  └───────────────────────────────────┘
```

#### A Complete, Working Summary Memory

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)


def transcript_from_messages(messages: list[BaseMessage]) -> str:
    role = {"human": "User", "ai": "Assistant", "system": "System"}
    return "\n".join(f"{role.get(m.type, m.type)}: {m.content}" for m in messages)


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You maintain a running summary of a conversation.\n"
     "Update the EXISTING SUMMARY with the NEW LINES.\n"
     "Rules: keep names, numbers, dates, decisions, constraints and open tasks; "
     "drop greetings and filler; stay under 120 words; never invent facts."),
    ("human",
     "EXISTING SUMMARY:\n{summary}\n\nNEW LINES:\n{new_lines}\n\nUPDATED SUMMARY:"),
])
summarizer = SUMMARY_PROMPT | llm | StrOutputParser()


class SummaryMemory:
    """Rolling summary + recent window. The archive keeps EVERYTHING."""

    def __init__(self, threshold: int = 12, keep_last: int = 4):
        assert keep_last % 2 == 0, "keep an even number so pairs stay intact"
        self.threshold = threshold          # condense when window exceeds this many messages
        self.keep_last = keep_last          # always keep this many raw messages
        self.summary = ""                   # the rolling digest
        self.window = InMemoryChatMessageHistory()
        self.archive: list[BaseMessage] = []          # full audit log (never summarized)
        self.summaries_made = 0

    def add_exchange(self, user_text: str, ai_text: str) -> None:
        for msg in (HumanMessage(content=user_text), AIMessage(content=ai_text)):
            self.window.add_message(msg)
            self.archive.append(msg)
        self._maybe_condense()

    def _maybe_condense(self) -> None:
        msgs = self.window.messages
        if len(msgs) <= self.threshold:
            return
        old, recent = msgs[:-self.keep_last], msgs[-self.keep_last:]
        self.summary = summarizer.invoke({
            "summary": self.summary or "(none yet)",
            "new_lines": transcript_from_messages(old),
        })
        self.summaries_made += 1
        self.window.clear()
        self.window.add_messages(recent)

    def messages_for_prompt(self, base_system: str) -> list[BaseMessage]:
        system = base_system
        if self.summary:
            system += f"\n\n## Summary of earlier conversation\n{self.summary}"
        return [SystemMessage(content=system), *self.window.messages]


# ---------- chat loop ----------
memory = SummaryMemory(threshold=8, keep_last=4)
BASE_SYSTEM = "You are a helpful travel assistant. Be concise."


def summary_chat(user_text: str) -> str:
    messages = memory.messages_for_prompt(BASE_SYSTEM) + [HumanMessage(content=user_text)]
    ai = llm.invoke(messages)
    memory.add_exchange(user_text, ai.content)
    return ai.content


turns = [
    "I'm Priya. I'm planning a 10-day trip to Japan in April.",
    "My budget is 150,000 INR excluding flights.",
    "I'm vegetarian and I dislike crowded places.",
    "Suggest a rough route.",
    "Swap Osaka for Kanazawa.",
    "What about hotels?",
    "How do I get between cities?",
    "Any must-try food?",
    "What was my budget and what city did I swap out?",     # tests long-range recall
]
for t in turns:
    print("You:", t)
    print("Bot:", summary_chat(t), "\n")

print("Summaries made:", memory.summaries_made)
print("Current summary:\n", memory.summary)
print(f"Window: {len(memory.window.messages)} msgs | Archive: {len(memory.archive)} msgs")
```

**Expected behavior:**

- After roughly turn 5 the first condense fires: `summaries_made >= 1` and the window shrinks to 4 raw messages
- The final answer still recalls **150,000 INR** and **Osaka → Kanazawa** — they now live in the **summary**, not the window
- `Archive` keeps all 18 messages while `Window` stays ≤ 8

#### Why the Summary Prompt Is Written This Way

| Prompt design choice | Reason |
|----------------------|--------|
| **Progressive** (existing summary + new lines) | Never re-reads the whole history → bounded cost per condense |
| "keep names, numbers, dates, decisions, constraints" | These are what summaries most often lose |
| "stay under 120 words" | Prevents the summary itself from growing unboundedly |
| "never invent facts" | Reduces summary hallucination |
| Temperature `0` | Stable, repeatable summaries |

#### Threshold Design

```
threshold = 12 messages, keep_last = 4

msgs: 1..12   → no action
msgs: 13      → condense m1..m9 into summary, keep m10..m13
```

Condensing on **every** turn wastes money; condensing **in batches** (when the window is 3× the kept size) amortizes the extra LLM call. Many teams also use a **token** threshold (`approx_tokens(window) > 3000`) instead of a message count.

#### Costs and Failure Modes

| Issue | Detail | Mitigation |
|-------|--------|------------|
| Extra LLM call | Latency + cost on condense turns | Batch; use a cheaper model; run async after replying |
| **Lossy** | Numbers, negations ("do **not**…"), conditions get dropped | Keep facts in a **profile** (Part 5) |
| **Drift** | Summary-of-a-summary degrades over many cycles | Periodically rebuild from the archive; cap cycles |
| **Hallucination** | Summarizer adds facts that weren't said | Strict prompt; temperature 0; let users view/edit |
| PII | Summaries persist sensitive info in a new place | Redact before summarizing |

**Best for:** long coaching sessions, tutoring, B2B copilots, any conversation past ~20 turns.

---

### Part 5: Entity / Profile Memory — Durable Facts

Summaries are fuzzy. But some facts must be **exact and permanent**: the user's name, language, allergy, plan tier. Pull them out into a small **structured profile** and inject it into the system prompt every turn.

```
 user message ─► [extractor LLM → structured JSON] ─► merge into profile dict
                                                          │
 system prompt = base instructions + "Known facts: …" ◄──┘
```

#### Define the Schema (Structured Output)

```python
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)


class ProfileUpdate(BaseModel):
    """Facts the USER explicitly stated about themselves in this message."""
    name: Optional[str] = Field(None, description="User's name, only if stated")
    location: Optional[str] = Field(None, description="City/country where the user lives")
    diet: Optional[str] = Field(None, description="Dietary preference or restriction")
    likes: list[str] = Field(default_factory=list, description="Things the user says they like/prefer")
    constraints: list[str] = Field(default_factory=list, description="Hard limits: budget, allergies, deadlines")


EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Extract ONLY facts the user explicitly states about themselves in the message below. "
     "Do not infer or guess. Leave fields empty/null if nothing is stated."),
    ("human", "{user_text}"),
])

extractor = EXTRACT_PROMPT | llm.with_structured_output(ProfileUpdate)
```

#### Merge and Render

```python
def merge_profile(profile: dict, update: ProfileUpdate) -> dict:
    """Scalars: newest non-empty value wins. Lists: union, order preserved."""
    merged = dict(profile)
    for key, value in update.model_dump(exclude_none=True).items():
        if isinstance(value, list):
            existing = merged.get(key, [])
            merged[key] = existing + [v for v in value if v not in existing]
        elif value not in ("", None):
            merged[key] = value
    return {k: v for k, v in merged.items() if v not in ([], "", None)}


def render_profile(profile: dict) -> str:
    if not profile:
        return "No known facts about the user yet."
    lines = []
    for key, value in profile.items():
        shown = ", ".join(value) if isinstance(value, list) else value
        lines.append(f"- {key}: {shown}")
    return "\n".join(lines)
```

Test the merge logic **without an LLM**:

```python
profile: dict = {}
profile = merge_profile(profile, ProfileUpdate(name="Priya", likes=["trekking"]))
profile = merge_profile(profile, ProfileUpdate(diet="vegetarian", likes=["trekking", "ramen"], location="Pune"))
profile = merge_profile(profile, ProfileUpdate(location="Bengaluru"))     # user moved → overwrite

print(profile)
# {'name': 'Priya', 'likes': ['trekking', 'ramen'], 'location': 'Bengaluru', 'diet': 'vegetarian'}
print(render_profile(profile))
# - name: Priya
# - likes: trekking, ramen
# - location: Bengaluru
# - diet: vegetarian
```

#### Inject Into the System Prompt

```python
from langchain_core.messages import HumanMessage, SystemMessage

BASE = "You are a personal assistant. Use the known facts naturally; never recite them unprompted."


def profile_chat(user_text: str, profile: dict, recent: list) -> tuple[str, dict]:
    # 1) update the profile from what the user just said
    try:
        profile = merge_profile(profile, extractor.invoke({"user_text": user_text}))
    except Exception as exc:                       # extraction must never break the chat
        print(f"[profile extraction skipped: {exc}]")

    # 2) inject the profile into the SYSTEM prompt
    system = f"{BASE}\n\n## Known facts about the user\n{render_profile(profile)}"
    ai = llm.invoke([SystemMessage(content=system), *recent[-4:], HumanMessage(content=user_text)])

    recent += [HumanMessage(content=user_text), ai]
    return ai.content, profile


profile, recent = {}, []
for text in [
    "Hi! I'm Priya and I live in Pune.",
    "I'm vegetarian, so keep that in mind.",
    "Tell me about the weather on Mars.",         # filler — profile unchanged
    "Fun facts about Saturn?",                    # filler — and 'Priya' is now outside recent[-4:]
    "Suggest a dinner idea for me.",
]:
    reply, profile = profile_chat(text, profile, recent)
    print(f"You: {text}\nBot: {reply}\n")

print("Final profile:", profile)
```

**Expected behavior:** the last answer is a **vegetarian** dinner idea, probably addressing Priya or referencing Pune — even though those facts are more than 4 messages old. The profile carried them.

#### Design Rules for Entity Memory

| Rule | Why |
|------|-----|
| Extract from the **user's** message, not the assistant's | The model can hallucinate; users are the source of truth |
| Use a **schema** (Pydantic) | Validated, predictable, mergeable |
| **Overwrite** scalars, **union** lists | Handles "I moved" and "I also like X" |
| Show the profile to the user; allow edit/delete | Trust, GDPR-style rights |
| Never auto-extract secrets (passwords, card numbers) | Don't build a vault of liabilities |
| Wrap extraction in `try/except` | A failed extraction should not fail the chat |
| Don't treat profile text as instructions | Defends against memory poisoning |

**Best for:** assistants with returning users, CRMs, onboarding flows, personalization.

---

### Part 6: Hybrid Memory — Summary + Last K + Profile

This is what most serious chat products converge to:

```
┌──────────────────────── PROMPT SENT TO MODEL ────────────────────────┐
│ SYSTEM  = base instructions                                          │
│         + ## Known facts (PROFILE)       ← exact, durable            │
│         + ## Summary of earlier chat     ← compressed, long-range    │
│ HUMAN/AI … last K messages VERBATIM      ← fresh, detailed           │
│ HUMAN  new user message                                              │
└──────────────────────────────────────────────────────────────────────┘
```

Each layer covers the previous layer's weakness: the profile fixes the summary's lossy facts, the summary fixes the window's amnesia, the window fixes the summary's lack of detail.

> The summary goes **inside the single system message** rather than as a second `SystemMessage`. Some providers (reached through LiteLLM) accept only one system message at the start of the conversation.

#### Complete Hybrid Memory + CLI Chatbot

```python
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)


# ---------- helpers ----------
def transcript_from_messages(messages: list[BaseMessage]) -> str:
    role = {"human": "User", "ai": "Assistant", "system": "System"}
    return "\n".join(f"{role.get(m.type, m.type)}: {m.content}" for m in messages)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class ProfileUpdate(BaseModel):
    """Facts the USER explicitly stated about themselves in this message."""
    name: Optional[str] = Field(None, description="User's name, only if stated")
    location: Optional[str] = Field(None, description="Where the user lives")
    diet: Optional[str] = Field(None, description="Dietary preference or restriction")
    likes: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list, description="Budget, allergies, deadlines")


SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You maintain a running summary of a conversation. Update the EXISTING SUMMARY with "
     "the NEW LINES. Keep names, numbers, dates, decisions, constraints, open tasks. "
     "Drop filler. Under 120 words. Never invent facts."),
    ("human", "EXISTING SUMMARY:\n{summary}\n\nNEW LINES:\n{new_lines}\n\nUPDATED SUMMARY:"),
])
EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extract ONLY facts the user explicitly states about themselves. Do not infer."),
    ("human", "{user_text}"),
])


class HybridMemory:
    def __init__(self, llm, base_system: str, keep_last: int = 6, condense_at: int = 14):
        assert keep_last % 2 == 0
        self.base_system = base_system
        self.keep_last, self.condense_at = keep_last, condense_at
        self.summary = ""
        self.profile: dict = {}
        self.recent = InMemoryChatMessageHistory()
        self.archive: list[BaseMessage] = []
        self._summarizer = SUMMARY_PROMPT | llm | StrOutputParser()
        self._extractor = EXTRACT_PROMPT | llm.with_structured_output(ProfileUpdate)

    # ----- profile -----
    def _update_profile(self, user_text: str) -> None:
        try:
            update = self._extractor.invoke({"user_text": user_text})
        except Exception:
            return                                    # never break the chat for memory
        for key, value in update.model_dump(exclude_none=True).items():
            if isinstance(value, list):
                have = self.profile.get(key, [])
                self.profile[key] = have + [v for v in value if v not in have]
            elif value:
                self.profile[key] = value
        self.profile = {k: v for k, v in self.profile.items() if v not in ([], "", None)}

    def _profile_text(self) -> str:
        if not self.profile:
            return "(none yet)"
        return "\n".join(
            f"- {k}: {', '.join(v) if isinstance(v, list) else v}" for k, v in self.profile.items()
        )

    # ----- summary -----
    def _maybe_condense(self) -> None:
        msgs = self.recent.messages
        if len(msgs) <= self.condense_at:
            return
        old, keep = msgs[:-self.keep_last], msgs[-self.keep_last:]
        self.summary = self._summarizer.invoke({
            "summary": self.summary or "(none yet)",
            "new_lines": transcript_from_messages(old),
        })
        self.recent.clear()
        self.recent.add_messages(keep)

    # ----- public API -----
    def build_messages(self, user_text: str) -> list[BaseMessage]:
        system = (
            f"{self.base_system}\n\n"
            f"## Known facts about the user\n{self._profile_text()}\n\n"
            f"## Summary of earlier conversation\n{self.summary or '(conversation just started)'}"
        )
        return [SystemMessage(content=system), *self.recent.messages, HumanMessage(content=user_text)]

    def remember(self, user_text: str, ai_text: str) -> None:
        self._update_profile(user_text)
        for msg in (HumanMessage(content=user_text), AIMessage(content=ai_text)):
            self.recent.add_message(msg)
            self.archive.append(msg)
        self._maybe_condense()

    def prompt_tokens(self) -> int:
        return sum(approx_tokens(m.content) + 4 for m in self.build_messages(""))


# ---------- CLI ----------
def main() -> None:
    mem = HybridMemory(llm, base_system="You are a helpful, concise personal assistant.",
                       keep_last=4, condense_at=10)
    print("Hybrid memory bot. Commands: /profile /summary /tokens /quit")
    while True:
        text = input("\nYou: ").strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/profile":
            print(mem.profile or "(empty)")
            continue
        if text == "/summary":
            print(mem.summary or "(no summary yet)")
            continue
        if text == "/tokens":
            print(f"~{mem.prompt_tokens()} prompt tokens | recent={len(mem.recent.messages)} "
                  f"archive={len(mem.archive)}")
            continue
        ai = llm.invoke(mem.build_messages(text))
        mem.remember(text, ai.content)
        print("Bot:", ai.content)


if __name__ == "__main__":
    main()
```

**Sample session:**

```
You: Hi, I'm Priya from Pune. I'm vegetarian.
Bot: Nice to meet you, Priya! How can I help?
You: /profile
{'name': 'Priya', 'location': 'Pune', 'diet': 'vegetarian'}
... (many turns later) ...
You: /summary
User Priya (Pune, vegetarian) is planning a 10-day Japan trip in April, budget 150k INR; chose Tokyo+Kyoto+Kanazawa.
You: /tokens
~640 prompt tokens | recent=4 archive=36
```

**Cost note:** this design makes **two extra LLM calls** in the worst case (profile extraction every turn, summary on condense turns). Reduce it with a cheaper model for utilities, extracting only when the message contains first-person statements ("I", "my"), and running memory updates **after** replying (or in a background task) so users don't wait.

---

### Part 7: Legacy Memory Classes — For Interviews and Old Codebases

You will meet these in older tutorials, StackOverflow answers, and **interviews**. Understand what they did and how they map to today's building blocks.

#### What They Looked Like

```python
# ────────────────────────────────────────────────────────────────────────────
# LEGACY API — shown for reading old code. NOT part of the modern LCEL stack.
# Install: pip install "langchain<1.0"   (or: pip install langchain-classic
#          and import from `langchain_classic.memory` / `langchain_classic.chains`)
# ────────────────────────────────────────────────────────────────────────────
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
)

# 1) BUFFER — stores every exchange, injects all of it
buffer = ConversationBufferMemory(return_messages=True)        # memory_key="history" by default

# 2) WINDOW — keeps the last k EXCHANGES (so k=2 → 4 messages)
window = ConversationBufferWindowMemory(k=2, return_messages=True)

# 3) SUMMARY — after each exchange an LLM updates a running summary
summary = ConversationSummaryMemory(llm=llm)                   # needs an LLM

# The memory was attached to a chain, which called it behind the scenes:
chain = ConversationChain(llm=llm, memory=buffer)
chain.predict(input="Hi, I'm Priya")
chain.predict(input="What's my name?")

# Under the hood, the chain did these two calls every turn:
buffer.load_memory_variables({})                               # → {"history": [HumanMessage, AIMessage, ...]}
buffer.save_context({"input": "Hi"}, {"output": "Hello!"})     # append the new exchange
```

| Legacy class | Behavior | Modern equivalent |
|--------------|----------|-------------------|
| `ConversationBufferMemory` | Keep all messages | `InMemoryChatMessageHistory` (Part 1) |
| `ConversationBufferWindowMemory(k)` | Last **k exchanges** (2k messages) | `window_messages(msgs, 2*k)` or `trim_messages(..., token_counter=len)` (Parts 2–3) |
| `ConversationTokenBufferMemory(max_token_limit)` | Last messages within token limit | `trim_messages(max_tokens=..., token_counter=llm)` |
| `ConversationSummaryMemory` | LLM updates running summary **every exchange** | `SummaryMemory` (Part 4) |
| `ConversationSummaryBufferMemory(max_token_limit)` | Recent verbatim up to limit; older → summary | `HybridMemory` without profile (Part 6) |
| `ConversationEntityMemory` | LLM extracts entities → dict of descriptions | Profile memory (Part 5) |
| `VectorStoreRetrieverMemory` | Retrieve relevant past turns by similarity | Retriever tool / RAG over history (Phase 9) |

#### A Runnable Re-Creation (So You Can *See* What They Did)

You don't need the deprecated package to understand the interface. This **self-contained** version reproduces the legacy contract — `save_context` and `load_memory_variables` — on top of modern primitives:

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage


class LegacyStyleBufferMemory:
    """Mimics ConversationBufferMemory's interface (educational)."""

    def __init__(self, memory_key: str = "history", return_messages: bool = False):
        self.chat_memory = InMemoryChatMessageHistory()
        self.memory_key = memory_key
        self.return_messages = return_messages

    def _visible(self) -> list[BaseMessage]:
        return self.chat_memory.messages                     # buffer: everything

    def save_context(self, inputs: dict, outputs: dict) -> None:
        self.chat_memory.add_user_message(inputs["input"])
        self.chat_memory.add_ai_message(outputs["output"])

    def load_memory_variables(self, _inputs: dict) -> dict:
        msgs = self._visible()
        if self.return_messages:
            return {self.memory_key: msgs}                   # list[BaseMessage] for chat prompts
        text = "\n".join(f"{'Human' if m.type == 'human' else 'AI'}: {m.content}" for m in msgs)
        return {self.memory_key: text}                       # one string for completion-style prompts


class LegacyStyleWindowMemory(LegacyStyleBufferMemory):
    """Mimics ConversationBufferWindowMemory: k = EXCHANGES, not messages."""

    def __init__(self, k: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def _visible(self) -> list[BaseMessage]:
        return self.chat_memory.messages[-2 * self.k:] if self.k > 0 else []


mem = LegacyStyleWindowMemory(k=1)
for i in range(3):
    mem.save_context({"input": f"question {i}"}, {"output": f"answer {i}"})

print(mem.load_memory_variables({}))
# {'history': 'Human: question 2\nAI: answer 2'}      ← only the last EXCHANGE (k=1 → 2 messages)
print(len(mem.chat_memory.messages))
# 6                                                    ← everything is still stored
```

#### Why LangChain Moved Away From Them

| Legacy design problem | Consequence | Modern fix |
|-----------------------|-------------|------------|
| Memory object **lives inside the chain** | One chain = one conversation; sharing across users **leaks** history | History fetched **per `session_id`** via `get_session_history` |
| Hidden I/O contract (`input`/`output` keys, `memory_key`) | Silent mismatches, hard debugging | Explicit `MessagesPlaceholder` and `input_messages_key` |
| Tied to `Chain` classes | Not composable with LCEL; weak streaming/async | Works with any `Runnable` |
| Summary updated **every** turn | Extra LLM call per message | You choose when (threshold-based) |
| `k` means exchanges, not messages | Frequent off-by-2 bugs | Explicit message/token budgets |
| In-process only | Lost on restart; not multi-worker | Pluggable stores (Chapter 8.3) |

#### Migration Cheat Sheet

```python
# ── BEFORE (legacy) ─────────────────────────────────────────────
memory = ConversationBufferWindowMemory(k=3, return_messages=True)
chain = ConversationChain(llm=llm, memory=memory)
chain.predict(input="Hi")

# ── AFTER (modern) ──────────────────────────────────────────────
store = {}
def get_session_history(session_id):
    return store.setdefault(session_id, InMemoryChatMessageHistory())

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
core = (
    RunnablePassthrough.assign(history=itemgetter("history") | RunnableLambda(lambda m: window_messages(m, 6)))  # k=3 → 6 msgs
    | prompt | llm | StrOutputParser()
)
bot = RunnableWithMessageHistory(core, get_session_history,
                                 input_messages_key="input", history_messages_key="history")
bot.invoke({"input": "Hi"}, config={"configurable": {"session_id": "u1"}})
```

**Interview framing:** *"Legacy memory bundled storage, selection policy, and prompt formatting into one object attached to a chain. Modern LangChain separates them: a **history store** (per session), a **selection policy** (window/trim/summary written as a runnable step), and a **prompt slot** (`MessagesPlaceholder`)."*

---

### Part 8: Strategy Decision Matrix

| Strategy | Recall of old facts | Token use | Extra LLM calls | Complexity | Best for | Main risk |
|----------|--------------------|-----------|-----------------|------------|----------|-----------|
| **Full buffer** | Perfect (until limit) | Grows ~N² cumulative | 0 | Low | Short chats, demos | Overflow, cost |
| **Window (N msgs)** | Recent only | Bounded | 0 | Low | FAQ/support bots | Forgets name/constraints |
| **Token trim** | Recent, budget-aware | Bounded exactly | 0 | Medium | General production chat | Same as window |
| **Summary** | Long-range, lossy | Low, bounded | 1 per condense | Medium–High | Long coaching/tutoring | Drift, dropped details |
| **Entity / profile** | Exact durable facts | Tiny | 1 per extraction | Medium | Returning users, CRM | Wrong/poisoned facts |
| **Hybrid** | Best overall | Controlled | Both | Highest | Enterprise assistants | Complexity, cost |

#### A Selector You Can Reason About

```python
def choose_strategy(expected_turns: int, early_facts_matter: bool,
                    exact_facts_required: bool, latency_sensitive: bool) -> str:
    if expected_turns <= 15:
        return "buffer"
    if exact_facts_required and early_facts_matter:
        return "hybrid (summary + last K + profile)"
    if exact_facts_required:
        return "window/trim + profile"
    if early_facts_matter:
        return "summary + last K" if not latency_sensitive else "trim + profile (avoid summary latency)"
    return "token trim"


print(choose_strategy(8, False, False, True))     # buffer
print(choose_strategy(60, True, True, False))     # hybrid (summary + last K + profile)
print(choose_strategy(100, False, True, True))    # window/trim + profile
```

---

## Execution Walkthrough

Trace one turn of **hybrid memory** (`keep_last=4`, `condense_at=10`), when the window already holds 10 messages:

```
User sends: "Swap Osaka for Kanazawa."

Step 1  build_messages()
        SYSTEM = base + "Known facts: name=Priya, diet=vegetarian" + "Summary: …Tokyo+Osaka plan…"
        + recent[10 messages] + Human("Swap Osaka for Kanazawa.")
Step 2  llm.invoke(...)  → AI: "Done — Kanazawa replaces Osaka."
Step 3  remember(user_text, ai_text)
        3a  _update_profile  → extractor sees ONLY the user line → no new personal facts
        3b  recent.add(Human, AI)  → recent has 12 messages;  archive has 12+ messages
        3c  _maybe_condense → 12 > 10 ✓
              old  = recent[:-4]  (8 msgs)      keep = recent[-4:]  (4 msgs)
              summary = summarizer("EXISTING SUMMARY + NEW LINES(old)") → updated digest
              recent  = keep (4 msgs)
Step 4  Next turn prompt ≈ system(+profile+summary) + 4 recent + new human  → small & stable
```

---

## Common Mistakes

### Mistake 1: Deleting from the Store Instead of Windowing the View

```python
# ❌ Destroys data you may need for audit, summaries, or user export
history.messages = history.messages[-6:]

# ✅ Keep the store intact; slice only what you send
prompt_messages = window_messages(history.messages, 6)
```

### Mistake 2: Window That Starts on an AI Message

```python
# ❌ Odd n can split a pair
history.messages[-5:]       # begins with an AIMessage

# ✅ Snap to a human turn
window_messages(history.messages, 5)
trim_messages(..., start_on="human")
```

### Mistake 3: Forgetting the `token_counter`

```python
# ❌ TypeError: trim_messages() missing 1 required keyword-only argument: 'token_counter'
trim_messages(msgs, max_tokens=500, strategy="last")

# ✅ Provide one
trim_messages(msgs, max_tokens=500, strategy="last", token_counter=approx_tokens)
```

### Mistake 4: Trimming Without Reserving Room for the Reply and System Prompt

```python
# ❌ max_tokens = the model's whole context window
trim_messages(msgs, max_tokens=128_000, ...)

# ✅ Subtract system prompt, new input, and reply reserve first
HISTORY_BUDGET = 128_000 - 1_500 - 1_000 - 2_000
```

### Mistake 5: Summarizing Without Keeping Recent Verbatim Turns

```python
# ❌ Summary only — the model can't see the exact question it was just asked
messages = [SystemMessage(summary), HumanMessage(user_text)]

# ✅ Summary + last K raw messages
messages = memory.messages_for_prompt(BASE) + [HumanMessage(user_text)]
```

### Mistake 6: Trusting the Summary for Exact Facts

```python
# ❌ "Budget: around 150k" — summary rounded, or dropped "excluding flights"
# ✅ Put exact numbers in the profile/constraints; summaries carry narrative only
```

### Mistake 7: One Policy for Every User and Conversation

A 4-turn FAQ chat and a 200-turn coaching relationship need different strategies. Choose by **conversation type**, or switch dynamically when tokens cross a threshold.

### Mistake 8: Using One Global Memory Object for All Users

```python
# ❌ HybridMemory instance created at import time and shared
memory = HybridMemory(...)

# ✅ One memory per session_id — fetched from a dict/DB
memories: dict[str, HybridMemory] = {}
def get_memory(session_id): return memories.setdefault(session_id, HybridMemory(llm, BASE))
```

This is the same flaw that doomed the legacy memory-inside-chain design.

---

## Debugging Guide

### Symptom: Bot forgets the user's name after ~6 turns

You are using a **window** or **trim**. Add profile memory, or a summary. Print the exact prompt (`[m.content for m in build_messages(...)]`) to confirm the name is absent.

### Symptom: Bot invents details about earlier turns

Likely **summary hallucination**. Inspect `memory.summary`. Tighten the prompt ("never invent facts"), lower temperature, or rebuild the summary from the archive.

### Symptom: `trim_messages` returns an empty list

`max_tokens` is smaller than even one message (or the last message alone exceeds it). Raise the budget, or use `allow_partial=True`. Print `token_counter([msg])` for the newest message.

### Symptom: Provider error "first message must be a user message" or "roles must alternate"

Your window started on an AI message, or you inserted a second system message. Use `start_on="human"` and merge the summary into one system message.

### Symptom: Token counts from `trim_messages` disagree with the provider bill

Local counters are **approximations** (different tokenizer, formatting overhead). Compare to `ai.usage_metadata["input_tokens"]` and leave a 10–20% safety margin.

### Symptom: Latency spikes every ~10 turns

That's the **condense** turn running an extra LLM call synchronously. Move summarization to a background task or run it after the reply is returned.

### Symptom: Profile contains wrong facts ("likes: spiders")

Extractor is seeing assistant text, or inferring. Extract from **user messages only**, use `temperature=0`, and show users their profile to correct it.

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Store the **full log**; shrink only the **prompt view** | Compliance, analytics, re-summarization |
| Always `strategy="last"`, `include_system=True`, `start_on="human"` for chat trims | Valid, coherent turn structure |
| Reserve tokens for system + reply when setting `max_tokens` | Avoid truncated answers |
| Summarize in **batches** with a threshold, not every turn | Controls cost and latency |
| Keep **last K verbatim** alongside any summary | Preserves fresh detail |
| Put **exact facts** in a structured profile | Summaries are lossy |
| Extract profile facts from **user** messages only | Reduces hallucinated memory |
| Put summary/profile **inside one system message** | Provider compatibility |
| One memory object **per session** | Isolation; no cross-user leakage |
| Wrap memory updates in `try/except` | Memory failures must not break chat |
| Let users **view, edit, and delete** profile/summary | Trust and privacy rights |
| Test with the **longest realistic** conversations | Catch overflow and drift before production |
| Log `usage_metadata` per turn | Verify your savings are real |
| Monitor "you forgot X" complaints | Signals a too-aggressive strategy |

---

## Interview Preparation

### Easy

**Q: What is window memory?**

> Only the last **N messages** (or exchanges) are placed in the prompt. Older turns are omitted from the prompt but can remain in permanent storage. It's cheap and bounded, but forgets older facts.

**Q: What is the difference between a buffer and a summary memory?**

> A buffer sends the **verbatim** history; a summary replaces older history with an LLM-written **digest**. Buffer is lossless but grows without bound; summary is bounded but lossy.

---

### Medium

**Q: Compare token trimming vs a fixed message window.**

> A message window counts messages and ignores size — one huge paste can blow the context. `trim_messages` counts **tokens** with a `token_counter`, so the history always fits a budget aligned to the model's limit. Set `strategy="last"`, `include_system=True`, and `start_on="human"` so the result is a valid conversation.

**Q: In legacy `ConversationBufferWindowMemory(k=3)`, how many messages are kept?**

> **Six** — `k` counts *exchanges* (a human message plus an AI reply), not individual messages.

**Q: How would you migrate `ConversationBufferMemory` to modern LangChain?**

> Replace the memory object with an `InMemoryChatMessageHistory` returned from a `get_session_history(session_id)` function; add `MessagesPlaceholder("history")` to the prompt; wrap the chain with `RunnableWithMessageHistory`, setting `input_messages_key` and `history_messages_key`. Pass a `session_id` through `config["configurable"]`.

---

### Hard

**Q: How does summary memory fail, and how do you mitigate it?**

> Summaries **drop nuance**: numbers, negations, conditions, and who-said-what; repeated summarization causes **drift**, and the summarizer may **hallucinate**. Mitigations: keep the last K messages verbatim, store exact facts in a **structured profile** extracted with a schema, use a strict progressive-summary prompt at temperature 0, periodically rebuild the summary from the archive, and let users inspect/edit it.

**Q: Why might `trim_messages` with `strategy="last"` still produce an invalid conversation, and how do you prevent it?**

> It can cut in the middle of a pair, leaving the history starting with an `AIMessage` (or a dangling `ToolMessage` without its call). Providers may reject that, or the model sees an answer without a question. Use `start_on="human"`, `end_on` where appropriate, and `include_system=True` to preserve instructions. For tool-calling agents, ensure tool-call/tool-result pairs are trimmed together.

**Q: Where should the summary and profile be injected, and why?**

> Into the **single system message** (or a clearly delimited block), above the verbatim turns. That keeps instructions + durable context together, works across providers that allow one system message, and prevents the model from treating summary text as something the user said. Always treat it as **data, not instructions**.

---

### Senior / System Design

**Q: Design memory for a coding assistant with 50-turn debugging sessions.**

> **Hybrid:** (1) structured **task state** — current file, failing test, stack trace, git branch — kept outside chat text; (2) a **rolling summary** of decisions and dead ends; (3) the **last 10–15 turns verbatim**; (4) **retrieval** (RAG) for relevant code/docs instead of stuffing logs; (5) a **token monitor** that triggers compaction at ~70% of the window; (6) for multi-step tool loops use **LangGraph checkpointing**, because message-only memory can't capture agent state. Keep the full log in a DB for audit and re-summarization.

**Q: You're running a support product with 1M conversations/day. How do you control the cost of memory?**

> Measure first: per-turn `usage_metadata`. Then: window/trim for short-lived chats; summarization only for sessions crossing a token threshold, using a **cheaper model** and **batched** condensing; extract profile facts only when the user message contains first-person statements; rely on **prompt caching** by keeping history append-only and the system prefix stable (note: rewriting the summary invalidates part of the cached prefix, so condense infrequently); set TTLs and archive cold logs; and A/B test strategies against resolution rate and "forgot context" complaints, not just token spend.

**Q: A user says "forget what I told you about my health." What must your memory system support?**

> **Deletion across every layer**: the raw archive, the recent window, the **summary** (which may embed the fact — regenerate it from the redacted archive), the **profile**, and any derived caches or vector indexes. Hence: design with a single deletion path, store provenance (which messages produced which profile field), and verify with tests that no derived artifact still contains the fact. Log the deletion for audit without storing the deleted content.

---

## Summary

| Strategy | Mechanism | Key API / Code |
|----------|-----------|----------------|
| Buffer | All messages in prompt | `InMemoryChatMessageHistory`, `RunnableWithMessageHistory` |
| Window | Last N messages | `window_messages(msgs, n)` (start on human) |
| Token trim | Budgeted by tokens | `trim_messages(max_tokens, token_counter, strategy="last", start_on="human")` |
| Summary | LLM condenses old turns | `SummaryMemory` (threshold, keep_last, progressive prompt) |
| Entity / profile | Structured facts → system prompt | `llm.with_structured_output(ProfileUpdate)` + `merge_profile` |
| Hybrid | Profile + summary + last K | `HybridMemory.build_messages / remember` |
| Legacy | Memory attached to chain | `Conversation*Memory` → migrate to history + placeholder |

---

## Cheat Sheet

```python
# BUFFER
history = InMemoryChatMessageHistory()
history.add_user_message("..."); history.add_ai_message("...")
history.messages                                    # list[BaseMessage]

# WINDOW (non-destructive)
tail = history.messages[-6:]                         # then snap to a human turn

# TOKEN TRIM
trim_messages(
    msgs,
    max_tokens=800,                  # budget for the history slice
    token_counter=approx_tokens,     # fn | chat model | len
    strategy="last",                 # keep newest
    include_system=True,             # keep system message
    start_on="human",                # valid first turn
    allow_partial=False,
)
trimmer = trim_messages(max_tokens=800, token_counter=..., strategy="last", start_on="human")  # as Runnable

# MODERN WIRING
chain = RunnablePassthrough.assign(history=itemgetter("history") | trimmer) | prompt | llm | StrOutputParser()
bot = RunnableWithMessageHistory(chain, get_session_history,
        input_messages_key="input", history_messages_key="history")
bot.invoke({"input": "hi"}, config={"configurable": {"session_id": "u1"}})

# SUMMARY (progressive)
summary = SUMMARY_PROMPT.pipe(llm).pipe(StrOutputParser()).invoke(
    {"summary": old_summary, "new_lines": transcript_from_messages(old_msgs)})

# PROFILE
extractor = EXTRACT_PROMPT | llm.with_structured_output(ProfileUpdate)
profile = merge_profile(profile, extractor.invoke({"user_text": text}))
system = f"{BASE}\n## Facts\n{render_profile(profile)}"

# HYBRID PROMPT
[SystemMessage(base + profile + summary), *last_K_messages, HumanMessage(new_input)]

# LEGACY → MODERN
ConversationBufferMemory          → InMemoryChatMessageHistory
ConversationBufferWindowMemory(k) → last 2*k messages
ConversationSummaryMemory         → rolling summary step
ConversationEntityMemory          → structured profile
ConversationChain(memory=...)     → RunnableWithMessageHistory(chain, get_session_history, ...)
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| What does a memory strategy decide? | What subset/transformation of history the model sees each turn |
| Does windowing delete stored messages? | No — it only changes the prompt view (if done correctly) |
| What does `trim_messages(strategy="last")` keep? | The most recent messages that fit `max_tokens` |
| What are valid `token_counter` values? | A chat model, a function `list[BaseMessage] -> int`, or `len` |
| Why set `start_on="human"`? | Avoid a history that begins with an orphaned AI reply |
| What does `include_system=True` do? | Preserves the leading `SystemMessage` when trimming from the end |
| What does `allow_partial` do? | Keeps part of a message that doesn't fit entirely |
| `ConversationBufferWindowMemory(k=2)` keeps how many messages? | 4 (2 exchanges) |
| What is a "progressive" summary? | Update the existing summary with only the new lines |
| Biggest weakness of summary memory? | Lossy — drops numbers/negations; can hallucinate or drift |
| What fixes summary's lossy facts? | A structured profile (entity memory) |
| Where should summary and profile be injected? | Inside the single system message |
| Which memory strategy is the production default? | Hybrid: profile + summary + last K verbatim |
| Why was legacy memory deprecated? | Memory tied to chain, not per-session; not LCEL-native; hidden I/O contract |
| How do you keep multi-user isolation? | One history/memory per `session_id` |
| How do you verify real token usage? | `ai_msg.usage_metadata["input_tokens"]` |

---

## Hands-on Exercises

### Exercise 1: Simulate 20 Turns — Buffer vs Window vs Trim vs Summary

Run this **offline** (no LLM needed) to see how each strategy shapes the prompt size. The summary column uses a deterministic fake condenser so you can focus on the **shape** of the curve.

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage, trim_messages

SYSTEM = SystemMessage(content="You are a helpful assistant. " * 8)   # ~60 tokens


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def count_tokens(messages: list[BaseMessage]) -> int:
    """~4 chars per token + 4 tokens of role overhead per message."""
    return sum(approx_tokens(m.content) + 4 for m in messages)


def view_buffer(msgs):
    return [SYSTEM, *msgs]


def view_window(msgs, n=6):
    tail = msgs[-n:]
    while tail and tail[0].type != "human":
        tail = tail[1:]
    return [SYSTEM, *tail]


def view_trim(msgs, budget=400):
    return trim_messages(
        [SYSTEM, *msgs], max_tokens=budget, token_counter=count_tokens,
        strategy="last", include_system=True, start_on="human",
    )


class SimSummary:
    """Fake condenser: shows the SHAPE of summary memory without an LLM."""

    def __init__(self, threshold=12, keep=4, summary_chars=300):
        self.summary, self.recent = "", []
        self.threshold, self.keep, self.summary_chars = threshold, keep, summary_chars

    def add(self, *msgs):
        self.recent.extend(msgs)
        if len(self.recent) > self.threshold:
            old, self.recent = self.recent[:-self.keep], self.recent[-self.keep:]
            digest = "; ".join(m.content[:20] for m in old)
            # A real LLM re-condenses, so its summary stays bounded.
            self.summary = (self.summary + " | " + digest)[-self.summary_chars:]

    def view(self):
        head = [SYSTEM]
        if self.summary:
            head.append(SystemMessage(content=f"Summary so far: {self.summary}"))
        return head + self.recent


full, summ = InMemoryChatMessageHistory(), SimSummary()
print(f"{'turn':>4} | {'buffer':>6} | {'window':>6} | {'trim':>5} | {'summary':>7}")
print("-" * 42)
for turn in range(1, 21):
    full.add_user_message(f"Turn {turn} question: " + "tell me more about topic " * 4)
    full.add_ai_message(f"Turn {turn} answer: " + "here is a detailed explanation " * 10)
    summ.add(*full.messages[-2:])
    msgs = full.messages
    print(f"{turn:>4} | {count_tokens(view_buffer(msgs)):>6} | {count_tokens(view_window(msgs)):>6} "
          f"| {count_tokens(view_trim(msgs)):>5} | {count_tokens(summ.view()):>7}")
```

**Expected output (exact, since the counter is deterministic):**

```
turn | buffer | window |  trim | summary
------------------------------------------
   1 |    180 |    180 |   180 |     180
   2 |    298 |    298 |   298 |     298
   3 |    416 |    416 |   298 |     416
   4 |    534 |    416 |   298 |     534
   5 |    652 |    416 |   298 |     652
   6 |    770 |    416 |   298 |     770
   7 |    888 |    416 |   298 |     361
   8 |   1006 |    416 |   298 |     479
   9 |   1124 |    416 |   298 |     597
  10 |   1242 |    416 |   298 |     715
  11 |   1360 |    416 |   298 |     833
  12 |   1478 |    416 |   298 |     381
  13 |   1596 |    416 |   298 |     499
  14 |   1714 |    416 |   298 |     617
  15 |   1832 |    416 |   298 |     735
  16 |   1950 |    416 |   298 |     853
  17 |   2068 |    416 |   298 |     381
  18 |   2186 |    416 |   298 |     499
  19 |   2304 |    416 |   298 |     617
  20 |   2422 |    416 |   298 |     735
```

**What to observe:**

| Column | Shape | Why |
|--------|-------|-----|
| buffer | Straight line up (+118/turn) | Everything is resent |
| window | Plateaus at 416 from turn 3 | Last 6 messages = 3 exchanges |
| trim | Plateaus at **298** (lower than window) | Budget 400: system + 2 exchanges fit, a 3rd would reach 416 > 400 |
| summary | **Sawtooth** between ~380 and ~850 | Grows until the threshold, then collapses into a bounded digest |

**Extension tasks:**
1. Change `budget` to 1000 — at which turn does `trim` plateau?
2. Change the window to 10 messages — where does the plateau move?
3. Add a fifth column for **hybrid** (profile of ~40 tokens + summary + last 4 messages). Is it larger or smaller than summary alone, and what do you gain for the difference?

### Exercise 2: Entity Memory Injected Into the System Prompt

Using the code from Part 5:

1. Run the 5-message conversation and print the final `profile`.
2. Add a user message `"Actually, I moved to Bengaluru."` and verify `location` is **overwritten**, not duplicated.
3. Add `"I'm allergic to peanuts"` and verify it lands in `constraints`.
4. Print the **exact system prompt** sent on the last turn.

**Expected behavior:**
- `profile["name"] == "Priya"`, `profile["location"] == "Bengaluru"`, and `constraints` contains a peanut-allergy entry
- The system prompt contains a `## Known facts about the user` block with those values
- A question like "What can't I eat?" is answered correctly even when you set `recent[-4:]` → `recent[-2:]` — the profile carries it

### Exercise 3: Prove the Window Forgets — Then Fix It

1. With `window_chat` from Part 2 (`WINDOW=6`), tell the bot your name, then send 6 unrelated messages, then ask your name.
2. Record the failure.
3. Fix it **three ways** and compare: (a) bigger window, (b) profile memory, (c) summary memory.

**Expected behavior:** (a) works until you add more filler; (b) always works for the name; (c) works but may phrase it as "the user (Priya)…" — list the pros and cons of each in 3 lines.

### Exercise 4: Legacy ↔ Modern Equivalence Test

Using `LegacyStyleWindowMemory(k=2)` and `window_messages(history.messages, 4)`, feed both the same 10 exchanges and `assert` they expose identical message content.

**Expected behavior:** the assertion passes; then change `k=2` to `window_messages(..., 2)` and observe the **off-by-2** bug that interviewers love to ask about.

---

## Challenge Project

### Hybrid Memory CLI Chatbot — Production-Style

Extend the `HybridMemory` CLI from Part 6 into a polished tool.

**Requirements:**

1. **Per-session memory:** `python bot.py --session priya` loads/creates `sessions/priya.json` containing `summary`, `profile`, `recent`, and `archive`
2. **Persistence:** save after every turn; reload on start (messages serialized with `langchain_core.messages.messages_to_dict` / `messages_from_dict`)
3. **Token-triggered condensing:** condense when `approx_tokens(recent) > 1500`, not by message count
4. **Commands:** `/profile`, `/summary`, `/tokens`, `/forget <field>` (remove one profile field), `/rebuild` (regenerate the summary from the full archive), `/export`, `/quit`
5. **Safety:** redact emails/phones with the `redact()` function from Chapter 8.1 before saving to the archive
6. **Metrics:** after each turn print `prompt_tokens (estimated)` vs `usage_metadata["input_tokens"]` (actual)
7. **Failure tolerance:** if the extractor or summarizer raises, the chat continues and logs a warning
8. **Tests (pytest, with a stubbed LLM):** condense triggers at the threshold; `/forget` removes a field; reload restores identical state; two sessions never share memory

**Acceptance demo:** hold a 40-turn conversation; at the end the bot must correctly answer (a) your name and diet (profile), (b) a decision you made at turn 8 (summary), and (c) the exact sentence you said two turns ago (recent window) — while the prompt stays under 2,000 estimated tokens.

---

## Homework

1. **Reading:** Read the LangChain how-tos on *trimming messages* and *migrating from legacy memory* (links below). List the three biggest differences between `ConversationSummaryBufferMemory` and the hybrid design in this chapter.
2. **Coding:** Complete Exercises 1–4. Save Exercise 1's table, and add a plot (matplotlib) of the four curves.
3. **Experiment:** Using a real conversation transcript of 30+ turns, compare answers to 5 recall questions across buffer, window(6), trim(500), and summary. Record which strategy failed which question and why.
4. **Analysis:** Write half a page: for **(a)** a mental-health companion app, **(b)** an airline customer-service bot, **(c)** a coding copilot — choose a strategy, justify the window/summary parameters, and state what you refuse to store.
5. **Debugging:** Fix this code (three bugs):

```python
memory = HybridMemory(llm, "Be helpful.")                  # (a)

def chat(user_id, text):
    msgs = memory.build_messages(text)
    msgs.insert(0, SystemMessage(content=memory.summary))  # (b)
    ai = llm.invoke(msgs[-6:])                             # (c)
    memory.remember(text, ai.content)
    return ai.content
```

*(Hints: (a) global memory shared by all users; (b) a second system message that may be rejected or duplicate the first; (c) slicing the list chops off the system message.)*

---

## Additional Resources

- [LangChain Docs — How to trim messages](https://python.langchain.com/docs/how_to/trim_messages/)
- [LangChain Docs — How to add message history](https://python.langchain.com/docs/how_to/message_history/)
- [LangChain Docs — Migrating off legacy memory](https://python.langchain.com/docs/versions/migrating_memory/)
- [LangGraph Docs — Add memory / persistence](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [API Reference — `trim_messages`](https://python.langchain.com/api_reference/core/messages/langchain_core.messages.utils.trim_messages.html)
- [MemGPT / Letta — LLMs as operating systems (Packer et al., 2023)](https://arxiv.org/abs/2310.08560)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Anthropic — Long context prompting tips](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips)

---

## What's Next

You can now decide **what the model sees**: buffer, window, trim, summary, profile, or hybrid — and you can read the legacy classes that older codebases still use.

But every memory object in this chapter lives in a **Python process**. Restart the server and it's gone; run two workers and they disagree. In **Chapter 8.3** we move history into **Redis and PostgreSQL**, so sessions survive restarts, scale across workers, expire with TTLs, and can be deleted on request.

> [← Previous: Why Memory?](chapter-32-why-memory.md) | [Next: Persistent Memory →](chapter-34-persistent-memory.md)

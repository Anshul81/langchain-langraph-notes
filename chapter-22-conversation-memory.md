# Chapter 5.1: Conversation Memory

> **Phase 5 — Memory & Chat History** | [← Previous: Streaming & Async Chains](../phase-04-chains-runnables/chapter-21-streaming-async-chains.md) | [Next: Document Loading & Text Splitting →](../phase-06-rag/chapter-23-document-loading-splitting.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand why LLMs are stateless and why memory matters
- ✅ Use `ChatMessageHistory` to store messages
- ✅ Use `RunnableWithMessageHistory` to add memory to any chain
- ✅ Manage multiple conversation sessions
- ✅ Understand different memory strategies (buffer, window, summary)
- ✅ Use persistent memory backends (Redis, SQL, MongoDB)

| | |
|---|---|
| **Prerequisites** | Chapter 4.2 (Sequential Chains), Chapter 3.3 (ChatPromptTemplate) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

LLMs have **zero memory** by default. Every call is independent:

```
You: "My name is Rahul."
LLM: "Nice to meet you, Rahul!"

You: "What's my name?"
LLM: "I don't know your name."  ← Forgot instantly!
```

Each `.invoke()` is a **fresh HTTP request** — no state is carried. To create chatbots, assistants, or any multi-turn app, YOU must manage the conversation history.

### The Solution: Send History with Every Request

```
Turn 1: "My name is Rahul"    → Send: [user: "My name is Rahul"]
Turn 2: "What's my name?"     → Send: [user: "My name is Rahul",
                                        ai: "Nice to meet you!",
                                        user: "What's my name?"]
```

The trick: **append every message** to a list and send the full list each time. LangChain automates this.

---

## Part 1: `ChatMessageHistory` — The Message Store

The simplest memory — stores messages in a Python list:

```python
from langchain_core.chat_history import InMemoryChatMessageHistory

# Create a history store
history = InMemoryChatMessageHistory()

# Add messages
history.add_user_message("My name is Rahul.")
history.add_ai_message("Nice to meet you, Rahul!")
history.add_user_message("What's my name?")

# View all messages
for msg in history.messages:
    print(f"{msg.type}: {msg.content}")
# human: My name is Rahul.
# ai: Nice to meet you, Rahul!
# human: What's my name?

# Clear history
history.clear()
```

---

## Part 2: `RunnableWithMessageHistory` — Memory for Chains

The **modern way** (LangChain v0.2+) to add memory to any chain:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "standard"),
    temperature=0.7,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=200
)

# Step 1: Prompt WITH a history placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant."),
    MessagesPlaceholder(variable_name="history"),  # ← history goes here
    ("human", "{input}")
])

# Step 2: Build the chain
chain = prompt | llm | StrOutputParser()

# Step 3: Session store
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Step 4: Wrap chain with memory
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Step 5: Use it!
config = {"configurable": {"session_id": "user_123"}}

r1 = chain_with_memory.invoke({"input": "My name is Rahul."}, config=config)
print(f"Turn 1: {r1}")

r2 = chain_with_memory.invoke({"input": "What's my name?"}, config=config)
print(f"Turn 2: {r2}")  # ← It remembers!
```

### How It Works Under the Hood

```
Turn 2 flow:
1. User sends: {"input": "What's my name?"}
2. RunnableWithMessageHistory looks up session "user_123"
3. Finds: [human: "My name is Rahul", ai: "Nice to meet you!"]
4. Injects into prompt's MessagesPlaceholder
5. LLM sees full context → responds: "Your name is Rahul!"
6. Both user message and AI response saved to history
```

---

## Part 3: Multiple Sessions

Different users = different histories:

```python
config_a = {"configurable": {"session_id": "alice"}}
config_b = {"configurable": {"session_id": "bob"}}

chain_with_memory.invoke({"input": "I love Python."}, config=config_a)
chain_with_memory.invoke({"input": "I hate mornings."}, config=config_b)

chain_with_memory.invoke({"input": "What do I love?"}, config=config_a)
# → "You said you love Python!"

chain_with_memory.invoke({"input": "What do I hate?"}, config=config_b)
# → "You said you hate mornings!"
```

---

## Part 4: Memory Strategies

### Strategy 1: Window Memory (Keep Last N Messages)

```python
def get_windowed_history(session_id: str, window_size: int = 10):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    history = store[session_id]
    if len(history.messages) > window_size:
        trimmed = history.messages[-window_size:]
        history.clear()
        for msg in trimmed:
            history.add_message(msg)
    return history
```

### Strategy 2: Token-Based Trimming

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=1000,
    strategy="last",
    token_counter=llm,
    include_system=True,
    start_on="human"
)

chain = trimmer | prompt | llm | StrOutputParser()
```

### Strategy 3: Summary Memory (Advanced)

Summarize old messages periodically and replace them with a condensed summary.

### Comparison

| Strategy | Pros | Cons | Best For |
|----------|------|------|----------|
| **Full history** | Perfect recall | Grows forever, hits token limit | Short conversations |
| **Window (last N)** | Fixed size, simple | Forgets old context | Customer support bots |
| **Token trimming** | Respects model limits | May cut mid-thought | General chatbots |
| **Summary** | Compact, preserves key facts | Lossy, extra LLM call | Long conversations |

---

## Part 5: Persistent Memory

`InMemoryChatMessageHistory` is lost when your app restarts. For production:

```python
# Redis (most common)
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_history(session_id: str):
    return RedisChatMessageHistory(session_id=session_id, url="redis://localhost:6379")

# SQLite (simple, file-based)
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_sql_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id, connection="sqlite:///chat_history.db")

# Swap into RunnableWithMessageHistory — everything else stays the same!
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_redis_history,  # ← just change this function
    input_messages_key="input",
    history_messages_key="history"
)
```

---

## Common Mistakes

### Mistake 1: Forgetting `MessagesPlaceholder`
```python
# ❌ No history placeholder — memory has nowhere to go!
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}")
])

# ✅ Include MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder(variable_name="history"),  # ← required!
    ("human", "{input}")
])
```

### Mistake 2: Forgetting `session_id` in config
```python
# ❌ No session_id — crashes!
chain_with_memory.invoke({"input": "hello"})

# ✅ Always pass session_id
chain_with_memory.invoke(
    {"input": "hello"},
    config={"configurable": {"session_id": "user_123"}}
)
```

### Mistake 3: Mismatched key names
```python
# ❌ Prompt uses "chat_history" but RunnableWithMessageHistory uses "history"
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
chain_with_memory = RunnableWithMessageHistory(
    chain, get_history,
    input_messages_key="input",       # but prompt uses "question"!
    history_messages_key="history"    # but prompt uses "chat_history"!
)

# ✅ Keys must match
chain_with_memory = RunnableWithMessageHistory(
    chain, get_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)
```

### Mistake 4: Using `and` to concatenate lists
```python
# ❌ `and` returns second operand if first is truthy — skips list_a!
for msg in session_a.messages and session_b.messages:
    print(msg)

# ✅ Use + to concatenate
for msg in session_a.messages + session_b.messages:
    print(msg)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Always use `MessagesPlaceholder` for history | Standard pattern for injecting history |
| Use `session_id` to isolate users | Prevents conversation leakage |
| Implement token trimming for production | Prevents context window overflow |
| Use persistent backends (Redis/SQL) in production | Survives app restarts |
| Place history AFTER system message, BEFORE user message | System instructions always apply |
| Match key names between prompt and `RunnableWithMessageHistory` | Prevents silent failures |

---

## Interview Preparation

### Easy
**Q: Why do LLMs need external memory management?**

> LLMs are **stateless** — each API call is independent. To maintain conversation context, you must send all previous messages with each request. LangChain's `RunnableWithMessageHistory` automates this.

### Medium
**Q: How does `RunnableWithMessageHistory` work?**

> It wraps any chain. Before each call: (1) looks up session history, (2) injects messages into `MessagesPlaceholder`. After: (3) saves user input and AI response. The chain itself doesn't know about memory.

### Hard
**Q: How would you handle memory for thousands of turns?**

> Layered approach: (1) `trim_messages` for context window limits, (2) periodic summarization of old messages, (3) full history in database for audit, (4) extracted key facts in a separate user profile always included in system prompt.

### Senior
**Q: `RunnableWithMessageHistory` vs LangGraph persistence?**

> `RunnableWithMessageHistory` is for simple chatbots — manages message history only. LangGraph's `MemorySaver` saves **entire graph state**, supports time-travel/checkpointing, and works with multi-agent workflows. Use LangGraph when you need stateful agents, interrupts, or complex state.

---

## Mini Assignment — Answer

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "standard"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=200
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor. Remember student details and adapt your teaching style."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# --- Student A (beginner) ---
config_a = {"configurable": {"session_id": "student_a"}}

r1 = chain_with_memory.invoke({"input": "I'm a beginner in Python"}, config=config_a)
print(f"A Turn 1: {r1}\n")

r2 = chain_with_memory.invoke({"input": "Explain loops"}, config=config_a)
print(f"A Turn 2: {r2}\n")

r3 = chain_with_memory.invoke({"input": "What level am I?"}, config=config_a)
print(f"A Turn 3: {r3}\n")

# --- Student B (expert) ---
config_b = {"configurable": {"session_id": "student_b"}}

r1 = chain_with_memory.invoke({"input": "I'm an expert in ML"}, config=config_b)
print(f"B Turn 1: {r1}\n")

r2 = chain_with_memory.invoke({"input": "Explain transformers"}, config=config_b)
print(f"B Turn 2: {r2}\n")

r3 = chain_with_memory.invoke({"input": "What level am I?"}, config=config_b)
print(f"B Turn 3: {r3}\n")

# --- Print histories ---
print("\n📚 Student A History:")
for msg in get_session_history("student_a").messages:
    print(f"  {msg.type}: {msg.content[:60]}...")

print("\n📚 Student B History:")
for msg in get_session_history("student_b").messages:
    print(f"  {msg.type}: {msg.content[:60]}...")
```

---

## Summary

| Component | What It Does |
|-----------|-------------|
| `InMemoryChatMessageHistory` | Stores messages in a Python list (in-memory) |
| `MessagesPlaceholder("history")` | Placeholder in prompt where history gets injected |
| `RunnableWithMessageHistory` | Wraps a chain to auto-manage message history |
| `session_id` in config | Isolates conversations per user/session |
| `trim_messages()` | Trims history to fit token limits |
| Redis/SQL backends | Persistent memory that survives restarts |

---

> [← Previous: Streaming & Async Chains](../phase-04-chains-runnables/chapter-21-streaming-async-chains.md) | [Next: Document Loading & Text Splitting →](../phase-06-rag/chapter-23-document-loading-splitting.md)

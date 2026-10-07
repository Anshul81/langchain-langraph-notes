# Chapter 10.4: Agent with Memory — Multi-Turn Tool-Using Conversations

> **Phase 10 — Agents** | [← Previous: Tool-Calling Agents](chapter-43-tool-calling-agents.md) | [Next: Agent Debugging →](chapter-45-agent-debugging.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Add **conversation memory** to LangGraph ReAct agents with checkpointers
- ✅ Separate conversations using **`thread_id`**
- ✅ Use **`state_modifier`** (system prompt) with persistent history
- ✅ Trim or summarize history to stay within context limits
- ✅ Compare **MemorySaver** vs production database checkpointers
- ✅ Build a **multi-turn travel assistant** that remembers prior preferences

| | |
|---|---|
| **Prerequisites** | Chapter 10.3 (Tool-Calling Agents), Phase 8 (Memory in LCEL) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction — Agents Without Memory Feel Broken

A user says: *"Book me a flight to Tokyo."* The agent searches and responds. Then: *"Make it business class."*

Without memory, the second message has **no link** to Tokyo. The agent asks *"Where to?"* — frustrating and wrong.

### The Problem

```
Turn 1: "Find flights to Tokyo next Friday"
        → Agent uses search_flights("Tokyo", ...)  ✓

Turn 2: "Business class only"
        → Agent sees ONLY this sentence
        → search_flights(???)  ✗
```

### The Solution — Checkpointed State

LangGraph **persists graph state** (including the full message list) per **`thread_id`**:

```
                    thread_id = "alice-session-1"
┌──────────────────────────────────────────────────┐
│  messages: [Human, AI, Tool, Human, AI, ...]      │
│  checkpointer saves after each graph step         │
└──────────────────────────────────────────────────┘
         Next invoke merges NEW user message
         into EXISTING history automatically
```

**Memory for agents = message history in graph state + a checkpointer.**

---

## Part 1: Setup

```bash
pip install langgraph langchain langchain-openai python-dotenv
```

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)
```

---

## Part 2: Tools — Travel Assistant

```python
@tool
def search_flights(destination: str, date: str, cabin: str = "economy") -> str:
    """Search flights. destination=city, date=YYYY-MM-DD, cabin=economy|business."""
    return (
        f"3 options to {destination.title()} on {date} ({cabin}): "
        f"Flight A $840 (08:00), Flight B $920 (14:30), Flight C $780 (22:15)."
    )

@tool
def search_hotels(city: str, nights: int, stars: int = 4) -> str:
    """Search hotels in a city for N nights at minimum star rating."""
    return (
        f"Hotels in {city.title()} ({nights} nights, {stars}+ stars): "
        f"Harbor Inn $189/night, Skyline $240/night, Zen Retreat $165/night."
    )

@tool
def remember_preference(key: str, value: str) -> str:
    """Store a user preference for this trip (airline, budget, dietary)."""
    return f"Saved preference: {key}={value}"

tools = [search_flights, search_hotels, remember_preference]
```

---

## Part 3: Agent + `MemorySaver`

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

SYSTEM = (
    "You are a travel assistant. Use tools for flights and hotels. "
    "Remember explicit preferences the user states. "
    "When the user refines a request ('business class', 'cheaper hotel'), "
    "use prior messages — do not ask for info already provided."
)

checkpointer = MemorySaver()

agent = create_react_agent(
    llm,
    tools,
    state_modifier=SYSTEM,
    checkpointer=checkpointer,
)

config = {"configurable": {"thread_id": "travel-user-001"}}
```

### Multi-Turn Invoke

```python
def ask(user_text: str) -> str:
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
    )
    return result["messages"][-1].content

print(ask("I need flights to Tokyo on 2025-11-14."))
print(ask("Business class only, please."))
print(ask("Also find a 4-star hotel for 3 nights there."))
```

The second turn should **inherit Tokyo and date** from checkpointed messages — no re-asking.

### Inspecting Checkpointed History

```python
state = agent.get_state(config)
print(f"Messages in thread: {len(state.values['messages'])}")
for m in state.values["messages"][-4:]:
    print(type(m).__name__, ":", (m.content or str(m.tool_calls))[:100])
```

---

## Part 4: Multiple Users — One Agent, Many Threads

```python
def ask_in_thread(thread_id: str, text: str) -> str:
    cfg = {"configurable": {"thread_id": thread_id}}
    out = agent.invoke({"messages": [HumanMessage(content=text)]}, config=cfg)
    return out["messages"][-1].content

print(ask_in_thread("user-alice", "Find hotels in Paris for 2 nights."))
print(ask_in_thread("user-bob", "Find hotels in Berlin for 2 nights."))
print(ask_in_thread("user-alice", "Something under $200 per night."))  # still Paris
```

```
thread_id isolation:
────────────────────
 user-alice  →  [Paris context ...]
 user-bob    →  [Berlin context ...]
 (no cross-leak if thread_id is unique per session)
```

**Production:** use `user_id + session_id` or UUID per chat tab.

---

## Part 5: Context Window Management

Long agent runs accumulate **tool outputs** — memory grows fast.

### Strategy 1: Message trimming before invoke (application layer)

```python
from langchain_core.messages import AIMessage, ToolMessage

MAX_MESSAGES = 20

def trim_messages(messages):
    if len(messages) <= MAX_MESSAGES:
        return messages
    # Keep system context via state_modifier; keep last N messages
    return messages[-MAX_MESSAGES:]

def ask_trimmed(thread_id: str, text: str) -> str:
    cfg = {"configurable": {"thread_id": thread_id}}
    prior = agent.get_state(cfg).values.get("messages", [])
    # Note: for production, implement trimming inside a custom graph or
    # use LangChain's trim_messages utility on the list before adding new input.
    result = agent.invoke({"messages": [HumanMessage(content=text)]}, config=cfg)
    return result["messages"][-1].content
```

### Strategy 2: Summarize older turns (pattern)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize this conversation for future context. Keep cities, dates, budgets:\n\n{history}"
)

def summarize_history(text_block: str) -> str:
    chain = summarize_prompt | llm | StrOutputParser()
    return chain.invoke({"history": text_block})
```

Use summaries as a **single system note** or inject as an early `HumanMessage` with label `"Conversation summary so far:"`.

| Approach | Pros | Cons |
|----------|------|------|
| Full history | Perfect fidelity | Hits token limits |
| Last-N messages | Simple | May drop early constraints |
| Summarization | Long sessions | Extra LLM call, summary drift |

---

## Part 6: Streaming With Memory

```python
cfg = {"configurable": {"thread_id": "stream-demo"}}

for event in agent.stream(
    {"messages": [HumanMessage(content="Hotels in Singapore, 5 nights.")]},
    config=cfg,
    stream_mode="updates",
):
    for node_name, update in event.items():
        if "messages" in update:
            last = update["messages"][-1]
            if hasattr(last, "content") and last.content:
                print(f"[{node_name}] {last.content[:120]}")
```

Streaming + memory gives responsive UIs while checkpoints persist between user messages.

---

## Part 6b: Production Checkpointers (Overview)

`MemorySaver` resets when your process restarts. Production deployments swap in durable savers:

| Checkpointer | Use case |
|--------------|----------|
| `MemorySaver` | Local dev, unit tests |
| `SqliteSaver` | Single-node apps, demos |
| Postgres saver (LangGraph ecosystem) | Horizontally scaled APIs |

```python
# Development only — pattern is identical for durable backends:
# checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
# agent = create_react_agent(llm, tools, checkpointer=checkpointer, state_modifier=SYSTEM)
```

Whichever backend you choose, treat **`thread_id` as a session primary key** in your API layer and never expose one user's thread to another.

---

## Part 6c: HumanMessage-Only Invokes (Important)

Each turn should pass **only the new** user message — not the full history manually:

```python
# ✅ Correct — checkpointer merges history
agent.invoke({"messages": [HumanMessage(content="Business class only.")]}, config=config)

# ❌ Wrong — duplicating full history each call inflates state and tokens
# agent.invoke({"messages": entire_history + [new_msg]}, ...)
```

If you manually append all messages every time, you will **duplicate** turns in checkpoint storage and blow the context window.

---

## Part 7: AgentExecutor Memory (Legacy Bridge)

If you maintain AgentExecutor (Chapter 10.2), use `RunnableWithMessageHistory` with `chat_history` in the prompt. For LangGraph agents, **checkpointers replace that pattern** — do not double-store history in two systems.

---

## Part 8: Evaluating Multi-Turn Agent Memory

Build three-turn scripts and assert **tool arguments** on later turns:

```python
def test_tokyo_follow_up():
    cfg = {"configurable": {"thread_id": "eval-tokyo-1"}}
    agent.invoke({"messages": [HumanMessage(content="Flights to Tokyo 2025-11-14.")]}, config=cfg)
    result = agent.invoke(
        {"messages": [HumanMessage(content="Business class only.")]},
        config=cfg,
    )
    # Inspect tool calls in result["messages"] for destination=Tokyo
    tool_msgs = [m for m in result["messages"] if hasattr(m, "tool_calls") and m.tool_calls]
    assert any("Tokyo" in str(tc) for m in tool_msgs for tc in (m.tool_calls or []))
```

Memory regressions show up as **wrong tool args**, not always as wrong natural language — test the structured steps.

---

## Common Mistakes

### Mistake 1: Reusing `thread_id` across all users
```python
# ❌ Everyone shares one conversation
config = {"configurable": {"thread_id": "default"}}

# ✅ Unique thread per user session
config = {"configurable": {"thread_id": f"user-{user_id}-{session_id}"}}
```

### Mistake 2: Creating a new agent without checkpointer each request
```python
# ❌ New MemorySaver every HTTP request — state lost
def handle(req):
    agent = create_react_agent(llm, tools, checkpointer=MemorySaver())

# ✅ Single compiled agent + shared checkpointer (or DB-backed saver)
agent = create_react_agent(llm, tools, checkpointer=shared_checkpointer)
```

### Mistake 3: Expecting tools to remember things
```python
# ❌ Tool-local dict without thread scope
_prefs = {}
@tool
def save_pref(k, v):
    _prefs[k] = v  # global — leaks across users!

# ✅ Let the LLM use message history, or store prefs in DB keyed by user_id
```

### Mistake 4: Unbounded tool logs in history
```python
# ❌ Tool returns 50 KB JSON every call → context explodes
# ✅ Truncate tool outputs; store full payload in object storage / DB
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| One `thread_id` per chat session | Isolation and correct pronoun resolution |
| Use `state_modifier` for stable rules | System instructions don't bloat user turns |
| Monitor message count per thread | Plan trimming/summary before limits |
| Use Postgres/SQLite checkpointer in prod | Survives process restarts |
| Pass `recursion_limit` on every invoke | Memory doesn't prevent runaway loops |
| Log thread_id with traces | Debug multi-turn failures in LangSmith |

---

## Interview Preparation

### Easy
**Q: How do you add memory to a LangGraph ReAct agent?**

> Pass a checkpointer (e.g. `MemorySaver` for development) to `create_react_agent`, then invoke with a `configurable.thread_id`. Each invoke appends to the checkpointed message list for that thread. The agent sees prior human messages, AI replies, and tool results on subsequent turns.

### Medium
**Q: What is the difference between `thread_id` and session memory in LCEL?**

> `thread_id` identifies a persisted graph state bucket in the checkpointer — it stores the full agent message state across turns. LCEL `RunnableWithMessageHistory` wraps chains with a chat history store for input/output messages. Conceptually both persist conversation context, but LangGraph checkpoints capture **tool calls and tool messages** too, which simple chat history wrappers may omit unless configured carefully.

### Hard
**Q: How would you prevent context overflow in a long agent conversation?**

> Track token or message count per thread; when over threshold, summarize older messages with a cheap model and replace them with one summary message, or keep last-N turns plus a rolling summary. Trim verbose tool outputs at the source. Optionally store structured slots (destination, dates, budget) in side state or a database via a dedicated tool. Re-evaluate after trimming with regression tests on multi-turn scenarios.

### Hard
**Q: Should tool outputs be stored forever in checkpoint memory?**

> Store enough for the LLM to continue reasoning, but truncate large payloads in `ToolMessage.content` and keep full blobs in object storage keyed by tool call ID. Checkpoints grow with every turn; oversized tool JSON is the most common cause of sudden context overflow in memory-enabled agents.

### Senior
**Q: How do you deploy durable agent memory at scale?**

> Use a database-backed checkpointer (Postgres), shard by tenant, encrypt at rest, TTL old threads, and enforce max history size. Separate **conversation state** from **user profile** (preferences in CRM/DB). Horizontal scaling requires checkpointer storage all instances can reach — not in-process `MemorySaver`. Add observability: thread_id on every trace, metrics on checkpoint size and load latency. Consider async summarization jobs for inactive sessions.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Checkpointer** | Persists graph state between invocations |
| **`MemorySaver`** | In-memory checkpointer for dev/tests |
| **`thread_id`** | Key for conversation isolation |
| **`state_modifier`** | System prompt separate from user turns |
| **Message history** | Includes tool calls — agent memory is not just chat text |
| **Trimming / summary** | Keeps long sessions within context limits |

---

## Hands-on Exercise

Extend the travel assistant with a **`set_budget(max_usd: float)`** tool that writes to a **thread-safe store keyed by `thread_id`** (dict in memory is fine for exercise). Require the agent to respect budget on hotel suggestions across three turns: destination → dates → budget cap. Verify turn 3 filters options without losing city/dates.

**Stretch goal:** Persist one conversation thread to disk (export `agent.get_state(config).values["messages"]` as JSON) and reload it into a new process by replaying messages into a fresh thread — note what you lose if the checkpointer itself is not durable.

---

## Part 9: Memory + `recursion_limit` Together

Memory increases message count every turn; tool-heavy threads hit context limits **before** recursion limits. Set both:

```python
config = {
    "configurable": {"thread_id": "ops-user-9"},
    "recursion_limit": 12,
}

result = agent.invoke(
    {"messages": [HumanMessage(content="Continue from yesterday's itinerary change.")]},
    config=config,
)
```

If you see good answers truncating mid-thought, trim history **before** raising recursion — more loops with bloated history only add cost.

---

## What's Next

Memory makes agents usable; **debugging** makes them shippable. **Chapter 10.5** covers errors, limits, tracing, and systematic agent debugging.

---

> [← Previous: Tool-Calling Agents](chapter-43-tool-calling-agents.md) | [Next: Agent Debugging →](chapter-45-agent-debugging.md)

# Chapter 10.2: AgentExecutor — The Legacy Agent Runtime (Still Worth Knowing)

> **Phase 10 — Agents** | [← Previous: Intro to Agents](chapter-41-intro-to-agents.md) | [Next: Tool-Calling Agents →](chapter-43-tool-calling-agents.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand what `AgentExecutor` is and where it sits in LangChain history
- ✅ Build an agent with `create_tool_calling_agent` + `AgentExecutor`
- ✅ Configure `max_iterations`, timeouts, and `handle_parsing_errors`
- ✅ Compare **AgentExecutor** vs **LangGraph `create_react_agent`**
- ✅ Migrate a legacy AgentExecutor pattern to LangGraph when needed
- ✅ Run a **multi-tool support agent** using the classic stack

| | |
|---|---|
| **Prerequisites** | Chapter 10.1 (Intro to Agents), Phase 9 (Tools & Tool Calling) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction — The Runtime That Powered a Million Demos

For years, **`AgentExecutor`** was *the* way to run LangChain agents. You defined an agent (ReAct, OpenAI functions, tool-calling), wrapped it in an executor, and called `.invoke()`. The executor owned the loop: call LLM → parse action → run tool → feed observation back → repeat.

### The Problem

You will still see `AgentExecutor` in:

- Older tutorials and Stack Overflow answers
- Enterprise codebases built on LangChain 0.1–0.2
- Internal tools that never migrated to LangGraph

If you only learn `create_react_agent`, you cannot debug or extend those systems.

### The Solution

Learn AgentExecutor as **historical + practical context**, then default to **LangGraph** for new work.

```
LEGACY STACK (LangChain agents package):
──────────────────────────────────────
Tools + Prompt + LLM  →  create_tool_calling_agent()
                              ↓
                        AgentExecutor  ← owns the while-loop
                              ↓
                         Final answer

MODERN STACK (LangGraph):
─────────────────────────
Tools + LLM  →  create_react_agent()  ← graph IS the loop
                      ↓
               Compiled StateGraph
```

**AgentExecutor = outer loop. LangGraph = explicit graph with the same loop, but observable and controllable.**

---

## Part 1: Setup

```bash
pip install langchain langchain-openai langchain-community python-dotenv
```

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)
```

---

## Part 2: Tools for Our Support Agent

```python
@tool
def lookup_order(order_id: str) -> str:
    """Look up order status by order ID. Use when the user mentions an order number."""
    orders = {
        "ORD-1001": "Shipped — expected delivery Oct 12",
        "ORD-1002": "Processing — payment confirmed",
        "ORD-1003": "Cancelled — refund issued",
    }
    return orders.get(order_id.upper(), f"No order found for {order_id}")

@tool
def search_policy(topic: str) -> str:
    """Search company policies: returns, shipping, warranty. Use for policy questions."""
    policies = {
        "return": "Returns accepted within 30 days with receipt. Open-box items: 15 days.",
        "shipping": "Free shipping on orders over $50. Express: 2 business days.",
        "warranty": "Electronics: 1 year manufacturer warranty. Extended plans available.",
    }
    key = topic.lower().strip()
    for k, v in policies.items():
        if k in key:
            return v
    return f"Policy snippet for '{topic}': contact support@example.com for details."

@tool
def calculate_refund(amount: float, days_since_purchase: int) -> str:
    """Calculate refund amount after restocking fee. Use for refund math."""
    if days_since_purchase > 30:
        return "Not eligible — outside 30-day return window."
    fee = 0.0 if days_since_purchase <= 7 else amount * 0.15
    refund = max(0, amount - fee)
    return f"Purchase ${amount:.2f}, day {days_since_purchase}: refund ${refund:.2f} (fee ${fee:.2f})"

tools = [lookup_order, search_policy, calculate_refund]
```

---

## Part 3: `create_tool_calling_agent` + `AgentExecutor`

Modern legacy agents use **native tool calling**, not text parsing of `Action:` / `Observation:`.

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful customer support agent. Use tools for order lookup, "
        "policies, and refund calculations. Be concise. If tools fail, say so clearly.",
    ),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=6,
    max_execution_time=60,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)
```

### What `AgentExecutor` Does Internally

```
┌─────────────────────────────────────────────────────────┐
│                   AgentExecutor loop                     │
│                                                          │
│  input + scratchpad  →  LLM (tool calls?)               │
│            ↑                    │                        │
│            │                    ▼                        │
│            │            Run ToolNode (your @tools)       │
│            │                    │                        │
│            └──── ToolMessage / observation ────────────┘
│                                                          │
│  Stops when: no tool calls OR max_iterations OR error   │
└─────────────────────────────────────────────────────────┘
```

The **`agent_scratchpad`** placeholder is where intermediate tool calls and results are injected — similar to LangGraph's message list, but hidden inside the executor.

### First Invocation

```python
result = executor.invoke({
    "input": "I bought ORD-1002 for $120 nineteen days ago. Can I return it and how much refund?",
})

print("Answer:", result["output"])
print("Steps:", len(result.get("intermediate_steps", [])))
for action, observation in result.get("intermediate_steps", []):
    print(f"  Tool: {action.tool} → {observation[:80]}...")
```

Expected behavior: lookup order → search return policy → `calculate_refund(120, 19)`.

---

## Part 4: Key Configuration Knobs

| Parameter | Purpose | Production tip |
|-----------|---------|----------------|
| `max_iterations` | Cap LLM↔tool rounds | Start at 5–8 |
| `max_execution_time` | Wall-clock timeout (seconds) | Pair with API timeouts |
| `handle_parsing_errors` | Recover from malformed outputs | `True` in dev; log in prod |
| `return_intermediate_steps` | Audit trail of tools | Essential for debugging |
| `early_stopping_method` | What to do when max hit | `"force"` or `"generate"` |

```python
# Stricter production executor
strict_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
    max_iterations=5,
    max_execution_time=45,
    handle_parsing_errors="Check your tool arguments and try again.",
    return_intermediate_steps=True,
)
```

### Streaming (Legacy)

```python
for chunk in executor.stream({"input": "What's the warranty on electronics?"}):
    if "actions" in chunk:
        for action in chunk["actions"]:
            print(f"[tool] {action.tool}")
    if "output" in chunk:
        print("[final]", chunk["output"])
```

LangGraph streaming is richer (per-node events); AgentExecutor streaming is coarser but still useful in older apps.

---

## Part 5: AgentExecutor vs LangGraph

| Dimension | AgentExecutor | `create_react_agent` (LangGraph) |
|-----------|---------------|----------------------------------|
| **Loop visibility** | Opaque | Explicit graph nodes/edges |
| **State** | Scratchpad + optional memory | Typed state + checkpointers |
| **Human-in-the-loop** | Awkward | First-class interrupts |
| **Persistence** | Custom memory classes | `MemorySaver`, DB savers |
| **Parallel tools** | Supported via tool calling | Supported |
| **Recommended for new code** | No | Yes |

```python
# Equivalent modern pattern (preview — full detail in Chapter 10.3)
from langgraph.prebuilt import create_react_agent

graph_agent = create_react_agent(llm, tools)
# graph_agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

**Migration rule:** Keep tool definitions and prompts; replace `AgentExecutor` with a LangGraph agent and move `chat_history` to a checkpointer + `thread_id`.

---

## Part 6: Optional Chat History with AgentExecutor

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

response = agent_with_history.invoke(
    {"input": "What's the return policy?"},
    config={"configurable": {"session_id": "user-42"}},
)
print(response["output"])

follow_up = agent_with_history.invoke(
    {"input": "And for open-box items?"},
    config={"configurable": {"session_id": "user-42"}},
)
print(follow_up["output"])
```

Chapter 10.4 covers memory for LangGraph agents in depth — prefer that path for new chat agents.

---

## Part 7: Parsing Errors and `handle_parsing_errors`

Early ReAct agents parsed **text** like `Action: search`. Tool-calling agents rarely need this, but legacy stacks still hit parse failures when models emit markdown or chit-chat instead of structured calls.

```python
executor_lenient = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=(
        "Your last output was not a valid tool call. "
        "Call exactly one tool with valid JSON arguments, or give the final answer."
    ),
    max_iterations=5,
)

# Boolean True gives a generic repair message:
# handle_parsing_errors=True
```

When debugging parse loops, enable `verbose=True` and inspect whether the model is **trying to answer without tools** while the prompt still demands tools — tighten the system message: *"Use tools for order IDs and policy facts."*

---

## Part 8: Observability Hook for Legacy Runs

```python
def log_executor_run(result: dict, user_input: str):
    steps = result.get("intermediate_steps", [])
    print(f"input={user_input[:60]!r} steps={len(steps)}")
    for i, (action, obs) in enumerate(steps, 1):
        print(f"  {i}. {action.tool} args={action.tool_input} obs_len={len(str(obs))}")

result = executor.invoke({"input": "Policy on returns for ORD-1001"})
log_executor_run(result, "returns ORD-1001")
print(result["output"])
```

Pipe the same structure into JSON logs (`tool`, `args`, `latency_ms`) so you can compare AgentExecutor traces with LangGraph/LangSmith spans during migration.

---

## Common Mistakes

### Mistake 1: Forgetting `agent_scratchpad` in the prompt
```python
# ❌ Missing scratchpad — tools never see prior observations
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
])

# ✅ Required for tool-calling agents
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
```

### Mistake 2: No iteration limit
```python
# ❌ Infinite loop risk on bad prompts or tool errors
executor = AgentExecutor(agent=agent, tools=tools)

# ✅ Always cap iterations and time
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=6, max_execution_time=60)
```

### Mistake 3: Tools that raise instead of returning errors
```python
# ❌ Exception kills the whole executor run
@tool
def bad_tool(x: str) -> str:
    raise ValueError("API down")

# ✅ Return actionable error strings
@tool
def good_tool(x: str) -> str:
    try:
        ...
    except Exception as e:
        return f"Tool error: {e}. Try a different query."
```

### Mistake 4: Building new products on AgentExecutor
```python
# ❌ New 2025+ greenfield app on legacy executor
executor = AgentExecutor(...)

# ✅ LangGraph for new work; AgentExecutor only for maintenance/migration
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm, tools)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use `create_tool_calling_agent`, not old ReAct text parsers | Fewer parsing failures |
| Set `max_iterations` and `max_execution_time` | Cost and latency guardrails |
| Enable `return_intermediate_steps` in staging | Replay tool sequences |
| Keep tool descriptions precise | Same as modern agents |
| Plan LangGraph migration for long-lived services | Better ops and HITL |
| Log `intermediate_steps` to your observability stack | Legacy agents are hard to debug |

---

## Interview Preparation

### Easy
**Q: What is `AgentExecutor`?**

> `AgentExecutor` is LangChain's legacy runtime that runs the agent loop: it repeatedly calls the LLM, executes any requested tools, appends observations to the scratchpad, and stops when the model returns a final answer or limits are hit. It wraps agents created with helpers like `create_tool_calling_agent`. New projects should use LangGraph instead, but many production systems still use AgentExecutor.

### Medium
**Q: What is the role of `agent_scratchpad` in a tool-calling agent prompt?**

> The scratchpad holds the sequence of prior tool calls and tool results (observations) within the current run. The placeholder `MessagesPlaceholder("agent_scratchpad")` injects that history so the LLM can reason about what it already tried. Without it, each LLM call would only see the user message and would not know previous tool outputs — breaking multi-step reasoning.

### Hard
**Q: How would you migrate an AgentExecutor-based agent to LangGraph?**

> (1) Keep the same `@tool` functions and system instructions. (2) Replace `create_tool_calling_agent` + `AgentExecutor` with `create_react_agent(llm, tools, state_modifier=system_prompt)`. (3) Change invocation from `{"input": "..."}` to `{"messages": [HumanMessage(...)]}`. (4) Map `max_iterations` to `recursion_limit` in config (remember ~2 graph steps per tool round). (5) Replace `RunnableWithMessageHistory` with a checkpointer and `thread_id`. (6) Replicate logging by streaming graph events or using LangSmith. (7) Run parallel evals on golden questions comparing tool traces and final answers.

### Hard
**Q: How do `max_iterations` and `max_execution_time` interact in AgentExecutor?**

> `max_iterations` caps how many times the executor completes an LLM→tool cycle — it prevents unbounded reasoning loops. `max_execution_time` caps wall-clock seconds regardless of iteration count — it catches slow tools or hung network calls. Use **both**: iterations guard model behavior; time guards infrastructure. When time expires, the run stops even if iterations remain; log which limit fired for tuning.

### Senior
**Q: Why did LangChain move agents from AgentExecutor to LangGraph?**

> AgentExecutor bundled control flow in a black box — hard to insert human approval, branch on tool results, or persist partial state. LangGraph models agents as explicit graphs with typed state, checkpoints, interrupts, and streaming per node. That enables production patterns: durable conversations, time travel debugging, conditional routing, and multi-agent orchestration. AgentExecutor remains for backward compatibility, but the graph model aligns with how teams operate agents in production.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **`AgentExecutor`** | Legacy loop runner for LangChain agents |
| **`create_tool_calling_agent`** | Builds a tool-calling agent from prompt + LLM + tools |
| **`agent_scratchpad`** | Injects tool call history into the prompt |
| **`max_iterations`** | Safety cap on agent loops |
| **`intermediate_steps`** | List of (action, observation) for auditing |
| **LangGraph** | Preferred replacement with explicit graph control |

---

## Hands-on Exercise

Build a **billing assistant** with three tools: `get_invoice(id)`, `apply_promo_code(code)`, and `estimate_tax(subtotal, state)`. Wire it with `create_tool_calling_agent` and `AgentExecutor`, set `max_iterations=5`, and log all intermediate steps. Test: *"Invoice INV-88 was $200 in CA — any active promo SAVE10?"* Then sketch (comments only) how you would rewrite the same agent using `create_react_agent`.

**Stretch goal:** Add a `max_execution_time=30` integration test that asserts the executor stops gracefully when a `@tool` sleeps 5 seconds and the model calls it repeatedly (mock the tool to detect call count).

---

## What's Next

You understand the **legacy runtime**. In **Chapter 10.3**, you build the same capabilities with **LangGraph tool-calling / ReAct agents** — the default for new LangChain projects.

---

> [← Previous: Intro to Agents](chapter-41-intro-to-agents.md) | [Next: Tool-Calling Agents →](chapter-43-tool-calling-agents.md)

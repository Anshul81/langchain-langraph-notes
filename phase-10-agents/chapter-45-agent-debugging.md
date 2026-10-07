# Chapter 10.5: Error Handling & Agent Debugging — Shipping Reliable Agents

> **Phase 10 — Agents** | [← Previous: Agent with Memory](chapter-44-agent-with-memory.md) | [Next: Intro to RAG →](../phase-11-rag/chapter-46-intro-to-rag.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Handle **`GraphRecursionError`**, tool failures, and LLM timeouts gracefully
- ✅ Use **streaming** and **verbose** modes to inspect agent steps
- ✅ Add **callbacks** and **LangSmith** tracing for production debugging
- ✅ Design **tool error contracts** that help the LLM recover
- ✅ Build guardrails: recursion limits, fallbacks, and user-facing messages
- ✅ Run a **debug playbook** on a failing multi-tool agent

| | |
|---|---|
| **Prerequisites** | Chapters 10.3–10.4, Phase 9 (Tools) |
| **Estimated Reading Time** | 28 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction — Agents Fail Differently Than Chains

Chains fail at a **fixed step**. Agents fail **anywhere in a loop** — wrong tool, bad JSON args, infinite re-search, or silent hallucination after a tool error.

### The Problem

```
User: "Compare AWS and Azure pricing for 500 VMs"

Agent loop:
  search → 429 rate limit  → LLM retries search → retries → ...
  OR tool returns ""       → LLM invents numbers
  OR recursion_limit hit   → hard crash to user
```

### The Solution — Defense in Depth

```
┌────────────────────────────────────────────────────────────┐
│ Layer 1: Tool design (return errors as strings)          │
│ Layer 2: Executor limits (recursion_limit, timeouts)       │
│ Layer 3: Exception handlers (GraphRecursionError, etc.)    │
│ Layer 4: Observability (stream, callbacks, LangSmith)      │
│ Layer 5: Evals (golden questions + expected tool paths)    │
└────────────────────────────────────────────────────────────┘
```

---

## Part 1: Setup

```bash
pip install langgraph langchain langchain-openai python-dotenv
```

```python
import os
import time
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
    timeout=30,
    max_retries=2,
)
```

Optional LangSmith (set in `.env`):

```bash
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=...
# LANGCHAIN_PROJECT=agent-debugging-lab
```

---

## Part 2: Tools With Explicit Error Contracts

```python
@tool
def search_docs(query: str) -> str:
    """Search internal documentation. Use for product and API questions."""
    if len(query.strip()) < 3:
        return "ERROR: Query too short. Provide at least 3 characters."
    # Simulate flaky API
    if "rate" in query.lower():
        return "ERROR: Upstream rate limited. Wait 2s or narrow the query."
    catalog = {
        "pricing": "Starter $29/dev/mo, Pro $79/dev/mo, Enterprise custom.",
        "sso": "SSO available on Pro+ via SAML 2.0 and OIDC.",
    }
    for k, v in catalog.items():
        if k in query.lower():
            return v
    return f"ERROR: No docs matched '{query}'. Try keywords: pricing, sso."

@tool
def calculate(expression: str) -> str:
    """Evaluate math expressions for comparisons."""
    import math
    try:
        val = eval(expression, {"__builtins__": {}, "math": math})
        return str(val)
    except Exception as e:
        return f"ERROR: Invalid expression ({e}). Use numbers and math.* only."

tools = [search_docs, calculate]
```

**Rule:** Tools return **`ERROR: ...`** strings — they do not raise — so the LLM can read the failure and adapt.

---

## Part 3: Agent With Safe Defaults

```python
from langgraph.prebuilt import create_react_agent

SYSTEM = (
    "You are a support agent. If a tool returns ERROR, explain the issue "
    "and try a different approach. Never invent pricing or policy facts."
)

agent = create_react_agent(llm, tools, state_modifier=SYSTEM)

DEFAULT_CONFIG = {
    "recursion_limit": 10,  # ~5 tool rounds (2 steps per round)
}
```

---

## Part 4: Handling `GraphRecursionError`

```python
from langgraph.errors import GraphRecursionError

def invoke_safe(user_text: str) -> str:
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=DEFAULT_CONFIG,
        )
        return result["messages"][-1].content
    except GraphRecursionError:
        return (
            "I hit the maximum reasoning steps for this question. "
            "Please simplify or split into smaller questions."
        )
    except Exception as e:
        return f"Something went wrong ({type(e).__name__}). Please try again."

print(invoke_safe("Search rate limits and pricing and compare annual cost for 10 devs"))
```

Tune `recursion_limit`: too low → premature stop; too high → cost and latency.

---

## Part 5: Streaming for Live Debugging

```python
def debug_stream(question: str):
    print(f"\n=== DEBUG RUN: {question[:60]}... ===")
    for event in agent.stream(
        {"messages": [HumanMessage(content=question)]},
        config=DEFAULT_CONFIG,
        stream_mode="updates",
    ):
        for node, data in event.items():
            msgs = data.get("messages", [])
            if not msgs:
                continue
            last = msgs[-1]
            if getattr(last, "tool_calls", None):
                for tc in last.tool_calls:
                    print(f"  [{node}] TOOL CALL {tc['name']}({tc['args']})")
            elif last.content:
                preview = last.content.replace("\n", " ")[:100]
                print(f"  [{node}] TEXT {preview}")

debug_stream("What is SSO and Pro pricing?")
```

ASCII view of what you are watching:

```
agent node  →  TOOL CALL search_docs(...)
tools node  →  (ToolMessage with result)
agent node  →  TOOL CALL search_docs(...)
tools node  →  ...
agent node  →  TEXT final answer
```

---

## Part 6: Custom Callback Handler

```python
from langchain_core.callbacks import BaseCallbackHandler

class AgentStepLogger(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        name = serialized.get("name", "?")
        print(f"[callback] tool_start {name} input={input_str[:80]}")

    def on_tool_end(self, output, **kwargs):
        print(f"[callback] tool_end output={str(output)[:80]}")

    def on_llm_error(self, error, **kwargs):
        print(f"[callback] llm_error {error}")

logger = AgentStepLogger()

result = agent.invoke(
    {"messages": [HumanMessage(content="Calculate 29*12 for Starter annual cost")]},
    config={**DEFAULT_CONFIG, "callbacks": [logger]},
)
```

Use callbacks to feed metrics (latency, tool frequency) into Datadog/Prometheus.

---

## Part 7: Retry Wrapper for Transient Tool Failures

```python
def invoke_with_tool_retry(user_text: str, max_attempts: int = 2) -> str:
    last_err = None
    for attempt in range(max_attempts):
        try:
            return invoke_safe(user_text)
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return f"Failed after retries: {last_err}"

# Better: retry inside the tool for idempotent reads, not the whole agent loop
@tool
def search_docs_resilient(query: str) -> str:
    """Resilient doc search with one retry."""
    for i in range(2):
        out = search_docs.invoke({"query": query})
        if not out.startswith("ERROR: Upstream rate"):
            return out
        time.sleep(2)
    return out
```

Avoid retrying **entire agent runs** unless idempotent — duplicates side effects.

---

## Part 8: Debugging Playbook

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Wrong tool chosen | Weak descriptions | Rewrite `@tool` docstrings |
| Infinite search loop | No stop condition in prompt | System: "max 2 searches" |
| `GraphRecursionError` | Limit too low OR loop | Raise limit or fix prompt/tools |
| Hallucination after tool error | Model ignores ERROR prefix | Enforce "never invent" + evals |
| Slow responses | Too many rounds | Lower limit, cache tool results |
| Works locally, fails in prod | Missing env / keys | Validate env at startup |

```python
def validate_agent_env():
    required = ["LITELLM_PROXY_API_KEY", "LITELLM_PROXY_API_BASE"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}")

validate_agent_env()
```

---

## Part 9: Minimal Golden-Test Harness

```python
GOLDEN = [
    {
        "q": "Pro plan price?",
        "must_contain": ["79"],
        "tools_expected": ["search_docs"],
    },
]

def run_golden(agent_fn):
    for case in GOLDEN:
        answer = agent_fn(case["q"])
        ok = all(s in answer for s in case["must_contain"])
        print(f"{'PASS' if ok else 'FAIL'}: {case['q'][:40]}")

run_golden(invoke_safe)
```

Expand with LangSmith datasets for regression after prompt changes.

---

## Part 10: Timeouts and Cancellation

```python
import signal

class AgentTimeout(Exception):
    pass

def invoke_with_wall_clock(user_text: str, seconds: int = 45) -> str:
    # Platform note: use asyncio.wait_for or task cancellation in async servers.
    start = time.time()
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config={**DEFAULT_CONFIG, "recursion_limit": 8},
        )
        if time.time() - start > seconds:
            return "Request timed out. Try a narrower question."
        return result["messages"][-1].content
    except GraphRecursionError:
        return invoke_safe(user_text)
```

In FastAPI, run agents in thread pools or async executors and cancel on client disconnect to avoid burning tokens after the user navigates away.

---

## Part 11: Redaction Before Logging

```python
import re

EMAIL = re.compile(r"[\\w.-]+@[\\w.-]+\\.[A-Za-z]{2,}")

def redact(text: str) -> str:
    return EMAIL.sub("[EMAIL]", text)

def log_trace(messages):
    for m in messages:
        content = getattr(m, "content", "") or ""
        print(redact(str(content))[:200])
```

Never ship full tool outputs containing PII to third-party tracing without scrubbing.

---

## Common Mistakes

### Mistake 1: Tools that raise on expected failures
```python
# ❌
@tool
def api_call(q: str) -> str:
    resp.raise_for_status()

# ✅ Return ERROR strings; reserve exceptions for programmer bugs
```

### Mistake 2: No recursion limit
```python
# ❌ agent.invoke({"messages": [...]})  # default may be high

# ✅ Always pass config={"recursion_limit": 10}
```

### Mistake 3: Debugging without seeing tool messages
```python
# ❌ Only print final answer
# ✅ Stream updates or log tool_calls + ToolMessage content (redact PII)
```

### Mistake 4: Changing prompts and tools simultaneously
```python
# ❌ Can't attribute regressions
# ✅ One change at a time; run golden set after each
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Standardize `ERROR:` prefix in tool outputs | LLM learns to react consistently |
| Set `recursion_limit` on every invoke | Prevents runaway cost |
| Enable LangSmith in staging | Time-travel through agent steps |
| Stream in internal admin UI | Faster RCA during incidents |
| Separate read vs write tools | Easier to add HITL on writes |
| Version prompts and tool schemas | Reproduce old behavior |

---

## Interview Preparation

### Easy
**Q: What is `GraphRecursionError`?**

> LangGraph raises it when the graph exceeds `recursion_limit` — the maximum number of super-steps. For ReAct agents, each tool round typically consumes multiple steps. Handling it with a user-friendly message prevents hard failures when the agent loops too long.

### Medium
**Q: Why should tools return error strings instead of raising exceptions?**

> The LLM needs the failure reason as an observation to choose a different tool, fix arguments, or explain the issue to the user. Uncaught exceptions abort the run. Structured error strings act as recoverable observations; reserve exceptions for unexpected bugs.

### Hard
**Q: Describe your agent observability stack in production.**

> Trace every run with project/user/thread IDs in LangSmith or OpenTelemetry. Log tool name, latency, and truncated output; redact secrets. Metrics: recursion depth, tool error rate, P50/P95 latency, tokens per session. Alerts on spike in recursion errors or tool ERROR rates. Golden evals on deploy. Dashboards per tool to spot flaky dependencies.

### Senior
**Q: How do you debug non-deterministic agent failures?**

> Capture full message traces for failing thread_ids, diff against successful runs on similar queries, classify failure mode (routing vs tool vs generation). Use counterfactual prompts in staging. A/B tool descriptions. Add constraint checks on outputs (citation required, numeric answers must cite tool). For intermittent issues, correlate with model version, temperature, and upstream 429s. Build minimal reproducers from LangSmith exports.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Tool error contract** | Return `ERROR: ...` instead of raising |
| **`recursion_limit`** | Cap graph steps / agent loops |
| **`GraphRecursionError`** | Signal to degrade gracefully |
| **Streaming `updates`** | See tool calls in real time |
| **Callbacks / LangSmith** | Production-grade traceability |
| **Golden tests** | Catch regressions when debugging prompts |

---

## Hands-on Exercise

Take the travel agent from Chapter 10.4. Inject a **`search_flights`** failure mode when `destination="TestFail"`. Implement `invoke_safe`, stream debugging, and one golden test. Document in comments your playbook entry for *"agent loops on repeated searches"*.

**Stretch goal:** Export one failing run to a JSON fixture (`messages`, `tool_calls`, final answer) and add a pytest that asserts the agent calls `search_docs` at most twice for a golden question.

---

## Part 12: Checklist Before Production Deploy

1. Every tool returns structured errors, never silent empty strings.
2. `recursion_limit` documented and enforced in API gateway.
3. LangSmith (or equivalent) project per environment.
4. Golden set of ≥20 queries with expected tool names.
5. PII redaction in logs and traces.
6. Runbook link in on-call docs for `GraphRecursionError` spikes.

---

## What's Next

You can build and debug **autonomous tool users**. **Phase 11** shifts to **RAG** — grounding LLMs in your documents — starting with **Chapter 11.1: Introduction to RAG**.

---

> [← Previous: Agent with Memory](chapter-44-agent-with-memory.md) | [Next: Intro to RAG →](../phase-11-rag/chapter-46-intro-to-rag.md)

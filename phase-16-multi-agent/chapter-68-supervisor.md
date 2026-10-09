# Chapter 16.1: Supervisor Architecture

> **Phase 16 — Multi-Agent Systems** | [← Previous: Streaming in LangGraph](../phase-15-langgraph-persistence/chapter-67-streaming-langgraph.md) | [Next: Swarm Architecture →](chapter-69-swarm.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Explain **when one agent is not enough** (and when it still is)
- ✅ Describe the **hub-and-spoke** supervisor pattern: supervisor routes, workers work, control returns to the supervisor
- ✅ Design **shared state** (`task`, `messages`, `results`) for a team of agents
- ✅ Build **real workers** — each a tool-using ReAct agent — under a supervisor
- ✅ Route with a **structured Pydantic decision** (`next` + `reason`)
- ✅ **Cap loops** with a turn counter and handle **worker failure** gracefully
- ✅ Choose between a supervisor and a single ReAct agent using a decision table

| | |
|---|---|
| **Prerequisites** | Phases 13–15 (StateGraph, conditional edges, loops, persistence) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction — The Problem

In Phase 13 you built a ReAct agent: one LLM, a handful of tools, a loop. That works great — until it doesn't:

```
One agent, 12 tools:
  ✗ Picks the wrong tool (descriptions blur together)
  ✗ One giant system prompt: "be a researcher AND an analyst AND a coder..."
  ✗ Every tool description is in context on every call (cost + confusion)
  ✗ Hard to test or improve one skill without breaking the others
```

The fix is the same one human teams use: **specialize**, then **coordinate**.

### The Solution — A Supervisor and Specialist Workers

```
                      ┌─────────────────────┐
   User task ───────► │     SUPERVISOR      │ ◄──────────────┐
                      │ "who goes next?"    │                │
                      └───┬─────────┬───────┘                │
              next=       │         │       next=            │ every worker
              researcher  │         │       analyst          │ reports BACK
                          ▼         ▼                        │ to supervisor
                   ┌───────────┐ ┌───────────┐               │
                   │ RESEARCHER│ │  ANALYST  │───────────────┤
                   │ web_search│ │ calculator│               │
                   └─────┬─────┘ └───────────┘               │
                         └───────────────────────────────────┘
                      next=FINISH ──► finalize ──► END
```

Three ideas to hold on to:

1. **Hub-and-spoke** — workers never talk to each other. All communication goes through the supervisor (the hub).
2. **Narrow workers** — each worker is a small ReAct agent with 1–3 tools and a one-paragraph prompt.
3. **Shared state is the whiteboard** — the supervisor reads it, workers write to it.

---

## Part 1: When Is One Agent Not Enough?

Don't reach for multi-agent by default. It costs more LLM calls and more latency. Use it when you see these signals:

| Signal | Why a team helps |
|---|---|
| Agent chooses wrong tools among many | Each worker sees only its own 1–3 tools |
| Prompt keeps growing ("also act as...") | Each worker gets a short, focused prompt |
| Sub-tasks need different models/temperature | Workers can use different LLMs |
| You want to test/tune one skill alone | Workers are independently invokable |
| Sub-tasks are clearly separable | The supervisor splits the job naturally |

### Supervisor vs. single ReAct agent

| Question | Single ReAct agent | Supervisor + workers |
|---|---|---|
| Tools | ≤ ~5, all related | Many, grouped by skill |
| LLM calls per task | Fewest | More (routing + each worker's loop) |
| Latency / cost | Lowest | Higher |
| Debuggability | One trace | Clear per-worker traces |
| Specialization | Weak (one prompt) | Strong (prompt + tools per role) |
| Failure handling | Agent-level only | Supervisor can re-route around a failure |
| **Start here when...** | **Almost always** | **Single agent demonstrably struggles** |

> **Rule of thumb:** build the single agent first. Split into a supervisor only when you can name the specific failure (wrong tools, prompt overload) the split fixes.

---

## Part 2: Shared State Design

State is the contract between the supervisor and workers. Keep it small:

```python
import ast
import operator as op
import os
from operator import add
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)


class TeamState(TypedDict):
    task: str                                  # the user's request (never changes)
    messages: Annotated[list, add_messages]    # conversation: user task + final answer
    results: Annotated[list[str], add]         # worker reports, appended (reducer = add)
    next: str                                  # supervisor's decision
    reason: str                                # why (great for debugging/logs)
    turns: int                                 # loop counter
```

Why these fields?

- **`task`** — a stable copy of the goal. Workers and supervisor always see the original ask.
- **`results`** — a list with an `add` reducer, so each worker *appends* its report instead of overwriting the previous one.
- **`messages`** — the user-facing conversation. Workers' internal tool chatter does **not** go here (it stays inside each worker's own graph). This keeps the supervisor's context clean.
- **`next` / `reason`** — the routing decision, stored so the graph and your logs can read it.
- **`turns`** — a safety counter (Part 5).

---

## Part 3: Real Workers — Tool-Using Agents

A worker is **not** `llm.invoke("You are a researcher...")`. That's just a prompt with a costume. A real worker can *act*: it calls tools, looks at results, and decides when it's done. That's exactly what `create_react_agent` gives us — a compiled LangGraph (a subgraph) with an LLM ↔ tools loop.

First, two tools. They're deterministic fakes so the demo runs the same every time:

```python
# --- Tool 1: a tiny fake "web search" -------------------------------------
KB = {
    "tokyo": "Tokyo metro area population: about 37.4 million people.",
    "paris": "Paris metro area population: about 11.2 million people.",
    "london": "London metro area population: about 9.6 million people.",
}


@tool
def web_search(query: str) -> str:
    """Search the web for a fact. Returns a short text snippet."""
    hits = [text for key, text in KB.items() if key in query.lower()]
    return " ".join(hits) or "No results found."


# --- Tool 2: a safe calculator (no eval!) ---------------------------------
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression such as '37.4 / 11.2'. Supports + - * / **."""
    try:
        return str(round(_eval(ast.parse(expression, mode="eval").body), 4))
    except Exception as e:
        return f"Error: {e}"
```

Now the two worker agents — each with **one** tool and a focused prompt:

```python
researcher_agent = create_react_agent(
    llm,
    tools=[web_search],
    prompt=(
        "You are a research specialist. Use web_search to find facts. "
        "Report the facts you found in 1-3 short sentences. Do not do math."
    ),
)

analyst_agent = create_react_agent(
    llm,
    tools=[calculator],
    prompt=(
        "You are a quantitative analyst. Use the calculator for ANY arithmetic. "
        "Use numbers from the teammate context. Report the result in 1-2 sentences."
    ),
)
```

> **Version note:** in LangGraph 0.3+ the system-prompt argument is `prompt=`. Older releases called it `state_modifier=`. In LangGraph v1 `create_react_agent` is being superseded by `langchain.agents.create_agent` with the same idea — the pattern in this chapter doesn't change.

### Wrapping an agent as a graph node

A worker agent has its own message-based state. Our team state is different. So we wrap each agent in a thin **node function** that (1) builds the worker's input from the shared state, (2) runs the agent, (3) writes a short report back into `results`:

```python
FAIL_WORKER = os.getenv("FAIL_WORKER", "")  # test hook: FAIL_WORKER=researcher


def make_worker_node(name: str, agent):
    def node(state: TeamState) -> dict:
        context = "\n".join(state["results"]) or "(none yet)"
        prompt = (
            f"TASK: {state['task']}\n\n"
            f"CONTEXT FROM TEAMMATES:\n{context}\n\n"
            "Do ONLY your part of the task and reply concisely."
        )
        try:
            if FAIL_WORKER == name:
                raise RuntimeError("simulated outage")
            out = agent.invoke({"messages": [HumanMessage(content=prompt)]})
            report = f"[{name}] {out['messages'][-1].content}"
        except Exception as e:
            # Don't crash the whole graph — tell the supervisor what happened.
            report = f"[{name}] FAILED: {type(e).__name__}: {e}"
        return {"results": [report]}

    return node


researcher_node = make_worker_node("researcher", researcher_agent)
analyst_node = make_worker_node("analyst", analyst_agent)
```

Notice the `try/except`. A failing worker returns a **report that says it failed** rather than raising. The supervisor reads it like any other result and decides what to do (Part 6).

---

## Part 4: The Supervisor — Structured Routing

The supervisor's only job is to answer: **"Who should act next, and why?"** Free-text answers ("I think the researcher should go") are brittle to parse. Instead, force a typed decision with Pydantic and `with_structured_output`:

```python
class Route(BaseModel):
    """The supervisor's decision."""
    next: Literal["researcher", "analyst", "FINISH"] = Field(
        description="Which worker acts next, or FINISH if the task is fully answered."
    )
    reason: str = Field(description="One sentence explaining the choice.")


router_llm = llm.with_structured_output(Route)

SUPERVISOR_PROMPT = """You are a supervisor managing two workers:
- researcher: looks up facts with web search
- analyst: does calculations with a calculator

Read the TASK and the WORK SO FAR, then choose the next worker, or FINISH
if the work so far fully answers the task.

Rules:
- Facts are needed before math that depends on them.
- Never call a worker whose part is already done.
- If a worker FAILED, retry it at most once; otherwise route to another worker
  or FINISH and clearly state what could not be completed."""

MAX_TURNS = 6


def supervisor(state: TeamState) -> dict:
    turns = state.get("turns", 0)

    # Hard safety cap — no LLM call needed (Part 5)
    if turns >= MAX_TURNS:
        return {"next": "FINISH", "reason": "turn limit reached", "turns": turns}

    done = "\n".join(state["results"]) or "(nothing yet)"
    route = router_llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"TASK: {state['task']}\n\nWORK SO FAR:\n{done}"),
    ])
    return {"next": route.next, "reason": route.reason, "turns": turns + 1}
```

Because `next` is a `Literal`, the model can't invent a worker that doesn't exist. If your model is weak at structured output, the failure is loud (a validation error) rather than silent misrouting.

### The finalize node

When the supervisor says `FINISH`, one last node composes the user-facing answer from the collected reports:

```python
def finalize(state: TeamState) -> dict:
    notes = "\n".join(state["results"]) or "(no worker results)"
    answer = llm.invoke([
        SystemMessage(content=(
            "Write the final answer to the task using ONLY the worker notes. "
            "If any worker failed or the work is incomplete, say so plainly."
        )),
        HumanMessage(content=f"TASK: {state['task']}\n\nWORKER NOTES:\n{notes}"),
    ])
    return {"messages": [AIMessage(content=answer.content)]}
```

---

## Part 5: Wire the Graph and Cap the Loop

The wiring is the hub-and-spoke diagram in code. Workers have a **fixed edge back to the supervisor** — that's what makes it a hub.

```python
def route_from_supervisor(state: TeamState) -> str:
    return state["next"]


builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher_node)
builder.add_node("analyst", analyst_node)
builder.add_node("finalize", finalize)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {"researcher": "researcher", "analyst": "analyst", "FINISH": "finalize"},
)
builder.add_edge("researcher", "supervisor")   # spokes return to the hub
builder.add_edge("analyst", "supervisor")
builder.add_edge("finalize", END)

app = builder.compile()
```

### Why cap the loop?

The supervisor is an LLM. LLMs sometimes keep saying "researcher again" forever. Two layers of defense:

| Layer | Where | What it does |
|---|---|---|
| **Turn counter** | `supervisor` node (`MAX_TURNS`) | After N routing decisions, force `FINISH` |
| **Recursion limit** | `app.invoke(..., config)` | LangGraph raises `GraphRecursionError` as a last resort |

The turn counter is the *graceful* one: the user still gets an answer (built from whatever was collected). The recursion limit is the seatbelt — a crash, but a safe one.

---

## Part 6: Run It, and Handle Failure

```python
def run(task: str):
    init = {
        "task": task,
        "messages": [HumanMessage(content=task)],
        "results": [],
        "turns": 0,
    }
    final = None
    for step in app.stream(init, stream_mode="updates", config={"recursion_limit": 25}):
        for node, update in step.items():
            if node == "supervisor":
                print(f"🧭 supervisor → {update['next']:<10} ({update['reason']})")
            elif node in ("researcher", "analyst"):
                print(f"🛠  {update['results'][-1][:110]}")
            elif node == "finalize":
                final = update["messages"][-1].content
    print("\n✅ FINAL:", final)


if __name__ == "__main__":
    run("Find the metro populations of Tokyo and Paris, "
        "then compute how many times bigger Tokyo is.")
```

> **All the code blocks in Parts 2–6 go in one file** (e.g. `supervisor_demo.py`), in order. Needs `pip install langgraph langchain-openai python-dotenv pydantic` and your LiteLLM env vars.

**Typical output** (wording varies by model):

```
🧭 supervisor → researcher (Population facts are needed before any math.)
🛠  [researcher] Tokyo metro: ~37.4 million. Paris metro: ~11.2 million.
🧭 supervisor → analyst    (Facts collected; now compute the ratio.)
🛠  [analyst] Tokyo is about 3.34 times larger than Paris (37.4 / 11.2).
🧭 supervisor → FINISH     (Both parts of the task are complete.)

✅ FINAL: Tokyo's metro area (~37.4M) is about 3.34× larger than Paris's (~11.2M).
```

Count the LLM-driven steps: 3 supervisor decisions + 2 worker agents (each with their own tool loop) + 1 finalize. That's the cost of specialization — which is why Part 1's table says *start with a single agent*.

### What happens when a worker fails?

Run it with the failure hook:

```bash
# PowerShell
$env:FAIL_WORKER = "researcher"; python supervisor_demo.py
```

```
🧭 supervisor → researcher (Need population facts first.)
🛠  [researcher] FAILED: RuntimeError: simulated outage
🧭 supervisor → researcher (Retrying the researcher once after failure.)
🛠  [researcher] FAILED: RuntimeError: simulated outage
🧭 supervisor → FINISH     (Researcher failed twice; cannot complete the calculation.)

✅ FINAL: I couldn't complete this: the research worker failed twice, so the
population figures (and therefore the ratio) are unavailable.
```

The failure flow is: **worker catches the exception → writes `FAILED:` into `results` → supervisor sees it → retries once or gives up → finalize reports honestly.** The graph never crashes, the user never gets a fabricated answer, and `MAX_TURNS` guarantees it ends even if the supervisor keeps retrying.

---

## Part 7: A Note on Prebuilt Supervisors

The LangGraph ecosystem also ships `langgraph-supervisor` (`create_supervisor(...)`), which generates this exact graph for you using handoff tools. It's convenient once you understand the pattern — but build it manually first so you know where state, routing, and failure handling live. Everything above is what that helper does under the hood.

---

## Common Mistakes

| Mistake | Why it hurts | Fix |
|---|---|---|
| "Workers" are bare `llm.invoke(prompt)` | No tools = no real capability; just roleplay | Use `create_react_agent` + `@tool` |
| No turn cap | Supervisor ping-pongs forever, burning tokens | `MAX_TURNS` → `FINISH`, plus `recursion_limit` |
| Parsing routing from free text | Typos, extra words → wrong/no route | `with_structured_output(Route)` with a `Literal` |
| Overwriting results (no reducer) | Second worker erases the first's report | `Annotated[list[str], add]` |
| Dumping worker tool chatter into supervisor context | Context bloat, supervisor confusion | Return only a short **report** from each worker |
| Worker exception crashes the graph | One flaky tool kills the whole run | `try/except` → `FAILED:` report |
| Supervisor has no failure rule | Retries forever or ignores the failure | Put retry/give-up policy in the prompt **and** the cap |
| 8 workers on day one | Routing accuracy drops, cost explodes | Start with 2–4; split further (Ch 16.3) only if needed |

---

## Best Practices

1. **Start single, split on evidence.** Name the failure the split fixes.
2. **2–4 workers.** Past that, the supervisor itself becomes the overloaded agent — use hierarchy (Chapter 16.3).
3. **One skill per worker**, 1–3 tools, a 2–3 line prompt.
4. **Structured routing** (`Literal` + `reason`). Log the `reason` — it's your best debugging tool.
5. **Workers return reports, not transcripts.** Keep supervisor context small.
6. **Always cap** with a turn counter *and* `recursion_limit`.
7. **Make failure a first-class result**, not an exception.
8. **Stream with `stream_mode="updates"`** during development so you can watch the routing.

---

## Interview Questions & Answers

### 🟢 Easy

**Q1: What is the supervisor pattern?**
A central supervisor LLM decides which specialist worker should act next. Workers do their task and return control to the supervisor, which either picks another worker or finishes. It's hub-and-spoke.

**Q2: Why use specialist workers instead of one agent with all the tools?**
Fewer tools per agent means better tool selection, shorter prompts, lower per-call context cost, and the ability to improve or test each skill independently.

**Q3: Why must workers have tools (or be real agents)?**
A worker that's only a system prompt can't act on the world or verify anything — it's the same LLM with a different hat. Tool-using workers have real, distinct capabilities.

### 🟡 Medium

**Q4: How do you make supervisor routing reliable?**
Use structured output: a Pydantic model with `next: Literal[...]` and `reason: str`. The `Literal` makes invalid workers impossible, and the schema is enforced by the model/API instead of string parsing.

**Q5: How do you stop a supervisor from looping forever?**
Keep a `turns` counter in state and force `FINISH` at a limit (graceful, still produces an answer), and set LangGraph's `recursion_limit` as a hard backstop.

**Q6: What should happen when a worker fails?**
The worker node catches the error and appends a `FAILED: ...` report to shared state. The supervisor, whose prompt defines a retry/give-up policy, retries once, reroutes, or finishes with an honest limitation. The graph itself never crashes.

### 🔴 Hard

**Q7: Supervisor vs a single ReAct agent — how do you decide?**
Start with a single agent. Move to a supervisor when you can show concrete failures from tool overload or prompt conflict, or need independent specialization/testing. The tradeoff is more LLM calls, higher latency, and more state design.

**Q8: How would you reduce supervisor cost/latency?**
Skip the LLM for deterministic steps (rule-based routing when the next step is obvious), use a cheaper model for routing, return short worker reports, run independent workers in parallel (Phase 14 fan-out), and cap turns.

**Q9: What are the scaling limits of a flat supervisor?**
Every worker description lives in the supervisor's prompt and every report flows through its context. Beyond ~4–6 workers, routing accuracy and cost degrade. The fix is hierarchy — supervisors of supervisors (Chapter 16.3) — or peer handoffs (Chapter 16.2).

---

## Summary

- A **supervisor** is a router LLM; **workers** are narrow tool-using agents; every worker returns to the supervisor (**hub-and-spoke**).
- **Shared state** is `task` + `messages` + `results` (appended via reducer) + `next`/`reason`/`turns`.
- Workers are **real agents** built with `create_react_agent` and `@tool`, wrapped in nodes that return short reports.
- Routing is **structured** (Pydantic `Route` with a `Literal`), not parsed from prose.
- **Cap loops** with `turns` → `FINISH` plus `recursion_limit`.
- **Failures become results**: the worker reports `FAILED`, the supervisor decides.
- Prefer a **single ReAct agent** until it demonstrably struggles.

---

## Hands-On Exercise

Pick **one** (or do both):

**Option A — Add a third "summarizer" worker.**
1. Create a `summarizer_agent` with `create_react_agent`. Give it a tool, e.g. `word_count(text: str) -> str`, and a prompt: "Write a 1-sentence summary and report its word count using the tool."
2. Wrap it with `make_worker_node("summarizer", summarizer_agent)`.
3. Add `"summarizer"` to the `Route.next` `Literal`, the supervisor prompt, the conditional-edge map, and add `add_edge("summarizer", "supervisor")`.
4. Update the task: *"Find the populations of Tokyo and Paris, compute the ratio, then give a one-sentence summary."* Confirm the supervisor visits all three workers in a sensible order.

**Option B — Force a failure path.**
1. Run with `FAIL_WORKER=analyst`. What does the supervisor do?
2. Lower `MAX_TURNS` to `2` and rerun the normal task. Does the final answer honestly say it's incomplete?
3. Bonus: make the hook fail only the **first** attempt (use a module-level counter) and confirm the retry succeeds.

**Success check:** your streamed output shows each routing decision with its `reason`, and the final answer never claims results a failed worker didn't produce.

---

## What's Next

A supervisor has a central boss. In **[Chapter 16.2: Swarm Architecture](chapter-69-swarm.md)** there is no boss — peer agents hand the conversation to each other using **handoff tools**, and you'll learn how to prevent them from passing it around forever.

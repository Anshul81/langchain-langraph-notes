# Chapter 16.1: Supervisor Architecture

> **Phase 16 — Multi-Agent Systems** | [← Previous: Streaming in LangGraph](../phase-15-langgraph-persistence/chapter-67-streaming-langgraph.md) | [Next: Swarm Architecture →](chapter-69-swarm.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Describe the **supervisor** pattern: one router, many workers
- ✅ Wire worker **subgraphs** or nodes under a central supervisor
- ✅ Pass shared state and **task assignments** between agents
- ✅ Know when supervisor vs single agent is justified
- ✅ Build a **research + coding** dual-worker supervisor demo

| | |
|---|---|
| **Prerequisites** | Phase 15 (Persistence & Streaming) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 65 minutes |

---

## Introduction — The Problem

One LLM with twelve tools confuses itself:

```
Agent picks wrong tool 40% of the time
Prompt exceeds context with all tool descriptions
No specialization — legal and code in one voice
```

**Supervisor architecture:** a **coordinator** LLM delegates to **specialists**, each with a narrow toolset and prompt.

```
                    ┌──────────────┐
User ──► Supervisor │ route task   │
                    └───┬──────┬───┘
                        │      │
                   Research  CodeGen
                        │      │
                        └──┬───┘
                           ▼
                      Final answer
```

### The Solution — Supervisor Node + Worker Nodes

Supervisor writes `next_worker` to state; conditional edges dispatch; workers return to supervisor until done.

---

## Part 1: State Design

```python
import os
from typing import TypedDict, Annotated, Literal
from operator import add
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)

Worker = Literal["researcher", "coder", "FINISH"]


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_worker: str
    task: str
    worker_notes: Annotated[list[str], add]
```

---

## Part 2: Supervisor Node

```python
def supervisor(state: SupervisorState) -> dict:
    prompt = [
        SystemMessage(content=(
            "You are a supervisor. Reply with ONE token: researcher, coder, or FINISH. "
            "Use researcher for facts, coder for Python snippets, FINISH when done."
        )),
        HumanMessage(content=state["task"]),
        *state.get("messages", []),
    ]
    decision = llm.invoke(prompt).content.strip().lower()
    if "coder" in decision:
        nxt = "coder"
    elif "finish" in decision:
        nxt = "FINISH"
    else:
        nxt = "researcher"
    return {
        "next_worker": nxt,
        "messages": [AIMessage(content=f"[supervisor] -> {nxt}")],
    }
```

---

## Part 3: Worker Nodes

```python
def researcher(state: SupervisorState) -> dict:
    prompt = [
        SystemMessage(content="Research assistant — bullet facts only."),
        HumanMessage(content=state["task"]),
    ]
    text = llm.invoke(prompt).content
    return {
        "worker_notes": [f"research: {text}"],
        "messages": [AIMessage(content=text)],
    }


def coder(state: SupervisorState) -> dict:
    prompt = [
        SystemMessage(content="Write minimal Python for the task."),
        HumanMessage(content=state["task"]),
    ]
    text = llm.invoke(prompt).content
    return {
        "worker_notes": [f"code: {text}"],
        "messages": [AIMessage(content=text)],
    }


def route_supervisor(state: SupervisorState) -> Worker:
    nxt = state.get("next_worker", "researcher")
    if nxt == "FINISH":
        return "FINISH"
    return nxt  # type: ignore[return-value]
```

---

## Part 4: Graph Topology

```python
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {"researcher": "researcher", "coder": "coder", "FINISH": END},
)
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")

supervisor_app = graph.compile()

if __name__ == "__main__":
    out = supervisor_app.invoke({
        "task": "Explain cosine similarity and show numpy code.",
    })
    print(out["worker_notes"])
```

Loop: supervisor → worker → supervisor until FINISH.

---

## Part 5: `langgraph-supervisor` / Prebuilt (Optional)

LangGraph ecosystem may ship supervisor helpers — prefer **manual graphs** first so interviews and debugging are clear, then adopt libraries for boilerplate.

---

## Part 6: Supervisor with Structured Routing

```python
from pydantic import BaseModel
from typing import Literal as Lit


class Route(BaseModel):
    next: Lit["researcher", "coder", "FINISH"]
    reason: str


router = llm.with_structured_output(Route)


def supervisor_structured(state: SupervisorState) -> dict:
    decision = router.invoke([
        HumanMessage(content=f"Task: {state['task']}\nNotes: {state.get('worker_notes')}")
    ])
    return {"next_worker": decision.next, "messages": [AIMessage(content=decision.reason)]}
```

Log `reason` to LangSmith for post-incident review.

### Part 7: Cost Control

Charge expensive `coder` only when task contains "code" or "python":

```python
def cheap_supervisor(state: SupervisorState) -> dict:
    if "code" not in state["task"].lower():
        return {"next_worker": "researcher"}
    return supervisor(state)
```

Hybrid rules + LLM reduces spend.

---

## Common Mistakes

### Mistake 1: Workers talk directly to user without supervisor

Breaks policy consistency — route outputs through supervisor synthesis node if needed.

### Mistake 2: No FINISH condition

Supervisor loops forever — cap turns in state.

### Mistake 3: Identical prompts for all workers

Defeats specialization — shrink tools per worker.

### Mistake 4: Huge shared message list

Pass `task` + summaries, not full chat history to every worker.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Turn counter in state | Hard stop |
| Structured supervisor output | Reliable routing |
| Worker-specific tools | Accuracy |
| Checkpoint threads | Multi-turn projects |
| Trace supervisor decisions | Audit trail |

---

## Interview Preparation

### Easy
**Q: What is the supervisor multi-agent pattern?**

> A central agent routes tasks to specialized workers and aggregates results, instead of one monolithic agent with all tools.

### Medium
**Q: Graph shape for supervisor?**

> START → supervisor → conditional edges to workers → each worker edges back to supervisor → FINISH to END when complete.

### Hard
**Q: Supervisor failure modes?**

> Wrong routing, infinite loops, context bloat, conflicting worker answers — mitigate with structured routing, turn limits, critic node, and explicit merge policy.

---

## Summary

| Piece | Role |
|-------|------|
| **Supervisor** | Plans / routes |
| **Workers** | Specialized execution |
| **Conditional edges** | Dynamic assignment |
| **Back-edges** | Iterative refinement |

---

## Exercises

1. Add turn limit 5 — force FINISH with partial notes.

2. Add `synthesizer` node before END to merge worker_notes.

3. Give coder a `@tool` stub and bind tools only in coder node.

4. Draw ASCII graph and compare to ReAct single loop.

---

## What's Next

[Chapter 16.2 — Swarm Architecture](chapter-69-swarm.md) explores peer agents with handoffs instead of a central supervisor.

---

> [← Previous: Streaming](../phase-15-langgraph-persistence/chapter-67-streaming-langgraph.md) | [Next: Swarm →](chapter-69-swarm.md)

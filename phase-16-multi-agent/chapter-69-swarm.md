# Chapter 16.2: Swarm Architecture

> **Phase 16 — Multi-Agent Systems** | [← Previous: Supervisor Architecture](chapter-68-supervisor.md) | [Next: Hierarchical Multi-Agent →](chapter-70-hierarchical.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Contrast **supervisor** (hub-and-spoke) vs **swarm** (peer handoffs)
- ✅ Implement agents that **transfer control** via state flags
- ✅ Use handoff messages to pass context between peers
- ✅ Recognize LangGraph **Swarm** / multi-agent prebuilt patterns
- ✅ Choose swarm when no single orchestrator fits

| | |
|---|---|
| **Prerequisites** | Chapter 16.1 |
| **Estimated Reading Time** | 28 minutes |
| **Estimated Coding Time** | 55 minutes |

---

## Introduction — The Problem

Strict supervisors bottleneck when experts must **collaborate laterally**:

```
Supervisor must know every skill upfront
Extra hop adds latency
Workers can't delegate sub-problems to each other
```

**Swarm:** agents are **peers**. Any agent can hand off to another with context.

```
User ──► Triage ──► Billing ──► (handoff) ──► Legal ──► END
              │                      ▲
              └──────► Tech ─────────┘
```

### The Solution — Active Agent + Handoff Edges

State tracks `active_agent`. Each node can set `handoff_to` for conditional routing.

---

## Part 1: Swarm State

```python
import os
from typing import TypedDict, Annotated, Literal
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

AgentName = Literal["triage", "billing", "tech", "legal", "END"]


class SwarmState(TypedDict):
    messages: Annotated[list, add_messages]
    active_agent: str
    handoff_to: str
    user_issue: str
```

---

## Part 2: Agent Factory

```python
def make_agent(name: str, peers: list[str]):
    peer_list = ", ".join(peers)

    def _run(state: SwarmState) -> dict:
        prompt = [
            SystemMessage(content=(
                f"You are the {name} agent. Help the user. "
                f"If another team must take over, reply with HANDOFF:<agent> "
                f"where agent is one of: {peer_list}, or HANDOFF:END if resolved."
            )),
            HumanMessage(content=state["user_issue"]),
            *state["messages"],
        ]
        text = llm.invoke(prompt).content
        handoff = "END"
        if "HANDOFF:" in text:
            _, target = text.split("HANDOFF:", 1)
            handoff = target.strip().split()[0].lower()
        return {
            "messages": [AIMessage(content=f"[{name}] {text}")],
            "handoff_to": handoff,
            "active_agent": name,
        }

    return _run


triage = make_agent("triage", ["billing", "tech", "legal"])
billing = make_agent("billing", ["legal", "tech"])
tech = make_agent("tech", ["billing"])
legal = make_agent("legal", [])
```

---

## Part 3: Router

```python
def route_swarm(state: SwarmState) -> AgentName:
    target = state.get("handoff_to", "END").lower()
    if target in ("billing", "tech", "legal", "triage"):
        return target  # type: ignore[return-value]
    return "END"
```

---

## Part 4: Graph

```python
graph = StateGraph(SwarmState)
for name, fn in [("triage", triage), ("billing", billing), ("tech", tech), ("legal", legal)]:
    graph.add_node(name, fn)

graph.add_edge(START, "triage")
for name in ("triage", "billing", "tech", "legal"):
    graph.add_conditional_edges(name, route_swarm)

swarm_app = graph.compile()

if __name__ == "__main__":
    result = swarm_app.invoke({
        "user_issue": "Charged twice and API webhook fails with 500.",
        "active_agent": "triage",
    })
    print(result["messages"][-1].content)
```

Each agent runs once per visit; handoff determines next node.

---

## Part 5: Swarm vs Supervisor

| | Supervisor | Swarm |
|---|------------|-------|
| Control | Central | Distributed |
| Best for | Clear task decomposition | Expert collaboration |
| Risk | Supervisor single point of failure | Routing loops |
| Observability | One decision log | Multiple handoffs |

---

## Part 6: LangGraph Swarm Package (Awareness)

Community / LangGraph docs describe multi-agent swarms with handoff tools — conceptually identical: **transfer function** sets next agent. Implement manually first, then adopt package utilities.

---

## Part 7: Visit Set Anti-Cycle

```python
class SwarmState(TypedDict):
    messages: Annotated[list, add_messages]
    active_agent: str
    handoff_to: str
    user_issue: str
    visited: Annotated[list[str], add]


def route_swarm_safe(state: SwarmState) -> AgentName:
    target = state.get("handoff_to", "END").lower()
    visits = state.get("visited", [])
    if visits.count(target) >= 2:
        return "END"
    return route_swarm(state)  # type: ignore
```

Append `active_agent` to `visited` each node exit.

### Part 8: Customer Support Swarm Storyboard

```
triage → billing (payment) → legal (contract clause) → END
triage → tech (500 error) → END
```

Document expected paths for QA scripts.

---

## Common Mistakes

### Mistake 1: Handoff without context in messages

Next agent must see prior `[billing]` messages in `messages` list.

### Mistake 2: Circular handoffs

billing → tech → billing forever — add visit counter.

### Mistake 3: Too many agents

Start with 3–4 peers max.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Structured HANDOFF syntax | Parse reliably |
| Cap hops | Cost control |
| Log active_agent | Traces |
| Default triage entry | Consistent UX |

---

## Interview Preparation

### Easy
**Q: Swarm vs supervisor?**

> Supervisor centralizes routing; swarm lets peer agents hand off control directly based on local decisions.

### Medium
**Q: How implement handoff in LangGraph?**

> Agents write handoff target to state; conditional edges route to the next agent node or END.

### Hard
**Q: Prevent runaway handoffs?**

> Max hops, detect cycles via visited set in state, escalate to human node.

---

## Summary

| Concept | Role |
|---------|------|
| **active_agent** | Current peer |
| **handoff_to** | Routing hint |
| **Peer nodes** | Equal specialists |
| **Conditional edges** | Dynamic transfers |

---

## Exercises

1. Add `hop_count` reducer — route to END when > 6.

2. Parse HANDOFF with regex instead of split.

3. Simulate user_issue that should visit billing then legal — print agent sequence.

4. Compare token use vs supervisor for same task (rough estimate).

---

## What's Next

[Chapter 16.3 — Hierarchical Multi-Agent](chapter-70-hierarchical.md) stacks managers over team leads and workers.

---

> [← Previous: Supervisor](chapter-68-supervisor.md) | [Next: Hierarchical →](chapter-70-hierarchical.md)

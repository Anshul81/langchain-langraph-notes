# Chapter 16.3: Hierarchical Multi-Agent

> **Phase 16 — Multi-Agent Systems** | [← Previous: Swarm Architecture](chapter-69-swarm.md) | [Next: Agent Handoff →](chapter-71-handoff.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Model **three-tier** hierarchies: executive → lead → worker
- ✅ Nest **subgraphs** as team modules
- ✅ Delegate subtasks down and **aggregate** results up
- ✅ Scope tools and prompts per hierarchy level
- ✅ Know when hierarchy beats flat supervisor

| | |
|---|---|
| **Prerequisites** | Chapters 16.1–16.2 |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction — The Problem

Flat supervisors don't scale to 20 specialists:

```
Supervisor prompt lists 20 roles → confusion
No team cohesion (SEO vs Paid Ads)
Middle management missing — exec wants summary not raw logs
```

**Hierarchical multi-agent:** executives route to **team leads**; leads manage **workers**.

```
CEO graph
 ├── Marketing lead subgraph
 │     ├── SEO worker
 │     └── Ads worker
 └── Engineering lead subgraph
       ├── Backend worker
       └── QA worker
```

### The Solution — Subgraphs + Layered Routing

Compile team graphs as **subgraphs** invoked from parent nodes, or simulate hierarchy with named tiers in one graph.

---

## Part 1: Team Subgraph (Workers)

```python
import os
from typing import TypedDict, Annotated
from operator import add
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)


class TeamState(TypedDict):
    brief: str
    deliverables: Annotated[list[str], add]


def seo_worker(state: TeamState) -> dict:
    text = llm.invoke([
        SystemMessage(content="SEO specialist — 3 keywords."),
        HumanMessage(content=state["brief"]),
    ]).content
    return {"deliverables": [f"SEO: {text}"]}


def ads_worker(state: TeamState) -> dict:
    text = llm.invoke([
        SystemMessage(content="Paid ads — 2 headline ideas."),
        HumanMessage(content=state["brief"]),
    ]).content
    return {"deliverables": [f"Ads: {text}"]}


def marketing_lead(state: TeamState) -> dict:
    joined = "\n".join(state.get("deliverables", []))
    summary = llm.invoke([
        SystemMessage(content="Marketing lead — synthesize team output."),
        HumanMessage(content=joined or state["brief"]),
    ]).content
    return {"deliverables": [f"Lead summary: {summary}"]}


team = StateGraph(TeamState)
team.add_node("seo", seo_worker)
team.add_node("ads", ads_worker)
team.add_node("lead", marketing_lead)
team.add_edge(START, "seo")
team.add_edge(START, "ads")
team.add_edge("seo", "lead")
team.add_edge("ads", "lead")
team.add_edge("lead", END)
marketing_team = team.compile()
```

Parallel workers → lead merge inside subgraph.

---

## Part 2: Executive Parent Graph

```python
class OrgState(TypedDict):
    goal: str
    department: str
    final_report: str
    packages: Annotated[list[str], add]


def ceo_route(state: OrgState) -> dict:
    goal = state["goal"].lower()
    dept = "marketing" if "launch" in goal or "campaign" in goal else "engineering"
    return {"department": dept}


def run_marketing(state: OrgState) -> dict:
    sub = marketing_team.invoke({"brief": state["goal"]})
    pkg = sub["deliverables"][-1]
    return {"packages": [pkg]}


def run_engineering(state: OrgState) -> dict:
    text = llm.invoke([
        SystemMessage(content="Engineering lead — high-level architecture bullets."),
        HumanMessage(content=state["goal"]),
    ]).content
    return {"packages": [f"Eng: {text}"]}


def ceo_synthesize(state: OrgState) -> dict:
    body = "\n".join(state["packages"])
    report = llm.invoke([
        SystemMessage(content="CEO — executive summary for board."),
        HumanMessage(content=body),
    ]).content
    return {"final_report": report}


from typing import Literal

def pick_dept(state: OrgState) -> Literal["marketing", "engineering"]:
    return state["department"]  # type: ignore[return-value]


org = StateGraph(OrgState)
org.add_node("ceo_route", ceo_route)
org.add_node("marketing", run_marketing)
org.add_node("engineering", run_engineering)
org.add_node("ceo_synthesize", ceo_synthesize)

org.add_edge(START, "ceo_route")
org.add_conditional_edges("ceo_route", pick_dept)
org.add_edge("marketing", "ceo_synthesize")
org.add_edge("engineering", "ceo_synthesize")
org.add_edge("ceo_synthesize", END)

org_app = org.compile()
```

---

## Part 3: Subgraph as Node (Native)

LangGraph allows adding compiled graphs as nodes:

```python
# org.add_node("marketing_team", marketing_team)
# Parent state may need adapter functions mapping OrgState ↔ TeamState
```

Adapters translate fields at boundaries — explicit is better than magic.

---

## Part 4: When to Use Hierarchy

```
USE hierarchy when:
├── 8+ specialists
├── Teams align to org structure
├── Executives need rolled-up summaries
└── Different compliance zones per department

AVOID when:
├── 2–3 tools total
├── Latency-sensitive simple Q&A
└── Team maintenance cost > benefit
```

---

## Part 5: State Adapter Pattern

```python
def marketing_adapter(state: OrgState) -> dict:
    sub = marketing_team.invoke({"brief": state["goal"]})
    return {"packages": sub["deliverables"][-1:]}
```

Adapters keep parent graph ignorant of team internals — stable contract.

### Part 6: Failure Isolation

If marketing subgraph fails, route to CEO with partial package:

```python
def run_marketing_safe(state: OrgState) -> dict:
    try:
        return run_marketing(state)
    except Exception as exc:
        return {"packages": [f"Marketing unavailable: {exc}"]}
```

Prevent one team from killing entire org run.

---

## Common Mistakes

### Mistake 1: Skipping lead synthesis

CEO gets raw worker noise.

### Mistake 2: Shared state schema everywhere

Subgraphs should own minimal local state.

### Mistake 3: Deep nesting without tests

Integration test each subgraph independently.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Subgraph per team | Modular deploy |
| Adapter nodes | Clean contracts |
| Parallel workers inside team | Speed |
| Executive summary node | Stakeholder-ready |
| Tool scoping per level | Security |

---

## Interview Preparation

### Easy
**Q: Hierarchical multi-agent?**

> Agents organized in tiers — executives delegate to team leads who coordinate workers — often via nested subgraphs.

### Medium
**Q: Subgraph benefits?**

> Encapsulation, independent testing, reuse across products, clearer ownership.

### Hard
**Q: Map enterprise support org to LangGraph.**

> Tier1 subgraph (triage bots) → Tier2 subgraph (domain experts) → escalation subgraph with HITL; state carries ticket id; checkpointer for thread.

---

## Summary

| Layer | Responsibility |
|-------|----------------|
| **Executive** | Goal routing + final report |
| **Lead** | Team synthesis |
| **Worker** | Narrow execution |
| **Subgraph** | Team boundary |

---

## Exercises

1. Add engineering subgraph with backend + QA workers mirroring marketing.

2. Write adapter if parent passes `goal` but team expects `brief`.

3. Force CEO to always request both departments — merge reports.

4. Diagram three-tier ASCII for your capstone idea.

---

## What's Next

[Chapter 16.4 — Agent Communication & Handoff](chapter-71-handoff.md) formalizes message protocols between agents.

---

> [← Previous: Swarm](chapter-69-swarm.md) | [Next: Handoff →](chapter-71-handoff.md)

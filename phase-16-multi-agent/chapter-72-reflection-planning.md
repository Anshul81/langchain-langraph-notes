# Chapter 16.5: Reflection & Planning Patterns

> **Phase 16 — Multi-Agent Systems** | [← Previous: Agent Handoff](chapter-71-handoff.md) | [Next: Multi-Agent Research Project →](chapter-73-multi-agent-research.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Implement **plan-and-execute** graphs (planner → worker → replanner)
- ✅ Build **reflection** loops (generate → critique → revise)
- ✅ Combine planning with multi-agent delegation
- ✅ Use structured plans stored in state
- ✅ Cap replans and reflection iterations for production

| | |
|---|---|
| **Prerequisites** | Chapters 14.2, 16.1–16.4 |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 65 minutes |

---

## Introduction — The Problem

Agents that "just act" drift on multi-step research:

```
Step 3 contradicts step 1
Missing citation for a claim
No global plan — redundant searches
```

**Planning** sets intent up front; **reflection** validates output before shipping.

```
PLAN-EXECUTE:
  plan → execute step i → replan? → ... → done

REFLECT:
  draft → critique → revise (loop)
```

### The Solution — Explicit Plan State + Reflection Nodes

Store `plan: list[str]` and `step_index` in state. Add critic nodes that gate progress.

---

## Part 1: Plan-and-Execute State

```python
import os
from typing import TypedDict, Annotated, Literal
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


class PlanState(TypedDict):
    objective: str
    plan: list[str]
    step_index: int
    results: Annotated[list[str], add]
    final_answer: str
```

---

## Part 2: Planner Node

```python
def planner(state: PlanState) -> dict:
    prompt = [
        SystemMessage(content=(
            "Return numbered plan lines only, 3-5 steps, one action per line."
        )),
        HumanMessage(content=state["objective"]),
    ]
    raw = llm.invoke(prompt).content
    lines = [ln.strip("0123456789.) ") for ln in raw.splitlines() if ln.strip()]
    return {"plan": lines[:5], "step_index": 0}


def executor(state: PlanState) -> dict:
    i = state["step_index"]
    plan = state.get("plan", [])
    if i >= len(plan):
        return {}
    step = plan[i]
    prompt = [
        SystemMessage(content="Execute ONE plan step. Be concise."),
        HumanMessage(content=f"Objective: {state['objective']}\nStep: {step}"),
    ]
    out = llm.invoke(prompt).content
    return {
        "results": [f"Step {i+1}: {out}"],
        "step_index": i + 1,
    }


def replanner(state: PlanState) -> dict:
    """Optional: adjust remaining steps based on results."""
    remaining = state["plan"][state["step_index"] :]
    if not remaining:
        return {}
    prompt = [
        SystemMessage(content="If results suffice, reply DONE. Else one new next step."),
        HumanMessage(content="\n".join(state["results"])),
    ]
    decision = llm.invoke(prompt).content
    if "DONE" in decision.upper():
        return {"step_index": len(state["plan"])}
    return {"plan": state["plan"][: state["step_index"]] + [decision.strip()]}


def synthesize(state: PlanState) -> dict:
    body = "\n".join(state["results"])
    ans = llm.invoke([
        SystemMessage(content="Final answer for user."),
        HumanMessage(content=body),
    ]).content
    return {"final_answer": ans}


def route_plan(state: PlanState) -> Literal["executor", "synthesize"]:
    if state["step_index"] < len(state.get("plan", [])):
        return "executor"
    return "synthesize"
```

Graph:

```
START → planner → executor ↔ (increment) → synthesize → END
```

```python
graph = StateGraph(PlanState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("synthesize", synthesize)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", route_plan)
graph.add_edge("synthesize", END)

plan_app = graph.compile()
```

---

## Part 3: Reflection Pattern

```python
class ReflectState(TypedDict):
    question: str
    answer: str
    critique: str
    passes: bool
    tries: int


def generate(state: ReflectState) -> dict:
    ans = llm.invoke([HumanMessage(content=state["question"])]).content
    return {"answer": ans, "tries": state.get("tries", 0) + 1}


def reflect(state: ReflectState) -> dict:
    raw = llm.invoke([
        SystemMessage(content="Reply PASS or FAIL|reason"),
        HumanMessage(content=f"Q: {state['question']}\nA: {state['answer']}"),
    ]).content
    ok = raw.strip().upper().startswith("PASS")
    critique = raw.split("|", 1)[-1].strip() if "|" in raw else raw
    return {"passes": ok, "critique": critique}


def route_reflect(state: ReflectState) -> Literal["generate", END]:
    if state.get("passes"):
        return END
    if state.get("tries", 0) >= 3:
        return END
    return "generate"
```

Reuse Chapter 14.2 loop mechanics.

---

## Part 4: Plan + Multi-Agent

```
Planner produces steps
Supervisor maps step → specialist worker
Reflection node validates each step before incrementing step_index
```

Keeps **global plan** in state while workers stay narrow.

---

## Part 5: Structured Plans (Pydantic)

```python
from pydantic import BaseModel


class Plan(BaseModel):
    steps: list[str]


planner_llm = llm.with_structured_output(Plan)


def planner_structured(state: PlanState) -> dict:
    plan: Plan = planner_llm.invoke([HumanMessage(content=state["objective"])])
    return {"plan": plan.steps, "step_index": 0}
```

---

## Part 6: Full Plan Graph with Replanner Hook

```python
graph.add_node("replanner", replanner)
graph.add_edge("executor", "replanner")
graph.add_conditional_edges("replanner", route_plan)
```

Insert when steps often fail mid-objective (open-ended research).

### Part 7: Plan State in Checkpoint

Resume long objectives:

```python
config = {"configurable": {"thread_id": "plan-42"}}
plan_app.invoke({"objective": "Launch checklist for EU"}, config=config)
# later
plan_app.invoke({"objective": "Launch checklist for EU"}, config=config)
```

`step_index` restores from checkpoint — continue execution without redoing completed steps.

---

## Common Mistakes

### Mistake 1: Plans too vague

"Research more" isn't executable — force concrete steps.

### Mistake 2: Never replanning

Stale plan after failed step — add replanner or reflection.

### Mistake 3: Reflection without feeding critique into generate

Same mistake as Chapter 14.2.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Structured plans | Parse reliability |
| Max steps / tries | Cost cap |
| Log results list | Audit |
| Separate planner model (optional) | Quality |
| Human review before final synthesize | Risky domains |

---

## Interview Preparation

### Easy
**Q: Plan-and-execute?**

> LLM creates step list, graph executes one step at a time, optionally replans, then synthesizes final output.

### Medium
**Q: Reflection vs replanning?**

> Reflection judges output quality of a step/answer; replanning changes future steps based on progress.

### Hard
**Q: Combine with supervisor?**

> Planner sets steps; supervisor assigns each step to workers; reflection gates step completion before index increment.

---

## Summary

| Pattern | Flow |
|---------|------|
| **Plan-execute** | plan → execute → synthesize |
| **Replan** | Adjust remaining steps |
| **Reflect** | generate → critique loop |
| **Structured plan** | Pydantic steps |

---

## Exercises

1. Wire `replanner` between executor iterations when step result contains "blocked".

2. Add reflection before `synthesize` in plan graph.

3. Compare step count with/without replanner on ambiguous objective.

4. Export final `results` as markdown report.

---

## What's Next

[Chapter 16.6 — Multi-Agent Research System](chapter-73-multi-agent-research.md) integrates routing, parallelism, planning, and reflection in one project.

---

> [← Previous: Handoff](chapter-71-handoff.md) | [Next: Research Project →](chapter-73-multi-agent-research.md)

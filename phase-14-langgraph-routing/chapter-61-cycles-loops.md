# Chapter 14.2: Cycles & Loops (Iterative Agents)

> **Phase 14 — LangGraph Routing & Control Flow** | [← Previous: Conditional Edges](chapter-60-conditional-edges.md) | [Next: Parallel Branches →](chapter-62-parallel-branches.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Build **cycles** safely with explicit exit conditions
- ✅ Track **iteration counts** and quality scores in state
- ✅ Implement a **self-correcting** draft → critique → revise loop
- ✅ Combine LLM critique with rule-based gates
- ✅ Avoid infinite loops in production graphs

| | |
|---|---|
| **Prerequisites** | Chapter 14.1 (Conditional Edges) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction — The Problem

One-shot LLM answers fail on complex tasks:

```
Draft answer (v1):  "Deploy on Tuesday"     ← missing rollback plan
Draft answer (v2):  still vague           ← no iteration control
Runaway loop:       47 revisions          ← no max_iterations
```

Agents need **loops with guardrails**: repeat until good enough **or** cap attempts.

```
START ──► draft ──► critique ──► good? ──► END
                      ▲            │
                      └── no ──────┘
                           (max 3)
```

### The Solution — Conditional Back-Edges

Point a conditional edge **back** to an earlier node. The router must expose a path to `END`.

---

## Part 1: Loop Mechanics

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END


def should_continue(state: LoopState) -> Literal["draft", END]:
    if state["score"] >= 0.8:
        return END
    if state["attempts"] >= 3:
        return END
    return "draft"
```

```
attempts / score drive exit — NEVER unbounded cycle
```

| Guard | Purpose |
|-------|---------|
| `max_attempts` | Token + latency ceiling |
| Quality threshold | Stop when "good enough" |
| Timeout (external) | Kill runaway jobs in API layer |
| Degrading fallback | Return best-so-far on exit |

---

## Part 2: Self-Correcting Agent State

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


class CorrectState(TypedDict):
    task: str
    draft: str
    critique: str
    score: float
    attempts: int
    history: Annotated[list[str], add]
```

---

## Part 3: Draft → Critique → Revise Nodes

```python
def draft_node(state: CorrectState) -> dict:
    attempt = state.get("attempts", 0) + 1
    feedback = state.get("critique", "")
    prompt = [
        SystemMessage(content="Write a short actionable answer."),
        HumanMessage(content=(
            f"Task: {state['task']}\n"
            f"Previous critique (if any): {feedback}\n"
            f"Attempt: {attempt}"
        )),
    ]
    text = llm.invoke(prompt).content
    return {
        "draft": text,
        "attempts": attempt,
        "history": [f"draft#{attempt}: {text[:80]}..."],
    }


def critique_node(state: CorrectState) -> dict:
    prompt = [
        SystemMessage(content=(
            "Score 0.0-1.0 completeness. Format: SCORE|one-line critique"
        )),
        HumanMessage(content=f"Task: {state['task']}\nDraft:\n{state['draft']}"),
    ]
    raw = llm.invoke(prompt).content
    try:
        score_str, critique = raw.split("|", 1)
        score = float(score_str.replace("SCORE", "").strip())
    except ValueError:
        score, critique = 0.5, "Could not parse critique."
    return {
        "score": max(0.0, min(1.0, score)),
        "critique": critique.strip(),
        "history": [f"critique: score={score}"],
    }


def route_loop(state: CorrectState) -> Literal["draft", END]:
    if state["score"] >= 0.85:
        return END
    if state["attempts"] >= 3:
        return END
    return "draft"
```

---

## Part 4: Compile the Cycle

```python
graph = StateGraph(CorrectState)
graph.add_node("draft", draft_node)
graph.add_node("critique", critique_node)

graph.add_edge(START, "draft")
graph.add_edge("draft", "critique")
graph.add_conditional_edges("critique", route_loop)

self_correct_app = graph.compile()

if __name__ == "__main__":
    result = self_correct_app.invoke({
        "task": "Explain zero-downtime deploy for a FastAPI service.",
    })
    print("Final draft:", result["draft"])
    print("Score:", result["score"], "Attempts:", result["attempts"])
    for line in result["history"]:
        print(" ", line)
```

ASCII topology:

```
    START
      │
      ▼
   draft ◄──────┐
      │         │
      ▼         │
  critique ─────┘ (if score low & attempts left)
      │
      └──► END (success or cap)
```

---

## Part 5: ReAct Loop as a Cycle (Conceptual)

Tool-calling agents are cycles:

```
agent ──► tools ──► agent ──► ... ──► END
         ▲          │
         └──────────┘
```

Your custom loops swap `tools` for `critique`, `validate`, or `retrieve`.

---

## Part 6: Best-Effort Exit Payload

When max attempts hit, surface status:

```python
def finalize_node(state: CorrectState) -> dict:
    if state["score"] < 0.85:
        return {
            "draft": state["draft"] + "\n\n[Note: best effort after 3 attempts]",
        }
    return {}


# Optional: route to finalize before END when attempts exhausted
```

---

## Part 7: Quality Gates Beyond LLM Score

Combine **LLM critique** with **deterministic validators**:

```python
def rule_gate(state: CorrectState) -> dict:
    draft = state.get("draft", "")
    penalties = []
    if len(draft.split()) < 40:
        penalties.append("too_short")
    if "rollback" not in draft.lower() and "deploy" in state["task"].lower():
        penalties.append("missing_rollback")
    cap = 0.4 if penalties else state.get("score", 0.0)
    return {
        "score": min(state.get("score", 0.0), cap),
        "history": [f"rules:{','.join(penalties) or 'ok'}"],
    }
```

Insert `rule_gate` between `critique` and `route_loop` so numeric LLM scores cannot bypass safety checks.

### Part 8: Streaming Loop Progress

```python
for mode in ("updates", "values"):
    print(f"\n--- stream_mode={mode} ---")
    for chunk in self_correct_app.stream(
        {"task": "Write runbook for database failover."},
        stream_mode=mode,
    ):
        print(chunk)
```

Use `updates` in CLI tools; use `values` when UI needs cumulative draft text.

### Part 9: Comparing to `create_react_agent`

| Self-correct loop | ReAct agent |
|-------------------|-------------|
| Explicit critique node | Implicit in next LLM turn |
| Fixed graph topology | Tool loop only |
| Easy max-attempt policy | Needs middleware |

Choose self-correct when **quality gates** are first-class requirements, not emergent behavior.

---

## Common Mistakes

### Mistake 1: Cycle with no END path

Every loop router must return `END` under some condition.

### Mistake 2: Resetting attempts incorrectly

Increment in **one** node only — usually the loop head (`draft`).

### Mistake 3: Critique without feeding back

If router sends you to `draft` but prompt ignores `critique`, loops waste tokens.

### Mistake 4: Using float equality

```python
# ❌ score == 0.8
# ✅ score >= 0.8
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Log each attempt in state | Debug + UX progress |
| Cap iterations (3–5 typical) | Cost control |
| Store best draft | Return max score version |
| Separate critique model (optional) | Cheaper/faster critic |
| Stream loop events | Show "revision 2/3" in UI |
| Test router function alone | Fast unit tests |

---

## Interview Preparation

### Easy
**Q: How do you create a loop in LangGraph?**

> Add a conditional edge from a later node back to an earlier node. The routing function must include a termination condition returning END or equivalent.

### Medium
**Q: Design a self-correcting writing agent.**

> State holds draft, critique, score, attempts. Nodes: draft (uses critique), critique (scores draft), router exits on score threshold or max attempts, else routes to draft.

### Hard
**Q: How prevent infinite loops if the LLM never gives a high score?**

> Hard cap on attempts, optional timeout at orchestration layer, fallback node returning best-scoring draft from history, and monitoring alerts on graphs that always hit attempt caps.

---

## Summary

| Concept | Usage |
|---------|--------|
| **Back-edge** | Conditional return to prior node |
| **Termination** | Score threshold + max attempts |
| **State** | attempts, critique, history |
| **Self-correct** | draft ↔ critique cycle |

---

## Exercises

1. Track `best_draft` / `best_score` across iterations; return best on exit.

2. Add a rule gate: draft must contain the word "rollback" for deploy tasks or score capped at 0.5.

3. Stream `history` updates with `self_correct_app.stream()` and print after each node.

4. Change cap to 5 and measure average token use — document tradeoff in 3 sentences.

---

## What's Next

[Chapter 14.3 — Parallel Branches](chapter-62-parallel-branches.md) runs independent nodes concurrently and merges results with reducers.

---

> [← Previous: Conditional Edges](chapter-60-conditional-edges.md) | [Next: Parallel Branches →](chapter-62-parallel-branches.md)

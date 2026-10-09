# Chapter 16.5: Reflection & Planning Patterns

> **Phase 16 — Multi-Agent Systems** | [← Previous: Agent Communication & Handoff](chapter-71-handoff.md) | [Next: Multi-Agent Research System →](chapter-73-multi-agent-research.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Build a **Plan-and-Execute** graph: planner → executor (tool agent) → replan / done
- ✅ Keep the plan as a **structured list of steps** in state
- ✅ Build a **Reflection loop**: draft → critique → revise, with a max-iteration cap
- ✅ Let a planner **delegate a step to a specialist tool-agent** (light multi-agent)
- ✅ Apply **production caps**: `MAX_REPLANS`, `MAX_REFLECTS`, step limits, recursion limits

| | |
|---|---|
| **Prerequisites** | Chapters 16.1–16.4, Phase 10 (`create_react_agent`), Phase 14 (conditional edges) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 55 minutes |

---

## Introduction — The Problem

A single ReAct agent decides **one step at a time**. Two failure modes appear on bigger tasks:

```
Task: "Which is denser, Japan or India, and by what factor?"

❌ Wanders:  searches India, searches again, forgets Japan, answers from memory
❌ One-shot: writes a confident answer once — numbers are wrong, nobody checks
```

Two classic fixes:

| Pattern | Idea | Fixes |
|---------|------|-------|
| **Plan-and-Execute** | Write a plan first, execute step by step, replan if needed | Wandering, forgotten sub-goals |
| **Reflection** | Draft, critique, revise | Unchecked first drafts |

### The Solution — Two Small Graphs

```
PLAN-AND-EXECUTE
START → planner → executor ──(steps left)──▶ executor   (tool agent per step)
                     │
                (plan empty)
                     ▼
                  replan ──▶ done ────────▶ END
                     ├─────▶ more steps ──▶ executor
                     └─────▶ cap hit ─────▶ finalize → END

REFLECTION
START → draft → critique ──▶ score ≥ threshold, or max reflects ──▶ END
                   ▲  └────▶ otherwise ──▶ revise ─┐
                   └───────────────────────────────┘
```

---

## Part 0: Shared Setup and Real Tools

All agents in this chapter use **real `@tool`s** over a tiny in-memory fact base plus a safe calculator.

```python
import ast
import json
import operator as op
import os
import re
import sys
from operator import add
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

FACTS = {  # approximate, for teaching
    "japan": {"population_millions": 124.5, "area_km2": 377975, "capital": "Tokyo"},
    "india": {"population_millions": 1428.6, "area_km2": 3287263, "capital": "New Delhi"},
    "france": {"population_millions": 68.2, "area_km2": 643801, "capital": "Paris"},
}


@tool
def lookup_fact(topic: str) -> str:
    """Look up facts (population, area, capital) about a country. Input: country name."""
    key = topic.strip().lower()
    for name, data in FACTS.items():
        if name in key or key in name:
            return json.dumps({"country": name, **data})
    return f"NO_RESULT: nothing for '{topic}'. Known topics: {', '.join(FACTS)}"


_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.USub: op.neg}


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
    """Evaluate arithmetic like '1428.6 * 1000000 / 3287263'. Supports + - * / and parentheses."""
    try:
        return str(round(_eval(ast.parse(expression, mode="eval").body), 4))
    except Exception as exc:
        return f"ERROR: {exc}"


@tool
def text_stats(text: str) -> str:
    """Count words and sentences in a text."""
    return json.dumps({"words": len(text.split()), "sentences": len(re.findall(r"[.!?]+", text))})


def run_agent(agent, task: str) -> str:
    """Invoke a tool agent with a hard cap on tool loops; return its final text."""
    result = agent.invoke({"messages": [("user", task)]}, {"recursion_limit": 12})
    return result["messages"][-1].content
```

---

## Part 1: Plan-and-Execute

### Theory

1. **Planner** — one LLM call produces an ordered list of steps (structured output).
2. **Executor** — runs **one** step using a **specialist tool-agent** chosen by the step's `owner`.
3. **Replan** — when the plan is empty, an LLM checks the results: *done* (answer) or *more steps*.
4. **Cap** — after `MAX_REPLANS`, stop and write a best-effort answer.

This is multi-agent in a light way: the planner **delegates** each step to either a `facts` agent or a `math` agent — each is a `create_react_agent` with only its own tool.

### 1a. Plan schema and specialists

```python
MAX_PLAN_STEPS = 4
MAX_REPLANS = 2


class Step(BaseModel):
    owner: Literal["facts", "math"] = Field(description="facts = look up data; math = compute")
    task: str = Field(description="One self-contained instruction naming any numbers it needs")


class Plan(BaseModel):
    steps: list[Step] = Field(description="2-4 ordered steps")


class Decision(BaseModel):
    done: bool = Field(description="True if the results already answer the question")
    answer: str = Field(default="", description="Final answer, only if done")
    new_steps: list[Step] = Field(default_factory=list, description="Remaining steps if not done")


planner_llm = llm.with_structured_output(Plan, method="function_calling")
replanner_llm = llm.with_structured_output(Decision, method="function_calling")

SPECIALISTS = {
    "facts": create_react_agent(llm, [lookup_fact], prompt=(
        "You are the Facts specialist. ALWAYS call lookup_fact; never answer from memory. "
        "Report numbers exactly. If the tool returns NO_RESULT, say so plainly.")),
    "math": create_react_agent(llm, [calculator], prompt=(
        "You are the Math specialist. Use calculator for EVERY computation. "
        "Reply with the final number and the expression you used.")),
}
```

### 1b. State and nodes

```python
class PlanState(TypedDict):
    question: str
    plan: list[dict]                          # remaining steps (structured)
    past_steps: Annotated[list[dict], add]    # step + result, append-only
    answer: str
    replans: int


def format_results(state: PlanState) -> str:
    return "\n".join(f"- [{p['owner']}] {p['task']} => {p['result']}" for p in state["past_steps"]) \
        or "(nothing done yet)"


def planner(state: PlanState) -> dict:
    plan = planner_llm.invoke([
        SystemMessage(content="Break the question into 2-4 ordered steps. Use 'facts' to look up "
                             "data and 'math' to compute. Each task must be self-contained."),
        HumanMessage(content=state["question"]),
    ])
    return {"plan": [s.model_dump() for s in plan.steps[:MAX_PLAN_STEPS]]}


def executor(state: PlanState) -> dict:
    step, rest = state["plan"][0], state["plan"][1:]
    task = f"Results so far:\n{format_results(state)}\n\nYour task: {step['task']}"
    result = run_agent(SPECIALISTS[step["owner"]], task)         # delegate to a tool agent
    return {"plan": rest, "past_steps": [{**step, "result": result}]}


def replan(state: PlanState) -> dict:
    d = replanner_llm.invoke([
        SystemMessage(content="Review progress. If the results answer the question, set done=true "
                             "and answer using ONLY the results. Otherwise set done=false and list "
                             "the remaining new_steps (max 3). Failed lookups need a different approach."),
        HumanMessage(content=f"Question: {state['question']}\nResults:\n{format_results(state)}"),
    ])
    if d.done:
        return {"answer": d.answer}
    if state["replans"] >= MAX_REPLANS:               # production cap
        return {"plan": []}
    return {"plan": [s.model_dump() for s in d.new_steps[:3]], "replans": state["replans"] + 1}


def finalize(state: PlanState) -> dict:
    text = llm.invoke([
        SystemMessage(content="Replan limit reached. Give the best partial answer from the results "
                             "and clearly state what is missing."),
        HumanMessage(content=f"Question: {state['question']}\nResults:\n{format_results(state)}"),
    ]).content
    return {"answer": "(partial) " + text}


def after_executor(state: PlanState) -> str:
    return "executor" if state["plan"] else "replan"


def after_replan(state: PlanState) -> str:
    if state.get("answer"):
        return END
    return "executor" if state["plan"] else "finalize"
```

### 1c. Assemble and run

```python
g = StateGraph(PlanState)
for name, fn in [("planner", planner), ("executor", executor), ("replan", replan), ("finalize", finalize)]:
    g.add_node(name, fn)
g.add_edge(START, "planner")
g.add_edge("planner", "executor")
g.add_conditional_edges("executor", after_executor, ["executor", "replan"])
g.add_conditional_edges("replan", after_replan, ["executor", "finalize", END])
g.add_edge("finalize", END)
plan_app = g.compile()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")        # Windows consoles: model text may contain ≈, ² etc.
    q = "Which is more densely populated, Japan or India, and by what factor?"
    for update in plan_app.stream(
        {"question": q, "plan": [], "past_steps": [], "answer": "", "replans": 0},
        stream_mode="updates",
    ):
        for node, out in update.items():
            if node == "planner":
                print("PLAN:", [f"{s['owner']}: {s['task']}" for s in out["plan"]])
            elif node == "executor":
                print("STEP:", out["past_steps"][0]["owner"], "->", out["past_steps"][0]["result"][:90])
            else:
                print(node.upper(), out)
```

### What good output looks like

```
PLAN: ['facts: Get population and area of Japan', 'facts: Get population and area of India',
       'math: Compute density (people/km²) for each and the ratio']
STEP: facts -> Japan: population 124.5 million, area 377,975 km².
STEP: facts -> India: population 1428.6 million, area 3,287,263 km².
STEP: math -> Japan ≈ 329.4/km², India ≈ 434.6/km²; ratio ≈ 1.32 (124.5e6/377975 …)
REPLAN {'answer': 'India is denser: ≈ 435 vs ≈ 329 people/km², about 1.3×.'}
```

The exact step split varies, but you should see **facts steps before the math step**, and `REPLAN` ending with an answer. Ask about a country **not** in `FACTS` and you will see `NO_RESULT`, extra replans, then `FINALIZE` with a `(partial)` answer — the cap working.

---

## Part 2: Reflection Loop

### Theory

```
draft ──▶ critique ──▶ score ≥ THRESHOLD? ── yes ──▶ END
              ▲              │ no
              │              ▼ iteration < MAX_REFLECTS?
              └──────── revise            no → END (ship best effort)
```

- The **writer** is a tool agent (`lookup_fact`) so the draft is grounded.
- The **critic** is a *different* tool agent (`lookup_fact` + `text_stats`) — it verifies numbers and length with tools rather than guessing. It ends with `SCORE: n`.
- Two stop conditions: **good enough** or **out of budget**. Never loop on score alone.

### 2a. State and agents

```python
THRESHOLD = 8        # accept at this score or higher
MAX_REFLECTS = 2     # at most 2 revisions


class ReflectState(TypedDict):
    topic: str
    draft: str
    critique: str
    score: int
    iteration: int
    history: Annotated[list[dict], add]       # one entry per critique, for observability


writer_agent = create_react_agent(llm, [lookup_fact], prompt=(
    "You are a concise writer for beginners. Ground EVERY number with lookup_fact. "
    "Output ONLY the finished text."))

critic_agent = create_react_agent(llm, [lookup_fact, text_stats], prompt=(
    "You are a strict reviewer. Verify every number with lookup_fact and measure length with "
    "text_stats. Rubric: numbers correct (density = population / area), both countries mentioned, "
    "under 80 words, beginner-friendly. First line exactly 'SCORE: <1-10>', then up to 3 bullet issues."))


def parse_score(text: str) -> int:
    m = re.search(r"SCORE:\s*(\d+)", text)
    return min(int(m.group(1)), 10) if m else 0     # unparseable → treat as failing, forces a revise
```

### 2b. Nodes and routing

```python
def draft(state: ReflectState) -> dict:
    return {"draft": run_agent(writer_agent, f"Write about: {state['topic']}"), "iteration": 0}


def critique(state: ReflectState) -> dict:
    review = run_agent(critic_agent, f"Topic: {state['topic']}\n\nDraft to review:\n{state['draft']}")
    score = parse_score(review)
    return {"critique": review, "score": score,
            "history": [{"iteration": state["iteration"], "score": score}]}


def revise(state: ReflectState) -> dict:
    task = (f"Topic: {state['topic']}\n\nPrevious draft:\n{state['draft']}\n\n"
            f"Reviewer feedback:\n{state['critique']}\n\nRewrite, fixing every issue.")
    return {"draft": run_agent(writer_agent, task), "iteration": state["iteration"] + 1}


def after_critique(state: ReflectState) -> str:
    if state["score"] >= THRESHOLD:
        return END
    if state["iteration"] >= MAX_REFLECTS:
        return END                               # budget exhausted: return best effort
    return "revise"


r = StateGraph(ReflectState)
r.add_node("draft", draft)
r.add_node("critique", critique)
r.add_node("revise", revise)
r.add_conditional_edges(START, lambda s: "critique" if s.get("draft") else "draft", ["draft", "critique"])
r.add_edge("draft", "critique")
r.add_conditional_edges("critique", after_critique, ["revise", END])
r.add_edge("revise", "critique")
reflect_app = r.compile()


if __name__ == "__main__":
    out = reflect_app.invoke({
        "topic": "Explain to a beginner whether Japan or India is more densely populated, with numbers.",
        "draft": "", "critique": "", "score": 0, "iteration": 0, "history": [],
    })
    print("SCORES:", out["history"])
    print("FINAL DRAFT:", out["draft"])
```

The `START` conditional edge is a small trick: if a `draft` is already provided, skip drafting and **only critique/improve it** — handy for Part 3.

### What good output looks like

```
SCORES: [{'iteration': 0, 'score': 6}, {'iteration': 1, 'score': 9}]
FINAL DRAFT: Japan has about 124.5 million people in 377,975 km² (≈329 per km²). India has ≈1,428.6
million in 3,287,263 km² (≈435 per km²). India is denser — about 1.3 times Japan.
```

Scores rising across iterations (6 → 9) is the point. If scores stay flat, the critic's feedback is vague — fix the **rubric**, not the loop.

---

## Part 3: Combining Plan, Delegate, and Reflect

```python
def plan_then_polish(question: str) -> str:
    # 1) planner delegates each step to a specialist tool-agent
    answer = plan_app.invoke({"question": question, "plan": [], "past_steps": [],
                              "answer": "", "replans": 0})["answer"]
    # 2) reflection loop critiques and improves the finished answer (draft is pre-filled)
    polished = reflect_app.invoke({"topic": question, "draft": answer, "critique": "",
                                   "score": 0, "iteration": 0, "history": []})
    return polished["draft"]
```

| Layer | Pattern | Who does the work |
|-------|---------|-------------------|
| Decompose | Plan | Planner LLM (structured output) |
| Act | Delegate | `facts` / `math` tool agents |
| Improve | Reflect | Writer + critic tool agents |

### Production caps

| Cap | In this chapter | Why |
|-----|-----------------|-----|
| `MAX_PLAN_STEPS` | 4 | A planner can emit 20 steps |
| `MAX_REPLANS` | 2 | Prevents plan → fail → plan loops |
| `MAX_REFLECTS` | 2 | Critics can always find "one more issue" |
| `recursion_limit` | 12 per agent | Bounds each tool loop |
| `THRESHOLD` | 8/10 | Defines "good enough" |

Budget the worst case **before** shipping. Reflection: 1 draft + 3 critiques + 2 revisions = 6 agent runs. Plan-and-Execute: 1 planner call + up to `MAX_PLAN_STEPS + 3×MAX_REPLANS` agent runs + `MAX_REPLANS + 1` replan calls + 1 finalize.

---

## Common Mistakes

### Mistake 1: Uncapped replanning
A failing tool makes the replanner try forever. Always keep a `MAX_REPLANS` and a `finalize` exit.

### Mistake 2: Reflection with no stop except score
Models sometimes never score above 7. Combine **threshold + max iterations**.

### Mistake 3: Plan as free text
`"1. do this 2. do that"` is hard to execute reliably. Store steps as structured data (`owner`, `task`).

### Mistake 4: Critic that can't verify
A critic with no tools just rephrases opinions. Give it tools (`lookup_fact`, `text_stats`) and an explicit rubric.

### Mistake 5: Executor lacks earlier results
Each step must receive prior results (see `format_results`), otherwise `math` never gets the numbers.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Structured `Plan` via `with_structured_output` | Reliable parsing |
| One specialist per `owner` with minimal tools | Fewer wrong tool calls |
| Replan only when the plan is empty | Cheaper than replanning every step |
| Keep `history` of scores | See whether reflection is actually helping |
| Return best effort at the cap | Users prefer partial answers over errors |
| Different prompts for writer and critic | Avoids self-agreement |

---

## Interview Preparation

### Easy
**Q: Plan-and-Execute vs ReAct?**
> ReAct chooses the next action step by step; Plan-and-Execute writes the whole plan first, executes it, and only replans when needed — better for multi-part tasks and cheaper per step.

### Medium
**Q: How do you prevent an infinite reflection loop?**
> Two exits: score ≥ threshold, or iteration ≥ `MAX_REFLECTS`. Track `iteration` in state and return the best draft when the budget is spent.

**Q: Where does multi-agent fit into planning?**
> The planner assigns each step an `owner`; the executor routes that step to a specialist tool-agent with only the tools it needs.

### Hard
**Q: How would you make reflection reliable in production?**
> A rubric-based critic with tools to verify facts, a numeric score plus parseable format, caps on iterations and tokens, logging of score history, and a fallback when the score can't be parsed.

---

## Summary

| Pattern | Loop | Stops when |
|---------|------|-----------|
| Plan-and-Execute | planner → executor ↺ → replan | `done`, or `MAX_REPLANS` → finalize |
| Reflection | draft → critique → revise ↺ | score ≥ threshold, or `MAX_REFLECTS` |
| Delegation | step `owner` → specialist | Step finishes |
| Caps | steps, replans, reflects, recursion | Always set |

---

## Hands-on: Change the Knobs and Observe

1. **Lower the threshold.** Set `THRESHOLD = 5`. How many revisions now? Print `out["history"]`.
2. **Change attempts.** Set `MAX_REFLECTS = 0`, then `4`. Compare final drafts and total scores.
3. **Raise the bar.** Set `THRESHOLD = 10`. Does the loop stop because of the cap?
4. **Force a replan.** Ask the plan graph about **Brazil** (not in `FACTS`). Watch `NO_RESULT`, replans, and the `(partial)` answer. Then add `"brazil"` to `FACTS` and re-run.
5. **Stretch:** add a third specialist `"words"` (tool: `text_stats`) and allow it in `Step.owner`.

---

## What's Next

[Chapter 16.6 — Multi-Agent Research System (Project)](chapter-73-multi-agent-research.md) assembles planning, parallel tool-using specialists, a writer, and a critic loop into one complete project.

---

> [← Previous: Handoff](chapter-71-handoff.md) | [Next: Multi-Agent Research System →](chapter-73-multi-agent-research.md)

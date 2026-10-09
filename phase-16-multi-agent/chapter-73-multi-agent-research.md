# Chapter 16.6: Multi-Agent Research System (Project)

> **Phase 16 — Multi-Agent Systems** | [← Previous: Reflection & Planning](chapter-72-reflection-planning.md) | [Next: FastAPI Integration →](../phase-17-production/chapter-74-fastapi-integration.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Assemble a **complete multi-agent research system** in ~200 lines of LangGraph
- ✅ Combine **planner → parallel tool-using specialists → writer → critic loop**
- ✅ Give specialists **real `@tool`s** (Wikipedia when available, a keyword-searched local knowledge base, a calculator)
- ✅ Separate **shared state** from **private agent state**
- ✅ Add **checkpointing (`MemorySaver`)** and **streamed progress**
- ✅ Handle the two classic failures: **bad plans** and **empty tool results**

| | |
|---|---|
| **Prerequisites** | Chapters 16.1–16.5, Phase 14 (`Send`, parallel branches) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 75 minutes |

---

## Introduction — The Problem

One agent asked a broad research question produces a long, vague, partly invented answer:

```
"Compare solar PV and onshore wind: how does each work, and which produces
 more energy per year from 100 MW of installed capacity?"

Single agent → mixes facts and math, guesses capacity factors, no one checks the sum.
```

### The Solution — A Small Research Team

```
                         ┌──────────────┐
  question ─────────────▶│   PLANNER    │  2–3 sub-questions, each tagged with a specialist
                         └──────┬───────┘
                   Send × N (parallel)
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │ FACTS agent  │  │NUMBERS agent │  │ FACTS agent  │   each = create_react_agent
      │ wiki_search  │  │ kb_search    │  │ ...          │   with REAL tools; its tool
      │ kb_search    │  │ calculator   │  │              │   chatter stays PRIVATE
      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
             └─────────────────┼─────────────────┘
                               ▼   findings (shared, append-only)
                         ┌──────────────┐
                         │    WRITER    │◀──────────────┐
                         └──────┬───────┘               │ revise
                                ▼                       │ (max 2)
                         ┌──────────────┐  score < 8    │
                         │    CRITIC    │───────────────┘
                         └──────┬───────┘
                                ▼ approved or out of revisions
                               END          (MemorySaver checkpoints every step)
```

| Node | Type | Tools |
|------|------|-------|
| `planner` | LLM, structured output | — |
| `specialist` ×N (parallel) | **Tool agents** (`create_react_agent`) | `wiki_search`, `kb_search`, `calculator` |
| `writer` | LLM synthesis from evidence only | — (deliberately: it may only use findings) |
| `critic` | LLM, structured output | — |

---

## Part 1: Real Tools

Three tools. Nothing here returns a canned string pretending to be a search: `kb_search` actually tokenizes and scores a knowledge base, `wiki_search` actually calls Wikipedia when enabled, and `calculator` actually evaluates expressions.

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
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

# ── Curated local knowledge base (approximate figures, for teaching) ──
KB = [
    {"title": "Solar photovoltaic power",
     "text": "Solar PV panels convert sunlight directly into electricity using semiconductor cells "
             "(the photovoltaic effect). They only produce in daylight. Typical utility-scale "
             "solar capacity factor: about 25%."},
    {"title": "Onshore wind power",
     "text": "Wind turbines use rotor blades to turn the kinetic energy of wind into electricity "
             "through a generator. Output varies with wind speed. Typical onshore wind capacity "
             "factor: about 35% (offshore about 45%)."},
    {"title": "Capacity factor",
     "text": "Capacity factor is actual energy produced divided by the maximum possible if the plant "
             "ran at full capacity all year. A year has 8,760 hours, so annual energy (MWh) = "
             "capacity (MW) x capacity factor x 8,760."},
    {"title": "Intermittency and storage",
     "text": "Solar and wind are intermittent. Grids balance them with batteries, pumped hydro, "
             "demand response and wider transmission."},
    {"title": "Nuclear power",
     "text": "Nuclear plants split uranium atoms to make steam. They run at roughly 90% capacity factor."},
]
STOP = {"the", "and", "how", "does", "what", "for", "from", "with", "per", "are", "which", "each", "using"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in STOP}


def _kb_lookup(query: str, top_k: int = 2) -> str:
    q = _tokens(query)
    scored = sorted(((len(q & _tokens(e["title"] + " " + e["text"])), e) for e in KB),
                    key=lambda x: x[0], reverse=True)
    hits = [f"[{e['title']}] {e['text']}" for score, e in scored[:top_k] if score > 0]
    return "\n".join(hits) if hits else f"NO_RESULT: knowledge base has nothing for '{query}'"


@tool
def kb_search(query: str) -> str:
    """Search the curated local knowledge base (solar, wind, capacity factor, storage, nuclear).
    Use short keyword queries like 'solar capacity factor'."""
    return _kb_lookup(query)


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a short summary of a concept. Falls back to the local knowledge
    base when Wikipedia is disabled or unreachable."""
    if os.getenv("USE_WIKIPEDIA", "0") == "1":
        try:
            import wikipedia  # pip install wikipedia
            return "[wikipedia] " + wikipedia.summary(query, sentences=3, auto_suggest=False)
        except Exception as exc:        # ImportError, network, disambiguation, page missing
            return f"[wikipedia failed: {type(exc).__name__}; using local KB]\n" + _kb_lookup(query)
    return "[local KB]\n" + _kb_lookup(query)


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
    """Evaluate arithmetic such as '100 * 0.25 * 8760'. Supports + - * / and parentheses."""
    try:
        return str(round(_eval(ast.parse(expression, mode="eval").body), 4))
    except Exception as exc:
        return f"ERROR: {exc}"
```

Quick offline check — tools work without any LLM:

```python
print(kb_search.invoke({"query": "solar capacity factor"}))   # real scored lookup
print(calculator.invoke({"expression": "100 * 0.25 * 8760"}))  # 219000.0
print(kb_search.invoke({"query": "quantum gravity"}))          # NO_RESULT: ...
```

---

## Part 2: Specialist Agents

Each specialist is a ReAct agent with **only the tools it needs** (less confusion, less risk).

```python
SPECIALISTS = {
    "facts": create_react_agent(llm, [wiki_search, kb_search], prompt=(
        "You are the FACTS specialist. Call wiki_search or kb_search before answering; never rely on "
        "memory. Answer in at most 3 sentences and name the tool you used. If every tool returns "
        "NO_RESULT, reply exactly: NO EVIDENCE FOUND")),
    "numbers": create_react_agent(llm, [kb_search, calculator], prompt=(
        "You are the NUMBERS specialist. Get input figures with kb_search (e.g. capacity factors), then "
        "compute EVERY result with calculator. Show the expressions and final numbers. If the needed "
        "figures are missing, reply exactly: NO EVIDENCE FOUND")),
}
```

---

## Part 3: State (Shared vs Private) and Planner

```python
MAX_SUBQUESTIONS = 3
MAX_REVISES = 2
APPROVE_SCORE = 8


class ResearchState(TypedDict):
    # ── SHARED: every node can read these ──
    question: str
    subquestions: list[dict]
    findings: Annotated[list[dict], add]      # parallel specialists append here
    report: str
    critique: str
    score: int
    revisions: int


class SubQuestion(BaseModel):
    question: str = Field(description="A self-contained sub-question")
    specialist: Literal["facts", "numbers"] = Field(
        description="facts = how/what/why background; numbers = needs figures + calculation")


class ResearchPlan(BaseModel):
    subquestions: list[SubQuestion] = Field(description="2-3 sub-questions")


planner_llm = llm.with_structured_output(ResearchPlan, method="function_calling")


def planner(state: ResearchState) -> dict:
    plan = planner_llm.invoke([
        SystemMessage(content="Decompose the research question into 2-3 non-overlapping sub-questions. "
                             "Assign each to 'facts' (concepts) or 'numbers' (figures + calculation)."),
        HumanMessage(content=state["question"]),
    ])
    subs = [s.model_dump() for s in plan.subquestions[:MAX_SUBQUESTIONS]]
    if not subs:                                   # failure mode: empty/bad plan → safe fallback
        subs = [{"question": state["question"], "specialist": "facts"}]
    return {"subquestions": subs}
```

### Shared vs private state in this project

| State | Scope | Lives in | Why |
|-------|-------|----------|-----|
| `question`, `subquestions` | **Shared** | `ResearchState` | Everyone needs the goal |
| `findings` | **Shared** (append-only reducer) | `ResearchState` | The *only* output of specialists |
| `report`, `critique`, `score`, `revisions` | **Shared** | `ResearchState` | Writer/critic loop |
| Specialist's tool calls, raw tool outputs, ReAct messages | **Private** | Inside `agent.invoke(...)` — discarded after the node | Keeps the shared state small; writer sees distilled answers |
| `sub` + `idx` | **Private input** | `Send` payload to one specialist run | Each parallel worker sees only its own question |

Principle from Chapter 16.4: **share conclusions, keep working notes private.**

---

## Part 4: Parallel Specialists

```python
class SpecialistInput(TypedDict):
    sub: dict
    idx: int


def fan_out(state: ResearchState) -> list[Send]:
    return [Send("specialist", {"sub": s, "idx": i}) for i, s in enumerate(state["subquestions"])]


def specialist(state: SpecialistInput) -> dict:
    sub = state["sub"]
    try:
        result = SPECIALISTS[sub["specialist"]].invoke(
            {"messages": [("user", sub["question"])]}, {"recursion_limit": 12})
        msgs = result["messages"]
        answer = msgs[-1].content
        evidence = [m for m in msgs if isinstance(m, ToolMessage)]        # private tool outputs
        tools_used = [m.name for m in evidence]
        # empty if: no tool was called (ungrounded), all tools found nothing, or agent gave up
        empty = (not evidence) or all("NO_RESULT" in m.content for m in evidence) \
            or "NO EVIDENCE FOUND" in answer
    except Exception as exc:                                              # one bad worker ≠ crashed run
        answer, tools_used, empty = f"AGENT ERROR: {type(exc).__name__}", [], True
    return {"findings": [{"id": state["idx"] + 1, "question": sub["question"],
                          "specialist": sub["specialist"], "answer": answer,
                          "tools_used": tools_used, "empty": empty}]}
```

Parallel results arrive in any order, so each finding carries an `id`; the writer sorts by it.

---

## Part 5: Writer and Critic

```python
def format_evidence(state: ResearchState) -> str:
    rows = []
    for f in sorted(state["findings"], key=lambda f: f["id"]):
        status = "NO EVIDENCE" if f["empty"] else "ok"
        rows.append(f"[{f['id']}] Q: {f['question']}\n    A: {f['answer']}\n"
                    f"    (specialist={f['specialist']}, tools={f['tools_used'] or 'none'}, {status})")
    return "\n".join(rows)


def writer(state: ResearchState) -> dict:
    revising = bool(state.get("critique"))
    human = f"Question: {state['question']}\n\nEvidence:\n{format_evidence(state)}"
    if revising:
        human += (f"\n\nPrevious report:\n{state['report']}\n\nCritic issues:\n{state['critique']}\n"
                  "Rewrite the report and fix every issue.")
    report = llm.invoke([
        SystemMessage(content="Write a research brief (max 200 words) using ONLY the evidence. Cite "
                             "sources as [1], [2]. For any NO EVIDENCE item write 'Not found: <question>' "
                             "— never guess. End with a one-sentence conclusion."),
        HumanMessage(content=human),
    ]).content
    return {"report": report, "revisions": state["revisions"] + (1 if revising else 0)}


class Critique(BaseModel):
    score: int = Field(ge=1, le=10, description="1-10 quality score")
    issues: list[str] = Field(description="Concrete problems to fix; empty if none")


critic_llm = llm.with_structured_output(Critique, method="function_calling")


def critic(state: ResearchState) -> dict:
    c = critic_llm.invoke([
        SystemMessage(content="Strict reviewer. Check: (1) every claim traces to the evidence, (2) no "
                             "invented numbers, (3) NO EVIDENCE items are acknowledged, (4) the report "
                             "answers the question. Score 1-10."),
        HumanMessage(content=f"Question: {state['question']}\n\nEvidence:\n{format_evidence(state)}"
                             f"\n\nReport:\n{state['report']}"),
    ])
    # never leave critique empty when failing, or the writer would not count a revision
    return {"score": c.score, "critique": "; ".join(c.issues) or "Tighten wording and citations."}


def after_critic(state: ResearchState) -> str:
    if state["score"] >= APPROVE_SCORE or state["revisions"] >= MAX_REVISES:
        return END
    return "writer"
```

The writer **cannot call tools** — it can only use evidence the specialists gathered. That is what makes "no invented numbers" enforceable by the critic.

---

## Part 6: Assemble, Stream, Checkpoint

```python
PAUSE = os.getenv("PAUSE", "0") == "1"      # optional: stop before writer to inspect findings

g = StateGraph(ResearchState)
g.add_node("planner", planner)
g.add_node("specialist", specialist)
g.add_node("writer", writer)
g.add_node("critic", critic)
g.add_edge(START, "planner")
g.add_conditional_edges("planner", fan_out, ["specialist"])
g.add_edge("specialist", "writer")           # waits for ALL parallel specialists
g.add_edge("writer", "critic")
g.add_conditional_edges("critic", after_critic, ["writer", END])
app = g.compile(checkpointer=MemorySaver(), interrupt_before=["writer"] if PAUSE else [])


def run(question: str, thread_id: str = "research-1") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"question": question, "subquestions": [], "findings": [], "report": "",
              "critique": "", "score": 0, "revisions": 0}
    while True:
        for update in app.stream(inputs, config, stream_mode="updates"):
            for node, out in update.items():
                if node.startswith("__"):
                    continue
                if node == "planner":
                    print(f">> planner     -> {[(s['specialist'], s['question'][:45]) for s in out['subquestions']]}")
                elif node == "specialist":
                    f = out["findings"][0]
                    print(f">> specialist  -> #{f['id']} {f['specialist']} tools={f['tools_used']} empty={f['empty']}")
                elif node == "writer":
                    print(f">> writer      -> revision #{out['revisions']}")
                elif node == "critic":
                    print(f">> critic      -> score={out['score']}")
        if not app.get_state(config).next:          # nothing left to run -> finished
            break
        print("|| paused before writer (state saved) - resuming from checkpoint...")
        inputs = None                               # None = continue from saved checkpoint
    return app.get_state(config).values


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")        # Windows consoles: model text may contain ≈, —, etc.
    final = run("Compare solar PV and onshore wind: how does each generate electricity, and which "
                "produces more energy per year from 100 MW of installed capacity?")
    print("\n=== REPORT ===\n" + final["report"])
    print(f"\nscore={final['score']}  revisions={final['revisions']}")
```

---

## How to Run

```bash
pip install langchain-openai langgraph python-dotenv pydantic
# optional real Wikipedia:   pip install wikipedia

# .env
LITELLM_PROXY_API_KEY=...
LITELLM_PROXY_API_BASE=...
LITE_LLM_MODEL=gpt-4o-mini

python research_system.py                 # offline: local KB + calculator
USE_WIKIPEDIA=1 python research_system.py # Wikipedia first, KB fallback
PAUSE=1 python research_system.py         # pause before EVERY writer run, then auto-resume from checkpoint
```

(PowerShell: `$env:PAUSE="1"; python research_system.py`)

**Resume:** `MemorySaver` lives in process memory, so `PAUSE=1` demonstrates resume *within* a run. For crash recovery across processes, swap in `SqliteSaver`/`PostgresSaver` and call `app.stream(None, config)` with the same `thread_id`.

---

## What Good Output Looks Like

```
>> planner     -> [('facts', 'How does solar PV generate electricity?'), ('facts', 'How does onshore wind ...'), ('numbers', 'Using typical capacity factors, how much ...')]
>> specialist  -> #1 facts tools=['wiki_search'] empty=False
>> specialist  -> #3 numbers tools=['kb_search', 'kb_search', 'calculator', 'calculator'] empty=False
>> specialist  -> #2 facts tools=['wiki_search'] empty=False
>> writer      -> revision #0
>> critic      -> score=9

=== REPORT ===
Solar PV converts sunlight into electricity with semiconductor cells [1]; wind turbines convert wind's
kinetic energy via rotor and generator [2]. With typical capacity factors of ~25% (solar) and ~35%
(onshore wind), 100 MW yields ≈219,000 MWh/year for solar versus ≈306,600 MWh/year for wind [3].
Conclusion: onshore wind produces about 40% more energy per year from the same capacity.
```

**Checks that it worked:**
- Planner produced **2–3** sub-questions with a mix of specialists
- Each specialist line shows **tools actually called** (a `numbers` finding with no `calculator` is a red flag)
- Math is correct: 100 × 0.25 × 8760 = **219,000**; 100 × 0.35 × 8760 = **306,600**
- Order of specialist lines may vary (they run in parallel); writer always comes after all of them
- `revisions ≤ 2`

---

## Failure Modes

| Failure | Symptom | Defense in this project |
|---------|---------|------------------------|
| **Bad plan** (empty / 8 vague questions) | No findings, or huge cost | `[:MAX_SUBQUESTIONS]` cap; empty-plan fallback to the original question |
| **Overlapping sub-questions** | Duplicate findings | Planner prompt says *non-overlapping*; critic can flag redundancy |
| **Empty tool result** | `NO_RESULT` | Finding flagged `empty=True`; specialist told to reply `NO EVIDENCE FOUND`; writer writes "Not found:" instead of guessing |
| **Ungrounded answer** (agent skipped tools) | `tools=[]` | `empty` is also true when no tool ran |
| **Specialist crash / loop** | Exception, `GraphRecursionError` | `recursion_limit=12` + `try/except` → error finding, run continues |
| **Critic never satisfied** | Endless rewrites | `MAX_REVISES = 2`, then ship best effort |
| **Critic gives no issues but low score** | Writer doesn't count a revision | Fallback issue text keeps the counter moving |
| **Wikipedia down** | Exception in tool | `wiki_search` falls back to local KB |

Try it: ask *"What does geothermal drilling cost in Iceland?"* → the KB has nothing on it → expect `empty=True` and a "Not found" line in the report (with `USE_WIKIPEDIA=1` the facts agent may legitimately find something — that is the real tool working).

---

## Common Mistakes

### Mistake 1: Specialists with every tool
Wrong tool choices rise with tool count. Two or three tools per specialist.

### Mistake 2: Putting raw tool output into shared state
Findings should be **short answers**, not 3 KB of search results.

### Mistake 3: Writer with free access to its own knowledge
Then citations are decoration. Force "use ONLY the evidence."

### Mistake 4: No order key on parallel results
Reducer order isn't guaranteed. Carry an `id` and sort.

### Mistake 5: Treating "no result" as success
Always detect and surface empties; silent gaps become hallucinations downstream.

---

## Interview Talking Points (Project)

- **Architecture in one breath:** *"Planner decomposes into sub-questions, parallel tool-using specialists gather evidence via `Send`, a writer synthesizes strictly from findings, and a critic loop with a revision cap improves it. MemorySaver checkpoints it."*
- **Why parallel specialists?** Independent sub-questions → latency ≈ slowest one, not the sum.
- **Why a writer without tools?** It makes grounding enforceable: anything not in `findings` is a critic-detectable violation.
- **Shared vs private:** specialists' ReAct traces stay private; only distilled findings are shared.
- **Failure handling:** caps (subquestions, revisions, recursion), empty-result flagging, per-worker try/except.
- **What you'd add for production:** persistent checkpointer, real search API, tracing (LangSmith), cost budget, tests with fixed `findings`.
- **Trade-off you can discuss:** more agents = more cost/latency; each added role must fix a measured failure.

---

## Summary

| Piece | Implementation |
|-------|---------------|
| Planner | `with_structured_output(ResearchPlan)` + cap + fallback |
| Specialists | `create_react_agent` with `wiki_search` / `kb_search` / `calculator`, run in parallel via `Send` |
| Merger/Writer | LLM using only `findings`, sorted by `id` |
| Critic loop | Structured score, `MAX_REVISES = 2` |
| Checkpoint | `MemorySaver` + `thread_id`, optional `interrupt_before` |
| Progress | `app.stream(..., stream_mode="updates")` printing node names |

---

## Hands-on: Swap in a New Specialist Tool

Add a tool that estimates how many homes the energy could power, and give it to the `numbers` specialist.

```python
@tool
def homes_powered(annual_mwh: float) -> str:
    """Estimate households powered by annual energy in MWh (assumes 10.5 MWh per household per year)."""
    return f"{round(annual_mwh / 10.5)} households (assuming 10.5 MWh/household/year)"


SPECIALISTS["numbers"] = create_react_agent(llm, [kb_search, calculator, homes_powered], prompt=(
    "You are the NUMBERS specialist. Get figures with kb_search, compute with calculator, and use "
    "homes_powered to translate MWh into households. If figures are missing reply: NO EVIDENCE FOUND"))
```

Re-run with: *"...which produces more energy from 100 MW, and how many homes does each power?"* Confirm `homes_powered` appears in a specialist's `tools=[...]` line.

**Then try:** add a brand-new `"economics"` specialist — add it to `SubQuestion.specialist`'s `Literal`, to `SPECIALISTS`, and describe it in the planner prompt. Which three edits were needed? (Answer: schema, registry, planner prompt — the graph itself did not change.)

## Challenge: Supervisor-Style Final Synthesizer

Add a `synthesizer` node **after** the critic approves that behaves like a supervisor (Chapter 16.1):

1. It reads `report` + `findings` and returns structured `Verdict(action: Literal["finish", "research_more"], extra_question: str, specialist: str)`.
2. If `research_more`, send **one** extra `Send("specialist", ...)` then go back to `writer` (cap with a new `extra_rounds` counter, max 1).
3. If `finish`, produce the final answer with an "Evidence quality" line (how many findings were `empty`).

Hints: reuse `fan_out`'s `Send` pattern; keep the idx unique (`len(findings)`); add `extra_rounds` to `ResearchState`; watch your caps — the supervisor is the easiest place to create an infinite loop.

---

## What's Next

Phase 17 takes agents to production. [Chapter 17.1 — FastAPI Integration](../phase-17-production/chapter-74-fastapi-integration.md) wraps graphs like this one in an API.

---

> [← Previous: Reflection & Planning](chapter-72-reflection-planning.md) | [Next: FastAPI Integration →](../phase-17-production/chapter-74-fastapi-integration.md)

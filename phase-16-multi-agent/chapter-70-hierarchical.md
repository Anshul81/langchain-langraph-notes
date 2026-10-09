# Chapter 16.3: Hierarchical Multi-Agent

> **Phase 16 — Multi-Agent Systems** | [← Previous: Swarm Architecture](chapter-69-swarm.md) | [Next: Agent Communication & Handoff →](chapter-71-handoff.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Explain **why a flat supervisor breaks down** as the team grows
- ✅ Organize agents as **Executive → Team Lead → Workers** (two levels is enough)
- ✅ Build a **team subgraph**, compile it, and call it from a parent node
- ✅ Keep team internals **isolated** and **aggregate results upward** as short reports
- ✅ Build a **Marketing team** (two tool-using workers + a lead who merges) under a **CEO router**
- ✅ Decide when hierarchy **helps** and when it is **overkill**

| | |
|---|---|
| **Prerequisites** | Chapters 16.1–16.2, Phase 13 (StateGraph, subgraph basics) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction — The Problem

Your Chapter 16.1 supervisor handles 2–4 workers nicely. Now the product grows:

```
Flat supervisor, 12 workers:

  Supervisor prompt = 12 worker descriptions
  Supervisor context = 12 workers' reports piling up
  Routing accuracy   ↓      Cost per decision   ↑      Debugging   ↓↓
  "Should this go to seo_writer, copywriter, brand_designer,
   market_analyst, or social_manager?"   ← the supervisor is now the overloaded agent
```

You've recreated the single-agent problem one level up. People solve this in organizations with **management layers**: a CEO doesn't pick which copywriter writes the slogan; the CEO asks the *Marketing team*, and Marketing's lead organizes the work.

### The Solution — Teams as Subgraphs

```
                        ┌────────────────────┐
        Task ─────────► │   CEO (executive)  │ ◄───────────────────┐
                        │ routes to a TEAM   │                     │
                        └───┬────────────┬───┘                     │
                 next=      │            │     next=               │ short
                 marketing  │            │     engineering         │ team report
                            ▼            ▼                         │
        ┌──────────────────────────┐   ┌───────────────┐           │
        │ MARKETING TEAM (subgraph)│   │ ENGINEERING   │───────────┤
        │                          │   │ TEAM (stub)   │           │
        │ market_analyst           │   └───────────────┘           │
        │      ↓                   │                               │
        │ copywriter               │                               │
        │      ↓                   │                               │
        │ team_lead (merge)        │───────────────────────────────┘
        └──────────────────────────┘
   CEO sees: 2 teams.  Marketing lead sees: 2 workers.  Nobody sees 12.
```

Three ideas:

1. **Each level routes among few options** (2–4), so routing stays accurate.
2. **A team is a compiled graph** used as a single node by its parent.
3. **Reports travel upward, compressed.** The CEO gets a 4-line report, not every tool call.

---

## Part 1: Why Flat Fails, and How Deep to Go

| | Flat (12 workers) | Hierarchical (3 teams × 4 workers) |
|---|---|---|
| Options the top router sees | 12 | 3 |
| Options each lead sees | — | 4 |
| Context at the top | All 12 reports | 3 compressed reports |
| Change one team's workers | Re-test the whole supervisor | Re-test one team |
| Parallel work | Awkward | Teams are independent units |
| LLM calls | Fewer | More (extra lead layer) |

### How deep?

```
✅ Level 1: Executive routes to teams
✅ Level 2: Team lead coordinates workers
❌ Level 3+: "VP of sub-teams of sub-teams" — latency, cost, lost information
```

Every extra layer adds LLM calls and **loses detail** (each summary drops information). Two levels cover almost every real system. If you want a third level, first ask whether your teams can simply be split into two smaller teams at level 1.

---

## Part 2: The Building Block — A Team Subgraph

A team is an ordinary `StateGraph` with its **own state**, compiled with `.compile()`. The parent never sees the team's internals.

Setup and tools first. Workers are **real tool-using agents** (same recipe as 16.1):

```python
import os
from operator import add
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

MARKET = {
    "water bottle": (
        "Reusable bottle market: $10.4B, growing ~6% per year. Top segments: "
        "students and daily commuters. Competitor price range: $25-$35. "
        "Top buying driver: keeps drinks cold 24h."
    ),
    "headphones": "Headphone market: $70B, growing ~9% per year. Top driver: battery life.",
}

BANNED_WORDS = {"best", "cheapest", "guaranteed", "#1"}


@tool
def get_market_stats(category: str) -> str:
    """Get market size, growth, segments, and competitor pricing for a product category."""
    hits = [text for key, text in MARKET.items() if key in category.lower()]
    return " ".join(hits) or "No market data found for that category."


@tool
def check_slogan(slogan: str) -> str:
    """Validate a slogan: max 6 words and no banned claims. Returns OK or what to fix."""
    words = slogan.replace("-", " ").split()
    problems = []
    if len(words) > 6:
        problems.append(f"too long ({len(words)} words, max 6)")
    banned = [w for w in words if w.lower().strip(".,!\"'") in BANNED_WORDS]
    if banned:
        problems.append(f"banned words: {banned}")
    return "OK" if not problems else "FIX: " + "; ".join(problems)
```

The two worker agents, each narrow, each with one tool:

```python
market_analyst_agent = create_react_agent(
    llm,
    tools=[get_market_stats],
    prompt=("You are a market analyst. Use get_market_stats for the product category, "
            "then report market size, target segments, and competitor pricing in 3 short lines."),
)

copywriter_agent = create_react_agent(
    llm,
    tools=[check_slogan],
    prompt=("You are a copywriter. Write ONE slogan using the market notes. You MUST validate it "
            "with check_slogan and revise until it returns OK. Reply with the final slogan "
            "and a one-line rationale."),
)
```

### The team's own state and nodes

```python
class MarketingState(TypedDict):
    brief: str            # input from the CEO
    market_notes: str     # written by market_analyst
    slogan_notes: str     # written by copywriter
    report: str           # written by team_lead  (the ONLY thing sent upward)


def market_analyst(state: MarketingState) -> dict:
    out = market_analyst_agent.invoke(
        {"messages": [HumanMessage(content=f"Product brief: {state['brief']}")]}
    )
    return {"market_notes": out["messages"][-1].content}


def copywriter(state: MarketingState) -> dict:
    out = copywriter_agent.invoke({"messages": [HumanMessage(content=(
        f"Product brief: {state['brief']}\n\nMarket notes:\n{state['market_notes']}"
    ))]})
    return {"slogan_notes": out["messages"][-1].content}


def team_lead(state: MarketingState) -> dict:
    """The lead MERGES worker outputs into one short report for the executive."""
    msg = llm.invoke([
        SystemMessage(content=(
            "You are the Marketing Lead. Merge the notes into a 4-line report with exactly "
            "these labels: Positioning, Target, Slogan, Risk. Be concrete and brief."
        )),
        HumanMessage(content=(
            f"BRIEF: {state['brief']}\n\nMARKET NOTES:\n{state['market_notes']}\n\n"
            f"COPY NOTES:\n{state['slogan_notes']}"
        )),
    ])
    return {"report": msg.content}
```

### Compile the team

```python
tb = StateGraph(MarketingState)
tb.add_node("market_analyst", market_analyst)
tb.add_node("copywriter", copywriter)
tb.add_node("team_lead", team_lead)

tb.add_edge(START, "market_analyst")
tb.add_edge("market_analyst", "copywriter")   # copywriter needs the analyst's notes
tb.add_edge("copywriter", "team_lead")
tb.add_edge("team_lead", END)

marketing_graph = tb.compile()
```

Here the lead's coordination is a **fixed pipeline** (analyst → copywriter → merge) because that order never changes. That's deliberate: don't spend an LLM routing call where the order is known. If a team's order *does* vary, make the lead a mini-supervisor using the Chapter 16.1 pattern — a team is just a graph, so any pattern fits inside.

You can test the team alone, before any CEO exists:

```python
if __name__ == "__main__":
    r = marketing_graph.invoke({"brief": "An eco-friendly insulated water bottle for students"})
    print(r["report"])
```

That independent testability is a major reason to use teams.

---

## Part 3: The CEO — Routing Among Teams

Now the top level. Its state is **different** from the team's state and holds only what an executive needs:

```python
class CompanyState(TypedDict):
    task: str
    results: Annotated[list[str], add]    # one compressed report per team
    next: str
    reason: str
    turns: int
    final: str
```

### Team nodes: calling a subgraph from a parent node

The parent node translates **parent state → team input**, calls the compiled team, and translates **team output → parent update**:

```python
def marketing_team_node(state: CompanyState) -> dict:
    try:
        out = marketing_graph.invoke({"brief": state["task"]})
        return {"results": [f"[marketing]\n{out['report']}"]}      # only the report goes up
    except Exception as e:
        return {"results": [f"[marketing] FAILED: {type(e).__name__}: {e}"]}


def engineering_team_node(state: CompanyState) -> dict:
    # STUB for now — you'll replace this with a real subgraph in the exercise.
    return {"results": [
        "[engineering] (stub) Feasible with existing stainless-steel suppliers; "
        "prototype estimate: 8 weeks."
    ]}
```

Why a wrapper function instead of `add_node("marketing", marketing_graph)`? Because the schemas differ. The wrapper is the **boundary**: the CEO's state stays small, and the team's `market_notes`, `slogan_notes`, and tool calls never leak upward. (When parent and subgraph *share* state keys, you can pass the compiled graph straight to `add_node` — simpler, but less isolation.)

> Team failures are handled exactly as in 16.1: catch the exception and report `FAILED` upward, so the CEO can decide what to do.

### The CEO router

The CEO knows **teams**, not workers. It has no idea the Marketing team contains an analyst and a copywriter — that's the point.

```python
class Route(BaseModel):
    next: Literal["marketing", "engineering", "FINISH"] = Field(
        description="Which team acts next, or FINISH when all needed teams have reported."
    )
    reason: str = Field(description="One sentence explaining the choice.")


router_llm = llm.with_structured_output(Route)

CEO_PROMPT = """You are the CEO. You delegate to TEAMS (never to individuals):
- marketing: positioning, target audience, slogans, market research
- engineering: technical feasibility, build estimates, materials

Read the TASK and the TEAM REPORTS so far. Pick the next team, or FINISH when every
part of the task is covered. Do not call a team that has already reported.
If a team FAILED, retry at most once, then FINISH and state the gap."""

MAX_TURNS = 4


def ceo(state: CompanyState) -> dict:
    turns = state.get("turns", 0)
    if turns >= MAX_TURNS:                                   # loop cap (see 16.1)
        return {"next": "FINISH", "reason": "turn limit reached", "turns": turns}
    reports = "\n\n".join(state["results"]) or "(none yet)"
    route = router_llm.invoke([
        SystemMessage(content=CEO_PROMPT),
        HumanMessage(content=f"TASK: {state['task']}\n\nTEAM REPORTS:\n{reports}"),
    ])
    return {"next": route.next, "reason": route.reason, "turns": turns + 1}
```

---

## Part 4: Aggregate Upward and Run It

When the CEO says `FINISH`, a final node merges the team reports into one executive answer:

```python
def executive_summary(state: CompanyState) -> dict:
    reports = "\n\n".join(state["results"]) or "(no reports)"
    msg = llm.invoke([
        SystemMessage(content=(
            "Write a concise executive summary (max 6 lines) answering the task using ONLY "
            "the team reports. Mention any team that failed or was missing."
        )),
        HumanMessage(content=f"TASK: {state['task']}\n\nTEAM REPORTS:\n{reports}"),
    ])
    return {"final": msg.content}


builder = StateGraph(CompanyState)
builder.add_node("ceo", ceo)
builder.add_node("marketing", marketing_team_node)
builder.add_node("engineering", engineering_team_node)
builder.add_node("summary", executive_summary)

builder.add_edge(START, "ceo")
builder.add_conditional_edges(
    "ceo",
    lambda s: s["next"],
    {"marketing": "marketing", "engineering": "engineering", "FINISH": "summary"},
)
builder.add_edge("marketing", "ceo")        # teams report BACK to the CEO
builder.add_edge("engineering", "ceo")
builder.add_edge("summary", END)

company = builder.compile()


def run(task: str):
    init = {"task": task, "results": [], "turns": 0}
    for step in company.stream(init, stream_mode="updates", config={"recursion_limit": 25}):
        for node, upd in step.items():
            if node == "ceo":
                print(f"👔 CEO → {upd['next']:<12} ({upd['reason']})")
            elif node in ("marketing", "engineering"):
                print(f"🏢 {upd['results'][-1]}\n")
            elif node == "summary":
                print(f"✅ SUMMARY:\n{upd['final']}")


if __name__ == "__main__":
    run("We're launching an eco-friendly insulated water bottle for students. "
        "Give me positioning and a slogan, plus a quick engineering feasibility view.")
```

> **All code in Parts 2–4 goes in one file** (e.g. `hierarchy_demo.py`), in order. Needs `pip install langgraph langchain-openai python-dotenv pydantic`.

**Typical output** (wording varies):

```
👔 CEO → marketing    (Positioning and slogan are a marketing task.)
🏢 [marketing]
Positioning: Premium-feel reusable bottle priced under the $25-$35 competitor band.
Target: Students and daily commuters in a $10.4B, ~6%-growth market.
Slogan: "Cold all day. Kind to the planet."
Risk: Crowded market; differentiate on 24h cold retention.

👔 CEO → engineering  (Feasibility is still uncovered.)
🏢 [engineering] (stub) Feasible with existing stainless-steel suppliers; prototype estimate: 8 weeks.

👔 CEO → FINISH       (Both marketing and engineering have reported.)
✅ SUMMARY:
Position the bottle as a ... Slogan: "Cold all day. Kind to the planet." Engineering ...
```

### What the CEO never saw

Inside the Marketing team there were: a market analyst agent (LLM + tool call), a copywriter agent (LLM + `check_slogan` loop, perhaps several revisions), and the lead's merge call. **None of that** is in `CompanyState` — only the 4-line report is. That's the context win hierarchy provides.

To watch inside a team while debugging, run and stream `marketing_graph` on its own (as in Part 2). If you instead add a compiled team directly as a node, `stream(..., subgraphs=True)` also surfaces the inner steps.

---

## Part 5: When Hierarchy Helps — and When It's Overkill

| Situation | Verdict |
|---|---|
| 2–4 workers, simple routing | ❌ **Overkill** — use a flat supervisor (16.1) |
| 6+ workers with clear groupings (marketing / eng / legal) | ✅ **Helps** |
| Workers in one group need to share context a lot | ✅ **Helps** — shared inside a team, hidden from others |
| Different teams owned by different people/repos | ✅ **Helps** — independent subgraphs |
| Conversation passes between specialists dynamically | ➡️ Consider a swarm (16.2) |
| You're adding a level to "organize" rather than fix a measured problem | ❌ **Overkill** |
| Latency-sensitive path (chat reply in < 2s) | ❌ Layers add calls |

**Cost check:** the demo above made roughly: 1 CEO call per decision (3), plus marketing's 2 agent loops + 1 lead call, plus engineering (stub, 0), plus 1 summary. Compare with what a single well-prompted agent would cost *before* committing to this structure.

---

## Common Mistakes

| Mistake | Why it hurts | Fix |
|---|---|---|
| 4–5 levels deep | Latency, cost, information loss at each summary | Max 2 levels: executive → team |
| CEO prompt lists individual workers | Recreates the flat-supervisor overload | CEO knows **teams** only |
| Team returns its whole transcript | Parent context bloats; defeats the point | Team lead returns a **short report** |
| Sharing one giant state across all levels | Everyone sees everything; hard to evolve teams | Separate team state; map at the boundary |
| Workers are bare `llm.invoke(prompt)` | No real capability | `@tool` + `create_react_agent` |
| Using an LLM router inside a team with fixed order | Wasted calls and nondeterminism | Fixed edges when order is known |
| No failure handling at the team boundary | One team's error kills the whole run | `try/except` → `FAILED` report upward |
| No loop cap at the top router | CEO re-calls the same team forever | `MAX_TURNS` + `recursion_limit` |
| Building hierarchy "for scale" at 3 workers | Complexity with no benefit | Add layers only after a flat design measurably fails |

---

## Best Practices

1. **Earn the hierarchy.** Start flat; split when routing accuracy or prompt size degrades.
2. **Two levels.** Executive → team lead → workers.
3. **Teams are subgraphs with their own state.** Test each team alone.
4. **Compress at every boundary.** The lead's merge node is where detail is deliberately dropped.
5. **Use fixed edges for fixed pipelines**; use a mini-supervisor only where order truly varies.
6. **The top router knows teams, not people.** Describe each team in one line.
7. **Handle failures at the boundary** and surface them in the final summary.
8. **Cap loops at each routing level.**

---

## Interview Questions & Answers

### 🟢 Easy

**Q1: What is hierarchical multi-agent architecture?**
Agents organized in layers: a top-level executive routes to team leads, and each team coordinates its own workers. Results flow back up as summaries.

**Q2: Why not just use one supervisor with many workers?**
A flat supervisor must hold every worker's description and every report in context. As workers grow, routing accuracy drops and cost rises. Teams let each router choose among a few options.

**Q3: How do you use a team as part of a larger graph?**
Compile the team's `StateGraph` and call it from a node in the parent graph (or add the compiled graph directly as a node when state keys are shared).

### 🟡 Medium

**Q4: Why wrap the subgraph in a function instead of adding it directly as a node?**
When parent and team schemas differ, the wrapper maps parent state to team input and team output to a parent update. It also gives you a place for error handling and ensures only a short report crosses the boundary.

**Q5: How deep should the hierarchy be?**
Typically two levels. Every layer adds LLM calls and compresses (loses) information. Before adding a third level, consider splitting a team into two level-1 teams.

**Q6: What does the team lead do?**
It coordinates workers (fixed pipeline, or a mini-supervisor if order varies) and **merges** their outputs into one short report for the executive.

### 🔴 Hard

**Q7: How does hierarchy affect context management and cost?**
Context: positive — the top level sees only compressed reports, and workers see only their team's context. Cost: negative — extra routing/merge calls. The net win appears at scale, when a flat router would be overloaded or accuracy suffers.

**Q8: How would you speed up a hierarchical system?**
Run independent teams in parallel (Phase 14 fan-out with the `Send` API or parallel edges, merging reports with a reducer), skip LLM routing where the order is deterministic, use cheaper models for routing/merge, and keep reports short.

**Q9: How do you debug a wrong final answer?**
Trace top-down: check the CEO's `reason` per decision, then the team report, then (testing the team graph alone) each worker's tool calls. Stream with the team invoked in isolation or enable tracing (e.g. LangSmith) to see the nested runs.

---

## Summary

- A flat supervisor overloads at scale; **hierarchy** gives each router only 2–4 options.
- Use **two levels**: Executive (CEO router) → Team (lead + workers).
- A **team is a compiled subgraph with its own state**, called from a parent node via a wrapper that maps state in and out.
- Workers are **real tool-using agents**; the lead **merges** into a short report; reports **aggregate upward**.
- The CEO knows **teams, not workers**; failures are caught at the boundary; loops are capped.
- Hierarchy helps with **6+ workers in natural groups**; it's **overkill** for small teams.

---

## Hands-On Exercise

**Replace the Engineering stub with a real team subgraph** (or add a second team of your choice, e.g. *Legal*).

1. Define `EngineeringState(TypedDict)` with `brief`, `findings`, `qa_notes`, `report`.
2. Create two tools:
   - `estimate_build_time(component: str) -> str` (fake: `"bottle body"` → `"4 weeks"`, `"lid"` → `"2 weeks"`)
   - `list_test_requirements(product_type: str) -> str` (fake: leak test, drop test, thermal test)
3. Build two agents with `create_react_agent`: a `feasibility_engineer` (uses `estimate_build_time`) and a `qa_engineer` (uses `list_test_requirements`).
4. Write nodes `feasibility_engineer_node`, `qa_engineer_node`, and an `eng_lead` merge node that outputs a 3-line report: `Feasibility`, `Timeline`, `QA plan`.
5. Compile it as `engineering_graph` and test it alone with `engineering_graph.invoke({"brief": "..."})`.
6. Rewrite `engineering_team_node` to call `engineering_graph.invoke(...)` inside a `try/except`, returning only `report` upward.
7. Rerun the demo. The CEO prompt and graph wiring should need **no changes**.

**Stretch:** make the CEO route to `marketing` and `engineering` in parallel (hint: return multiple destinations with `Send`, and rely on the `results` reducer to collect both reports).

**Success check:** the CEO's streamed output shows only team-level decisions, and the final summary uses both real team reports.

---

## What's Next

You now have three coordination shapes: **supervisor** (central), **swarm** (peers), and **hierarchy** (layers). In **[Chapter 16.4: Agent Communication & Handoff](chapter-71-handoff.md)** we zoom into the mechanics they all share — how control and context pass between agents, what to hand over, and how to do it safely.

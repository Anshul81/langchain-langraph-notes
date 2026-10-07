# Chapter 16.6: Multi-Agent Research System (Project)

> **Phase 16 — Multi-Agent Systems** | [← Previous: Reflection & Planning](chapter-72-reflection-planning.md) | [Next: FastAPI Integration →](../phase-17-production/chapter-74-fastapi-integration.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Assemble a **multi-agent research pipeline** end-to-end
- ✅ Combine **planning**, **parallel retrieval**, **specialist writers**, and **reflection**
- ✅ Use **checkpointing** for long research sessions
- ✅ Stream progress to the console
- ✅ Document architecture decisions for interviews and capstone reuse

| | |
|---|---|
| **Prerequisites** | Phase 13–16 |
| **Estimated Reading Time** | 35 minutes |
| **Estimated Coding Time** | 90 minutes |

---

## Introduction — The Problem

Single-shot RAG answers complex research poorly:

```
"Compare LangGraph checkpoint backends for enterprise"
→ one retrieval pass, no plan, no critique, shallow answer
```

**Project goal:** build a **Research Team Graph** that plans sub-questions, gathers evidence in parallel, drafts, and reflects before delivery.

```
User query
    │
    ▼
 Planner ──► Research team (parallel sources)
    │              │
    ▼              ▼
 Writer ◄── evidence bundle
    │
    ▼
 Critic ──► (revise loop) ──► Final report
```

### The Solution — Composed LangGraph

One `StateGraph` with reducers, conditional loops, and optional `MemorySaver`.

---

## Part 1: Architecture Overview

| Node | Role |
|------|------|
| `planner` | Decompose query into sub-questions |
| `search_a` / `search_b` / `search_c` | Parallel mock retrievers |
| `merge_evidence` | Normalize snippets |
| `writer` | Draft report |
| `critic` | Score completeness |
| `revise` | Improve draft using critique |

State keys:

```
objective, subquestions[], snippets[], draft, score, attempts, messages[]
```

---

## Part 2: Full Implementation

```python
import os
from operator import add
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)


class ResearchState(TypedDict):
    objective: str
    subquestions: list[str]
    snippets: Annotated[list[str], add]
    draft: str
    critique: str
    score: float
    attempts: int
    messages: Annotated[list, add_messages]


def planner(state: ResearchState) -> dict:
    prompt = [
        SystemMessage(content=(
            "Split the research objective into exactly 3 sub-questions. "
            "One per line, no numbering."
        )),
        HumanMessage(content=state["objective"]),
    ]
    raw = llm.invoke(prompt).content
    subs = [ln.strip() for ln in raw.splitlines() if ln.strip()][:3]
    return {
        "subquestions": subs,
        "messages": [AIMessage(content=f"Plan: {subs}")],
        "attempts": 0,
    }


def _search(label: str, subquestions: list[str]) -> str:
    q = subquestions[0] if subquestions else "general"
    return f"[{label}] Evidence about '{q}' — mock paragraph with citations {label}-1."


def search_docs(state: ResearchState) -> dict:
    return {"snippets": [_search("docs", state.get("subquestions", []))]}


def search_web(state: ResearchState) -> dict:
    return {"snippets": [_search("web", state.get("subquestions", []))]}


def search_papers(state: ResearchState) -> dict:
    return {"snippets": [_search("papers", state.get("subquestions", []))]}


def merge_evidence(state: ResearchState) -> dict:
    joined = "\n".join(state.get("snippets", []))
    return {"messages": [AIMessage(content=f"Merged {len(state.get('snippets', []))} snippets.")]}


def writer(state: ResearchState) -> dict:
    critique = state.get("critique", "")
    prompt = [
        SystemMessage(content="Write a structured research brief with headings."),
        HumanMessage(content=(
            f"Objective: {state['objective']}\n"
            f"Subquestions: {state.get('subquestions')}\n"
            f"Evidence:\n" + "\n".join(state.get("snippets", [])) + "\n"
            f"Critique to address: {critique}"
        )),
    ]
    draft = llm.invoke(prompt).content
    return {
        "draft": draft,
        "attempts": state.get("attempts", 0) + 1,
        "messages": [AIMessage(content="Draft updated.")],
    }


def critic(state: ResearchState) -> dict:
    prompt = [
        SystemMessage(content="Score 0-1 completeness. Format: SCORE|critique"),
        HumanMessage(content=state.get("draft", "")),
    ]
    raw = llm.invoke(prompt).content
    try:
        score_str, crit = raw.split("|", 1)
        score = float(score_str.replace("SCORE", "").strip())
    except ValueError:
        score, crit = 0.6, raw
    return {"score": max(0.0, min(1.0, score)), "critique": crit.strip()}


def route_quality(state: ResearchState) -> Literal["revise", END]:
    if state.get("score", 0) >= 0.85:
        return END
    if state.get("attempts", 0) >= 3:
        return END
    return "revise"


def revise(state: ResearchState) -> dict:
    return {"messages": [AIMessage(content=f"Revise pass {state.get('attempts')}")]}


graph = StateGraph(ResearchState)
graph.add_node("planner", planner)
graph.add_node("search_docs", search_docs)
graph.add_node("search_web", search_web)
graph.add_node("search_papers", search_papers)
graph.add_node("merge_evidence", merge_evidence)
graph.add_node("writer", writer)
graph.add_node("critic", critic)
graph.add_node("revise", revise)

graph.add_edge(START, "planner")
for s in ("search_docs", "search_web", "search_papers"):
    graph.add_edge("planner", s)
    graph.add_edge(s, "merge_evidence")
graph.add_edge("merge_evidence", "writer")
graph.add_edge("writer", "critic")
graph.add_conditional_edges("critic", route_quality, {"revise": "revise", END: END})
graph.add_edge("revise", "writer")

memory = MemorySaver()
research_app = graph.compile(checkpointer=memory)
```

---

## Part 3: Run & Stream

```python
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "research-project-1"}}
    objective = "Compare SQLite vs PostgreSQL LangGraph checkpointers for SaaS."

    print("=== stream updates ===")
    for chunk in research_app.stream(
        {"objective": objective},
        config=config,
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            print(f"  [{node}] keys={list(update.keys())}")

    final = research_app.invoke({"objective": objective}, config=config)
    print("\n=== REPORT ===\n")
    print(final.get("draft", "")[:2000])
    print("\nScore:", final.get("score"), "Attempts:", final.get("attempts"))
```

---

## Part 4: Extending the Project

```
PRODUCTION UPGRADES:
├── Replace mock search with real retrievers (Phase 11–12)
├── Supervisor assigns subquestions to worker subgraphs
├── interrupt_before writer for legal review
├── PostgresSaver + thread per user
├── LangSmith traces per node
└── Export PDF via tool node
```

---

## Part 5: Testing Strategy

| Test | Type |
|------|------|
| `route_quality` at score 0.9 | Unit |
| Reducers append snippets | Unit |
| Full graph smoke | Integration (mock LLM) |
| Thread resume | Checkpoint test |

---

## Part 6: Interview Story

> "We built a LangGraph research system: planner decomposes the question, three parallel retrievers fan-in, writer+critic loop until score ≥ 0.85 or three attempts. State uses reducers for snippets and messages; MemorySaver enables resumable threads. Next we'd swap mock search for hybrid RAG and add HITL before external publish."

---

## Part 7: File Layout for the Project

```
research_team/
├── state.py          # ResearchState TypedDict
├── nodes/
│   ├── planner.py
│   ├── search.py
│   ├── writer.py
│   └── critic.py
├── graph.py          # compile()
└── main.py           # CLI
```

Splitting files mirrors Phase 17 FastAPI import paths.

### Part 8: Environment Variables

```
LITE_LLM_MODEL=gpt-4o-mini
LITELLM_PROXY_API_KEY=...
LITELLM_PROXY_API_BASE=...
LANGCHAIN_TRACING_V2=true   # optional LangSmith
```

Document in README before capstone demo.

### Part 9: Rubric for Self-Assessment

| Criterion | Pass |
|-----------|------|
| Parallel search | 3+ branches merge once |
| Reflection | critic loop with cap |
| Reducers | snippets append |
| Checkpoint | thread resume works |
| Stream | updates visible |

---

## Common Mistakes

### Mistake 1: Skipping merge node after parallel search

Writer runs multiple times or with partial evidence — always fan-in once.

### Mistake 2: No attempt cap on critic loop

Runaway cost — mirror `attempts >= 3`.

### Mistake 3: Giant draft in messages

Keep `draft` in dedicated state key; messages for UX summaries only.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Modular nodes | Swap retrievers independently |
| Stream updates | Demo-friendly |
| Checkpoint thread | Long research |
| Structured planner output | Fewer parse bugs |
| Version graph in git | Reproducible runs |

---

## Interview Preparation

### Easy
**Q: Components of this research graph?**

> Planner, parallel search nodes, merge, writer, critic with conditional revise loop, checkpoint optional.

### Medium
**Q: Why parallel search nodes?**

> Independent IO/latency; reducers merge snippets before single synthesis LLM call.

### Hard
**Q: Scale to 10 sources?**

> Dynamic Send API, rate-limit per source, dedupe snippets, rerank merge node, cap tokens into writer.

---

## Summary

| Stage | Output |
|-------|--------|
| Plan | subquestions |
| Retrieve | snippets (parallel) |
| Write | draft |
| Critique | score + feedback |
| Revise loop | quality gate |

---

## Exercises

1. Add fourth subquestion dynamically if score < 0.5 after first critic pass.

2. Persist `objective` across two invokes on same thread with follow-up "go deeper on Postgres".

3. Replace mock search with Chroma collection from Phase 11.

4. Add supervisor node that picks only 2 of 3 searchers based on query topic.

5. Write README section: architecture diagram + env vars — prep for Phase 17 FastAPI wrap.

---

## What's Next

[Chapter 17.1 — FastAPI Integration](../phase-17-production/chapter-74-fastapi-integration.md) exposes this graph as a production HTTP API with streaming and session management.

---

> [← Previous: Reflection & Planning](chapter-72-reflection-planning.md) | [Next: FastAPI Integration →](../phase-17-production/chapter-74-fastapi-integration.md)

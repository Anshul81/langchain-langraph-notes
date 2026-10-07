# Chapter 14.3: Parallel Branches

> **Phase 14 — LangGraph Routing & Control Flow** | [← Previous: Cycles & Loops](chapter-61-cycles-loops.md) | [Next: Error Recovery →](chapter-63-error-recovery.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Fan out from one node to **multiple parallel** successors
- ✅ Merge parallel outputs with **reducers**
- ✅ Build a **multi-source research** fan-out / fan-in graph
- ✅ Understand execution ordering and merge semantics
- ✅ Know when parallelism helps vs adds complexity

| | |
|---|---|
| **Prerequisites** | Chapters 13.3 (Reducers), 14.2 (Loops) |
| **Estimated Reading Time** | 28 minutes |
| **Estimated Coding Time** | 55 minutes |

---

## Introduction — The Problem

Sequential retrieval from three sources takes 3× latency:

```
wiki (2s) ──► docs (2s) ──► tickets (2s)  = 6s total
```

Independent IO-bound steps should run **together**, then merge.

```
           ┌──► search_wiki ────┐
START ──►  ├──► search_docs ────┼──► synthesize ──► END
           └──► search_tickets ┘
                 (parallel)
```

### The Solution — Parallel Edges + Reducers

LangGraph can schedule multiple edges from the same node. Downstream **fan-in** waits for branches; list fields use reducers to combine partial updates.

---

## Part 1: Fan-Out / Fan-In Pattern

```python
from langgraph.graph import StateGraph, START, END

graph.add_edge(START, "planner")
graph.add_edge("planner", "branch_a")
graph.add_edge("planner", "branch_b")
graph.add_edge("planner", "branch_c")
graph.add_edge("branch_a", "merge")
graph.add_edge("branch_b", "merge")
graph.add_edge("branch_c", "merge")
graph.add_edge("merge", END)
```

```
planner sends same state to A, B, C
merge runs once all predecessors complete
```

---

## Part 2: State for Parallel Merges

```python
import os
from operator import add
from typing import TypedDict, Annotated
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


class ParallelState(TypedDict):
    question: str
    snippets: Annotated[list[str], add]
    sources_used: Annotated[list[str], add]
```

Each branch **only** returns its chunk — reducers concatenate.

---

## Part 3: Branch Nodes (Mock Retrieval)

```python
def planner(state: ParallelState) -> dict:
    return {"sources_used": ["planner:ok"]}


def search_wiki(state: ParallelState) -> dict:
    # Replace with real HTTP / vector search in production
    snippet = f"Wiki stub for: {state['question'][:40]}"
    return {"snippets": [snippet], "sources_used": ["wiki"]}


def search_docs(state: ParallelState) -> dict:
    snippet = "Internal docs: configure LITELLM_PROXY for routing."
    return {"snippets": [snippet], "sources_used": ["docs"]}


def search_tickets(state: ParallelState) -> dict:
    snippet = "Ticket #4412: similar issue resolved by cache flush."
    return {"snippets": [snippet], "sources_used": ["tickets"]}


def synthesize(state: ParallelState) -> dict:
    body = "\n".join(f"- {s}" for s in state["snippets"])
    prompt = [
        SystemMessage(content="Synthesize bullets into a short answer."),
        HumanMessage(content=body),
    ]
    answer = llm.invoke(prompt).content
    return {"snippets": [f"FINAL: {answer}"]}
```

---

## Part 4: Build & Run

```python
graph = StateGraph(ParallelState)
graph.add_node("planner", planner)
graph.add_node("search_wiki", search_wiki)
graph.add_node("search_docs", search_docs)
graph.add_node("search_tickets", search_tickets)
graph.add_node("synthesize", synthesize)

graph.add_edge(START, "planner")
for branch in ("search_wiki", "search_docs", "search_tickets"):
    graph.add_edge("planner", branch)
    graph.add_edge(branch, "synthesize")
graph.add_edge("synthesize", END)

parallel_app = graph.compile()

if __name__ == "__main__":
    out = parallel_app.invoke({"question": "API returns 502 after deploy"})
    print("Sources:", out["sources_used"])
    print(out["snippets"][-1])
```

---

## Part 5: Send API (Dynamic Parallelism)

When branch count is **runtime-dependent**, LangGraph supports dynamic sends (map over a list in state):

```python
# Conceptual pattern — check LangGraph version docs for Send import
# from langgraph.types import Send
#
# def fan_out(state):
#     return [Send("search_one", {"query": q}) for q in state["queries"]]
```

Use when queries come from an LLM planner rather than fixed nodes.

---

## Part 6: Parallelism vs Thread Pools

| Approach | When |
|----------|------|
| Graph parallel branches | Declarative, checkpoint-friendly |
| `asyncio` inside one node | Fine for internal micro-parallelism |
| External job queue | Heavy batch workloads |

Graph-level parallelism keeps **one thread id** and unified state merge.

---

## Part 7: Simulating Latency (Benchmark)

```python
import time


def slow_branch(label: str, delay: float):
    def _node(state: ParallelState) -> dict:
        time.sleep(delay)
        return {"snippets": [f"{label} after {delay}s"], "sources_used": [label]}
    return _node
```

Swap branch nodes in a benchmark graph with `delay=0.5`. Sequential wiring would take ~1.5s; parallel fan-out should complete in ~0.5s plus merge overhead.

### Part 8: Deduping Snippets in Merge

Custom reducer to avoid duplicate URLs:

```python
def union_unique(left: list[str] | None, right: list[str] | None) -> list[str]:
    left = left or []
    right = right or []
    seen = set()
    out = []
    for item in left + right:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


class DedupState(TypedDict):
    snippets: Annotated[list[str], union_unique]
```

### Part 9: Failure in One Branch

If `search_docs` raises, entire super-step may fail unless you catch inside the node (Chapter 14.4). Pattern: return `{"snippets": [], "sources_used": ["docs:error"]}` instead of raising so merge still runs with partial evidence.

---

## Common Mistakes

### Mistake 1: Parallel writes without reducers

Two branches updating the same key without `Annotated[..., add]` — last writer wins.

### Mistake 2: Assuming merge order

List order from parallel branches may vary — don't rely on branch order for logic.

### Mistake 3: Heavy LLM in every branch

Parallel **retrieval** + single **synthesis** LLM call is cheaper than 3 LLM calls.

### Mistake 4: Fan-in before all branches complete

Wire all branches into the merge node — don't shortcut one branch to END.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Reducers on all parallel keys | Safe merges |
| Idempotent branch nodes | Retries won't duplicate side effects |
| One synthesis step | Coherent final answer |
| Tag snippets with source | Debugging |
| Limit branch count | Diminishing returns + rate limits |
| Mock IO in tests | Fast CI without network |

---

## Interview Preparation

### Easy
**Q: Why use parallel branches in LangGraph?**

> To reduce latency when steps are independent — e.g., querying multiple retrieval backends — then merging results into shared state.

### Medium
**Q: What state design supports parallel merges?**

> Use Annotated list fields with reducers like operator.add so each branch appends snippets or citations without overwriting siblings.

### Hard
**Q: How do reducers interact with parallel branch completion order?**

> Branches may finish in any order; reducers should be order-insensitive when possible (e.g., append + sort by source tag). For non-commutative merges, serialize via a single merge node that reads structured branch payloads.

---

## Summary

| Concept | Role |
|---------|------|
| **Fan-out** | Multiple edges from one node |
| **Fan-in** | One node waits for all inputs |
| **Reducer** | Combines parallel updates |
| **Synthesize** | Single LLM merge step |

---

## Exercises

1. Add a fourth branch `search_web` and verify four snippets merge.

2. Change `snippets` to overwrite — run once and document the bug.

3. Implement `merge_metadata` reducer merging dicts from each branch.

4. Time sequential vs parallel mock branches (sleep 0.5s each) — report speedup.

---

## What's Next

[Chapter 14.4 — Error Handling, Retries, Recovery](chapter-63-error-recovery.md) adds resilience: catch failures, retry nodes, and graceful degradation.

---

> [← Previous: Cycles & Loops](chapter-61-cycles-loops.md) | [Next: Error Recovery →](chapter-63-error-recovery.md)

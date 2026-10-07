# Chapter 13.3: State & Reducers

> **Phase 13 — LangGraph Fundamentals** | [← Previous: StateGraph Basics](chapter-57-stategraph-basics.md) | [Next: Nodes & Edges →](chapter-59-nodes-edges.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Define graph **state** with `TypedDict` and optional defaults
- ✅ Use **`Annotated`** to attach **reducer** functions to state keys
- ✅ Apply `add_messages`, `operator.add`, and custom reducers correctly
- ✅ Understand **partial updates** vs overwrite semantics
- ✅ Build an **accumulating research scratchpad** graph

| | |
|---|---|
| **Prerequisites** | Chapter 13.2 (StateGraph, START, END) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 55 minutes |

---

## Introduction — The Problem

Two nodes run in sequence. Both return a list of findings:

```
Node A returns: findings = ["fact A"]
Node B returns: findings = ["fact B"]

What is in state after both run?
```

**Without a reducer:** Node B **replaces** the list — you lose `"fact A"`.

**With a reducer:** LangGraph **merges** updates — you get `["fact A", "fact B"]`.

```
OVERWRITE (default):     A ──► [A] ──► B ──► [B]     ❌ lost A

REDUCER (e.g. append):   A ──► [A] ──► B ──► [A,B]  ✅
```

### The Solution — Reducers on State Keys

Reducers are merge functions declared on each state field:

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    findings: Annotated[list[str], add]  # concat lists
```

---

## Part 1: State Shape & Partial Updates

Nodes return **partial** state — only keys they change:

```python
def node_a(state: State) -> dict:
    return {"step": state.get("step", 0) + 1}  # other keys untouched
```

LangGraph applies each return value through the key's reducer (or default overwrite).

```
STATE BEFORE:  { messages: [...], step: 1, tags: ["x"] }
NODE RETURNS:  { step: 2 }
STATE AFTER:   { messages: [...], step: 2, tags: ["x"] }
```

| Rule | Detail |
|------|--------|
| Missing keys in return | Unchanged |
| Key without reducer | New value replaces old |
| Key with reducer | `reducer(old, new)` |
| First write to key | `reducer(None, new)` or similar — know your reducer |

---

## Part 2: Built-In Reducers

### `add_messages` — Chat History

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]


def user_turn(state: ChatState) -> dict:
    return {"messages": [HumanMessage(content="Summarize reducers in one line.")]}


def assistant_turn(state: ChatState) -> dict:
    # In real graphs, call LLM here
    return {"messages": [AIMessage(content="Reducers define how updates merge into state.")]}
```

`add_messages` also **deduplicates by message id** when resuming checkpoints — critical for HITL.

### `operator.add` — Lists & Numbers

```python
from operator import add


class RunState(TypedDict):
    logs: Annotated[list[str], add]
    total_tokens: Annotated[int, add]
```

```python
def log_start(state: RunState) -> dict:
    return {"logs": ["started"], "total_tokens": 0}


def log_step(state: RunState) -> dict:
    return {"logs": ["retrieved docs"], "total_tokens": 120}
```

After both nodes: `logs == ["started", "retrieved docs"]`, `total_tokens == 120`.

For integers, `add` sums — useful for aggregating usage across parallel branches (Phase 14).

---

## Part 3: Custom Reducers

Merge dicts, keep max score, or cap list length:

```python
from typing import Any


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    left = left or {}
    right = right or {}
    return {**left, **right}


def keep_max(left: float | None, right: float | None) -> float:
    left = left if left is not None else float("-inf")
    right = right if right is not None else float("-inf")
    return max(left, right)


class ScoredState(TypedDict):
    metadata: Annotated[dict[str, Any], merge_dicts]
    best_score: Annotated[float, keep_max]
```

```python
def retrieve(state: ScoredState) -> dict:
    return {
        "metadata": {"source": "kb-1"},
        "best_score": 0.72,
    }


def rerank(state: ScoredState) -> dict:
    return {
        "metadata": {"reranker": "cross-encoder"},
        "best_score": 0.81,
    }
```

Result: `metadata == {"source": "kb-1", "reranker": "cross-encoder"}`, `best_score == 0.81`.

---

## Part 4: Project — Research Scratchpad Graph

Accumulate sources, notes, and messages while an LLM expands a topic.

```python
import os
from operator import add
from typing import TypedDict, Annotated
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


class ResearchState(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]
    sources: Annotated[list[str], add]
    notes: Annotated[list[str], add]
    iteration: int


def plan_search(state: ResearchState) -> dict:
    topic = state["topic"]
    prompt = [
        SystemMessage(content="Return ONE search query only, no quotes."),
        HumanMessage(content=f"Topic: {topic}"),
    ]
    query = llm.invoke(prompt).content.strip()
    return {
        "messages": [AIMessage(content=f"Searching: {query}")],
        "sources": [f"mock://search?q={query.replace(' ', '+')}"],
        "notes": [f"Planned query: {query}"],
        "iteration": state.get("iteration", 0) + 1,
    }


def synthesize(state: ResearchState) -> dict:
    notes = "\n".join(state.get("notes", []))
    prompt = [
        SystemMessage(content="Write 3 bullet findings from the notes."),
        HumanMessage(content=notes or "No notes yet."),
    ]
    summary = llm.invoke(prompt).content
    return {
        "messages": [AIMessage(content=summary)],
        "notes": ["--- synthesis pass ---"],
    }


graph = StateGraph(ResearchState)
graph.add_node("plan_search", plan_search)
graph.add_node("synthesize", synthesize)
graph.add_edge(START, "plan_search")
graph.add_edge("plan_search", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()

if __name__ == "__main__":
    out = app.invoke({"topic": "LangGraph reducers"})
    print("Sources:", out["sources"])
    print("Notes:", out["notes"])
    print("Messages:", len(out["messages"]))
```

---

## Part 5: Reducers & Parallel Updates (Preview)

When two branches finish together, LangGraph may apply **multiple updates** to one key in a single step. Reducers must be **associative** in practice: merging A then B should match merging B then A when order doesn't matter.

```
        START
          │
     ┌────┴────┐
     ▼         ▼
  branch_a   branch_b
     │         │
     └────┬────┘
          ▼
        merge
```

Use `operator.add` for lists, custom reducers for conflicts — covered deeply in Chapter 14.3.

---

## Part 6: Input State vs Thread State

Invocation input can be a **subset** of state:

```python
app.invoke({"topic": "RAG", "iteration": 0})
```

Checkpointed threads rehydrate full state; reducers replay consistently when you resume (Phase 15).

---

## Part 7: Debugging Reducer Surprises

Print after each `invoke` when learning:

```python
for step in app.stream({"topic": "test"}, stream_mode="updates"):
    print(step)
```

If a list shrinks, you overwrote instead of reduced. If messages duplicate on resume, check `add_messages` vs raw list append.

### Part 8: Optional Keys and `total=False`

```python
class PartialState(TypedDict, total=False):
    required_field: str
    optional_notes: Annotated[list[str], add]
```

`total=False` marks keys optional at type level — still use `.get()` at runtime for counters.

### Part 9: Reducer Reference Table

| Reducer | Input types | Result |
|---------|-------------|--------|
| `add_messages` | Message-like | Appended, deduped chat |
| `operator.add` | list / int | Concat / sum |
| `merge_dicts` | dict | Shallow merge |
| `keep_max` | float | Maximum |
| Custom | any | Your policy |

---

## Common Mistakes

### Mistake 1: Forgetting `Annotated` on accumulating lists

```python
# ❌ Last node wins
class State(TypedDict):
    logs: list

# ✅ Append semantics
class State(TypedDict):
    logs: Annotated[list, add]
```

### Mistake 2: Returning full state from every node

Duplicated keys without reducers can **stale-copy** other fields if you manually spread `state`.

### Mistake 3: Using `add_messages` for non-message data

Use `add_messages` only for message sequences — use `add` or custom reducers for logs and metrics.

### Mistake 4: Non-commutative reducers on parallel keys

```python
# ❌ "keep first" breaks when branch order varies
# ✅ Document ordering assumptions or serialize merges
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| One reducer purpose per key | Easier to reason about merges |
| Prefer immutable-style returns | Return new chunks, let reducer combine |
| Initialize counters with `.get()` | First node may not set them |
| Type your lists (`list[str]`) | Catches bad merges early |
| Test reducers in isolation | Pure functions — unit test without graph |
| Keep message state separate | `messages` vs `artifacts` fields |

---

## Interview Preparation

### Easy
**Q: What is a reducer in LangGraph?**

> A function that defines how a state key combines an existing value with an update from a node. Without it, updates overwrite; with it, you can append messages, sum counters, or merge dicts.

### Medium
**Q: When would you use `add_messages` vs `operator.add`?**

> `add_messages` is specialized for chat message lists (including ID-based dedup on replay). `operator.add` concatenates generic lists or sums numbers — use for logs, citations, or token totals.

### Hard
**Q: How do reducers interact with checkpoint resume?**

> On resume, LangGraph replays or merges checkpointed state with new node outputs using the same reducers. Message reducers prevent duplicate messages when partial steps re-run; custom reducers must handle `None` left values and parallel writes consistently.

---

## Summary

| Concept | Meaning |
|---------|---------|
| **Partial update** | Node returns only changed keys |
| **Default merge** | Overwrite previous value |
| **`Annotated[..., reducer]`** | Custom merge per key |
| **`add_messages`** | Chat-safe message accumulation |
| **`operator.add`** | List concat / int sum |
| **Custom reducer** | Dict merge, max, cap, etc. |

---

## Exercises

1. Add a `citations: Annotated[list[str], add]` field. Push two citations from different nodes and verify both survive.

2. Implement `cap_list_10(left, right)` that concatenates then keeps the last 10 items. Attach it to a `events` field.

3. Write a reducer that merges sets: union of string tags. Use it in a two-node graph.

4. **Debug:** Remove `Annotated` from `notes` in the project graph. Run twice and explain the lost data in one paragraph.

---

## What's Next

[Chapter 13.4 — Nodes & Edges](chapter-59-nodes-edges.md) covers wiring nodes, static edges, conditional routing basics, and composing multi-node workflows.

---

> [← Previous: StateGraph Basics](chapter-57-stategraph-basics.md) | [Next: Nodes & Edges →](chapter-59-nodes-edges.md)

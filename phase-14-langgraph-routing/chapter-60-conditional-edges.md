# Chapter 14.1: Conditional Edges & Dynamic Routing

> **Phase 14 — LangGraph Routing & Control Flow** | [← Previous: Nodes & Edges](../phase-13-langgraph-fundamentals/chapter-59-nodes-edges.md) | [Next: Cycles & Loops →](chapter-61-cycles-loops.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Route with `add_conditional_edges` and typed return literals
- ✅ Build a **router function** that reads state and picks the next node
- ✅ Map LLM classifications to graph paths safely
- ✅ Implement a **Customer Support Router** (billing / tech / sales)
- ✅ Handle fallback paths when classification is ambiguous

| | |
|---|---|
| **Prerequisites** | Chapter 13.4 (Nodes & Edges) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction — The Problem

A single support bot tries to answer everything with one prompt. Result: wrong tone, wrong tools, and hallucinated refund policies.

```
ONE PROMPT TO RULE THEM ALL:
User: "My API key returns 401"
Bot:  "Your invoice #9921 will be refunded"  ❌
```

You need **dynamic routing**: classify the ticket, then run a **specialized subgraph path**.

```
                    ┌──► billing_handler ──► END
START ──► classify ─┼──► tech_handler ─────► END
                    └──► sales_handler ────► END
```

### The Solution — Conditional Edges

A **routing function** inspects state and returns the **name of the next node** (or `END`). The graph compiler wires possible targets explicitly.

---

## Part 1: Anatomy of Conditional Edges

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END

# router returns a key that maps to downstream nodes
def route(state: MyState) -> Literal["path_a", "path_b"]:
    if state["score"] > 0.5:
        return "path_a"
    return "path_b"

graph.add_conditional_edges("classifier", route)
# Implicit mapping: return value must match a node name or END
```

Optional explicit map:

```python
graph.add_conditional_edges(
    "classifier",
    route,
    {"path_a": "node_a", "path_b": "node_b"},
)
```

```
classifier
    │ route(state)
    ├─ "path_a" ──► node_a
    └─ "path_b" ──► node_b
```

---

## Part 2: LLM Classification Router

```python
import os
from typing import TypedDict, Literal
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

RouteLabel = Literal["billing", "tech", "sales", "fallback"]


class TicketState(TypedDict):
    user_message: str
    category: str
    response: str


def classify(state: TicketState) -> dict:
    prompt = [
        SystemMessage(content=(
            "Classify the ticket as exactly one word: billing, tech, sales. "
            "If unclear, reply: fallback"
        )),
        HumanMessage(content=state["user_message"]),
    ]
    label = llm.invoke(prompt).content.strip().lower()
    if label not in ("billing", "tech", "sales"):
        label = "fallback"
    return {"category": label}


def route_ticket(state: TicketState) -> RouteLabel:
    return state["category"]  # type: ignore[return-value]
```

Always **normalize** LLM output before routing — never trust free-form strings.

---

## Part 3: Customer Support Router Project

Three handlers share structure but different system prompts and (mock) tools.

```python
def make_handler(role: str):
    def _node(state: TicketState) -> dict:
        prompt = [
            SystemMessage(content=f"You are the {role} support specialist. Be concise."),
            HumanMessage(content=state["user_message"]),
        ]
        text = llm.invoke(prompt).content
        return {"response": f"[{role}] {text}"}
    return _node


def billing_handler(state: TicketState) -> dict:
    return make_handler("billing")(state)


def tech_handler(state: TicketState) -> dict:
    return make_handler("tech")(state)


def sales_handler(state: TicketState) -> dict:
    return make_handler("sales")(state)


def fallback_handler(state: TicketState) -> dict:
    return {
        "response": (
            "[fallback] Thanks — a human will route your ticket. "
            "Please add your account email."
        )
    }


graph = StateGraph(TicketState)
graph.add_node("classify", classify)
graph.add_node("billing_handler", billing_handler)
graph.add_node("tech_handler", tech_handler)
graph.add_node("sales_handler", sales_handler)
graph.add_node("fallback_handler", fallback_handler)

graph.add_edge(START, "classify")
graph.add_conditional_edges(
    "classify",
    route_ticket,
    {
        "billing": "billing_handler",
        "tech": "tech_handler",
        "sales": "sales_handler",
        "fallback": "fallback_handler",
    },
)
for node in ("billing_handler", "tech_handler", "sales_handler", "fallback_handler"):
    graph.add_edge(node, END)

router_app = graph.compile()
```

```python
if __name__ == "__main__":
    demo = router_app.invoke({
        "user_message": "We want to upgrade 50 seats — what pricing tiers exist?",
    })
    print(demo["category"], demo["response"][:200])
```

---

## Part 4: Structured Routing (Production Pattern)

Replace string parsing with JSON / tool schema when stakes are high:

```python
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage


class RouteDecision(BaseModel):
    category: Literal["billing", "tech", "sales", "fallback"]
    confidence: float = Field(ge=0, le=1)


structured_llm = llm.with_structured_output(RouteDecision)


def classify_structured(state: TicketState) -> dict:
    decision: RouteDecision = structured_llm.invoke([
        HumanMessage(content=state["user_message"]),
    ])
    cat = decision.category
    if decision.confidence < 0.6:
        cat = "fallback"
    return {"category": cat}
```

```
confidence < threshold  ──► fallback path (safe default)
```

---

## Part 5: Multi-Factor Routing

Combine rules + LLM:

```python
def route_hybrid(state: TicketState) -> RouteLabel:
    msg = state["user_message"].lower()
    if "invoice" in msg or "refund" in msg:
        return "billing"
    if "401" in msg or "api" in msg:
        return "tech"
    return route_ticket(state)
```

Order matters: **deterministic guards first**, LLM second.

---

## Part 6: Visualizing Routes

```python
print(router_app.get_graph().draw_ascii())
```

Use ASCII output in PRs so reviewers see dead-end nodes and missing edges.

---

## Part 7: Customer Support Router — Integration Test Harness

Treat routing as **pure functions** plus **one** LLM classification integration test:

```python
def test_route_ticket_billing():
    state = {"user_message": "x", "category": "billing", "response": ""}
    assert route_ticket(state) == "billing"


def test_classify_normalizes_garbage():
    # Monkeypatch llm in tests; here we simulate return dict
    fake = {"category": "BILLING!!!"}
    label = fake["category"].strip().lower()
    if label not in ("billing", "tech", "sales"):
        label = "fallback"
    assert label == "fallback" or label == "billing"
```

End-to-end smoke (requires API keys):

```python
cases = [
    "Refund for duplicate charge on invoice 441",
    "401 unauthorized on /v1/embeddings",
    "Enterprise pricing for 200 seats",
]
for msg in cases:
    out = router_app.invoke({"user_message": msg, "category": "", "response": ""})
    print(msg[:50], "→", out["category"])
```

### Part 8: Observability Fields

Extend `TicketState` for production dashboards:

```python
class TicketState(TypedDict):
    user_message: str
    category: str
    response: str
    route_confidence: float
    router_model: str
    handler_latency_ms: int
```

Set these inside `classify_structured` and each handler (use `time.perf_counter()` around LLM calls). LangSmith will pick up node spans automatically when tracing is enabled.

### Part 9: Subgraph per Handler (Scale-Up)

When handlers grow beyond one LLM call, compile each as a subgraph:

```
classify ──► billing_subgraph ──► END
         ──► tech_subgraph ─────► END
```

Parent graph keeps routing simple; teams own subgraph files (`billing_graph.py`). Pass `TicketState` through adapter nodes that map to subgraph-specific TypedDicts.

---

## Common Mistakes

### Mistake 1: Router returns a node not in the map

```python
# ❌ Typo "billling" → runtime error
# ✅ Literal types + explicit map keys
```

### Mistake 2: No fallback path

Always route unknown labels to `fallback` or END — never leave users in a void.

### Mistake 3: Classifier and router disagree

Keep classification **in state** in the classifier node; router reads state — don't re-call LLM in router unless intentional.

### Mistake 4: Too many paths in one router

```
If map > ~7 paths → consider hierarchical routing (Phase 16)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Normalize LLM labels | Prevents brittle routing |
| Log `category` + confidence | Debug misroutes |
| Default to safe path | fallback / human |
| Structured output for prod | Schema validation |
| Unit-test router with fixed state | No LLM needed for tests |
| Name handlers after domains | Matches ops teams |

---

## Interview Preparation

### Easy
**Q: What does `add_conditional_edges` do?**

> It connects a source node to multiple possible targets based on a routing function that inspects state and returns the next node key (or END).

### Medium
**Q: How do you implement a support ticket router in LangGraph?**

> Classifier node writes `category` to state. Conditional edges from classifier use a function returning handler node names. Each handler node produces a reply and edges to END. Add fallback for low confidence or invalid labels.

### Hard
**Q: How would you A/B test routing strategies?**

> Store `router_version` in state or config. Router function selects path based on version. Use checkpointer thread metadata for assignment. Compare resolution rate and escalation metrics per version in observability tooling.

---

## Summary

| Concept | Role |
|---------|------|
| **Conditional edge** | Dynamic next step |
| **Router function** | Pure(state) → node name |
| **Path map** | LLM label → node |
| **Fallback** | Safety for ambiguous input |
| **Structured classification** | Production-grade routing |

---

## Exercises

1. Add a `priority` field: if message contains "urgent", route to handlers with a shorter LLM max_tokens config (mock via state flag).

2. Wire **two-stage** routing: language detect → then category within language-specific handlers.

3. Write three unit tests for `route_hybrid` without invoking an LLM.

4. Draw the graph ASCII on paper and verify every node has a path to END.

---

## What's Next

[Chapter 14.2 — Cycles & Loops](chapter-61-cycles-loops.md) adds iterative agents: validate, retry, and self-correct until quality gates pass.

---

> [← Previous: Nodes & Edges](../phase-13-langgraph-fundamentals/chapter-59-nodes-edges.md) | [Next: Cycles & Loops →](chapter-61-cycles-loops.md)

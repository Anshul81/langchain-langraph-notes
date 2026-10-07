# Chapter 16.4: Agent Communication & Handoff

> **Phase 16 — Multi-Agent Systems** | [← Previous: Hierarchical Multi-Agent](chapter-70-hierarchical.md) | [Next: Reflection & Planning →](chapter-72-reflection-planning.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Define **handoff payloads** (context, artifacts, constraints)
- ✅ Use messages vs dedicated state fields for agent IPC
- ✅ Implement **tool-based handoff** patterns
- ✅ Preserve audit trails across transfers
- ✅ Avoid context loss during agent switches

| | |
|---|---|
| **Prerequisites** | Chapters 16.1–16.3 |
| **Estimated Reading Time** | 28 minutes |
| **Estimated Coding Time** | 55 minutes |

---

## Introduction — The Problem

Agent A finishes; Agent B starts blind:

```
A: collected customer_id=9912, refund_eligible=True
B: "Please provide your account ID"   ❌
```

Handoffs must move **structured context**, not just chat text.

```
HANDOFF PACKET:
├── from_agent / to_agent
├── user_visible_summary
├── internal_facts (structured)
├── open_tasks
└── policy_flags
```

### The Solution — Handoff State + Messages

Combine `HandoffMessage` in `messages` with a `handoff_payload` dict in state.

---

## Part 1: Handoff Schema

```python
import os
from typing import TypedDict, Annotated, Any
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


class HandoffState(TypedDict):
    messages: Annotated[list, add_messages]
    handoff_payload: dict[str, Any]
    current_agent: str
    next_agent: str
```

---

## Part 2: Agent A Creates Handoff

```python
def intake_agent(state: HandoffState) -> dict:
    user = state["messages"][-1].content if state.get("messages") else ""
    # Mock extraction
    payload = {
        "customer_id": "9912",
        "issue": user,
        "refund_eligible": "refund" in user.lower(),
    }
    summary = f"Customer {payload['customer_id']} — refund eligible={payload['refund_eligible']}"
    return {
        "handoff_payload": payload,
        "current_agent": "intake",
        "next_agent": "billing",
        "messages": [
            AIMessage(content=f"[intake] Handing off to billing. {summary}"),
        ],
    }
```

---

## Part 3: Agent B Consumes Handoff

```python
def billing_agent(state: HandoffState) -> dict:
    p = state.get("handoff_payload", {})
    prompt = [
        SystemMessage(content=(
            "Billing agent. Use INTERNAL facts below — do not re-ask for customer_id."
        )),
        HumanMessage(content=f"Internal: {p}\nUser issue: {p.get('issue', '')}"),
    ]
    reply = llm.invoke(prompt).content
    return {
        "current_agent": "billing",
        "next_agent": "END",
        "messages": [AIMessage(content=f"[billing] {reply}")],
    }


def route_next(state: HandoffState) -> str:
    nxt = state.get("next_agent", "END")
    return nxt if nxt in ("billing", "END") else "END"
```

---

## Part 4: Graph

```python
graph = StateGraph(HandoffState)
graph.add_node("intake", intake_agent)
graph.add_node("billing", billing_agent)
graph.add_edge(START, "intake")
graph.add_conditional_edges("intake", route_next, {"billing": "billing", "END": END})
graph.add_edge("billing", END)

handoff_app = graph.compile()
```

---

## Part 5: Tool-Based Handoff (Pattern)

```python
from langchain_core.tools import tool


@tool
def transfer_to_billing(customer_id: str, note: str) -> str:
    """Transfer conversation to billing with customer_id."""
    return f"HANDOFF billing {customer_id}: {note}"
```

LLM calls tool → tool node writes `handoff_payload` → graph routes — mirrors OpenAI swarm handoff tools.

---

## Part 6: Communication Channels

| Channel | Use |
|---------|-----|
| `messages` | User-visible transcript |
| `handoff_payload` | Structured internal |
| `artifacts` | Files, JSON blobs |
| Checkpointer | Durability across pauses |

Never expose raw `handoff_payload` to end users if it contains PII.

---

## Part 7: Handoff Versioning

```python
handoff_payload = {
    "schema_version": 2,
    "customer_id": "9912",
    "refund_eligible": True,
}
```

Readers ignore unknown versions; writers bump on breaking changes.

### Part 8: Multi-Agent Handoff Chain

```
intake → billing → legal → END
```

Each agent sets `next_agent`; router validates allowed transitions matrix:

```python
ALLOWED = {
    "intake": {"billing", "tech"},
    "billing": {"legal", "END"},
    "tech": {"END"},
    "legal": {"END"},
}
```

Reject illegal jumps — prevents LLM hallucinated routes.

---

## Common Mistakes

### Mistake 1: Handoff only in natural language

Next agent parses badly — use dict + short summary.

### Mistake 2: Stale payload after edits

Clear or version payload on each handoff.

### Mistake 3: Missing from/to metadata

Debugging multi-agent flows becomes impossible.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Schema for payload | Validation |
| Audit log list | Compliance |
| Idempotent handoff | Retries safe |
| Minimize payload size | Token cost |

---

## Interview Preparation

### Easy
**Q: Why structured handoff?**

> Preserves IDs and flags so the next agent doesn't repeat work or ask redundant questions.

### Medium
**Q: messages vs handoff_payload?**

> Messages are conversational; payload is machine-oriented context and permissions.

### Hard
**Q: Handoff with HITL?**

> Pause before sensitive transfer; supervisor approves payload; checkpoint stores payload for resume.

---

## Summary

| Element | Purpose |
|---------|---------|
| **payload** | Structured context |
| **next_agent** | Route target |
| **summary message** | UX continuity |
| **tools** | LLM-initiated transfer |

---

## Exercises

1. Add `handoff_audit: Annotated[list, add]` logging each transfer.

2. Route to `tech` if `refund_eligible` is False.

3. Implement tool-based handoff in a mini ReAct subgraph.

4. Redact PII from user-visible messages while keeping payload full (dev only).

---

## What's Next

[Chapter 16.5 — Reflection & Planning](chapter-72-reflection-planning.md) covers plan-execute and critique loops at multi-agent scale.

---

> [← Previous: Hierarchical](chapter-70-hierarchical.md) | [Next: Reflection →](chapter-72-reflection-planning.md)

# Chapter 16.4: Agent Communication & Handoff

> **Phase 16 — Multi-Agent Systems** | [← Previous: Hierarchical Multi-Agent](chapter-70-hierarchical.md) | [Next: Reflection & Planning →](chapter-72-reflection-planning.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Say exactly **what must travel** in a handoff: goal, constraints, artifacts, what's already done
- ✅ Choose between **messages** and a dedicated **`handoff_payload`** state field
- ✅ Implement the **tool-based handoff** pattern (`transfer_to_X`)
- ✅ Combine **shared memory** and **private per-agent scratchpads** in one small graph
- ✅ Keep an **audit trail** of every handoff (including rejected ones)
- ✅ Spot and fix **context loss** with a checklist and a validator

| | |
|---|---|
| **Prerequisites** | Chapters 16.1–16.3, Phase 9 (tools), Phase 14 (StateGraph) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction — The Problem

Agent A finishes its part; Agent B starts blind:

```
User:    "I'm customer C-9912. My latest order arrived broken — refund it."
Triage:  looks up customer → latest order is O-1001, $49, 12 days old, pro plan
Triage:  transfer_to_billing(reason="refund request")        ← all the facts stay behind
Billing: "Sure! Which order? What is your customer ID?"       ← asks the user AGAIN
```

Triage **knew** the answer. The handoff **dropped** it. This is *context loss*, the #1 bug in multi-agent systems.

### The Solution — A Structured Handoff Contract

```
┌──────────┐  transfer_to_billing(goal, constraints, done, artifacts)  ┌──────────┐
│  Triage  │ ───────────────────────────────────────────────────────▶ │ Billing  │
│  agent   │            validate → audit → switch active_agent        │  agent   │
└──────────┘                                                          └──────────┘
     │ private scratch                                                     │ private scratch
     └───────────────────────── shared: payload + audit ──────────────────┘
```

A handoff is an **API call between agents**. Treat it like one: typed fields, validation, logging.

---

## Part 1: What Must Travel in a Handoff

Four things. If any is missing, the next agent has to guess or re-ask.

| Field | Question it answers | Example |
|-------|--------------------|---------|
| `goal` | What should you accomplish? | `"Refund order O-1001"` |
| `constraints` | What must you NOT violate? | `["Max refund $100", "Be polite"]` |
| `done` | What has already been done? (don't repeat) | `["Verified customer C-9912"]` |
| `artifacts` | What concrete data do I need? | `{"customer_id": "C-9912", "order_id": "O-1001"}` |

Add `from_agent` / `to_agent` for tracing. Start with a validator — it is plain Python and needs no LLM:

```python
REQUIRED_KEYS = ("goal", "constraints", "done", "artifacts")


def validate_payload(p: dict) -> list[str]:
    """Return a list of problems. Empty list = payload is valid."""
    problems = [f"missing key: {k}" for k in REQUIRED_KEYS if k not in p]
    if problems:
        return problems
    if not str(p["goal"]).strip():
        problems.append("goal is empty")
    if not isinstance(p["constraints"], list):
        problems.append("constraints must be a list")
    if not isinstance(p["done"], list):
        problems.append("done must be a list")
    if not isinstance(p["artifacts"], dict) or not p["artifacts"]:
        problems.append("artifacts must be a non-empty dict")
    return problems


# The context-loss bug from the introduction, caught offline:
print(validate_payload({"goal": "refund request"}))
# ['missing key: constraints', 'missing key: done', 'missing key: artifacts']
assert validate_payload({
    "goal": "Refund order O-1001", "constraints": ["max $100"],
    "done": ["verified customer"], "artifacts": {"order_id": "O-1001"},
}) == []
```

---

## Part 2: Messages vs `handoff_payload`

| | Messages only | Dedicated `handoff_payload` field |
|---|---------------|-----------------------------------|
| Format | Free text in the transcript | Typed dict in state |
| Next agent finds it by | Re-reading the whole chat | Reading one field |
| Can be validated? | ❌ hard | ✅ trivially |
| Survives long chats / trimming? | ❌ may be summarized away | ✅ never trimmed |
| Good for | Human-visible conversation | Machine-to-machine contract |

**Rule of thumb:** `messages` is for the *conversation*; `handoff_payload` is for the *contract*. Use both — the payload goes into the next agent's system prompt.

---

## Part 3: The Shared + Private Graph (Runnable)

We build **one small graph** that shows every idea in this chapter:

```
START → agent ──tool_calls?──┬── none ───────────────▶ END
          ▲                  ├── normal tool ─▶ tools ─┐
          │                  └── transfer_to_* ▶ handoff ─┤
          └──────────────────────────────────────────────┘
```

- **Shared memory:** `messages`, `handoff_payload`, `audit`
- **Private memory:** `scratch[agent_name]` — only that agent's prompt sees it
- **Two tool-using agents:** `triage` (`lookup_customer`, `transfer_to_billing`) and `billing` (`check_refund_eligibility`, `issue_refund`)

### 3a. Setup, data, and real tools

```python
import json
import os
import sys
from operator import add
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

# A tiny in-memory "database" the tools really read and write
CUSTOMERS = {"C-9912": {"name": "Asha Rao", "plan": "pro", "orders": ["O-1001", "O-1002"]}}
ORDERS = {
    "O-1001": {"amount": 49.0, "days_ago": 12, "refunded": False},
    "O-1002": {"amount": 199.0, "days_ago": 90, "refunded": False},
}
REFUND_LEDGER: list[dict] = []


@tool
def lookup_customer(customer_id: str) -> str:
    """Look up a customer by ID. Returns name, plan, and their orders (id, amount, days_ago)."""
    cust = CUSTOMERS.get(customer_id)
    if not cust:
        return f"NOT_FOUND: no customer {customer_id}"
    orders = [{"order_id": o, **ORDERS[o]} for o in cust["orders"]]
    return json.dumps({"customer_id": customer_id, "name": cust["name"],
                       "plan": cust["plan"], "orders": orders})


@tool
def check_refund_eligibility(order_id: str) -> str:
    """Check if an order is refundable (policy: within 30 days and not already refunded)."""
    o = ORDERS.get(order_id)
    if not o:
        return f"NOT_FOUND: no order {order_id}"
    ok = o["days_ago"] <= 30 and not o["refunded"]
    return json.dumps({"order_id": order_id, "eligible": ok, "amount": o["amount"],
                       "reason": "ok" if ok else "older than 30 days or already refunded"})


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a refund. Only call after check_refund_eligibility says eligible."""
    o = ORDERS.get(order_id)
    if not o or o["refunded"] or amount > o["amount"]:
        return "REFUND_FAILED: invalid order, already refunded, or amount too large"
    o["refunded"] = True
    REFUND_LEDGER.append({"order_id": order_id, "amount": amount})
    return f"REFUND_OK: {amount} refunded for {order_id}"


@tool
def transfer_to_billing(goal: str, constraints: list[str], done: list[str],
                        artifacts: dict[str, str]) -> str:
    """Hand the case to the Billing agent.
    goal: what billing must accomplish. constraints: rules to respect.
    done: steps already completed. artifacts: concrete ids/data billing needs (e.g. order_id)."""
    return "handoff requested"  # never executed — the handoff node intercepts it (see 3c)
```

Why intercept `transfer_to_billing` instead of letting it run? A tool can only return a string; the handoff must **change state** (active agent, payload, audit). So the tool defines the *schema the LLM must fill*, and a graph node does the work.

### 3b. State and agent registry

```python
def merge_scratch(old: dict, new: dict) -> dict:
    """Reducer: append each agent's notes under its own key."""
    merged = {k: list(v) for k, v in (old or {}).items()}
    for agent, notes in (new or {}).items():
        merged.setdefault(agent, []).extend(notes)
    return merged


class HandoffState(TypedDict):
    # ── SHARED ──
    messages: Annotated[list, add_messages]          # full transcript (UI / debugging)
    handoff_payload: dict                            # the contract for the next agent
    audit: Annotated[list[dict], add]                # append-only handoff log
    # ── PRIVATE ──
    scratch: Annotated[dict, merge_scratch]          # {"triage": [...], "billing": [...]}
    # ── ROUTING ──
    active_agent: str
    view_start: int                                  # where the active agent's own messages begin
    hops: int


AGENTS = {
    "triage": {
        "prompt": ("You are the Triage agent. Identify the customer's problem using your tools. "
                   "For refund or billing issues, look up the customer first, resolve which order "
                   "they mean, then call transfer_to_billing with ALL fields filled "
                   "(goal, constraints, done, artifacts incl. customer_id and order_id)."),
        "tools": [lookup_customer, transfer_to_billing],
    },
    "billing": {
        "prompt": ("You are the Billing agent. Call check_refund_eligibility, then issue_refund "
                   "only if eligible. Respect the handoff constraints. "
                   "Never ask the user for anything already in the handoff. Finish with a short reply."),
        "tools": [check_refund_eligibility, issue_refund],
    },
}
```

### 3c. Nodes: agent, tools, handoff

```python
MAX_HOPS = 4


def build_view(state: HandoffState) -> list:
    """What the ACTIVE agent sees. Shared contract + its OWN private notes + its own turn."""
    name = state["active_agent"]
    system = AGENTS[name]["prompt"]

    p = state.get("handoff_payload") or {}
    if p.get("to_agent") == name:
        system += (
            f"\n\n=== HANDOFF from {p['from_agent']} ===\n"
            f"Goal: {p['goal']}\nConstraints: {p['constraints']}\n"
            f"Already done: {p['done']}\nArtifacts: {p['artifacts']}"
        )
    notes = state.get("scratch", {}).get(name, [])
    if notes:
        system += "\n\nYour private notes:\n- " + "\n- ".join(notes)

    first_user_msg = state["messages"][0]
    own_turn = state["messages"][state["view_start"]:]    # excludes the previous agent's tool chatter
    return [SystemMessage(content=system), first_user_msg, *own_turn]


def agent_node(state: HandoffState) -> dict:
    model = llm.bind_tools(AGENTS[state["active_agent"]]["tools"])
    return {"messages": [model.invoke(build_view(state))]}


def tools_node(state: HandoffState) -> dict:
    """Run normal tools for the active agent and record private notes."""
    name = state["active_agent"]
    by_name = {t.name: t for t in AGENTS[name]["tools"]}
    out, notes = [], []
    for call in state["messages"][-1].tool_calls:
        tool_fn = by_name.get(call["name"])
        result = tool_fn.invoke(call["args"]) if tool_fn else f"ERROR: unknown tool {call['name']}"
        out.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        notes.append(f"{call['name']}({call['args']}) -> {str(result)[:150]}")
    return {"messages": out, "scratch": {name: notes}}


def handoff_node(state: HandoffState) -> dict:
    """Validate payload → append audit event → switch agent (or reject)."""
    last = state["messages"][-1]
    call = next(c for c in last.tool_calls if c["name"].startswith("transfer_to_"))
    target = call["name"].removeprefix("transfer_to_")
    source = state["active_agent"]

    payload = {k: call["args"][k] for k in REQUIRED_KEYS if k in call["args"]}
    problems = validate_payload(payload)
    if target not in AGENTS:
        problems.append(f"unknown agent: {target}")

    reply = ("HANDOFF REJECTED: " + "; ".join(problems) + ". Call the tool again with every field filled."
             if problems else f"Handoff to {target} accepted.")
    # Every tool_call needs a ToolMessage reply, or the next LLM call fails.
    tool_msgs = [ToolMessage(content=reply if c is call else "skipped: handoff in progress",
                             tool_call_id=c["id"]) for c in last.tool_calls]

    hops = state["hops"] + 1
    update = {
        "messages": tool_msgs,
        "hops": hops,
        "audit": [{"hop": hops, "from": source, "to": target,
                   "status": "rejected" if problems else "ok",
                   "problems": problems, "goal": payload.get("goal")}],
    }
    if not problems:
        update["active_agent"] = target
        update["handoff_payload"] = {**payload, "from_agent": source, "to_agent": target}
        update["view_start"] = len(state["messages"]) + len(tool_msgs)   # target starts fresh
    return update
```

### 3d. Routing and assembly

```python
def route_after_agent(state: HandoffState) -> str:
    calls = state["messages"][-1].tool_calls
    if not calls:
        return END
    return "handoff" if any(c["name"].startswith("transfer_to_") for c in calls) else "tools"


def route_after_handoff(state: HandoffState) -> str:
    return "agent" if state["hops"] <= MAX_HOPS else END     # cap rejected-retry loops


g = StateGraph(HandoffState)
g.add_node("agent", agent_node)
g.add_node("tools", tools_node)
g.add_node("handoff", handoff_node)
g.add_edge(START, "agent")
g.add_conditional_edges("agent", route_after_agent, ["tools", "handoff", END])
g.add_edge("tools", "agent")
g.add_conditional_edges("handoff", route_after_handoff, ["agent", END])
app = g.compile()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")        # Windows consoles: model text may contain symbols
    question = "I'm customer C-9912. My latest order arrived broken — please refund it."
    result = app.invoke(
        {"messages": [HumanMessage(content=question)], "handoff_payload": {}, "audit": [],
         "scratch": {}, "active_agent": "triage", "view_start": 1, "hops": 0},
        {"recursion_limit": 25},
    )
    print("FINAL:", result["messages"][-1].content)
    print("AUDIT:", json.dumps(result["audit"], indent=2))
    print("PAYLOAD:", json.dumps(result["handoff_payload"], indent=2))
    print("PRIVATE KEYS:", {k: len(v) for k, v in result["scratch"].items()})
    print("LEDGER:", REFUND_LEDGER)
```

### What good output looks like

```
FINAL: Your refund of $49.0 for order O-1001 has been issued. Sorry about the damaged item!
AUDIT: [{"hop": 1, "from": "triage", "to": "billing", "status": "ok", "problems": [],
         "goal": "Refund order O-1001 for customer C-9912 (arrived broken)"}]
PAYLOAD: {"goal": "...", "constraints": ["Refund only within 30 days"],
          "done": ["Verified customer C-9912"],
          "artifacts": {"customer_id": "C-9912", "order_id": "O-1001"},
          "from_agent": "triage", "to_agent": "billing"}
PRIVATE KEYS: {'triage': 1, 'billing': 2}
LEDGER: [{'order_id': 'O-1001', 'amount': 49.0}]
```

(Wording varies; the **structure** should not.) The ledger entry proves a real tool executed a real side effect.

### Reading the graph — shared vs private

| Memory | Where | Who sees it | Why |
|--------|-------|-------------|-----|
| `messages` | shared | Only `messages[0]` + the active agent's own turn go into a prompt; the rest is for UI/debug | Prevents the previous agent's tool noise leaking in |
| `handoff_payload` | shared | The receiving agent (in its system prompt) | The contract |
| `audit` | shared | Humans, tests, observability | Accountability |
| `scratch["triage"]` | private | Only triage's prompt | Raw lookups stay out of billing |
| `scratch["billing"]` | private | Only billing's prompt | Same, reversed |

**Rule:** share *conclusions* (payload), keep *working notes* private.

---

## Part 4: The Audit Trail

`handoff_node` appends one event per attempt — **including rejected ones** — via the `add` reducer, so nothing is overwritten:

```python
for e in result["audit"]:
    print(f"hop {e['hop']}: {e['from']} -> {e['to']} [{e['status']}] {e['problems']}")
# hop 1: triage -> billing [ok] []
```

Uses: debugging ("why did billing refund the wrong order?"), compliance, loop detection (`hops` + `MAX_HOPS`), and tests (`assert result["audit"][0]["status"] == "ok"`).

---

## Part 5: Avoiding Context Loss

### Checklist before every handoff

- [ ] **Goal** is a full sentence, not `"help"`
- [ ] **Identifiers** (customer_id, order_id, ticket) are in `artifacts`
- [ ] **Constraints** include limits and policy the user already stated
- [ ] **Done** lists what NOT to repeat
- [ ] Payload **validated** before the switch
- [ ] Handoff **logged** to `audit`
- [ ] Receiver's prompt says: *"Do not ask for anything in the handoff."*

### The example bug

```python
# ❌ Naive schema — only a reason string
@tool
def transfer_to_billing(reason: str) -> str:
    """Hand off to billing."""

# Result: billing sees only "refund request". The order id lived in triage's
# lookup_customer output, which build_view() deliberately hides → billing asks the user again.
```

```python
# ✅ Fix — require the full contract in the schema AND validate it
@tool
def transfer_to_billing(goal: str, constraints: list[str], done: list[str],
                        artifacts: dict[str, str]) -> str: ...
```

Two defenses: the **schema** (LLM sees required fields) and the **validator** (code rejects empty ones, and the agent retries — visible as a `rejected` audit event).

---

## Common Mistakes

### Mistake 1: Handoff = "just keep the chat history"
Long chats get trimmed or summarized, and the receiver must re-derive facts. Use an explicit payload.

### Mistake 2: Letting the receiving agent see everything
Full transcript sharing leaks irrelevant tool output and bloats tokens. Build a **view** per agent.

### Mistake 3: No validation
An LLM will happily call `transfer_to_billing(goal="help")`. Validate in code.

### Mistake 4: Unanswered tool calls
After an AI message with `tool_calls`, **every** call needs a `ToolMessage`, or the next model call errors. Notice `handoff_node` answers all of them.

### Mistake 5: Unlimited handoff loops
A ↔ B ping-pong burns money. Cap `hops` and log the audit trail.

---

## Best Practices

| Practice | Why |
|----------|-----|
| Fixed payload schema (`goal`, `constraints`, `done`, `artifacts`) | Predictable contract |
| `transfer_to_X` tools with typed args | LLM fills the contract; schema documents it |
| Intercept transfers in a graph node | Tools can't mutate state; nodes can |
| Validate, then log, then switch | Fail early, stay accountable |
| Private scratch + shared payload | Less noise, less token cost |
| `MAX_HOPS` | Bounded cost |

---

## Interview Preparation

### Easy
**Q: What should a handoff include?**
> Goal, constraints, what's already done, and artifacts (ids/data) — so the next agent never re-asks or repeats work.

### Medium
**Q: Why a `handoff_payload` field instead of messages?**
> It's typed, validatable, and never trimmed or summarized away; messages remain for the human-visible conversation.

**Q: How do you implement handoff with tools in LangGraph?**
> Give the agent a `transfer_to_X` tool whose arguments are the payload. A routing function detects that call, a handoff node validates it, logs it to `audit`, updates `active_agent` + `handoff_payload`, and answers the tool call with a `ToolMessage`.

### Hard
**Q: How do you stop context loss and context bloat at the same time?**
> Share only the distilled contract (payload) plus the original user request; keep each agent's raw tool outputs in a private scratchpad. Validate payloads in code and keep an audit trail to detect regressions.

---

## Summary

| Concept | Takeaway |
|---------|----------|
| Handoff contents | goal, constraints, done, artifacts |
| Messages vs payload | Conversation vs contract |
| Tool-based handoff | `transfer_to_X` schema + interceptor node |
| Shared vs private | Share conclusions, keep notes private |
| Audit | Append every attempt via `add` reducer |
| Context loss | Checklist + schema + validator |

---

## Hands-on: Payload Validation with Per-Target Requirements

`validate_payload` only checks that keys exist. Extend it so billing refuses payloads missing the data it needs:

```python
TARGET_REQUIRED_ARTIFACTS = {"billing": ["customer_id", "order_id"]}


def validate_for_target(p: dict, target: str) -> list[str]:
    problems = validate_payload(p)
    if problems:
        return problems
    # TODO: for each key in TARGET_REQUIRED_ARTIFACTS.get(target, []),
    #       add f"artifacts missing {key}" if it's absent or empty.
    return problems


good = {"goal": "Refund O-1001", "constraints": [], "done": [],
        "artifacts": {"customer_id": "C-9912", "order_id": "O-1001"}}
assert validate_for_target(good, "billing") == []
assert validate_for_target({**good, "artifacts": {"customer_id": "C-9912"}}, "billing") == ["artifacts missing order_id"]
assert "missing key: done" in validate_for_target({"goal": "x", "constraints": [], "artifacts": {"a": "b"}}, "billing")
```

Then:
1. Swap `validate_payload(payload)` for `validate_for_target(payload, target)` in `handoff_node`.
2. Re-run with the user message *"Refund my latest order"* (no customer ID). Does triage ask the user, or does the audit show a `rejected` event first?
3. Add a `transfer_to_triage` tool so billing can escalate back; watch `hops`.

---

## What's Next

[Chapter 16.5 — Reflection & Planning Patterns](chapter-72-reflection-planning.md) teaches agents to plan before acting and critique their own output.

---

> [← Previous: Hierarchical](chapter-70-hierarchical.md) | [Next: Reflection & Planning →](chapter-72-reflection-planning.md)

# Chapter 16.2: Swarm Architecture

> **Phase 16 — Multi-Agent Systems** | [← Previous: Supervisor Architecture](chapter-68-supervisor.md) | [Next: Hierarchical Multi-Agent →](chapter-70-hierarchical.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Explain the **swarm** pattern: peer agents, no central boss
- ✅ Contrast **swarm vs supervisor** and choose between them
- ✅ Implement **handoffs with a handoff tool** that returns a LangGraph `Command`
- ✅ Track the **active agent** in state so conversations resume with the right peer
- ✅ Build a **Triage → Billing / Tech Support** swarm where each agent has its own tools
- ✅ **Prevent handoff loops** with a hop counter and a visited list
- ✅ Recognize why parsing `HANDOFF:` out of text is an anti-pattern

| | |
|---|---|
| **Prerequisites** | Chapter 16.1 (Supervisor), Phase 14 (conditional edges), Phase 15 (checkpointing) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction — The Problem

The supervisor from Chapter 16.1 is great for *task decomposition*: "do research, then math, then summarize." But consider a customer-support chat:

```
User: "I was charged twice for invoice INV-1002."
  → the BILLING specialist should now own this conversation.

User: "Also, login shows error 503."
  → the TECH specialist should take over — directly.
```

With a supervisor, **every** message goes: user → supervisor → worker → supervisor → user. The supervisor is an extra LLM call on every turn, and the *worker never talks to the user directly* — the supervisor does.

In a support-style flow what you really want is a **handoff**: the agent that's currently talking to the user realizes "this isn't my area" and **passes the conversation** to the right peer. The new agent speaks to the user directly.

### The Solution — Peer Agents with Handoff Tools

```
              SUPERVISOR (16.1)                         SWARM (this chapter)

            ┌────────────┐                      ┌─────────┐
            │ supervisor │ ◄──┐                 │ TRIAGE  │
            └─┬────┬────┬┘    │                 └──┬───┬──┘
              ▼    ▼    ▼     │           handoff  │   │  handoff
             W1   W2   W3 ────┘                    ▼   ▼
        (workers never talk                 ┌────────┐ ┌────────┐
         to each other or the user)         │BILLING │◄►│  TECH  │
                                            └────────┘ └────────┘
                                         (peers hand off directly;
                                          whoever is active talks to the user)
```

Key idea: **a handoff is just a tool call.** The agent calls `transfer_to_billing`, and the tool tells LangGraph "go to the `billing` node next, and make `billing` the active agent."

---

## Part 1: Swarm vs Supervisor

| | Supervisor (16.1) | Swarm (16.2) |
|---|---|---|
| Control | Central router decides every step | Each agent decides who's next |
| Who talks to the user | Supervisor (workers return reports) | Whichever agent is **active** |
| Extra LLM call per hop | Yes (routing call) | No (the handoff *is* the agent's tool call) |
| Agent awareness | Workers know only their job | Agents know their **peers** (names + when to hand off) |
| Best for | Decomposing a task into steps | Conversations that move between specialties |
| Typical examples | Research + calculate + summarize | Support desk, sales → onboarding, concierge |
| Main risk | Supervisor bottleneck/overload | **Handoff loops**, unclear ownership |
| Debugging | One place shows all routing | Follow the handoff trail |

> **Rule of thumb:** *"Do these steps in order and combine the results"* → supervisor. *"Whoever is best placed should own this conversation right now"* → swarm.

---

## Part 2: State — Who Is Active?

The swarm needs only three things beyond messages:

```python
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)


class SwarmState(MessagesState):          # parent graph state
    active_agent: str                     # who owns the conversation now
    hops: int                             # handoffs so far THIS user turn
    path: list[str]                       # agents already visited THIS user turn


class SwarmAgentState(AgentState):        # each agent's inner state (same extra keys)
    active_agent: str
    hops: int
    path: list[str]
```

Why two classes? `create_react_agent` needs its state to include `remaining_steps` (that's what `AgentState` adds), while the parent graph only needs `messages` + our swarm keys. The extra keys are identical so they flow in and out of each agent.

- **`active_agent`** is saved by the checkpointer. On the **next** user message, the graph resumes at the agent who was last talking — not at triage again.
- **`hops` / `path`** are loop-prevention bookkeeping (Part 5). We reset them at the start of each user turn.

---

## Part 3: The Handoff Tool

Handoff is the heart of a swarm. A **handoff tool** is a normal `@tool` — but instead of returning text, it returns a LangGraph **`Command`**:

```python
MAX_HOPS = 3


def make_handoff_tool(target: str, description: str):
    """Create a tool that transfers the conversation to `target`."""

    @tool(f"transfer_to_{target}", description=description)
    def handoff(
        state: Annotated[SwarmAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        current = state.get("active_agent", "triage")
        path = state.get("path", [])
        hops = state.get("hops", 0)

        # Loop prevention (Part 5): refuse, and tell the LLM why.
        if hops >= MAX_HOPS or target in set(path) | {current}:
            return (
                f"Handoff to '{target}' refused (hop limit reached or agent already "
                "involved). Answer the customer yourself with what you know."
            )

        tool_msg = ToolMessage(
            content=f"Transferred to {target}.",
            name=f"transfer_to_{target}",
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=target,                 # go to this node in the PARENT graph
            graph=Command.PARENT,        # not inside the agent's own mini-graph
            update={
                "messages": state["messages"] + [tool_msg],
                "active_agent": target,  # the receiving agent is now in charge
                "hops": hops + 1,
                "path": path + [current],
            },
        )

    return handoff
```

Three things to understand:

1. **`InjectedState`** gives the tool the agent's current state (hidden from the LLM — the model only sees the tool's name and description).
2. **`InjectedToolCallId`** gives the tool the id of *this* call so we can reply with a matching `ToolMessage` (chat APIs require every tool call to get a response).
3. **`Command(goto=..., graph=Command.PARENT, update=...)`** does three jobs at once: *route* to the target node, *escape* from the agent's inner graph to the swarm graph, and *update* shared state — including the full message history so the receiving agent sees what happened.

We pass `state["messages"] + [tool_msg]` because the agent's inner messages (including the AI's tool call) haven't been merged into the parent yet. `add_messages` dedupes by message id, so nothing is duplicated.

Now create the three handoff tools the swarm needs:

```python
to_billing = make_handoff_tool("billing", "Transfer to billing for invoices, charges, and refunds.")
to_tech    = make_handoff_tool("tech", "Transfer to tech support for errors, outages, and login problems.")
```

### Anti-pattern: parsing handoffs from text

You'll see tutorials where agents write `HANDOFF: billing` in their reply and a regex routes on it:

```python
# ❌ FRAGILE — don't build your swarm on this
if "HANDOFF:" in reply.content:
    target = reply.content.split("HANDOFF:")[1].strip().split()[0]
```

Why it breaks: the model writes `Handoff - Billing`, or mentions "HANDOFF:" while *explaining* the policy to the user, or adds punctuation, or hands off to a name that doesn't exist. Tool calls are schema-validated, named, logged, and traceable. Prefer tools.

---

## Part 4: Agents With Their Own Tools

Each peer is a real ReAct agent with **domain tools plus handoff tools** for its peers. First the domain tools (deterministic fakes):

```python
INVOICES = {
    "INV-1001": {"amount": 49.00, "status": "paid"},
    "INV-1002": {"amount": 49.00, "status": "paid", "note": "duplicate of INV-1001"},
}


@tool
def lookup_invoice(invoice_id: str) -> str:
    """Look up an invoice by id, e.g. 'INV-1002'."""
    inv = INVOICES.get(invoice_id.upper())
    return str(inv) if inv else f"Invoice {invoice_id} not found."


@tool
def issue_refund(invoice_id: str, reason: str) -> str:
    """Issue a refund for an invoice. Only use for confirmed billing errors."""
    inv = INVOICES.get(invoice_id.upper())
    if not inv:
        return f"Cannot refund: {invoice_id} not found."
    return f"Refund RF-9001 issued for ${inv['amount']:.2f} on {invoice_id} ({reason})."


@tool
def check_service_status(service: str) -> str:
    """Check the status of a service: 'login', 'api', or 'dashboard'."""
    status = {
        "login": "DEGRADED - error 503, incident INC-77, fix ETA 30 minutes",
        "api": "operational",
        "dashboard": "operational",
    }
    return status.get(service.lower(), f"Unknown service '{service}'.")


@tool
def create_ticket(summary: str) -> str:
    """Create a support ticket for engineering."""
    return f"Ticket TCK-4521 created: {summary}"
```

Now the agents. Notice each prompt tells the agent **who its peers are and when to hand off** — that's what replaces the supervisor's routing:

```python
def make_agent(tools, prompt):
    return create_react_agent(llm, tools=tools, prompt=prompt, state_schema=SwarmAgentState)


triage_agent = make_agent(
    [to_billing, to_tech],
    "You are the TRIAGE agent for a software company. You do NOT solve problems. "
    "Billing, charges, invoices, refunds -> call transfer_to_billing. "
    "Errors, outages, login, bugs -> call transfer_to_tech. "
    "If the request is just a greeting, reply briefly and ask how you can help.",
)

billing_agent = make_agent(
    [lookup_invoice, issue_refund, to_tech],
    "You are the BILLING agent. Look up invoices before refunding. Refund only "
    "confirmed duplicates/errors. If the customer raises a technical problem, "
    "call transfer_to_tech. Reply to the customer in 2-3 sentences.",
)

tech_agent = make_agent(
    [check_service_status, create_ticket, to_billing],
    "You are the TECH SUPPORT agent. Check service status before answering. "
    "Create a ticket only if the service is operational but the user still has a problem. "
    "If the customer raises a billing question, call transfer_to_billing. "
    "Reply to the customer in 2-3 sentences.",
)
```

Why do billing and tech have handoff tools to *each other* but not to triage? Triage is a one-way front door. And `target in visited` (Part 5) would refuse a hop back anyway.

---

## Part 5: Wire the Swarm and Prevent Loops

```python
def route_to_active(state: SwarmState) -> str:
    """On every new user message, resume with whoever was last in charge."""
    return state.get("active_agent") or "triage"


builder = StateGraph(SwarmState)
builder.add_node("triage", triage_agent)
builder.add_node("billing", billing_agent)
builder.add_node("tech", tech_agent)

builder.add_conditional_edges(START, route_to_active, ["triage", "billing", "tech"])
# NOTE: no edges between agents and no edges to END.
#  - Handoffs happen via Command(goto=...) from inside the handoff tools.
#  - If an agent replies WITHOUT handing off, the node simply finishes the turn.

app = builder.compile(checkpointer=MemorySaver())
```

That's the whole graph — compare with the supervisor's router + conditional map. There's **no central router**; the routing intelligence lives in agent prompts and handoff tools.

### Loop prevention — three layers

| Guard | Where | What it stops |
|---|---|---|
| **Hop limit** (`MAX_HOPS`) | inside the handoff tool | Endless A → B → C → ... chains |
| **Visited list** (`path` + current) | inside the handoff tool | Ping-pong A → B → A |
| **`recursion_limit`** | `config` on `invoke` | Anything that slips through (hard stop) |

When a handoff is refused, the tool *returns a normal string* — so the LLM reads "Handoff refused... answer yourself" and responds to the user. The conversation degrades gracefully instead of crashing.

Reset the counters each **user turn** (otherwise a long chat eventually hits `MAX_HOPS`):

```python
def chat(text: str, thread_id: str = "customer-1"):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    result = app.invoke(
        {"messages": [HumanMessage(content=text)], "hops": 0, "path": []},
        config,
    )
    show_turn(result)
    return result
```

---

## Part 6: Run It

A helper that prints this turn's trail — handoffs, tool calls, and the final reply:

```python
def show_turn(result: dict):
    msgs = result["messages"]
    last_user = max(i for i, m in enumerate(msgs) if m.type == "human")
    print(f"\n👤 {msgs[last_user].content}")
    for m in msgs[last_user + 1:]:
        if m.type == "ai" and m.tool_calls:
            for tc in m.tool_calls:
                tag = "🔀" if tc["name"].startswith("transfer_to_") else "🔧"
                print(f"{tag} {tc['name']}({tc['args']})")
        elif m.type == "ai" and m.content:
            print(f"🤖 {m.content}")
    print(f"   [active_agent={result['active_agent']}  hops={result['hops']}  path={result['path']}]")


if __name__ == "__main__":
    chat("I was charged twice for invoice INV-1002. Can I get a refund?")
    chat("Thanks! Also, login keeps failing with error 503.")   # same thread!
```

> **All code blocks in Parts 2–6 go in one file** (e.g. `swarm_demo.py`), in order. Needs `pip install langgraph langchain-openai python-dotenv`.

**Typical output** (wording varies):

```
👤 I was charged twice for invoice INV-1002. Can I get a refund?
🔀 transfer_to_billing({})
🔧 lookup_invoice({'invoice_id': 'INV-1002'})
🔧 issue_refund({'invoice_id': 'INV-1002', 'reason': 'duplicate of INV-1001'})
🤖 You're right, INV-1002 duplicated INV-1001. I've issued refund RF-9001 for $49.00.
   [active_agent=billing  hops=1  path=['triage']]

👤 Thanks! Also, login keeps failing with error 503.
🔀 transfer_to_tech({})
🔧 check_service_status({'service': 'login'})
🤖 Login is currently degraded (incident INC-77). The fix ETA is about 30 minutes.
   [active_agent=tech  hops=1  path=['billing']]
```

Look at turn 2: the first agent to speak was **billing** (resumed from the checkpoint), and it handed **directly to tech** — triage was never involved. That's peer-to-peer behavior a supervisor can't give you without an extra routing call.

### What a refused handoff looks like

Suppose Tech wrongly tries to send the customer back to Billing, who already handled them this turn:

```
🔀 transfer_to_billing({})
   → "Handoff to 'billing' refused (hop limit reached or agent already involved)..."
🤖 I can't transfer you back, but here is what I know: ...
```

---

## Part 7: When to Choose a Swarm

| Choose **swarm** when... | Choose **supervisor** when... |
|---|---|
| The conversation naturally moves between specialists | The task is a pipeline of sub-tasks |
| The active specialist should talk to the user directly | You want one consistent voice to the user |
| Agents can decide "this isn't mine" locally | You need global control/auditing of every step |
| You want fewer LLM calls per hop | You need guaranteed coverage (e.g. *always* run compliance check) |
| 3–6 peers with clear domains | Peers' domains overlap and routing needs a referee |

A common production combo: **swarm at the front** (support conversation), with **supervisors inside** a specialty for multi-step jobs.

Also: LangGraph's ecosystem has `langgraph-swarm` (`create_swarm(...)` + `create_handoff_tool`) that packages exactly this pattern. Learn the manual version first — you just built its core.

---

## Common Mistakes

| Mistake | Why it hurts | Fix |
|---|---|---|
| Parsing `HANDOFF:` from reply text | Fragile; wrong/no routes; prompt-injection risk | Use handoff **tools** returning `Command` |
| Forgetting the `ToolMessage` reply | Chat APIs error: tool call has no response | Include `ToolMessage` with `tool_call_id` |
| Missing `graph=Command.PARENT` | `goto` looks for the target *inside* the agent | Set `graph=Command.PARENT` |
| No hop limit / visited list | A → B → A → B ... burns tokens | `MAX_HOPS` + visited check |
| Not storing `active_agent` | Every user message restarts at triage | Save in state + checkpointer, route from `START` |
| Never resetting `hops` | Long chats eventually refuse all handoffs | Reset `hops`/`path` each user turn |
| Vague peer descriptions | Agents hand off randomly | One clear sentence per handoff tool |
| Agents with overlapping domains | Ownership ping-pong | Distinct domains; refuse-and-answer fallback |
| Giving every agent every handoff | Dense mesh = chaotic routing | Hand off only along sensible edges |

---

## Best Practices

1. **Handoff = tool.** Typed, logged, testable.
2. **Tell each agent its peers** in its prompt ("if X → call `transfer_to_tech`").
3. **Keep a sparse graph.** Triage → specialists, specialists ↔ only related peers.
4. **Persist `active_agent`** with a checkpointer so follow-ups reach the right owner.
5. **Guard against loops** with hops, visited, and `recursion_limit`.
6. **Refuse softly:** return a string telling the agent to answer itself.
7. **Each agent owns its tools.** Don't share all tools everywhere — that's the problem swarms solve.
8. **Log the handoff trail** (`path`) — it's your main debugging aid.

---

## Interview Questions & Answers

### 🟢 Easy

**Q1: What is a swarm architecture?**
A group of peer agents with no central controller. The agent currently handling the conversation can hand it to another agent, which then becomes the active agent and talks to the user directly.

**Q2: How is a handoff implemented in LangGraph?**
As a tool that returns a `Command(goto=<agent>, graph=Command.PARENT, update={...})`. The command routes to the target agent node and updates shared state (including `active_agent`).

**Q3: How is a swarm different from a supervisor?**
A supervisor centrally routes every step and workers return to it. In a swarm each agent decides handoffs itself, and the active agent responds to the user directly — no routing LLM call per hop.

### 🟡 Medium

**Q4: Why not parse `HANDOFF: billing` from the agent's text?**
Text is unstructured: formatting drift, hallucinated agent names, or the phrase appearing in normal prose would misroute. Tool calls are schema-validated and traceable.

**Q5: How do you prevent infinite handoffs?**
Track `hops` and a `path`/visited list in state; the handoff tool refuses when `hops >= MAX_HOPS` or the target was already involved, returning a message so the agent answers itself. Add `recursion_limit` as a backstop.

**Q6: How does the swarm resume with the right agent on the next user message?**
`active_agent` is part of state and saved by the checkpointer. A conditional edge from `START` reads it and routes to that agent.

### 🔴 Hard

**Q7: Why must the handoff tool include a `ToolMessage`?**
Chat model APIs require every assistant tool call to be followed by a tool response. Since the tool is escaping to the parent graph, we construct that `ToolMessage` ourselves (with the `tool_call_id`) and include it in the update so the receiving agent has a valid history.

**Q8: What are the failure modes of swarms at scale, and mitigations?**
Handoff loops, ambiguous ownership, and growing peer-awareness in prompts (O(n²) edges). Mitigate with sparse handoff graphs, hop/visited guards, clear domain boundaries, and — beyond ~6 peers — introduce hierarchy (Chapter 16.3).

**Q9: Would you ever combine swarm and supervisor?**
Yes. Use a swarm as the conversational front (peers hand off based on the user's evolving needs) and give a specialty agent an internal supervisor-with-workers subgraph for multi-step jobs. Each pattern is used where it fits.

---

## Summary

- A **swarm** is peer agents with no boss; the **active agent** owns the conversation.
- A **handoff tool** returns `Command(goto=..., graph=Command.PARENT, update=...)` and includes a matching `ToolMessage`.
- `active_agent` in state + a **checkpointer** + a conditional edge from `START` resumes the right peer next turn.
- Each agent is a real **tool-using agent** with its own domain tools plus handoff tools to relevant peers.
- **Loop prevention:** `MAX_HOPS`, visited list, `recursion_limit`; refusal is a *string*, not a crash.
- Don't parse `HANDOFF:` from prose.
- Swarm for **conversations that change owners**; supervisor for **task decomposition**.

---

## Hands-On Exercise

**Add a third peer: a `sales` agent.**

1. Create tools: `get_plan_price(plan: str) -> str` (return fake prices for `"basic"`, `"pro"`) and `create_quote(plan: str, seats: int) -> str`.
2. Create `to_sales = make_handoff_tool("sales", "Transfer to sales for pricing, plans, and upgrades.")`.
3. Build `sales_agent = make_agent([get_plan_price, create_quote, to_billing], "...")` with a prompt that says to hand off to billing for payment issues.
4. Give **triage** the `to_sales` tool (and update its prompt), and give **billing** the `to_sales` tool for upgrade questions.
5. Register the node and add `"sales"` to the list in `add_conditional_edges(START, ...)`.
6. Test a three-turn thread:
   - "How much is the pro plan for 10 seats?" → triage → sales
   - "Great, I was also double-charged on INV-1002." → sales → billing
   - "Actually, what would the pro plan cost?" → billing → sales is allowed, because `path` resets each user turn. 
   - Now try one message that needs sales → billing → sales (e.g. "Quote me the pro plan, fix my duplicate charge on INV-1002, then update the quote"). Watch the visited rule refuse the second sales handoff. Is that the behavior you want? How would you relax it safely (e.g. allow revisits but keep `MAX_HOPS`)?

**Success check:** `[active_agent=... path=...]` lines show the expected trail, and no thread ever exceeds `MAX_HOPS` handoffs in one turn.

---

## What's Next

Peers and a single supervisor both get unwieldy as the team grows. In **[Chapter 16.3: Hierarchical Multi-Agent](chapter-70-hierarchical.md)** you'll organize agents into **teams** with their own leads, compile each team as a **subgraph**, and let an executive router delegate to whole teams.

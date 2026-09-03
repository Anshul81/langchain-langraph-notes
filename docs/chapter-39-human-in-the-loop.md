# Chapter 10.5: Human-in-the-Loop Agents

> **Phase 10 — Agents** | [← Previous: Custom Agents](chapter-38-custom-agents.md) | [Next: Phase 11 — RAG →](../phase-10-rag/chapter-40-intro-to-rag.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand why human-in-the-loop (HITL) is essential for production agents
- ✅ Use `interrupt_before` and `interrupt_after` in LangGraph
- ✅ Implement approval flows for dangerous tool calls
- ✅ Build agents that can ask for user clarification mid-execution
- ✅ Use checkpointers to resume interrupted agents
- ✅ Build a **safe order management agent** with approval workflows

| | |
|---|---|
| **Prerequisites** | Chapter 10.3 (LangGraph Fundamentals), Chapter 10.4 (Custom Agents) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction — Why Human-in-the-Loop?

Agents are powerful but **unpredictable**. Without human oversight, they can:

```
🚨 WHAT CAN GO WRONG:
├── Send an email to the wrong person
├── Delete database records irreversibly
├── Approve a refund they shouldn't have
├── Deploy untested code to production
├── Share confidential information
└── Execute a harmful shell command
```

**Human-in-the-loop (HITL)** adds a checkpoint: the agent pauses before executing risky actions and waits for human approval.

```
WITHOUT HITL:
User → Agent → [calls delete_records()] → Data deleted! ❌ No undo!

WITH HITL:
User → Agent → [wants to call delete_records()]
              → PAUSE ⏸️
              → "I want to delete 500 records. Approve? [Y/N]"
              → Human: "No! Only delete the 3 inactive ones."
              → Agent continues with corrected action ✅
```

---

## Part 1: `interrupt_before` — Pause Before a Node

### Basic Interrupt

```python
import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)


# --- Tools ---

@tool
def search_orders(customer_name: str) -> str:
    """Search for customer orders by name. This is SAFE (read-only)."""
    orders = {
        "rahul": "ORD-1001: Laptop ($999), Status: Delivered",
        "priya": "ORD-1002: Headphones ($49), Status: Shipped",
    }
    return orders.get(customer_name.lower(), f"No orders found for {customer_name}")


@tool
def process_refund(order_id: str, amount: float, reason: str) -> str:
    """Process a refund for an order. This is DANGEROUS (modifies data, irreversible)."""
    return f"✅ Refund of ${amount:.2f} processed for {order_id}. Reason: {reason}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer. This is DANGEROUS (irreversible communication)."""
    return f"✅ Email sent to {to}. Subject: {subject}"


# Create agent with interrupt_before on the tools node
memory = MemorySaver()

agent = create_react_agent(
    llm,
    [search_orders, process_refund, send_email],
    checkpointer=memory,
    interrupt_before=["tools"]  # ⏸️ Pause BEFORE executing ANY tool
)
```

### Running with Interrupts

```python
config = {"configurable": {"thread_id": "demo-1"}}

# Start the agent
print("🚀 Starting agent...")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Process a $49 refund for order ORD-1002, reason: defective item"}]},
    config=config
)

# The agent is now PAUSED before the tools node
# Check what tool it wants to call:
last_message = result["messages"][-1]
if last_message.tool_calls:
    print("\n⏸️ Agent wants to call:")
    for tc in last_message.tool_calls:
        print(f"   🔧 {tc['name']}({tc['args']})")
    
    # Human decision: approve or reject
    approved = input("\n   Approve? (y/n): ").lower() == "y"
    
    if approved:
        # Resume — continue execution
        print("\n✅ Approved! Resuming...")
        result = agent.invoke(None, config=config)  # None = continue from checkpoint
        print(f"💬 {result['messages'][-1].content}")
    else:
        # Reject — send feedback
        print("\n❌ Rejected! Sending feedback...")
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "No, don't process that refund. Just look up the order status instead."}]},
            config=config
        )
        print(f"💬 {result['messages'][-1].content}")
```

---

## Part 2: Selective Interrupts — Only Pause for Dangerous Tools

Pausing before EVERY tool call is annoying. Let's only pause for dangerous ones:

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# Define which tools are dangerous
DANGEROUS_TOOLS = {"process_refund", "send_email", "delete_record"}
SAFE_TOOLS = {"search_orders", "get_weather", "calculate"}

all_tools = [search_orders, process_refund, send_email]
tool_map = {t.name: t for t in all_tools}
llm_with_tools = llm.bind_tools(all_tools)


def agent_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def safe_tool_node(state: AgentState) -> dict:
    """Execute only safe tools."""
    last_msg = state["messages"][-1]
    results = []
    for tc in last_msg.tool_calls:
        if tc["name"] in SAFE_TOOLS:
            result = tool_map[tc["name"]].invoke(tc["args"])
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


def dangerous_tool_node(state: AgentState) -> dict:
    """Execute dangerous tools (only reached after human approval)."""
    last_msg = state["messages"][-1]
    results = []
    for tc in last_msg.tool_calls:
        if tc["name"] in DANGEROUS_TOOLS:
            result = tool_map[tc["name"]].invoke(tc["args"])
            results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}


def route_tools(state: AgentState):
    last_msg = state["messages"][-1]
    
    if not last_msg.tool_calls:
        return END
    
    # Check if any tool call is dangerous
    has_dangerous = any(
        tc["name"] in DANGEROUS_TOOLS 
        for tc in last_msg.tool_calls
    )
    
    if has_dangerous:
        return "dangerous_tools"
    return "safe_tools"


# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("safe_tools", safe_tool_node)
graph.add_node("dangerous_tools", dangerous_tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route_tools)
graph.add_edge("safe_tools", "agent")
graph.add_edge("dangerous_tools", "agent")

# Only interrupt before dangerous tools!
memory = MemorySaver()
safe_agent = graph.compile(
    checkpointer=memory,
    interrupt_before=["dangerous_tools"]  # Only pause here
)

# Test — safe query flows through without interruption
config = {"configurable": {"thread_id": "demo-2"}}
result = safe_agent.invoke(
    {"messages": [HumanMessage(content="Look up orders for Rahul")]},
    config=config
)
print(f"💬 Safe query (no interrupt): {result['messages'][-1].content}")

# Test — dangerous query pauses for approval
config = {"configurable": {"thread_id": "demo-3"}}
result = safe_agent.invoke(
    {"messages": [HumanMessage(content="Refund $49 for order ORD-1002")]},
    config=config
)
print(f"\n⏸️ Paused! Wants to call: {result['messages'][-1].tool_calls}")
```

---

## Part 3: `interrupt_after` — Pause After Execution

Sometimes you want to execute first, then ask for confirmation before continuing:

```python
# interrupt_after: Execute the node, THEN pause
# Useful for: "Here's what I found. Should I continue?"

agent_review = create_react_agent(
    llm,
    [search_orders, process_refund],
    checkpointer=MemorySaver(),
    interrupt_after=["tools"]  # Execute tools, then pause to show results
)

config = {"configurable": {"thread_id": "review-1"}}

result = agent_review.invoke(
    {"messages": [HumanMessage(content="Look up Rahul's orders and process a refund if appropriate")]},
    config=config
)

# Agent searched orders, then paused AFTER showing results
print("Tool executed. Results:")
for msg in result["messages"]:
    if hasattr(msg, 'content') and msg.content:
        print(f"  {msg.content[:200]}")

# Human reviews and decides to continue
print("\nHuman approves continuation...")
result = agent_review.invoke(None, config=config)
print(f"💬 {result['messages'][-1].content}")
```

---

## Part 4: Editing Tool Calls Before Execution

The most powerful HITL pattern — review AND modify the agent's planned actions:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

agent = create_react_agent(
    llm,
    [search_orders, process_refund, send_email],
    checkpointer=memory,
    interrupt_before=["tools"]
)

config = {"configurable": {"thread_id": "edit-1"}}

# Agent plans its action
result = agent.invoke(
    {"messages": [HumanMessage(content="Send an email to rahul@example.com about his refund")]},
    config=config
)

# Review what the agent wants to do
last_msg = result["messages"][-1]
print("⏸️ Agent wants to:")
for tc in last_msg.tool_calls:
    print(f"   {tc['name']}({tc['args']})")

# Option 1: Approve as-is
# result = agent.invoke(None, config=config)

# Option 2: Modify the tool call before executing
# Get the current state
state = agent.get_state(config)

# Modify the last message's tool calls
modified_message = last_msg.model_copy()
for tc in modified_message.tool_calls:
    if tc["name"] == "send_email":
        tc["args"]["subject"] = "Refund Confirmation — Order ORD-1002"  # Fix subject
        tc["args"]["body"] = "Dear Rahul, your refund has been processed."

# Update the state with modified message
agent.update_state(config, {"messages": [modified_message]})

# Resume with the modified action
result = agent.invoke(None, config=config)
print(f"\n💬 {result['messages'][-1].content}")
```

---

## Part 5: Complete Project — Safe Order Management Agent

```python
import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

# --- Simulated Database ---
ORDERS = {
    "ORD-1001": {"customer": "Rahul", "email": "rahul@example.com",
                 "item": "Laptop", "amount": 999.00, "status": "delivered"},
    "ORD-1002": {"customer": "Priya", "email": "priya@example.com",
                 "item": "Headphones", "amount": 49.99, "status": "shipped"},
    "ORD-1003": {"customer": "Amit", "email": "amit@example.com",
                 "item": "Keyboard", "amount": 79.99, "status": "processing"},
}

# --- Safe Tools (read-only) ---

@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by ID. SAFE — read-only operation."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"Order {order_id} not found."
    return (f"Order {order_id}: {order['item']} (${order['amount']}) "
            f"for {order['customer']} ({order['email']}). Status: {order['status']}")

@tool
def search_customer(name: str) -> str:
    """Search for a customer's orders by name. SAFE — read-only."""
    matches = [(oid, o) for oid, o in ORDERS.items() 
               if name.lower() in o["customer"].lower()]
    if not matches:
        return f"No orders found for customer '{name}'."
    return "\n".join(f"{oid}: {o['item']} (${o['amount']}) — {o['status']}"
                     for oid, o in matches)

@tool
def check_refund_eligibility(order_id: str) -> str:
    """Check if an order is eligible for a refund. SAFE — read-only."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"Order {order_id} not found."
    if order["status"] == "delivered":
        return f"Order {order_id} IS eligible for refund (delivered). Amount: ${order['amount']}"
    elif order["status"] == "shipped":
        return f"Order {order_id} is NOT eligible yet (still in transit). Wait for delivery."
    else:
        return f"Order {order_id} can be cancelled instead of refunded (still {order['status']})."

# --- Dangerous Tools (modifies data) ---

@tool
def process_refund(order_id: str, amount: float, reason: str) -> str:
    """Process a refund for an order. ⚠️ DANGEROUS — irreversible financial operation."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"Error: Order {order_id} not found."
    if amount > order["amount"]:
        return f"Error: Refund amount ${amount} exceeds order total ${order['amount']}."
    ORDERS[order_id.upper()]["status"] = "refunded"
    return f"✅ Refund of ${amount:.2f} processed for {order_id}. Reason: {reason}."

@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order. ⚠️ DANGEROUS — cannot be undone."""
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"Error: Order {order_id} not found."
    if order["status"] == "delivered":
        return f"Error: Cannot cancel delivered order. Use refund instead."
    ORDERS[order_id.upper()]["status"] = "cancelled"
    return f"✅ Order {order_id} has been cancelled."

@tool
def send_notification(email: str, message: str) -> str:
    """Send a notification email to a customer. ⚠️ DANGEROUS — sends real email."""
    return f"✅ Email sent to {email}: {message}"


# --- Build the Safe Agent ---

all_tools = [lookup_order, search_customer, check_refund_eligibility,
             process_refund, cancel_order, send_notification]

memory = MemorySaver()

safe_agent = create_react_agent(
    llm,
    all_tools,
    checkpointer=memory,
    interrupt_before=["tools"],  # Pause before ALL tool executions
    state_modifier=SystemMessage(content="""You are a customer support agent for an online store.

Safety Rules:
1. Always look up the order BEFORE processing any refund or cancellation.
2. Check refund eligibility before processing refunds.
3. Never process a refund exceeding the order amount.
4. Always confirm the action with the customer before proceeding.
5. After a refund or cancellation, notify the customer via email.

Be friendly, professional, and thorough.""")
)


# --- Interactive Session ---

class SafeAgentSession:
    """Interactive session with the safe agent."""
    
    def __init__(self):
        self.thread_count = 0
    
    def chat(self, message: str, auto_approve_safe: bool = True):
        """Chat with the agent, handling interrupts."""
        self.thread_count += 1
        config = {"configurable": {"thread_id": f"session-{self.thread_count}"}}
        
        print(f"\n👤 Customer: {message}")
        
        # Initial invocation
        result = safe_agent.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config
        )
        
        # Handle interrupt loop
        max_rounds = 10
        for round_num in range(max_rounds):
            last_msg = result["messages"][-1]
            
            # Check if we're at an interrupt (has tool calls)
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                safe_tools = {"lookup_order", "search_customer", "check_refund_eligibility"}
                dangerous_tools = {"process_refund", "cancel_order", "send_notification"}
                
                requested_tools = [tc["name"] for tc in last_msg.tool_calls]
                has_dangerous = any(t in dangerous_tools for t in requested_tools)
                
                if has_dangerous:
                    # Dangerous tools — require approval
                    print(f"\n⚠️ Agent wants to perform DANGEROUS action(s):")
                    for tc in last_msg.tool_calls:
                        risk = "🔴 DANGEROUS" if tc["name"] in dangerous_tools else "🟢 Safe"
                        print(f"   {risk}: {tc['name']}({tc['args']})")
                    
                    # In a real app, this would be a UI prompt
                    approved = input("   Approve? (y/n): ").strip().lower() == "y"
                    
                    if approved:
                        print("   ✅ Approved!")
                        result = safe_agent.invoke(None, config=config)
                    else:
                        print("   ❌ Rejected!")
                        result = safe_agent.invoke(
                            {"messages": [HumanMessage(content="The supervisor rejected this action. Please suggest an alternative.")]},
                            config=config
                        )
                elif auto_approve_safe:
                    # Safe tools — auto-approve
                    print(f"   🟢 Auto-approving safe tools: {requested_tools}")
                    result = safe_agent.invoke(None, config=config)
                else:
                    # Manual approval for all
                    print(f"\n🔧 Agent wants to call: {requested_tools}")
                    result = safe_agent.invoke(None, config=config)
            else:
                # No tool calls — final answer
                print(f"\n🤖 Agent: {last_msg.content}")
                return result
        
        print("⚠️ Max rounds reached")
        return result


# --- Demo ---
session = SafeAgentSession()

# Safe query — auto-approved
session.chat("Look up order ORD-1001")

# Dangerous query — needs approval
session.chat("Process a full refund for order ORD-1001 because the laptop was defective")

# Multi-step query
session.chat("Check if order ORD-1002 is eligible for a refund")
```

---

## Common Mistakes

### Mistake 1: Forgetting the checkpointer
```python
# ❌ interrupt_before without checkpointer — can't resume!
agent = create_react_agent(llm, tools, interrupt_before=["tools"])

# ✅ Always pair interrupts with a checkpointer
memory = MemorySaver()
agent = create_react_agent(llm, tools, checkpointer=memory, interrupt_before=["tools"])
```

### Mistake 2: Not passing `None` to resume
```python
# ❌ Passing new messages to resume — creates a new conversation
result = agent.invoke(
    {"messages": [HumanMessage(content="continue")]},
    config=config
)

# ✅ Pass None to resume from the interrupt point
result = agent.invoke(None, config=config)
```

### Mistake 3: Interrupting on every tool (too aggressive)
```python
# ❌ Pausing for search, calculate, etc. — annoying!
agent = create_react_agent(llm, tools, interrupt_before=["tools"])

# ✅ Build a custom graph that only interrupts for dangerous tools
# Or classify tools and route to different nodes
```

### Mistake 4: Not handling the rejection case
```python
# ❌ Only handling approval — what if the human says no?
if approved:
    result = agent.invoke(None, config=config)
# What happens if not approved? The agent is stuck!

# ✅ Handle rejection with feedback
if approved:
    result = agent.invoke(None, config=config)
else:
    result = agent.invoke(
        {"messages": [HumanMessage(content="Action was rejected. Please try a different approach.")]},
        config=config
    )
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Always use a checkpointer with interrupts | Required to resume from interrupt |
| Classify tools by risk level (safe vs dangerous) | Only interrupt for dangerous tools |
| Show the user exactly what the agent wants to do | Transparency builds trust |
| Allow editing tool calls, not just approve/reject | More flexible than binary choices |
| Log all approvals and rejections | Audit trail for compliance |
| Auto-approve safe (read-only) tools | Better UX — less friction |
| Provide rejection feedback to the agent | Agent can try alternative approaches |
| Use thread_ids to manage separate conversations | Prevent state corruption |

---

## Interview Preparation

### Easy
**Q: What is human-in-the-loop in AI agents?**

> Human-in-the-loop (HITL) is a pattern where an AI agent pauses before executing potentially risky actions and waits for human approval. In LangGraph, this is implemented using `interrupt_before` or `interrupt_after` on specific nodes. The agent's state is checkpointed so it can resume after the human approves, rejects, or modifies the planned action. HITL is essential for production agents that handle irreversible actions like sending emails, processing refunds, or deleting data.

### Medium
**Q: How do you implement selective interrupts in LangGraph?**

> Build a custom graph with separate nodes for safe and dangerous tools. The agent node calls the LLM; a routing function checks if the requested tools are dangerous. Safe tools route to a `safe_tools` node (auto-executed). Dangerous tools route to a `dangerous_tools` node with `interrupt_before=["dangerous_tools"]`. This way, read-only operations (search, lookup) flow through without interruption, while write operations (refund, delete, email) require human approval. Always use a checkpointer to enable resumption after the interrupt.

### Hard
**Q: Design a production-ready approval workflow for an AI agent handling financial operations.**

> Multi-layer architecture: (1) **Risk classification** — categorize tools as low-risk (lookups, read-only), medium-risk (status updates, notifications), and high-risk (refunds, cancellations, payments). (2) **Approval routing** — low-risk: auto-execute. Medium-risk: require single human approval. High-risk: require dual approval (two different humans). (3) **Context presentation** — show the approver the full context: what the agent wants to do, why, the conversation history, and the potential impact. (4) **Modification** — allow approvers to edit tool arguments before approving (e.g., change refund amount). (5) **Audit logging** — log every action with timestamp, approver, decision, and reasoning. (6) **Time-based expiry** — if no approval within N minutes, auto-reject with notification. (7) **Escalation** — if an agent repeatedly gets rejected, escalate to a human agent entirely. (8) **Testing** — test with simulated approval/rejection flows in CI/CD.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Human-in-the-loop** | Agent pauses for human approval before risky actions |
| **`interrupt_before`** | Pause BEFORE executing a node |
| **`interrupt_after`** | Pause AFTER executing a node |
| **Checkpointer** | Required — saves state so agent can resume |
| **`MemorySaver`** | In-memory checkpointer for development |
| **Resume with `None`** | `agent.invoke(None, config)` continues from interrupt |
| **Edit tool calls** | Modify the agent's planned action before executing |
| **Selective interrupts** | Only pause for dangerous tools, auto-approve safe ones |
| **Thread ID** | Identifies which conversation to resume |

---

> [← Previous: Custom Agents](chapter-38-custom-agents.md) | [Next: Phase 11 — RAG →](../phase-10-rag/chapter-40-intro-to-rag.md)

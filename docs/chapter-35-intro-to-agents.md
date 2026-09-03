# Chapter 10.1: Introduction to Agents — When LLMs Think for Themselves

> **Phase 10 — Agents** | [← Previous: MCP](../phase-08-tools-tool-calling/chapter-34-mcp.md) | [Next: ReAct Agents →](chapter-36-react-agents.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand what agents are and how they differ from chains
- ✅ Know the agent loop: Observe → Think → Act → Repeat
- ✅ Understand different agent architectures (ReAct, Plan-and-Execute, etc.)
- ✅ Know when to use agents vs chains
- ✅ Understand the risks and limitations of agents
- ✅ Preview the agent-building journey in this phase

| | |
|---|---|
| **Prerequisites** | Phase 9 (Tools & Tool Calling) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 15 minutes (conceptual chapter) |

---

## Introduction — From Chains to Agents

In chains, **you** decide the steps:

```python
# CHAIN: Developer controls the flow
chain = retrieve_docs | format_prompt | llm | parse_output
# Step 1 → Step 2 → Step 3 → Step 4  (always, every time)
```

In agents, **the LLM** decides the steps:

```python
# AGENT: LLM controls the flow
agent = create_react_agent(llm, tools=[search, calculator, wiki])
# LLM thinks: "What tool should I use? Do I need more info? Am I done?"
# The LLM LOOPS until it has enough information to answer.
```

### The Key Difference

```
CHAIN (Fixed Pipeline):
──────────────────────
User → Step A → Step B → Step C → Answer
        ↓         ↓         ↓
     (always)  (always)  (always)

AGENT (Dynamic Loop):
─────────────────────
User → Think → Act → Observe → Think → Act → Observe → Answer
         ↓      ↓       ↓         ↓      ↓       ↓
        (LLM)  (tool)  (result)  (LLM)  (tool)  (result)
         └──────────────────────────┘
               Loops until done
```

**Agents are LLMs that can autonomously decide which tools to use, in what order, and when to stop.**

---

## Part 1: What Is an Agent?

### Definition

An **agent** is a system where an LLM:
1. **Observes** the current state (user query + previous results)
2. **Thinks** about what to do next (reason about the problem)
3. **Acts** by choosing a tool to call (or decides to respond)
4. **Repeats** until it has enough information to answer

```
┌──────────────────────────────────────────────────┐
│                   AGENT LOOP                      │
│                                                   │
│   ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│   │ OBSERVE  │───→│  THINK   │───→│   ACT     │  │
│   │          │    │          │    │           │  │
│   │ Read     │    │ Reason   │    │ Call tool │  │
│   │ context  │    │ about    │    │ OR answer │  │
│   │ + results│    │ next step│    │ directly  │  │
│   └──────────┘    └──────────┘    └─────┬─────┘  │
│        ↑                                │        │
│        │          ┌──────────┐          │        │
│        └──────────│ OBSERVE  │←─────────┘        │
│                   │ (result) │                    │
│                   └──────────┘                    │
│                                                   │
│   Loop continues until LLM decides: "I'm done"   │
└──────────────────────────────────────────────────┘
```

### A Real Example

```
User: "What's the population of India and how does it compare to the EU total?"

AGENT LOOP:

  Round 1 — THINK: "I need India's population. Let me search."
            ACT:   search("India population 2025")
            OBSERVE: "India's population is approximately 1.44 billion"

  Round 2 — THINK: "Now I need the EU's total population."
            ACT:   search("European Union total population 2025")
            OBSERVE: "EU population is approximately 448 million"

  Round 3 — THINK: "Now I need to compare them. Let me calculate."
            ACT:   calculator("1440000000 / 448000000")
            OBSERVE: "3.214"

  Round 4 — THINK: "I have all the data. I can answer now."
            ACT:   RESPOND
            
  Answer: "India's population (~1.44 billion) is about 3.2x larger 
           than the entire EU (~448 million)."
```

### Agents vs Chains — Comparison

| Aspect | Chain | Agent |
|--------|-------|-------|
| **Control flow** | Fixed, developer-defined | Dynamic, LLM-decided |
| **Steps** | Always the same | Varies per query |
| **Tool usage** | Hardcoded tool order | LLM picks tools as needed |
| **Loops** | No (single pass) | Yes (iterates until done) |
| **Predictability** | High (same input → same flow) | Low (non-deterministic) |
| **Cost** | Fixed (known LLM calls) | Variable (1-10+ LLM calls) |
| **Debugging** | Easy | Hard (emergent behavior) |
| **Best for** | Known workflows | Open-ended questions |

---

## Part 2: Agent Architectures

### ReAct (Reasoning + Acting)

The most common and foundational agent pattern:

```
Thought: I need to find the current weather in Mumbai.
Action: search("weather Mumbai today")
Observation: Current weather in Mumbai: 32°C, Partly Cloudy

Thought: The user also asked about Delhi. Let me search for that too.
Action: search("weather Delhi today")
Observation: Current weather in Delhi: 38°C, Sunny

Thought: I now have weather for both cities. I can answer.
Final Answer: Mumbai is 32°C and Partly Cloudy, while Delhi is 38°C and Sunny.
```

**ReAct = interleaved Reasoning and Acting**. The LLM thinks out loud (chain-of-thought), then acts.

### Plan-and-Execute

A two-step approach — plan first, then execute:

```
PLANNER:
  Step 1: Search for India's GDP
  Step 2: Search for China's GDP
  Step 3: Calculate the percentage difference
  Step 4: Compile the answer

EXECUTOR:
  Executing Step 1... ✅ India's GDP: $3.7T
  Executing Step 2... ✅ China's GDP: $17.8T
  Executing Step 3... ✅ Difference: 381%
  Executing Step 4... ✅ Final answer composed
```

### Self-Ask

The agent breaks complex questions into sub-questions:

```
Question: "Was the inventor of the telephone born before or after 
           the American Civil War ended?"

Sub-question 1: "Who invented the telephone?"
Answer: Alexander Graham Bell

Sub-question 2: "When was Alexander Graham Bell born?"
Answer: March 3, 1847

Sub-question 3: "When did the American Civil War end?"
Answer: April 9, 1865

Final: Bell was born in 1847, the Civil War ended in 1865.
       So Bell was born BEFORE the Civil War ended.
```

### Tool-Calling Agent (Modern Default)

Uses native tool calling APIs (not text-based Thought/Action parsing):

```
Messages:
  User: "What's 2^32?"
  AI: [tool_call: calculator(expression="2**32")]
  Tool: "4294967296"
  AI: "2 to the power of 32 is 4,294,967,296."
```

This is what LangChain and LangGraph use by default — cleaner and more reliable than ReAct text parsing.

### Comparison

| Architecture | Reasoning | Planning | Tool Calls | Best For |
|-------------|-----------|----------|------------|----------|
| **ReAct** | Step-by-step | No upfront plan | After each thought | General-purpose |
| **Plan-and-Execute** | Upfront | Yes | Per plan step | Complex, multi-step tasks |
| **Self-Ask** | Decomposition | As sub-questions | Per sub-question | Complex factual queries |
| **Tool-Calling** | Implicit | No | Via API | Production (most reliable) |

---

## Part 3: When to Use Agents vs Chains

### Use Chains When:

```
✅ The workflow is KNOWN and FIXED:
   "Always: retrieve docs → format prompt → generate answer"

✅ Predictability matters:
   "Every customer gets the same 3-step onboarding flow"

✅ Cost must be controlled:
   "Budget: exactly 2 LLM calls per request"

✅ The task is straightforward:
   "Summarize this document"
   "Translate this text"
   "Extract entities from this paragraph"
```

### Use Agents When:

```
✅ The workflow is UNKNOWN or VARIABLE:
   "Answer any question using whatever tools are needed"

✅ Multiple tools might be needed (or none):
   "User asks about weather → search. User asks about math → calculator.
    User says hi → no tool."

✅ The number of steps varies:
   "Simple question = 1 step. Complex question = 5 steps."

✅ The task requires reasoning about WHICH tools to use:
   "Given these 10 tools, figure out the right approach"
```

### The Decision Framework

```
Is the workflow fixed and known?
├── YES → Use a CHAIN
│         (faster, cheaper, more predictable)
│
└── NO  → Does the LLM need to choose between tools?
          ├── YES → Use an AGENT
          │         (flexible, but costlier and less predictable)
          │
          └── NO  → Use a simple LLM call
                    (no tools needed)
```

---

## Part 4: The Risks of Agents

### Risk 1: Infinite Loops

```python
# The agent calls a tool, gets an unhelpful result, tries again...
# Round 1: search("latest AI news") → "No results found"
# Round 2: search("AI news today") → "No results found"
# Round 3: search("artificial intelligence news") → "No results found"
# Round 4: search("AI updates") → "No results found"
# ... forever!

# MITIGATION: Set max_iterations
agent = create_react_agent(llm, tools)
result = agent.invoke(
    {"messages": [...]},
    config={"recursion_limit": 10}  # Stop after 10 rounds
)
```

### Risk 2: Cost Explosion

```
Each agent round = 1 LLM API call
Complex question might need 5-10 rounds
Each round costs ~$0.01-0.05 (GPT-4o)

10 rounds × $0.03 = $0.30 per question!
1000 users × 10 questions × $0.30 = $3,000/day

MITIGATION:
├── Set max_iterations (cap at 5-10)
├── Use cheaper models for tool selection (GPT-4o-mini)
├── Cache common tool results
├── Use chains for predictable workflows
└── Monitor and alert on high-iteration queries
```

### Risk 3: Hallucinated Tool Calls

```python
# The LLM might:
# ❌ Call a tool that doesn't exist
# ❌ Pass wrong arguments to a tool
# ❌ Misinterpret tool results
# ❌ Make up results instead of calling a tool

# MITIGATION:
# - Use strong models (GPT-4o > GPT-3.5 for tool selection)
# - Write excellent tool descriptions
# - Validate tool call arguments
# - Check for invalid_tool_calls
```

### Risk 4: Unsafe Actions

```python
# With tools like shell, email, database write:
# The agent might autonomously:
# ❌ Delete database records
# ❌ Send emails to customers
# ❌ Execute destructive commands

# MITIGATION: Human-in-the-loop
# LangGraph provides interrupt_before for dangerous tools
# "Pause and ask the human before executing this tool"
```

---

## Part 5: Agent Building Blocks in LangChain

### The Evolution

```
2023 (Legacy):
  AgentExecutor + AgentType.ZERO_SHOT_REACT_DESCRIPTION
  → Deprecated. Don't use for new projects.

2024+ (Current):
  LangGraph + create_react_agent()
  → The recommended way to build agents.
```

### LangGraph — The Agent Framework

LangGraph is LangChain's framework for building **stateful, multi-step** AI workflows:

```python
from langgraph.prebuilt import create_react_agent

# The simplest agent — 3 lines!
agent = create_react_agent(llm, tools)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

### What You'll Build in This Phase

| Chapter | What You'll Build |
|---------|------------------|
| **10.1** (this) | Understanding agents conceptually |
| **10.2** | ReAct agents with `create_react_agent` |
| **10.3** | LangGraph fundamentals (StateGraph, nodes, edges) |
| **10.4** | Custom agent architectures with LangGraph |
| **10.5** | Human-in-the-loop agents |

---

## Part 6: Quick Agent Demo

A preview of what's coming in the next chapters:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

# Tools
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather = {
        "Mumbai": "32°C, Partly Cloudy, Humidity 78%",
        "Delhi": "38°C, Sunny, Humidity 45%",
        "London": "18°C, Overcast, Humidity 72%",
    }
    return weather.get(city, f"Weather data not available for {city}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Input should be a valid Python math expression."""
    import math
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {str(e)}"

# Create agent — THIS IS ALL IT TAKES!
agent = create_react_agent(llm, [get_weather, calculate])

# The agent DECIDES which tools to use (or none)
queries = [
    "What's the weather in Mumbai?",                    # → uses get_weather
    "What is 2 to the power of 20?",                    # → uses calculate
    "Compare the weather in Mumbai and Delhi",          # → uses get_weather TWICE
    "Hello! What's your name?",                         # → no tool needed
    "What's the temperature in London in Fahrenheit?",  # → get_weather + calculate
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"❓ {query}")
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    print(f"💬 {result['messages'][-1].content}")
```

---

## Common Mistakes

### Mistake 1: Using agents for everything
```python
# ❌ Using an agent for a fixed workflow
agent = create_react_agent(llm, [retriever_tool])
result = agent.invoke("Summarize my documents")
# Overkill! The agent just calls the retriever every time anyway.

# ✅ Use a chain for fixed workflows — simpler, faster, cheaper
chain = retriever | format_prompt | llm | parse
result = chain.invoke("Summarize my documents")
```

### Mistake 2: No iteration limit
```python
# ❌ Agent can loop forever
result = agent.invoke({"messages": [...]})

# ✅ Always set a recursion limit
result = agent.invoke(
    {"messages": [...]},
    config={"recursion_limit": 10}
)
```

### Mistake 3: Using the legacy AgentExecutor
```python
# ❌ DEPRECATED — don't use in new projects
from langchain.agents import initialize_agent, AgentType
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# ✅ Use LangGraph's create_react_agent
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm, tools)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Start with chains, graduate to agents when needed | Chains are simpler, faster, and cheaper |
| Always set `recursion_limit` | Prevents infinite loops and cost explosions |
| Use GPT-4o-class models for agents | Better tool selection than smaller models |
| Limit to 3-5 focused tools | Too many tools confuse the agent |
| Write excellent tool descriptions | Agents live or die by description quality |
| Log all agent steps | Essential for debugging non-deterministic behavior |
| Use human-in-the-loop for risky actions | Safety for write/delete/send operations |
| Monitor cost per agent invocation | Agents can be expensive at scale |

---

## Interview Preparation

### Easy
**Q: What is an AI agent?**

> An AI agent is a system where an LLM autonomously decides which tools to use, in what order, and when to stop. Unlike chains (fixed pipelines), agents have a dynamic loop: the LLM observes the current state, thinks about what to do next, acts (calls a tool or responds), and repeats until it has enough information to answer. Agents are ideal for open-ended questions where the workflow isn't known in advance.

### Medium
**Q: What is the ReAct agent pattern?**

> ReAct (Reasoning + Acting) is an agent pattern where the LLM alternates between reasoning (thinking out loud in chain-of-thought) and acting (calling tools). The cycle is: Thought → Action → Observation → Thought → Action → Observation → ... → Final Answer. The "thought" step makes the agent's reasoning transparent and debuggable. In LangChain/LangGraph, `create_react_agent` implements this pattern using native tool calling APIs rather than text-based parsing.

### Hard
**Q: When would you use a chain vs an agent, and what are the trade-offs?**

> Use **chains** when the workflow is known, fixed, and predictable — like RAG (always: retrieve → format → generate). Chains are faster (fewer LLM calls), cheaper, more debuggable, and deterministic. Use **agents** when the workflow is unknown or varies per query — like a research assistant that might need search, calculation, or no tools at all. Trade-offs: agents are more flexible but non-deterministic, costlier (multiple LLM rounds), harder to debug, and risk infinite loops. The rule of thumb: start with chains, only switch to agents when you genuinely need the LLM to make routing decisions.

### Senior
**Q: How would you make agents production-ready?**

> Production agents need multiple layers: (1) **Iteration limits** — cap at 5-10 rounds, with graceful degradation. (2) **Cost controls** — budget per query, alerts on high-iteration queries, use cheaper models for tool selection. (3) **Human-in-the-loop** — interrupt before dangerous tools (delete, send, deploy). (4) **Observability** — log every thought/action/observation, trace latency per tool, track total tokens used. (5) **Error handling** — catch and handle tool failures, invalid tool calls, and timeout. (6) **Caching** — cache common tool results to reduce API calls. (7) **Testing** — evaluation suite with expected tool sequences for known queries. (8) **Fallback** — if agent exceeds limits, fall back to direct LLM response with disclaimer. (9) **Guardrails** — input/output filtering to prevent prompt injection and harmful outputs.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Agent** | LLM autonomously decides which tools to use and when to stop |
| **Chain** | Fixed pipeline — developer controls the flow |
| **Agent loop** | Observe → Think → Act → Repeat until done |
| **ReAct** | Reasoning + Acting — interleaved thinking and tool use |
| **Plan-and-Execute** | Plan all steps first, then execute sequentially |
| **Tool-Calling Agent** | Uses native API tool calls (modern, recommended) |
| **LangGraph** | The framework for building agents (replaces AgentExecutor) |
| **`create_react_agent`** | Simplest way to create an agent in LangGraph |
| **Recursion limit** | Max iterations to prevent infinite loops |
| **Cost risk** | Agents can make 5-10+ LLM calls per query |

---

> [← Previous: MCP](../phase-08-tools-tool-calling/chapter-34-mcp.md) | [Next: ReAct Agents →](chapter-36-react-agents.md)

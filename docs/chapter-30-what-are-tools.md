# Chapter 9.1: What Are Tools? Why LLMs Need Them

> **Phase 9 — Tools & Tool Calling** | [← Previous: Cloud Vector DBs](../phase-07-vector-databases/chapter-29-cloud-vector-dbs.md) | [Next: Built-in Tools →](chapter-31-builtin-tools.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand why LLMs are fundamentally limited without tools
- ✅ Know the tool calling architecture (LLM decides → system executes → LLM reads result)
- ✅ Understand the difference between tool calling and function calling
- ✅ Know the taxonomy of tools (retrieval, action, computation, communication)
- ✅ Understand how tools fit into agents, chains, and RAG

| | |
|---|---|
| **Prerequisites** | Phase 6 (Chains & Runnables), Phase 8 (Memory) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 15 minutes (conceptual chapter) |

---

## Introduction — The Problem with Naked LLMs

LLMs are incredibly powerful at **language**. But they're terrible at everything else:

```
❌ "What's the weather in Mumbai right now?"      → Makes up a number
❌ "What's 7,847 × 3,291?"                        → Confidently gives wrong answer
❌ "Search the latest news about LangChain"        → Hallucinates articles
❌ "Send an email to john@example.com"             → Can't. No ability to act.
❌ "What's in my database?"                        → Can't access external systems
❌ "What time is it?"                              → Doesn't know. Frozen in training.
```

### The Root Cause

LLMs are **text-in, text-out** functions. They:
- ✅ **Can**: Reason, summarize, translate, generate, classify, extract
- ❌ **Cannot**: Access the internet, do math reliably, read files, call APIs, take actions

```
┌──────────────────────────────────────┐
│              LLM Brain               │
│                                      │
│  ✅ Language understanding           │
│  ✅ Reasoning & planning             │
│  ✅ Knowledge (from training data)   │
│                                      │
│  ❌ No internet access               │
│  ❌ No calculator                    │
│  ❌ No file system access            │
│  ❌ No API calling ability           │
│  ❌ No real-time information         │
│  ❌ No ability to take actions       │
└──────────────────────────────────────┘
```

**Tools are the bridge.** They give LLMs **hands** to interact with the real world.

---

## Part 1: What Is a Tool?

A tool is a **function that an LLM can decide to call**. The LLM doesn't execute it — it tells your code *which* tool to call and *with what arguments*. Your code executes it and returns the result.

```
User: "What's the weather in Mumbai?"
         │
         ↓
┌─────────────────────────────┐
│          LLM Brain          │
│                             │
│  "I need real-time weather  │
│   data. I should use the    │
│   get_weather tool."        │
│                             │
│  Decision:                  │
│  CALL get_weather(          │
│    city="Mumbai"            │
│  )                          │
└─────────────┬───────────────┘
              │
              ↓  (LLM outputs a tool call, NOT the answer)
┌─────────────────────────────┐
│     Your Code (Executor)    │
│                             │
│  result = get_weather(      │
│    city="Mumbai"            │
│  )                          │
│  → {"temp": 32, "humid": 78│
│     "condition": "Cloudy"}  │
└─────────────┬───────────────┘
              │
              ↓  (Result sent back to LLM)
┌─────────────────────────────┐
│          LLM Brain          │
│                             │
│  "The weather in Mumbai is  │
│   32°C and cloudy with 78%  │
│   humidity."                │
└─────────────────────────────┘
              │
              ↓
User sees: "The weather in Mumbai is 32°C and cloudy with 78% humidity."
```

### The Three-Step Dance

| Step | Who | What |
|------|-----|------|
| **1. Decide** | LLM | Reads user query + available tools. Decides which tool to call and with what args. |
| **2. Execute** | Your code | Runs the actual function (API call, DB query, calculation, etc.) |
| **3. Respond** | LLM | Reads the tool result, formulates a natural language answer. |

**Key insight**: The LLM never executes anything. It only **decides** and **interprets**. Your code does the actual work.

---

## Part 2: Tool Calling vs Function Calling

These terms are often used interchangeably, but there's a nuance:

| Term | Meaning |
|------|---------|
| **Function calling** | OpenAI's original name (June 2023). The LLM outputs structured JSON specifying which function to call. |
| **Tool calling** | The broader, standardized term. Same concept but provider-agnostic. Anthropic, Google, and others use this term. |
| **Tool use** | Anthropic's term for the same concept. |

```python
# They all mean the same thing:
# The LLM outputs: {"tool": "get_weather", "args": {"city": "Mumbai"}}
# Instead of: "The weather in Mumbai is..."
```

In LangChain, the unified term is **tool calling**. It works the same regardless of the LLM provider.

---

## Part 3: Anatomy of a Tool

Every tool has four components:

```python
# 1. NAME — how the LLM identifies the tool
name = "get_weather"

# 2. DESCRIPTION — tells the LLM WHEN to use this tool
description = "Get the current weather for a city. Use when the user asks about weather conditions."

# 3. SCHEMA — tells the LLM WHAT arguments the tool accepts
schema = {
    "city": {"type": "string", "description": "The city name, e.g., 'Mumbai'"},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
}

# 4. FUNCTION — the actual code that runs
def get_weather(city: str, unit: str = "celsius") -> dict:
    # Call weather API, query database, etc.
    return {"temp": 32, "condition": "Cloudy", "humidity": 78}
```

### The Description Is Critical

The LLM uses the **description** to decide when to call a tool. Bad descriptions = wrong tool choices:

```python
# ❌ Vague description — LLM doesn't know when to use it
description = "A useful tool"

# ❌ Too technical — LLM can't match user intent to this
description = "Invokes the OpenWeatherMap API v3 endpoint with geocoding"

# ✅ Clear, intent-focused description
description = "Get the current weather (temperature, conditions, humidity) for any city worldwide. Use when the user asks about weather, temperature, or climate conditions."
```

### The Schema Matters Too

```python
# ❌ No descriptions on parameters — LLM guesses
schema = {"q": "string", "n": "int"}

# ✅ Clear parameter descriptions
schema = {
    "city": {
        "type": "string",
        "description": "The city name, e.g., 'Mumbai', 'New York'"
    },
    "unit": {
        "type": "string",
        "description": "Temperature unit: 'celsius' or 'fahrenheit'",
        "enum": ["celsius", "fahrenheit"]
    }
}
```

---

## Part 4: Taxonomy of Tools

### By Category

| Category | Examples | What They Do |
|----------|---------|-------------|
| **Retrieval** | Search, Wikipedia, RAG, database query | Fetch information |
| **Computation** | Calculator, Python REPL, data analysis | Process/compute data |
| **Action** | Send email, create ticket, write file, deploy | Change external state |
| **Communication** | Slack message, API call, webhook | Interact with services |
| **Observation** | Get weather, check stock price, read sensor | Observe real-time state |

### By Risk Level

```
LOW RISK (read-only, safe to auto-execute):
├── Search the web
├── Look up Wikipedia
├── Calculate 2 + 2
├── Get weather
└── Query database (SELECT only)

MEDIUM RISK (modifies data, needs review):
├── Update database record
├── Create a file
├── Send a notification
└── Schedule a meeting

HIGH RISK (irreversible, needs human approval):
├── Send email to customer
├── Delete database records
├── Deploy to production
├── Transfer money
└── Modify user permissions
```

**This is why "human-in-the-loop" matters in agents** — you don't want an LLM auto-executing high-risk tools.

---

## Part 5: How Tool Calling Works in the LLM API

### What the LLM Actually Sees

When you bind tools to an LLM, the system message includes tool schemas:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Mumbai?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

### What the LLM Responds With

Instead of a text response, the LLM returns a **tool call**:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Mumbai\"}"
      }
    }
  ]
}
```

### Your Code Executes and Sends the Result Back

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"temp\": 32, \"condition\": \"Cloudy\", \"humidity\": 78}"
}
```

### Then the LLM Generates the Final Answer

```json
{
  "role": "assistant",
  "content": "The weather in Mumbai is currently 32°C and cloudy with 78% humidity."
}
```

### The Full Flow (Sequence Diagram)

```
User          Your App         LLM API
  │               │               │
  │──"Weather?"──→│               │
  │               │──messages +──→│
  │               │  tools        │
  │               │               │──(decides to call tool)
  │               │←─tool_call───│
  │               │  get_weather  │
  │               │  city=Mumbai  │
  │               │               │
  │               │──(executes)   │
  │               │  get_weather()│
  │               │  → {temp: 32} │
  │               │               │
  │               │──tool result─→│
  │               │               │──(reads result)
  │               │←─"32°C..."──│
  │←─"32°C..."──│               │
```

### Parallel Tool Calls

Modern LLMs can call **multiple tools at once**:

```
User: "What's the weather in Mumbai AND Delhi?"

LLM returns TWO tool calls simultaneously:
  1. get_weather(city="Mumbai")
  2. get_weather(city="Delhi")

Your code executes both (in parallel for speed),
sends both results back, LLM combines them into one answer.
```

---

## Part 6: Tools in the LangChain Ecosystem

### Where Tools Fit

```
┌─────────────────────────────────────────────┐
│              LangChain Ecosystem             │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Chains   │  │  Agents  │  │ LangGraph │  │
│  │(LCEL |)  │  │(ReAct)   │  │(StateGraph)│  │
│  └────┬─────┘  └────┬─────┘  └─────┬─────┘  │
│       │              │              │         │
│       └──────────────┼──────────────┘         │
│                      │                        │
│               ┌──────┴──────┐                 │
│               │    Tools    │                 │
│               │             │                 │
│               │ • Search    │                 │
│               │ • Calculator│                 │
│               │ • Custom fn │                 │
│               │ • RAG       │                 │
│               │ • APIs      │                 │
│               └─────────────┘                 │
└───────────────────────────────────────────────┘
```

### Tools in Chains vs Agents

| Approach | How Tools Are Used | Decision-Making |
|----------|--------------------|-----------------|
| **Chain with tools** | You hardcode WHICH tool to call | Developer decides |
| **Agent with tools** | LLM DECIDES which tool to call | LLM decides |

```python
# CHAIN: You decide to always call the search tool
chain = prompt | llm.bind_tools([search_tool]) | parse_tool_call | execute_tool

# AGENT: LLM decides whether to call search, calculator, or nothing
agent = create_react_agent(llm, [search_tool, calculator_tool])
# User asks math question → LLM picks calculator
# User asks factual question → LLM picks search
# User asks opinion → LLM picks no tool
```

---

## Part 7: Tool Calling Support by Provider

| Provider | Model | Tool Calling | Parallel Calls | Streaming |
|----------|-------|-------------|----------------|-----------|
| **OpenAI** | GPT-4o, GPT-4o-mini | ✅ | ✅ | ✅ |
| **Anthropic** | Claude 3.5 Sonnet/Opus | ✅ | ✅ | ✅ |
| **Google** | Gemini 1.5 Pro/Flash | ✅ | ✅ | ✅ |
| **Meta** | Llama 3.1 (via Ollama/vLLM) | ✅ | 🟡 | ✅ |
| **Mistral** | Mistral Large | ✅ | ✅ | ✅ |
| **Cohere** | Command R+ | ✅ | ✅ | ✅ |

**All modern LLMs support tool calling.** It's a standard capability now.

---

## Part 8: Preview — What's Coming in This Phase

| Chapter | What You'll Build |
|---------|------------------|
| **9.1** (this) | Understanding tools conceptually |
| **9.2** | Use built-in tools (Search, Wikipedia, Calculator) |
| **9.3** | Create custom tools with `@tool` decorator |
| **9.4** | Tool calling protocol deep dive (`bind_tools`, `tool_calls`) |
| **9.5** | MCP — Model Context Protocol (new standard) |

After this phase, you'll move to **Phase 10 — Agents**, where LLMs autonomously choose and execute tools in loops.

---

## Common Mistakes

### Mistake 1: Thinking the LLM executes the tool
```python
# ❌ Misconception: LLM calls the weather API directly
# The LLM CANNOT execute code or make HTTP requests!

# ✅ Reality: LLM outputs JSON saying "call get_weather with city=Mumbai"
# YOUR CODE executes the function and returns the result to the LLM
```

### Mistake 2: Bad tool descriptions
```python
# ❌ LLM doesn't know when to use this
@tool
def fetch(url: str) -> str:
    """Fetches stuff."""  # What stuff? When? Why?

# ✅ Clear, intent-based description
@tool
def fetch_webpage(url: str) -> str:
    """Fetch and return the text content of a webpage. Use when the user
    provides a URL and wants to read or analyze its contents."""
```

### Mistake 3: Too many tools
```python
# ❌ Giving LLM 50 tools — it gets confused about which to use
tools = [tool_1, tool_2, ..., tool_50]

# ✅ Start with 3-5 focused tools. Add more only when needed.
# Group related tools or use routing to select relevant tool subsets.
```

### Mistake 4: Not handling tool errors
```python
# ❌ Tool crashes → entire chain crashes
def get_weather(city: str) -> dict:
    response = requests.get(f"https://api.weather.com/{city}")
    return response.json()  # What if API is down?

# ✅ Always handle errors gracefully
def get_weather(city: str) -> str:
    try:
        response = requests.get(f"https://api.weather.com/{city}", timeout=5)
        response.raise_for_status()
        return json.dumps(response.json())
    except Exception as e:
        return f"Error fetching weather for {city}: {str(e)}"
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Write clear, intent-focused tool descriptions | LLM uses description to decide when to call |
| Add descriptions to every parameter | LLM needs to know what each arg means |
| Return strings from tools (not dicts) | LLMs process text; convert dicts to JSON strings |
| Handle errors inside tools | Return error message instead of crashing |
| Start with few tools, add incrementally | Too many tools confuse the LLM |
| Classify tools by risk level | High-risk tools need human approval |
| Use Pydantic models for tool schemas | Type safety + auto-generated descriptions |
| Log all tool calls in production | Debugging, audit trail, cost tracking |

---

## Interview Preparation

### Easy
**Q: What is tool calling in LLMs?**

> Tool calling is a mechanism where an LLM, instead of generating a text response, outputs a structured request to call a specific function with specific arguments. The LLM doesn't execute the function — it only decides which function to call. Your application code executes the function and sends the result back to the LLM, which then generates a natural language response. This allows LLMs to interact with external systems like APIs, databases, and calculators.

### Medium
**Q: What is the difference between tool calling in a chain vs an agent?**

> In a **chain**, the developer hardcodes the tool-calling flow — the tool is always called in a fixed sequence. In an **agent**, the LLM autonomously decides whether to call a tool, which tool to call, and can loop (call multiple tools iteratively) until it has enough information to answer. Chains are predictable but rigid; agents are flexible but less predictable. Use chains when the workflow is known; use agents when the LLM needs to reason about which tools to use.

### Hard
**Q: How would you handle tool calling for high-risk actions like sending emails or deleting data?**

> Implement a **human-in-the-loop** pattern: (1) LLM decides to call the dangerous tool and outputs the tool call with arguments. (2) Instead of auto-executing, your code pauses and presents the proposed action to a human for approval. (3) Only after human approval does the tool execute. In LangGraph, this is built-in via `interrupt_before` on specific nodes. For lower-risk actions, use confirmation prompts. For production, maintain an audit log of all tool calls, implement rate limiting, and use role-based permissions to control which users can trigger which tools.

### Senior
**Q: An LLM has access to 20 tools but frequently picks the wrong one. How do you fix this?**

> Systematic approach: (1) **Improve descriptions** — make each tool's description clearly state WHEN to use it and when NOT to use it. Include examples. (2) **Reduce ambiguity** — if two tools overlap in purpose, merge them or make their descriptions mutually exclusive. (3) **Use tool routing** — classify the user's intent first, then only present relevant tools (3-5) to the LLM. (4) **Few-shot examples** — include examples in the system prompt showing correct tool selection. (5) **Use a stronger model** — GPT-4o is much better at tool selection than GPT-3.5. (6) **Evaluate systematically** — create a test set of queries with expected tool calls, measure tool selection accuracy, iterate.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Tool** | A function the LLM can decide to call |
| **Tool calling** | LLM outputs structured JSON specifying function + args |
| **Three-step dance** | LLM decides → your code executes → LLM interprets result |
| **Description** | How the LLM knows WHEN to use a tool (critical for accuracy) |
| **Schema** | Parameter types and descriptions (what args the tool accepts) |
| **Parallel tool calls** | LLM can request multiple tools at once |
| **Chain vs Agent** | Chain = hardcoded tool flow; Agent = LLM decides autonomously |
| **Risk levels** | Read-only (safe) → modify data (review) → irreversible (human approval) |

---

> [← Previous: Cloud Vector DBs](../phase-07-vector-databases/chapter-29-cloud-vector-dbs.md) | [Next: Built-in Tools →](chapter-31-builtin-tools.md)

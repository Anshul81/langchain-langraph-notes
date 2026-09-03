# Chapter 9.4: Tool Calling Deep Dive — `bind_tools`, `tool_calls` & The Protocol

> **Phase 9 — Tools & Tool Calling** | [← Previous: Custom Tools](chapter-32-custom-tools.md) | [Next: MCP →](chapter-34-mcp.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand the internal mechanics of `bind_tools()` and `tool_calls`
- ✅ Parse `AIMessage.tool_calls` and construct `ToolMessage` responses correctly
- ✅ Force the LLM to use a specific tool (or no tool at all)
- ✅ Handle parallel tool calls (multiple tools in one response)
- ✅ Stream tool call outputs in real time
- ✅ Understand provider differences (OpenAI vs Anthropic vs Google)
- ✅ Build a **tool-calling pipeline** with automatic execution and retry logic

| | |
|---|---|
| **Prerequisites** | Chapter 9.2 (Built-in Tools), Chapter 9.3 (Custom Tools) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction

In the previous chapters, you used tools at a high level — `@tool` decorator, `bind_tools()`, and manual execution loops. Now we go **under the hood**.

Understanding the tool-calling protocol is critical for:
- Debugging when tool calls fail silently
- Building robust agents that handle edge cases
- Optimizing for speed (streaming, parallel calls)
- Working across different LLM providers

```
This chapter answers:
├── What exactly does bind_tools() do to the API request?
├── What does the raw tool_calls response look like?
├── How do I force the LLM to call (or not call) a tool?
├── How do parallel tool calls work?
├── How do I stream tool call chunks?
└── What's different between OpenAI, Anthropic, and Google?
```

---

## Part 1: What `bind_tools()` Actually Does

### The Before and After

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 32°C, Sunny"

@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    return f"Search results for: {query}"

# bind_tools returns a NEW LLM instance with tools baked in
llm_with_tools = llm.bind_tools([get_weather, search_web])
```

### What's Happening Inside

```python
# bind_tools() does NOT modify the original LLM
# It creates a new Runnable that adds tool schemas to every API call

# Equivalent to:
# llm_with_tools = llm.bind(tools=[...tool_schemas...])

# What gets sent to the API:
# {
#   "model": "gpt-4o-mini",
#   "messages": [...],
#   "tools": [
#     {
#       "type": "function",
#       "function": {
#         "name": "get_weather",
#         "description": "Get the current weather for a city.",
#         "parameters": {
#           "type": "object",
#           "properties": {
#             "city": {"type": "string", "description": "..."}
#           },
#           "required": ["city"]
#         }
#       }
#     },
#     {
#       "type": "function",
#       "function": {
#         "name": "search_web",
#         "description": "Search the web for current information.",
#         "parameters": {
#           "type": "object",
#           "properties": {
#             "query": {"type": "string", "description": "..."}
#           },
#           "required": ["query"]
#         }
#       }
#     }
#   ]
# }
```

### Inspecting the Bound Tools

```python
# You can see what tools are bound
print(llm_with_tools.kwargs)
# {'tools': [{'type': 'function', 'function': {'name': 'get_weather', ...}}, ...]}

# The original LLM is untouched
print(llm.kwargs)  # {} — no tools
```

---

## Part 2: The `AIMessage.tool_calls` Structure

### What the LLM Returns

```python
from langchain_core.messages import HumanMessage

response = llm_with_tools.invoke([
    HumanMessage(content="What's the weather in Mumbai?")
])

# The response is an AIMessage
print(type(response))
# <class 'langchain_core.messages.ai.AIMessage'>

print(f"Content:    '{response.content}'")        # Usually empty when tool is called
print(f"Tool calls: {response.tool_calls}")         # Parsed tool calls (LangChain format)
print(f"Raw:        {response.additional_kwargs}")  # Raw provider response
```

### `tool_calls` — The Parsed Format

```python
# response.tool_calls is a list of dicts:
[
    {
        "name": "get_weather",           # Tool name
        "args": {"city": "Mumbai"},      # Parsed arguments (dict, not JSON string)
        "id": "call_abc123def456",       # Unique ID for this tool call
        "type": "tool_call"              # Always "tool_call"
    }
]
```

### `invalid_tool_calls` — When Parsing Fails

```python
# Sometimes the LLM generates malformed JSON for tool args
# LangChain catches this and puts it in invalid_tool_calls

print(response.invalid_tool_calls)
# Usually []. If populated:
# [
#     {
#         "name": "get_weather",
#         "args": "{'city': Mumbai}",  # Invalid JSON — missing quotes
#         "id": "call_xyz789",
#         "error": "JSONDecodeError: ..."
#     }
# ]

# Always check for invalid_tool_calls in production!
if response.invalid_tool_calls:
    print(f"⚠️ {len(response.invalid_tool_calls)} invalid tool call(s)")
    for itc in response.invalid_tool_calls:
        print(f"   Tool: {itc['name']}, Error: {itc['error']}")
```

### The Raw Provider Response

```python
# response.additional_kwargs contains the raw response from the provider
# Useful for debugging provider-specific issues

print(response.additional_kwargs)
# OpenAI format:
# {
#     "tool_calls": [
#         {
#             "id": "call_abc123",
#             "type": "function",
#             "function": {
#                 "name": "get_weather",
#                 "arguments": "{\"city\": \"Mumbai\"}"  # JSON STRING (not dict!)
#             }
#         }
#     ]
# }
```

---

## Part 3: `ToolMessage` — Sending Results Back

### The Correct Format

```python
from langchain_core.messages import ToolMessage

# After executing a tool, send the result back
tool_message = ToolMessage(
    content="Weather in Mumbai: 32°C, Sunny, Humidity 78%",
    tool_call_id="call_abc123def456"  # MUST match the id from tool_calls!
)
```

### Why `tool_call_id` Is Critical

```python
# The LLM API uses tool_call_id to match:
# "This result corresponds to THAT specific tool call"
# 
# This matters when there are MULTIPLE tool calls:
# 
# Tool call 1: get_weather(city="Mumbai")    → id: "call_AAA"
# Tool call 2: get_weather(city="Delhi")     → id: "call_BBB"
# 
# Result 1: ToolMessage(content="32°C", tool_call_id="call_AAA")
# Result 2: ToolMessage(content="28°C", tool_call_id="call_BBB")
#
# Without matching IDs, the LLM can't tell which result goes with which call!
```

### The Complete Message Flow

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# The full conversation for one tool-calling round:
messages = [
    # 1. User asks a question
    HumanMessage(content="What's the weather in Mumbai?"),
    
    # 2. LLM decides to call a tool (content is empty/null)
    AIMessage(
        content="",
        tool_calls=[{
            "name": "get_weather",
            "args": {"city": "Mumbai"},
            "id": "call_abc123",
            "type": "tool_call"
        }]
    ),
    
    # 3. Tool result (you execute the tool and create this)
    ToolMessage(
        content="Weather in Mumbai: 32°C, Sunny",
        tool_call_id="call_abc123"
    ),
    
    # 4. Send all messages back to LLM → it generates final answer
]

final = llm_with_tools.invoke(messages)
print(final.content)
# "The weather in Mumbai is currently 32°C and sunny."
```

---

## Part 4: Controlling Tool Choice

### `tool_choice` — Force or Prevent Tool Calls

```python
# AUTO (default): LLM decides whether to call a tool
llm_auto = llm.bind_tools(tools, tool_choice="auto")

# NONE: Force LLM to NOT call any tool (respond with text only)
llm_no_tools = llm.bind_tools(tools, tool_choice="none")

# REQUIRED: Force LLM to call at LEAST one tool
llm_must_use = llm.bind_tools(tools, tool_choice="required")

# SPECIFIC: Force LLM to call a SPECIFIC tool
llm_weather_only = llm.bind_tools(tools, tool_choice="get_weather")
# Or with dict syntax:
llm_weather_only = llm.bind_tools(
    tools, 
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)
```

### When to Use Each

| `tool_choice` | When to Use |
|---------------|-------------|
| `"auto"` (default) | Normal usage — let LLM decide |
| `"none"` | Force text response (e.g., final answer after tools ran) |
| `"required"` | Guarantee at least one tool call (e.g., first step of a pipeline) |
| `"get_weather"` | Force a specific tool (e.g., routing already decided which tool) |

### Practical Example: Two-Step Pipeline

```python
# Step 1: Force tool call to get data
llm_step1 = llm.bind_tools([get_weather], tool_choice="required")

# Step 2: Force text response to interpret the data
llm_step2 = llm.bind_tools([get_weather], tool_choice="none")

# Execute
response1 = llm_step1.invoke([HumanMessage(content="Weather in Delhi")])
# → Always returns a tool call

# Execute tool...
result = get_weather.invoke(response1.tool_calls[0]["args"])

# Get interpretation
messages = [
    HumanMessage(content="Weather in Delhi"),
    response1,
    ToolMessage(content=result, tool_call_id=response1.tool_calls[0]["id"]),
]
response2 = llm_step2.invoke(messages)
# → Always returns text, never another tool call
print(response2.content)
```

---

## Part 5: Parallel Tool Calls

Modern LLMs can request **multiple tools at once** when they're independent:

### How Parallel Calls Work

```python
@tool
def get_population(country: str) -> str:
    """Get the current population of a country."""
    populations = {
        "India": "1.44 billion",
        "China": "1.43 billion",
        "USA": "334 million",
        "Brazil": "216 million",
    }
    return populations.get(country, f"Population data not available for {country}")

@tool
def get_gdp(country: str) -> str:
    """Get the GDP of a country in USD."""
    gdps = {
        "India": "$3.7 trillion",
        "China": "$17.8 trillion",
        "USA": "$25.5 trillion",
        "Brazil": "$2.1 trillion",
    }
    return gdps.get(country, f"GDP data not available for {country}")


llm_with_tools = llm.bind_tools([get_population, get_gdp])

# This question needs BOTH tools for the SAME country
response = llm_with_tools.invoke([
    HumanMessage(content="What is India's population and GDP?")
])

# The LLM returns TWO tool calls in parallel!
print(f"Number of tool calls: {len(response.tool_calls)}")
for tc in response.tool_calls:
    print(f"  🔧 {tc['name']}({tc['args']}) — id: {tc['id']}")
```

**Output:**
```
Number of tool calls: 2
  🔧 get_population({'country': 'India'}) — id: call_aaa111
  🔧 get_gdp({'country': 'India'}) — id: call_bbb222
```

### Executing Parallel Calls Efficiently

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def execute_tool_calls_parallel(tool_calls, tool_map):
    """Execute multiple tool calls in parallel using threads."""
    results = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for tc in tool_calls:
            tool = tool_map[tc["name"]]
            future = executor.submit(tool.invoke, tc["args"])
            futures[future] = tc
        
        for future, tc in futures.items():
            try:
                result = future.result(timeout=30)
            except Exception as e:
                result = f"Error: {str(e)}"
            
            results.append(ToolMessage(
                content=str(result),
                tool_call_id=tc["id"]
            ))
    
    return results


# Usage
tool_map = {"get_population": get_population, "get_gdp": get_gdp}

response = llm_with_tools.invoke([
    HumanMessage(content="Compare India and China's population and GDP")
])

# Execute all tool calls in parallel
tool_results = execute_tool_calls_parallel(response.tool_calls, tool_map)

# Send results back
messages = [
    HumanMessage(content="Compare India and China's population and GDP"),
    response,
    *tool_results
]
final = llm_with_tools.invoke(messages)
print(final.content)
```

### Disabling Parallel Tool Calls

```python
# Some scenarios require sequential tool calls (e.g., output of tool 1 → input of tool 2)
# OpenAI supports disabling parallel calls:

llm_sequential = llm.bind_tools(
    tools,
    parallel_tool_calls=False  # OpenAI-specific parameter
)
```

---

## Part 6: Streaming Tool Calls

### Why Stream Tool Calls?

In a chat UI, you want to show the user what's happening in real time — not wait 5 seconds silently.

### Streaming with `stream()`

```python
from langchain_core.messages import HumanMessage

llm_with_tools = llm.bind_tools([get_weather, search_web])

# Stream the response
for chunk in llm_with_tools.stream([
    HumanMessage(content="What's the weather in Mumbai?")
]):
    # Each chunk is an AIMessageChunk
    if chunk.tool_call_chunks:
        for tc_chunk in chunk.tool_call_chunks:
            print(f"Tool chunk: name={tc_chunk.get('name', '')}, "
                  f"args={tc_chunk.get('args', '')}")
    if chunk.content:
        print(f"Content: {chunk.content}", end="")
```

### Accumulating Streamed Tool Calls

```python
from langchain_core.messages import AIMessageChunk

# Tool calls arrive in chunks — you need to accumulate them
full_response = None

for chunk in llm_with_tools.stream([
    HumanMessage(content="What's the weather in Mumbai and Delhi?")
]):
    if full_response is None:
        full_response = chunk
    else:
        full_response = full_response + chunk  # AIMessageChunk supports addition!

# Now full_response has the complete tool_calls
print(f"Accumulated {len(full_response.tool_calls)} tool calls:")
for tc in full_response.tool_calls:
    print(f"  🔧 {tc['name']}({tc['args']})")
```

### Async Streaming

```python
import asyncio

async def stream_with_tools():
    """Stream tool calls asynchronously."""
    full_response = None
    
    async for chunk in llm_with_tools.astream([
        HumanMessage(content="Search for latest AI news")
    ]):
        if full_response is None:
            full_response = chunk
        else:
            full_response = full_response + chunk
        
        # Show progress
        if chunk.tool_call_chunks:
            print(".", end="", flush=True)
    
    print()  # newline
    return full_response

# Run
response = asyncio.run(stream_with_tools())
print(f"Tool calls: {response.tool_calls}")
```

---

## Part 7: The Complete Tool-Calling Pipeline

### Production-Ready Implementation

```python
import os
import json
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage
)

load_dotenv()


class ToolCallingPipeline:
    """A robust, production-ready tool-calling pipeline with retry logic."""
    
    def __init__(self, tools: list, model: str = "gpt-4o-mini", max_rounds: int = 5):
        self.llm = ChatOpenAI(
            model=model,
            temperature=0,
            openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
            openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
        )
        self.tools = tools
        self.tool_map = {t.name: t for t in tools}
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.max_rounds = max_rounds
        self.call_log = []  # Audit log
    
    def invoke(
        self, 
        question: str, 
        system_prompt: Optional[str] = None,
        verbose: bool = True
    ) -> dict:
        """
        Execute the full tool-calling pipeline.
        
        Returns:
            {
                "answer": str,
                "tool_calls_made": list,
                "rounds": int,
                "messages": list
            }
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=question))
        
        tool_calls_made = []
        
        for round_num in range(self.max_rounds):
            # Get LLM response
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Check for invalid tool calls
            if response.invalid_tool_calls:
                if verbose:
                    print(f"⚠️ Round {round_num + 1}: Invalid tool calls detected")
                for itc in response.invalid_tool_calls:
                    if verbose:
                        print(f"   Error: {itc.get('error', 'Unknown')}")
                    messages.append(ToolMessage(
                        content=f"Error: Invalid tool call — {itc.get('error', 'malformed arguments')}. "
                                f"Please try again with valid JSON arguments.",
                        tool_call_id=itc.get("id", "unknown")
                    ))
                continue
            
            # No tool calls — final answer
            if not response.tool_calls:
                if verbose:
                    print(f"✅ Final answer after {round_num + 1} round(s)")
                return {
                    "answer": response.content,
                    "tool_calls_made": tool_calls_made,
                    "rounds": round_num + 1,
                    "messages": messages
                }
            
            # Execute tool calls
            for tc in response.tool_calls:
                name = tc["name"]
                args = tc["args"]
                tc_id = tc["id"]
                
                if verbose:
                    print(f"🔧 Round {round_num + 1}: {name}({json.dumps(args)})")
                
                # Check if tool exists
                if name not in self.tool_map:
                    error_msg = (
                        f"Error: Tool '{name}' not found. "
                        f"Available tools: {', '.join(self.tool_map.keys())}"
                    )
                    messages.append(ToolMessage(content=error_msg, tool_call_id=tc_id))
                    tool_calls_made.append({
                        "round": round_num + 1, "tool": name,
                        "args": args, "result": error_msg, "status": "error"
                    })
                    continue
                
                # Execute with error handling
                try:
                    result = self.tool_map[name].invoke(args)
                    result_str = str(result)
                    
                    # Truncate very long results to save tokens
                    if len(result_str) > 4000:
                        result_str = result_str[:4000] + "\n\n[... truncated — result too long]"
                    
                    if verbose:
                        preview = result_str[:150].replace('\n', ' ')
                        print(f"   📋 Result: {preview}...")
                    
                    status = "success"
                    
                except Exception as e:
                    result_str = f"Error executing tool '{name}': {str(e)}"
                    if verbose:
                        print(f"   ❌ {result_str}")
                    status = "error"
                
                messages.append(ToolMessage(content=result_str, tool_call_id=tc_id))
                tool_calls_made.append({
                    "round": round_num + 1, "tool": name,
                    "args": args, "result": result_str[:500], "status": status
                })
                
                # Log for audit
                self.call_log.append({
                    "question": question, "round": round_num + 1,
                    "tool": name, "args": args, "status": status
                })
        
        # Exhausted max rounds
        if verbose:
            print(f"⚠️ Max rounds ({self.max_rounds}) reached")
        
        return {
            "answer": response.content or "Unable to generate a final answer.",
            "tool_calls_made": tool_calls_made,
            "rounds": self.max_rounds,
            "messages": messages
        }
    
    def get_audit_log(self) -> list:
        """Get the audit log of all tool calls."""
        return self.call_log


# --- Demo ---

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Use for weather questions."""
    weather_data = {
        "Mumbai": "32°C, Partly Cloudy, Humidity 78%",
        "Delhi": "38°C, Sunny, Humidity 45%",
        "Bangalore": "26°C, Rainy, Humidity 85%",
        "London": "18°C, Overcast, Humidity 72%",
    }
    return weather_data.get(city, f"Weather data not available for {city}")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Use for math questions.
    
    Args:
        expression: A Python math expression, e.g., '2 ** 10' or 'math.sqrt(144)'
    """
    import math
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"

@tool
def search_facts(query: str) -> str:
    """Search for factual information. Use for knowledge questions."""
    facts = {
        "python creator": "Python was created by Guido van Rossum, first released in 1991.",
        "transformer": "The Transformer architecture was introduced in 'Attention Is All You Need' (2017) by Vaswani et al.",
        "langchain": "LangChain is an open-source framework for building LLM applications, created by Harrison Chase.",
    }
    query_lower = query.lower()
    for key, value in facts.items():
        if key in query_lower:
            return value
    return f"No specific facts found for '{query}'. Try a different search query."


# Create pipeline
pipeline = ToolCallingPipeline(
    tools=[get_weather, calculate, search_facts],
    max_rounds=5
)

# Test various scenarios
print("\n" + "=" * 60)
result = pipeline.invoke("What's the weather in Mumbai?")
print(f"\n💬 {result['answer']}")

print("\n" + "=" * 60)
result = pipeline.invoke("What is 2^32 and what is the square root of 256?")
print(f"\n💬 {result['answer']}")

print("\n" + "=" * 60)
result = pipeline.invoke("Who created Python?")
print(f"\n💬 {result['answer']}")

print("\n" + "=" * 60)
result = pipeline.invoke("Hello! How are you?")
print(f"\n💬 {result['answer']}")

# View audit log
print("\n📊 Audit Log:")
for entry in pipeline.get_audit_log():
    print(f"   [{entry['round']}] {entry['tool']}({entry['args']}) → {entry['status']}")
```

---

## Part 8: Provider Differences

### The Promise of LangChain's Abstraction

LangChain normalizes tool calling across providers, but there are differences:

### OpenAI (GPT-4o, GPT-4o-mini)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# OpenAI-specific features:
llm_with_tools = llm.bind_tools(
    tools,
    tool_choice="auto",           # auto, none, required, or specific tool
    parallel_tool_calls=True,     # Allow multiple tools at once (default: True)
    strict=True,                  # Enable Structured Outputs for tool schemas
)
```

### Anthropic (Claude 3.5)

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
llm_with_tools = llm.bind_tools(tools)

# Anthropic differences:
# - Uses "tool_use" blocks in the response
# - tool_choice options: "auto", "any" (= required), or {"name": "specific_tool"}
# - No parallel_tool_calls parameter (Claude decides based on query)

llm_with_tools = llm.bind_tools(
    tools,
    tool_choice={"type": "tool", "name": "get_weather"}  # Force specific tool
)
```

### Google (Gemini)

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
llm_with_tools = llm.bind_tools(tools)

# Google differences:
# - Uses "functionCall" in the response format
# - tool_choice options: "auto", "none", or "any"
# - Supports function calling configuration
```

### Comparison Table

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| `tool_choice="auto"` | ✅ | ✅ | ✅ |
| `tool_choice="none"` | ✅ | ✅ | ✅ |
| `tool_choice="required"` | ✅ | ✅ (`"any"`) | ✅ (`"any"`) |
| Force specific tool | ✅ | ✅ | ✅ |
| Parallel tool calls | ✅ (configurable) | ✅ (auto) | ✅ (auto) |
| `strict` mode | ✅ | ❌ | ❌ |
| Streaming tool calls | ✅ | ✅ | ✅ |
| `tool_calls` on AIMessage | ✅ | ✅ | ✅ |

### The Beauty of LangChain's Abstraction

```python
# Same code works across ALL providers!
# Only the import changes.

from langchain_core.messages import HumanMessage, ToolMessage

def run_tool_loop(llm_with_tools, tools, question):
    """This function works with ANY LLM provider."""
    tool_map = {t.name: t for t in tools}
    messages = [HumanMessage(content=question)]
    
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    
    if not response.tool_calls:
        return response.content
    
    for tc in response.tool_calls:
        result = tool_map[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    
    final = llm_with_tools.invoke(messages)
    return final.content

# Works identically with OpenAI, Anthropic, or Google!
```

---

## Part 9: Advanced Patterns

### Pattern 1: Tool Call Retry on Error

```python
def invoke_with_retry(llm_with_tools, messages, tool_map, max_retries=2):
    """Retry failed tool calls with error feedback."""
    
    for attempt in range(max_retries + 1):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return response.content, messages
        
        all_succeeded = True
        for tc in response.tool_calls:
            try:
                result = tool_map[tc["name"]].invoke(tc["args"])
                messages.append(ToolMessage(
                    content=str(result), 
                    tool_call_id=tc["id"]
                ))
            except Exception as e:
                all_succeeded = False
                messages.append(ToolMessage(
                    content=f"Error: {str(e)}. Please try different arguments.",
                    tool_call_id=tc["id"]
                ))
        
        if all_succeeded or attempt == max_retries:
            break
    
    # Get final answer
    final = llm_with_tools.invoke(messages)
    return final.content, messages
```

### Pattern 2: Tool Result Validation

```python
from pydantic import BaseModel, ValidationError

class WeatherResult(BaseModel):
    """Expected structure of weather tool output."""
    temperature: float
    condition: str
    humidity: float

def validated_tool_call(tool, args, result_model=None):
    """Execute a tool and optionally validate its output."""
    result = tool.invoke(args)
    
    if result_model:
        try:
            parsed = result_model.parse_raw(result)
            return {"status": "valid", "data": parsed.dict(), "raw": result}
        except ValidationError as e:
            return {"status": "invalid", "errors": str(e), "raw": result}
    
    return {"status": "raw", "data": result}
```

### Pattern 3: Tool Call Routing

```python
@tool
def route_to_tool(user_intent: str) -> str:
    """Classify user intent to determine which tools are relevant.
    
    Categories:
    - 'weather': weather-related questions
    - 'math': calculations and math
    - 'knowledge': factual questions
    - 'general': general conversation (no tools needed)
    """
    # In practice, this could use an LLM or classifier
    intent_lower = user_intent.lower()
    if any(w in intent_lower for w in ["weather", "temperature", "rain"]):
        return "weather"
    elif any(w in intent_lower for w in ["calculate", "math", "compute", "sum", "average"]):
        return "math"
    elif any(w in intent_lower for w in ["who", "what is", "when", "history"]):
        return "knowledge"
    return "general"

# Use routing to select tool subset before calling LLM
intent = route_to_tool.invoke({"user_intent": "What's 2+2?"})
if intent == "math":
    relevant_tools = [calculate]
elif intent == "weather":
    relevant_tools = [get_weather]
else:
    relevant_tools = [search_facts]

llm_focused = llm.bind_tools(relevant_tools)
# Now LLM only sees 1-2 relevant tools instead of all 10+
```

---

## Common Mistakes

### Mistake 1: Not matching tool_call_id
```python
# ❌ Mismatched IDs — LLM can't correlate results
for tc in response.tool_calls:
    result = tool_map[tc["name"]].invoke(tc["args"])
    messages.append(ToolMessage(content=result, tool_call_id="wrong_id"))

# ✅ Always use the ID from the tool call
for tc in response.tool_calls:
    result = tool_map[tc["name"]].invoke(tc["args"])
    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
```

### Mistake 2: Forgetting to append the AIMessage
```python
# ❌ Missing the AIMessage in the conversation — API error!
response = llm_with_tools.invoke(messages)
# Execute tools...
messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
final = llm_with_tools.invoke(messages)  # ERROR: ToolMessage without AIMessage

# ✅ Always append the AIMessage BEFORE the ToolMessages
response = llm_with_tools.invoke(messages)
messages.append(response)  # Add the AIMessage with tool_calls!
# Then add ToolMessages
messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
final = llm_with_tools.invoke(messages)  # Works!
```

### Mistake 3: Not handling the "no tool call" case
```python
# ❌ Assumes every response has tool calls
result = tool_map[response.tool_calls[0]["name"]].invoke(...)  # IndexError if no tool calls!

# ✅ Always check
if response.tool_calls:
    # Execute tools
else:
    # LLM answered directly
    return response.content
```

### Mistake 4: Ignoring invalid_tool_calls
```python
# ❌ Only checking tool_calls — missing parse errors
if response.tool_calls:
    execute(response.tool_calls)

# ✅ Check both
if response.tool_calls:
    execute(response.tool_calls)
if response.invalid_tool_calls:
    handle_errors(response.invalid_tool_calls)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Always append AIMessage before ToolMessages | Required message ordering for the API |
| Match `tool_call_id` exactly | Correlates results with specific tool calls |
| Check both `tool_calls` and `invalid_tool_calls` | Catch malformed LLM outputs |
| Truncate long tool results (>4000 chars) | Saves tokens, prevents context overflow |
| Use `tool_choice="required"` for guaranteed tool use | Useful for forced-retrieval patterns |
| Execute parallel calls concurrently | Performance improvement for I/O-bound tools |
| Log all tool calls for auditing | Debugging, cost tracking, compliance |
| Use streaming for chat UIs | Better user experience — shows progress |

---

## Interview Preparation

### Easy
**Q: What is `bind_tools()` and what does it do?**

> `bind_tools()` is a method on LangChain LLM instances that attaches tool schemas (name, description, parameter types) to every API call. It returns a new Runnable — it doesn't modify the original LLM. When the LLM receives a query, it can see the available tools and decide whether to call one. The LLM returns an `AIMessage` with a `tool_calls` list containing the tool name, arguments, and a unique ID. The tool schemas are derived from the function's name, docstring, and type hints.

### Medium
**Q: Explain the message flow for a complete tool-calling round.**

> The flow has four message types: (1) `HumanMessage` — the user's question. (2) `AIMessage` with `tool_calls` — the LLM's decision to call a tool (content is usually empty). (3) `ToolMessage` — the result of executing the tool, with a `tool_call_id` matching the original call's ID. (4) Final `AIMessage` with `content` — the LLM reads the tool result and generates a natural language answer. The critical rule is that `ToolMessage` must always follow an `AIMessage` that contains the matching `tool_call_id`, and the AIMessage must be appended to the message list before the ToolMessage.

### Hard
**Q: How would you handle parallel tool calls efficiently in a production system?**

> Use concurrent execution: (1) When the LLM returns multiple `tool_calls`, execute them in parallel using `ThreadPoolExecutor` or `asyncio.gather()` since they're independent. (2) Set timeouts on each tool execution to prevent hanging. (3) Handle partial failures — if one tool fails, still send successful results back and let the LLM work with incomplete data. (4) Match each result to its `tool_call_id` correctly. (5) Consider rate limiting if tools call external APIs. (6) Log execution times per tool for monitoring. (7) If tools have dependencies (output of tool A feeds into tool B), disable parallel calls using `parallel_tool_calls=False` (OpenAI) or design your tools to be independent.

### Senior
**Q: What are the key differences in tool calling across OpenAI, Anthropic, and Google, and how does LangChain abstract them?**

> Key differences: (1) **Format** — OpenAI uses `function` type tool calls, Anthropic uses `tool_use` content blocks, Google uses `functionCall` parts. (2) **tool_choice naming** — OpenAI uses `"required"`, Anthropic uses `"any"`, Google uses `"any"`. (3) **Parallel calls** — OpenAI has an explicit `parallel_tool_calls` parameter; others decide automatically. (4) **Strict mode** — only OpenAI supports `strict=True` for guaranteed schema compliance. LangChain abstracts all of this: `bind_tools()` accepts the same arguments across providers, `AIMessage.tool_calls` has a normalized format, and `ToolMessage` works identically. The only differences are in provider-specific features like `strict` mode. This allows swapping providers without changing application code.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **`bind_tools()`** | Attaches tool schemas to every LLM API call |
| **`tool_calls`** | Parsed list of tool calls on AIMessage (name, args, id) |
| **`invalid_tool_calls`** | Failed-to-parse tool calls (malformed JSON args) |
| **`ToolMessage`** | Sends tool results back to LLM (must include `tool_call_id`) |
| **`tool_choice`** | Control: auto, none, required, or specific tool name |
| **Parallel calls** | LLM can request multiple independent tools at once |
| **Streaming** | Tool call chunks arrive incrementally — accumulate with `+` |
| **Message ordering** | HumanMsg → AIMsg(tool_calls) → ToolMsg → AIMsg(content) |
| **Provider abstraction** | Same code works across OpenAI, Anthropic, Google |

---

> [← Previous: Custom Tools](chapter-32-custom-tools.md) | [Next: MCP →](chapter-34-mcp.md)

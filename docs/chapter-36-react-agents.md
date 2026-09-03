# Chapter 10.2: ReAct Agents — Building Your First Agent

> **Phase 10 — Agents** | [← Previous: Intro to Agents](chapter-35-intro-to-agents.md) | [Next: LangGraph Fundamentals →](chapter-37-langgraph-fundamentals.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Build a ReAct agent using `create_react_agent` from LangGraph
- ✅ Understand the message flow inside a ReAct agent
- ✅ Add system prompts and memory to agents
- ✅ Configure recursion limits, timeouts, and error handling
- ✅ Stream agent steps in real time
- ✅ Build a **multi-tool research agent** that searches, calculates, and reasons

| | |
|---|---|
| **Prerequisites** | Chapter 10.1 (Intro to Agents), Phase 9 (Tools & Tool Calling) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 55 minutes |

---

## Introduction

`create_react_agent` is the **fastest way to build an agent** in LangChain. It's a pre-built LangGraph agent that implements the ReAct (Reasoning + Acting) pattern using native tool calling.

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(llm, tools)  # That's it!
```

Three lines of code — and you have an agent that:
- Receives a user query
- Decides which tool(s) to call
- Executes tools and reads results
- Loops until it has enough information
- Returns a final answer

---

## Part 1: Setup

```bash
pip install langgraph langchain langchain-openai langchain-community
pip install duckduckgo-search wikipedia
```

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
```

---

## Part 2: Your First Agent

### Creating Tools

```python
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Use when the user asks about weather, temperature, or climate conditions.
    """
    weather_data = {
        "Mumbai": "32°C, Partly Cloudy, Humidity 78%",
        "Delhi": "38°C, Sunny, Humidity 45%",
        "Bangalore": "26°C, Rainy, Humidity 85%",
        "London": "18°C, Overcast, Humidity 72%",
        "New York": "28°C, Clear, Humidity 60%",
        "Tokyo": "30°C, Humid, Humidity 82%",
    }
    city_title = city.strip().title()
    return weather_data.get(city_title, f"Weather data not available for '{city}'")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    
    Use when the user needs math calculations, conversions, or comparisons.
    Input should be a valid Python math expression.
    
    Args:
        expression: A Python math expression like '2**10', 'math.sqrt(144)', '100 * 1.18'
    """
    import math
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"

@tool
def search_knowledge(query: str) -> str:
    """Search for factual information and knowledge.
    
    Use when the user asks factual questions about people, places,
    concepts, history, science, or technology.
    
    Args:
        query: The search query
    """
    knowledge_base = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability with significant whitespace.",
        "langchain": "LangChain is an open-source framework for building LLM-powered applications. Created by Harrison Chase in 2022.",
        "transformer": "The Transformer architecture was introduced in 'Attention Is All You Need' (2017) by Vaswani et al. at Google. It uses self-attention mechanisms.",
        "react": "ReAct is an agent pattern that interleaves Reasoning and Acting. The LLM thinks step-by-step and calls tools between thoughts.",
        "india": "India is a country in South Asia. Population: ~1.44 billion (2024). Capital: New Delhi. Largest city: Mumbai.",
        "machine learning": "Machine learning is a subset of AI where systems learn from data without explicit programming. Types: supervised, unsupervised, reinforcement learning.",
    }
    
    query_lower = query.lower()
    for key, value in knowledge_base.items():
        if key in query_lower:
            return value
    
    return f"No specific information found for '{query}'. Try rephrasing your question."
```

### Building the Agent

```python
from langgraph.prebuilt import create_react_agent

# Create the agent
tools = [get_weather, calculate, search_knowledge]
agent = create_react_agent(llm, tools)

# Invoke the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Mumbai?"}]
})

# The last message is the final answer
print(result["messages"][-1].content)
```

### What Happens Internally

```
Input: "What's the weather in Mumbai?"

Message Flow:
┌─────────────────────────────────────────────────────────────┐
│ messages[0]: HumanMessage("What's the weather in Mumbai?")  │
│                                                              │
│ → Agent Node (LLM call #1)                                  │
│                                                              │
│ messages[1]: AIMessage(tool_calls=[                          │
│   {name: "get_weather", args: {city: "Mumbai"}}             │
│ ])                                                           │
│                                                              │
│ → Tool Node (executes get_weather)                          │
│                                                              │
│ messages[2]: ToolMessage("32°C, Partly Cloudy...")           │
│                                                              │
│ → Agent Node (LLM call #2)                                  │
│                                                              │
│ messages[3]: AIMessage("The weather in Mumbai is currently   │
│              32°C with partly cloudy skies and 78%           │
│              humidity.")                                     │
│                                                              │
│ → No more tool calls → DONE                                 │
└─────────────────────────────────────────────────────────────┘
```

### Inspecting All Messages

```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Mumbai?"}]
})

# View the complete message trace
for i, msg in enumerate(result["messages"]):
    print(f"\n--- Message {i} [{msg.__class__.__name__}] ---")
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  🔧 Tool call: {tc['name']}({tc['args']})")
    if msg.content:
        print(f"  💬 {msg.content[:200]}")
```

---

## Part 3: Adding a System Prompt

### Method 1: System Message in `state_modifier`

```python
from langchain_core.messages import SystemMessage

# Add a system prompt that shapes the agent's personality
agent = create_react_agent(
    llm,
    tools,
    state_modifier=SystemMessage(content="""You are a helpful research assistant.

Rules:
- Always verify facts using your tools before answering.
- If you're not sure, say so — don't make things up.
- Be concise but thorough.
- When comparing data, show the numbers side by side.
- For weather, always mention temperature, condition, and humidity.
""")
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Compare weather in Mumbai and London"}]
})
print(result["messages"][-1].content)
```

### Method 2: Prompt Template as `state_modifier`

```python
from langchain_core.prompts import ChatPromptTemplate

# More complex system prompt with dynamic elements
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant with access to these tools:
- Weather lookup: For weather information
- Calculator: For math computations  
- Knowledge search: For factual questions

Today's date: {date}
User's name: {user_name}

Be friendly and address the user by name."""),
    ("placeholder", "{messages}")
])

# Use with agent — the prompt processes messages before the LLM
from datetime import date
agent = create_react_agent(
    llm,
    tools,
    state_modifier=prompt.partial(
        date=str(date.today()),
        user_name="Rahul"
    )
)
```

---

## Part 4: Streaming Agent Steps

### Stream All Steps

```python
# Stream shows you EVERY step as it happens
for step in agent.stream(
    {"messages": [{"role": "user", "content": "What's 2^16 and what's the weather in Delhi?"}]}
):
    # Each step is a dict with the node name as key
    for node_name, node_output in step.items():
        print(f"\n🔄 Node: {node_name}")
        
        if "messages" in node_output:
            for msg in node_output["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"   🔧 Tool: {tc['name']}({tc['args']})")
                elif msg.content:
                    print(f"   💬 {msg.content[:200]}")
```

### Stream with `stream_mode="messages"`

```python
# Stream individual message tokens (for chat UIs)
for message_chunk, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "Tell me about Python"}]},
    stream_mode="messages"
):
    # Only print AI content tokens (skip tool calls)
    if message_chunk.content and metadata.get("langgraph_node") == "agent":
        print(message_chunk.content, end="", flush=True)

print()  # newline
```

### Async Streaming

```python
import asyncio

async def stream_agent():
    async for step in agent.astream(
        {"messages": [{"role": "user", "content": "Weather in Tokyo?"}]}
    ):
        for node_name, node_output in step.items():
            print(f"🔄 {node_name}")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if msg.content:
                        print(f"   {msg.content[:150]}")

asyncio.run(stream_agent())
```

---

## Part 5: Configuration and Error Handling

### Recursion Limit

```python
# Set max iterations to prevent infinite loops
result = agent.invoke(
    {"messages": [{"role": "user", "content": "..."}]},
    config={"recursion_limit": 10}  # Max 10 steps (including tool executions)
)

# Note: recursion_limit counts ALL graph steps, not just LLM calls
# A single tool-calling round = 2 steps (agent node + tool node)
# So recursion_limit=10 means ~5 LLM rounds
```

### Handling Agent Errors

```python
from langgraph.errors import GraphRecursionError

try:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "..."}]},
        config={"recursion_limit": 6}
    )
    answer = result["messages"][-1].content
    
except GraphRecursionError:
    answer = "I wasn't able to find a complete answer within the allowed steps. Please try a simpler question."

except Exception as e:
    answer = f"An error occurred: {str(e)}"

print(answer)
```

### Timeouts

```python
import asyncio

async def agent_with_timeout(question: str, timeout_seconds: int = 30):
    """Run agent with a timeout."""
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": question}]}),
            timeout=timeout_seconds
        )
        return result["messages"][-1].content
    except asyncio.TimeoutError:
        return "The request timed out. Please try again."
```

---

## Part 6: Agent with Memory (Multi-Turn Conversations)

### Using LangGraph's Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver

# Create a checkpointer for conversation memory
memory = MemorySaver()

# Create agent with memory
agent_with_memory = create_react_agent(
    llm,
    tools,
    checkpointer=memory,
    state_modifier=SystemMessage(content="You are a helpful assistant. Remember previous conversations.")
)

# Thread ID groups messages into a conversation
config = {"configurable": {"thread_id": "user-123"}}

# Turn 1
result = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Mumbai?"}]},
    config=config
)
print(f"Turn 1: {result['messages'][-1].content}")

# Turn 2 — agent remembers Turn 1!
result = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "How about in Delhi?"}]},
    config=config
)
print(f"Turn 2: {result['messages'][-1].content}")

# Turn 3 — refers to both previous turns
result = agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "Which city is hotter?"}]},
    config=config
)
print(f"Turn 3: {result['messages'][-1].content}")
```

### Separate Conversations with Thread IDs

```python
# Different thread_id = different conversation
config_alice = {"configurable": {"thread_id": "alice-session"}}
config_bob = {"configurable": {"thread_id": "bob-session"}}

# Alice's conversation
agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "Weather in London?"}]},
    config=config_alice
)

# Bob's conversation — completely separate memory
agent_with_memory.invoke(
    {"messages": [{"role": "user", "content": "Weather in Tokyo?"}]},
    config=config_bob
)
```

---

## Part 7: Real-World Tools — DuckDuckGo + Wikipedia

### Agent with Real Search

```python
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Real tools
search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
    top_k_results=1,
    doc_content_chars_max=1500
))

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Use for calculations.
    
    Args:
        expression: A Python math expression, e.g., '2**32' or '1440/448'
    """
    import math
    try:
        result = eval(expression, {"__builtins__": {}, "math": math})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Create agent with real tools
agent = create_react_agent(
    llm,
    [search, wiki, calculate],
    state_modifier=SystemMessage(content="""You are a thorough research assistant.

Tool usage guidelines:
- Use DuckDuckGo Search for current events, recent news, and real-time data
- Use Wikipedia for established facts, biographies, scientific concepts
- Use Calculator for math operations and comparisons
- Always verify information before presenting it
- Cite your sources when possible
""")
)

# Test with complex multi-step questions
questions = [
    "Who is the current CEO of OpenAI and when was the company founded?",
    "What is the population of Japan and what percentage is that of the world population?",
    "What programming language was LangChain built with and who created it?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"❓ {q}\n")
    
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config={"recursion_limit": 15}
        )
        print(f"💬 {result['messages'][-1].content}")
    except Exception as e:
        print(f"❌ Error: {e}")
```

---

## Part 8: Complete Project — Research Agent with Reporting

```python
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


class ResearchAgent:
    """A research agent that can search, calculate, and maintain conversation history."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
            openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
        )
        
        # Tools
        self.tools = [
            DuckDuckGoSearchRun(),
            WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
                top_k_results=2,
                doc_content_chars_max=1500
            )),
            self._create_calculator(),
            self._create_note_tool(),
        ]
        
        # Memory
        self.memory = MemorySaver()
        
        # Agent
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=self.memory,
            state_modifier=SystemMessage(content=f"""You are an expert research assistant.

Today's date: {datetime.now().strftime('%Y-%m-%d')}

Your approach:
1. Break complex questions into smaller parts
2. Use tools to gather accurate, up-to-date information
3. Verify facts when possible
4. Provide well-structured, comprehensive answers
5. Use the note_taking tool to save important findings for later reference

Tool preferences:
- Current events → DuckDuckGo Search
- Historical facts, biographies → Wikipedia
- Math → Calculator
- Important findings → Note Taking

Always cite your sources and be transparent about what you found vs what you know.""")
        )
        
        # Research notes
        self.notes = []
        self.query_log = []
    
    def _create_calculator(self):
        @tool
        def calculate(expression: str) -> str:
            """Evaluate a mathematical expression.
            
            Use for any math: arithmetic, percentages, comparisons, conversions.
            
            Args:
                expression: A Python math expression, e.g., '2**32', '(100/7)*3'
            """
            import math
            try:
                result = eval(expression, {"__builtins__": {}, "math": math})
                return f"{expression} = {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        return calculate
    
    def _create_note_tool(self):
        notes_ref = self.notes
        
        @tool
        def take_note(note: str) -> str:
            """Save an important finding or piece of information.
            
            Use when you discover key facts worth remembering for the conversation.
            
            Args:
                note: The information to save
            """
            notes_ref.append({
                "note": note,
                "timestamp": datetime.now().isoformat()
            })
            return f"Note saved. Total notes: {len(notes_ref)}"
        return take_note
    
    def ask(self, question: str, thread_id: str = "default") -> dict:
        """Ask the research agent a question."""
        print(f"\n{'='*60}")
        print(f"❓ {question}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        try:
            # Stream the steps for visibility
            steps = []
            final_result = None
            
            for step in self.agent.stream(
                {"messages": [{"role": "user", "content": question}]},
                config={
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": 15
                }
            ):
                for node_name, node_output in step.items():
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    step_info = f"🔧 {tc['name']}({json.dumps(tc['args'])[:100]})"
                                    print(f"   {step_info}")
                                    steps.append(step_info)
                            elif msg.content and node_name == "agent":
                                final_result = msg.content
            
            duration = (datetime.now() - start_time).total_seconds()
            
            print(f"\n💬 Answer ({duration:.1f}s, {len(steps)} tool calls):")
            print(f"{final_result}")
            
            # Log the query
            self.query_log.append({
                "question": question,
                "answer": final_result,
                "steps": len(steps),
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "answer": final_result,
                "steps": steps,
                "duration": duration
            }
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return {"answer": f"Error: {str(e)}", "steps": [], "duration": 0}
    
    def get_notes(self) -> list:
        """Get all saved research notes."""
        return self.notes
    
    def get_stats(self) -> dict:
        """Get usage statistics."""
        if not self.query_log:
            return {"total_queries": 0}
        
        return {
            "total_queries": len(self.query_log),
            "total_tool_calls": sum(q["steps"] for q in self.query_log),
            "avg_duration": sum(q["duration"] for q in self.query_log) / len(self.query_log),
            "avg_tool_calls": sum(q["steps"] for q in self.query_log) / len(self.query_log),
        }


# --- Demo ---
agent = ResearchAgent()

# Simple question (1 tool)
agent.ask("What is the weather in Tokyo?")

# Factual question (Wikipedia)
agent.ask("Tell me about the Transformer architecture in deep learning")

# Multi-step question (multiple tools)
agent.ask("What is India's GDP and what percentage of the US GDP is it?")

# Follow-up question (uses memory)
agent.ask("Based on what we discussed, which country has the larger economy?")

# Question needing no tools
agent.ask("Explain the SOLID principles in software engineering")

# Stats
print(f"\n📊 Session Stats: {json.dumps(agent.get_stats(), indent=2)}")
print(f"📝 Notes: {agent.get_notes()}")
```

---

## Common Mistakes

### Mistake 1: Confusing recursion_limit with number of LLM calls
```python
# ❌ Thinking recursion_limit=10 means 10 LLM calls
# Reality: Each tool-calling round = 2 graph steps
# recursion_limit=10 → ~5 LLM calls max

# ✅ Set recursion_limit = 2 × desired_max_rounds + 1
# Want max 5 rounds? Set recursion_limit = 11
result = agent.invoke(
    {"messages": [...]},
    config={"recursion_limit": 11}
)
```

### Mistake 2: Not using streaming for long-running agents
```python
# ❌ User waits 30 seconds with no feedback
result = agent.invoke({"messages": [...]})

# ✅ Stream steps to show progress
for step in agent.stream({"messages": [...]}):
    # Show each tool call as it happens
    print(f"Step: {step}")
```

### Mistake 3: Too many tools
```python
# ❌ 15 tools — agent gets confused
agent = create_react_agent(llm, [tool1, tool2, ..., tool15])

# ✅ 3-5 focused tools — much better accuracy
agent = create_react_agent(llm, [search, wiki, calculator])
```

### Mistake 4: Not handling GraphRecursionError
```python
# ❌ App crashes when agent loops too long
result = agent.invoke({"messages": [...]}, config={"recursion_limit": 6})

# ✅ Catch and handle gracefully
from langgraph.errors import GraphRecursionError
try:
    result = agent.invoke({"messages": [...]}, config={"recursion_limit": 6})
except GraphRecursionError:
    result = {"messages": [AIMessage(content="I couldn't find a complete answer. Please try a simpler question.")]}
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Always set `recursion_limit` in config | Prevents infinite loops and cost explosions |
| Use streaming for chat UIs | Shows progress to users during multi-step reasoning |
| Start with 3-5 tools | More tools = more confusion for the LLM |
| Use `MemorySaver` for conversations | Enables multi-turn interactions |
| Use `state_modifier` for system prompts | Shapes agent behavior and tool usage |
| Log tool calls and durations | Essential for debugging and cost monitoring |
| Handle `GraphRecursionError` | Graceful degradation when agent loops too long |
| Use strong models (GPT-4o/4o-mini) | Smaller models make poor tool decisions |

---

## Interview Preparation

### Easy
**Q: How do you create a ReAct agent in LangChain?**

> Use `create_react_agent` from `langgraph.prebuilt`. Pass an LLM and a list of tools: `agent = create_react_agent(llm, tools)`. Invoke with `agent.invoke({"messages": [{"role": "user", "content": "..."}]})`. The agent automatically handles the Observe → Think → Act loop, calling tools as needed and returning a final answer. Add `state_modifier` for system prompts and `checkpointer=MemorySaver()` for conversation memory.

### Medium
**Q: How does the message flow work inside a ReAct agent?**

> The agent alternates between two nodes: the **agent node** (LLM) and the **tool node** (executor). Starting from a HumanMessage, the agent node calls the LLM which returns either an AIMessage with `tool_calls` (need more info) or an AIMessage with `content` (final answer). If tool calls exist, the tool node executes them and creates ToolMessages. These go back to the agent node, which calls the LLM again with the full message history. This loops until the LLM returns content without tool calls. Each round is 2 graph steps, so `recursion_limit=10` allows ~5 rounds.

### Hard
**Q: How would you build a production research agent with proper error handling?**

> Key components: (1) **Recursion limit** — set to 10-15 to cap at 5-7 rounds. (2) **Error handling** — catch `GraphRecursionError`, `TimeoutError`, and general exceptions with graceful fallbacks. (3) **Streaming** — use `agent.stream()` to show progress to users in real time. (4) **Memory** — use `MemorySaver` (development) or database-backed checkpointer (production) for multi-turn conversations. (5) **System prompt** — explicit guidelines on when to use each tool, plus instructions to not hallucinate. (6) **Logging** — track every tool call, its duration, and result for debugging and cost analysis. (7) **Tool error handling** — tools should return error strings, not raise exceptions. (8) **Rate limiting** — implement per-user limits on agent invocations. (9) **Cost tracking** — monitor token usage per agent run.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **`create_react_agent`** | Pre-built LangGraph agent with tool-calling loop |
| **Agent loop** | Agent node (LLM) → Tool node (execute) → repeat |
| **`state_modifier`** | System prompt for the agent |
| **`recursion_limit`** | Max graph steps (2 steps per tool round) |
| **`MemorySaver`** | In-memory checkpointer for conversation history |
| **`thread_id`** | Separates different conversations |
| **`agent.stream()`** | Stream agent steps for real-time visibility |
| **`GraphRecursionError`** | Raised when recursion_limit is exceeded |

---

> [← Previous: Intro to Agents](chapter-35-intro-to-agents.md) | [Next: LangGraph Fundamentals →](chapter-37-langgraph-fundamentals.md)

# Chapter 10.3: LangGraph Fundamentals — StateGraph, Nodes & Edges

> **Phase 10 — Agents** | [← Previous: ReAct Agents](chapter-36-react-agents.md) | [Next: Custom Agents →](chapter-38-custom-agents.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand LangGraph's core concepts: State, Nodes, Edges
- ✅ Build custom graphs using `StateGraph`
- ✅ Implement conditional routing with `conditional_edges`
- ✅ Use reducers to manage state updates
- ✅ Build a **multi-step document processing pipeline** with LangGraph
- ✅ Understand how `create_react_agent` works under the hood

| | |
|---|---|
| **Prerequisites** | Chapter 10.2 (ReAct Agents), Phase 6 (Chains & Runnables) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction — Why LangGraph?

`create_react_agent` is powerful but inflexible — it's a fixed pattern: call LLM → maybe call tools → repeat. For real-world applications, you need **custom control flow**:

```
What if you need:
├── A human approval step before executing a dangerous tool?
├── Different LLMs for different steps (GPT-4 for reasoning, GPT-3.5 for extraction)?
├── Parallel tool execution with result merging?
├── Custom retry logic with fallback models?
├── A pipeline: classify → route → process → validate → respond?
└── State that accumulates across multiple steps?
```

**LangGraph** lets you build any workflow as a **directed graph** with full control over state, routing, and execution.

```
LCEL Chains:     A → B → C → D          (linear, fixed)
create_react_agent: [LLM ↔ Tools] loop  (pre-built pattern)
LangGraph:       Any graph you can imagine (full control)
```

---

## Part 1: Core Concepts

### The Three Building Blocks

```
┌─────────────────────────────────────────────────┐
│               LangGraph                          │
│                                                  │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│   │  STATE   │   │  NODES   │   │  EDGES   │   │
│   │          │   │          │   │          │   │
│   │ Shared   │   │ Functions│   │ Connections│  │
│   │ data that│   │ that     │   │ between   │   │
│   │ flows    │   │ process  │   │ nodes     │   │
│   │ through  │   │ the      │   │ (can be   │   │
│   │ the graph│   │ state    │   │ conditional│  │
│   └──────────┘   └──────────┘   └──────────┘   │
└─────────────────────────────────────────────────┘
```

| Concept | What It Is | Analogy |
|---------|-----------|---------|
| **State** | A dictionary/TypedDict that holds all data | The clipboard being passed around |
| **Node** | A function that reads/updates the state | A worker at a station |
| **Edge** | A connection between nodes (normal or conditional) | The conveyor belt between stations |

---

## Part 2: Your First StateGraph

### Step 1: Define the State

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# State is a TypedDict — defines the shape of data flowing through the graph
class MyState(TypedDict):
    messages: Annotated[list, add_messages]  # Chat messages (with reducer)
    step_count: int                           # How many steps we've taken
```

### Step 2: Define Nodes (Functions)

```python
# Each node is a function that:
#   - Takes the CURRENT state as input
#   - Returns a PARTIAL state update (only the keys it wants to change)

def greet_node(state: MyState) -> dict:
    """First node — adds a greeting."""
    return {
        "messages": [{"role": "assistant", "content": "Hello! How can I help?"}],
        "step_count": state.get("step_count", 0) + 1
    }

def process_node(state: MyState) -> dict:
    """Second node — processes the messages."""
    msg_count = len(state["messages"])
    return {
        "messages": [{"role": "assistant", "content": f"I've processed {msg_count} messages."}],
        "step_count": state.get("step_count", 0) + 1
    }
```

### Step 3: Build the Graph

```python
# Create the graph with our state type
graph = StateGraph(MyState)

# Add nodes
graph.add_node("greet", greet_node)
graph.add_node("process", process_node)

# Add edges (connections)
graph.add_edge(START, "greet")       # Start → greet
graph.add_edge("greet", "process")   # greet → process
graph.add_edge("process", END)       # process → end

# Compile the graph into a runnable
app = graph.compile()

# Run it!
result = app.invoke({
    "messages": [{"role": "user", "content": "Hi there!"}],
    "step_count": 0
})

print(f"Steps taken: {result['step_count']}")
for msg in result["messages"]:
    role = msg.type if hasattr(msg, 'type') else msg.get("role", "?")
    content = msg.content if hasattr(msg, 'content') else msg.get("content", "")
    print(f"  [{role}] {content}")
```

### Visualizing the Graph

```python
# LangGraph can generate a visual representation
print(app.get_graph().draw_ascii())
```

```
         +-----------+
         | __start__ |
         +-----------+
               |
               v
          +--------+
          | greet  |
          +--------+
               |
               v
         +---------+
         | process |
         +---------+
               |
               v
          +-------+
          | __end__|
          +-------+
```

---

## Part 3: Conditional Edges — Dynamic Routing

The real power of LangGraph — route to different nodes based on the state:

### Basic Conditional Routing

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


class RouterState(TypedDict):
    query: str
    category: str
    result: str


def classify_node(state: RouterState) -> dict:
    """Classify the query into a category."""
    query = state["query"].lower()
    
    if any(w in query for w in ["weather", "temperature", "rain"]):
        category = "weather"
    elif any(w in query for w in ["calculate", "math", "compute", "sum"]):
        category = "math"
    elif any(w in query for w in ["hello", "hi", "hey", "thanks"]):
        category = "greeting"
    else:
        category = "general"
    
    return {"category": category}


def weather_node(state: RouterState) -> dict:
    """Handle weather queries."""
    return {"result": f"🌤️ Weather info for: {state['query']}"}


def math_node(state: RouterState) -> dict:
    """Handle math queries."""
    return {"result": f"🔢 Math result for: {state['query']}"}


def greeting_node(state: RouterState) -> dict:
    """Handle greetings."""
    return {"result": "👋 Hello! How can I help you today?"}


def general_node(state: RouterState) -> dict:
    """Handle general queries."""
    return {"result": f"💬 General response for: {state['query']}"}


# The routing function — returns the NAME of the next node
def route_query(state: RouterState) -> Literal["weather", "math", "greeting", "general"]:
    """Route to the appropriate handler based on category."""
    return state["category"]


# Build the graph
graph = StateGraph(RouterState)

# Add nodes
graph.add_node("classify", classify_node)
graph.add_node("weather", weather_node)
graph.add_node("math", math_node)
graph.add_node("greeting", greeting_node)
graph.add_node("general", general_node)

# Edges
graph.add_edge(START, "classify")

# CONDITIONAL edge — route based on classification
graph.add_conditional_edges(
    "classify",                    # From this node...
    route_query,                   # Use this function to decide...
    {                              # Map return values to node names:
        "weather": "weather",
        "math": "math",
        "greeting": "greeting",
        "general": "general"
    }
)

# All handlers → END
graph.add_edge("weather", END)
graph.add_edge("math", END)
graph.add_edge("greeting", END)
graph.add_edge("general", END)

# Compile
app = graph.compile()

# Test
queries = [
    "What's the temperature in Mumbai?",
    "Calculate 2 to the power of 10",
    "Hello there!",
    "Tell me about Python programming",
]

for q in queries:
    result = app.invoke({"query": q, "category": "", "result": ""})
    print(f"  Q: {q}")
    print(f"  A: {result['result']}\n")
```

### Visualized

```
          +-----------+
          | __start__ |
          +-----------+
                |
                v
          +-----------+
          | classify  |
          +-----------+
           /   |   |   \
          /    |   |    \
         v     v   v     v
   weather  math  greet  general
         \    |    |    /
          \   |    |   /
           v  v    v  v
          +---------+
          | __end__ |
          +---------+
```

---

## Part 4: Reducers — How State Updates Work

### The Problem Without Reducers

```python
# Without a reducer, each node OVERWRITES the state key:
# Node 1 returns: {"messages": ["Hello"]}
# Node 2 returns: {"messages": ["How are you?"]}
# Result: {"messages": ["How are you?"]}  ← "Hello" is GONE!
```

### The `add_messages` Reducer

```python
from typing import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    # The Annotated type tells LangGraph to use add_messages as the reducer
    messages: Annotated[list, add_messages]
    # This means: APPEND new messages instead of replacing

# Now:
# Node 1 returns: {"messages": ["Hello"]}
# Node 2 returns: {"messages": ["How are you?"]}  
# Result: {"messages": ["Hello", "How are you?"]}  ← Both preserved! ✅
```

### Custom Reducers

```python
from operator import add

class AccumulatorState(TypedDict):
    # add reducer: appends lists together
    items: Annotated[list, add]         # [1,2] + [3,4] = [1,2,3,4]
    
    # No reducer: overwrites
    current_step: str                    # "step_a" then "step_b" → "step_b"
    
    # Custom reducer function
    total_cost: Annotated[float, lambda old, new: old + new]  # Accumulates


# Custom reducer function example
def merge_dicts(old: dict, new: dict) -> dict:
    """Merge two dictionaries, keeping all keys."""
    merged = {**old, **new}
    return merged

class MergingState(TypedDict):
    metadata: Annotated[dict, merge_dicts]  # Merges instead of replacing
```

---

## Part 5: Loops in Graphs

### Creating a Loop

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class LoopState(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int
    max_iterations: int
    is_satisfactory: bool


def process_node(state: LoopState) -> dict:
    """Process and generate a result."""
    iteration = state.get("iteration", 0) + 1
    
    # Simulate processing that might need retries
    is_good = iteration >= 3  # Becomes satisfactory on 3rd try
    
    return {
        "messages": [{"role": "assistant", "content": f"Attempt {iteration}: {'Good result!' if is_good else 'Not good enough, refining...'}"}],
        "iteration": iteration,
        "is_satisfactory": is_good
    }


def should_continue(state: LoopState) -> Literal["process", "__end__"]:
    """Decide whether to continue the loop."""
    if state.get("is_satisfactory", False):
        return END
    if state.get("iteration", 0) >= state.get("max_iterations", 5):
        return END
    return "process"


# Build graph with loop
graph = StateGraph(LoopState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_conditional_edges("process", should_continue)

app = graph.compile()

result = app.invoke({
    "messages": [{"role": "user", "content": "Generate something good"}],
    "iteration": 0,
    "max_iterations": 5,
    "is_satisfactory": False,
})

print(f"Finished after {result['iteration']} iterations")
for msg in result["messages"]:
    content = msg.content if hasattr(msg, 'content') else msg.get("content", "")
    print(f"  {content}")
```

---

## Part 6: How `create_react_agent` Works Under the Hood

Now that you understand StateGraph, here's what `create_react_agent` builds:

```python
# This is a SIMPLIFIED version of what create_react_agent does internally:

from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState) -> dict:
    """Call the LLM with tools."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute tool calls from the last AI message."""
    last_message = state["messages"][-1]
    results = []
    
    for tool_call in last_message.tool_calls:
        tool = tool_map[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": results}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """If the last message has tool calls, go to tools. Otherwise, end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")  # After tools, go back to agent

agent = graph.compile()
```

### The ReAct Agent Graph

```
          +-----------+
          | __start__ |
          +-----------+
                |
                v
          +-----------+
     ┌───→|   agent   |──── has tool_calls? ────┐
     │    +-----------+                          │
     │          |                                │
     │          | (no tool calls = final answer) │
     │          v                                │
     │    +-----------+                          │
     │    |  __end__  |                          │
     │    +-----------+                          │
     │                                           │
     │    +-----------+                          │
     └────|   tools   |←─────────────────────────┘
          +-----------+
            (execute tool calls, 
             send results back to agent)
```

---

## Part 7: Complete Project — Document Processing Pipeline

Build a multi-step document processing system:

```python
import os
from typing import TypedDict, Annotated, Literal
from operator import add
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)


# --- State ---
class DocState(TypedDict):
    document: str           # Input document
    doc_type: str           # Classified type
    language: str           # Detected language
    summary: str            # Generated summary
    key_points: list        # Extracted key points
    sentiment: str          # Detected sentiment
    word_count: int         # Word count
    processing_log: Annotated[list, add]  # Log of processing steps


# --- Nodes ---

def classify_document(state: DocState) -> dict:
    """Classify the document type."""
    response = llm.invoke([
        SystemMessage(content="Classify this document as one of: 'technical', 'business', 'academic', 'casual'. Respond with just the category."),
        HumanMessage(content=state["document"][:500])
    ])
    
    doc_type = response.content.strip().lower()
    return {
        "doc_type": doc_type,
        "processing_log": [f"✅ Classified as: {doc_type}"]
    }


def detect_language(state: DocState) -> dict:
    """Detect the document language."""
    response = llm.invoke([
        SystemMessage(content="What language is this text written in? Respond with just the language name."),
        HumanMessage(content=state["document"][:200])
    ])
    
    language = response.content.strip()
    return {
        "language": language,
        "processing_log": [f"✅ Language: {language}"]
    }


def count_words(state: DocState) -> dict:
    """Count words in the document."""
    count = len(state["document"].split())
    return {
        "word_count": count,
        "processing_log": [f"✅ Word count: {count}"]
    }


def summarize_technical(state: DocState) -> dict:
    """Summarize a technical document."""
    response = llm.invoke([
        SystemMessage(content="Summarize this technical document in 2-3 concise sentences. Focus on the technical contributions and key findings."),
        HumanMessage(content=state["document"])
    ])
    return {
        "summary": response.content,
        "processing_log": ["✅ Technical summary generated"]
    }


def summarize_business(state: DocState) -> dict:
    """Summarize a business document."""
    response = llm.invoke([
        SystemMessage(content="Summarize this business document in 2-3 sentences. Focus on key business decisions, metrics, and action items."),
        HumanMessage(content=state["document"])
    ])
    return {
        "summary": response.content,
        "processing_log": ["✅ Business summary generated"]
    }


def summarize_general(state: DocState) -> dict:
    """Summarize any other document type."""
    response = llm.invoke([
        SystemMessage(content="Summarize this document in 2-3 clear, concise sentences."),
        HumanMessage(content=state["document"])
    ])
    return {
        "summary": response.content,
        "processing_log": ["✅ General summary generated"]
    }


def extract_key_points(state: DocState) -> dict:
    """Extract key points from the document."""
    response = llm.invoke([
        SystemMessage(content="Extract 3-5 key points from this document. Return each point on a new line, prefixed with '•'."),
        HumanMessage(content=state["document"])
    ])
    
    points = [line.strip().lstrip("•-").strip() 
              for line in response.content.split("\n") 
              if line.strip() and line.strip().startswith(("•", "-", "1", "2", "3", "4", "5"))]
    
    return {
        "key_points": points or [response.content],
        "processing_log": [f"✅ Extracted {len(points)} key points"]
    }


def analyze_sentiment(state: DocState) -> dict:
    """Analyze document sentiment."""
    response = llm.invoke([
        SystemMessage(content="Analyze the sentiment of this text. Respond with one of: 'positive', 'negative', 'neutral', 'mixed'."),
        HumanMessage(content=state["document"][:500])
    ])
    
    sentiment = response.content.strip().lower()
    return {
        "sentiment": sentiment,
        "processing_log": [f"✅ Sentiment: {sentiment}"]
    }


# --- Routing ---

def route_to_summarizer(state: DocState) -> Literal["summarize_technical", "summarize_business", "summarize_general"]:
    """Route to the appropriate summarizer based on document type."""
    doc_type = state.get("doc_type", "")
    if "technical" in doc_type or "academic" in doc_type:
        return "summarize_technical"
    elif "business" in doc_type:
        return "summarize_business"
    else:
        return "summarize_general"


# --- Build the Graph ---

graph = StateGraph(DocState)

# Add all nodes
graph.add_node("classify", classify_document)
graph.add_node("detect_language", detect_language)
graph.add_node("count_words", count_words)
graph.add_node("summarize_technical", summarize_technical)
graph.add_node("summarize_business", summarize_business)
graph.add_node("summarize_general", summarize_general)
graph.add_node("extract_key_points", extract_key_points)
graph.add_node("analyze_sentiment", analyze_sentiment)

# Entry point
graph.add_edge(START, "classify")

# After classification: run language detection, word count, and routing in sequence
graph.add_edge("classify", "detect_language")
graph.add_edge("detect_language", "count_words")

# Conditional routing to appropriate summarizer
graph.add_conditional_edges("count_words", route_to_summarizer)

# After summarization: extract key points and analyze sentiment
graph.add_edge("summarize_technical", "extract_key_points")
graph.add_edge("summarize_business", "extract_key_points")
graph.add_edge("summarize_general", "extract_key_points")

graph.add_edge("extract_key_points", "analyze_sentiment")
graph.add_edge("analyze_sentiment", END)

# Compile
pipeline = graph.compile()

# --- Test ---

document = """
The Transformer architecture, introduced in "Attention Is All You Need" by Vaswani et al. 
in 2017, revolutionized natural language processing. Unlike recurrent neural networks (RNNs), 
transformers process entire sequences in parallel using self-attention mechanisms. This 
architectural innovation enabled the development of large language models (LLMs) like GPT, 
BERT, and T5. The key components include multi-head attention, positional encoding, and 
feed-forward neural networks. Transformers have achieved state-of-the-art results on tasks 
including machine translation, text generation, and question answering. The architecture's 
scalability has led to models with billions of parameters, trained on massive text corpora.
"""

print("🔄 Processing document...\n")

# Stream the steps
for step in pipeline.stream({
    "document": document,
    "doc_type": "",
    "language": "",
    "summary": "",
    "key_points": [],
    "sentiment": "",
    "word_count": 0,
    "processing_log": [],
}):
    for node_name, output in step.items():
        if "processing_log" in output:
            for log in output["processing_log"]:
                print(f"  {log}")

# Get final result
result = pipeline.invoke({
    "document": document,
    "doc_type": "",
    "language": "",
    "summary": "",
    "key_points": [],
    "sentiment": "",
    "word_count": 0,
    "processing_log": [],
})

print(f"\n{'='*60}")
print(f"📄 Document Analysis Report")
print(f"{'='*60}")
print(f"  Type:      {result['doc_type']}")
print(f"  Language:  {result['language']}")
print(f"  Words:     {result['word_count']}")
print(f"  Sentiment: {result['sentiment']}")
print(f"\n📝 Summary:")
print(f"  {result['summary']}")
print(f"\n🔑 Key Points:")
for i, point in enumerate(result['key_points'], 1):
    print(f"  {i}. {point}")
print(f"\n📋 Processing Log:")
for log in result['processing_log']:
    print(f"  {log}")
```

---

## Common Mistakes

### Mistake 1: Forgetting to handle state keys that haven't been set yet
```python
# ❌ KeyError if 'count' hasn't been initialized
def my_node(state):
    return {"count": state["count"] + 1}

# ✅ Use .get() with default values
def my_node(state):
    return {"count": state.get("count", 0) + 1}
```

### Mistake 2: Returning full state instead of updates
```python
# ❌ Overwrites ALL state keys not mentioned
def my_node(state):
    return {
        "messages": state["messages"] + ["new"],
        "count": state["count"],      # Have to repeat everything!
        "category": state["category"],
    }

# ✅ Return only what changed — LangGraph merges it
def my_node(state):
    return {"messages": [{"role": "assistant", "content": "new"}]}
    # Other keys stay untouched!
```

### Mistake 3: Not using reducers for lists
```python
# ❌ Each node overwrites the messages list
class State(TypedDict):
    messages: list  # No reducer — overwrites!

# ✅ Use add_messages reducer — appends
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Appends!
```

### Mistake 4: Creating cycles without an exit condition
```python
# ❌ Infinite loop — no way to exit
graph.add_edge("process", "validate")
graph.add_edge("validate", "process")  # Always loops back!

# ✅ Use conditional edge with an exit path
graph.add_conditional_edges("validate", should_continue)
# should_continue returns either "process" (retry) or END (done)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use `Annotated[list, add_messages]` for message lists | Prevents accidental overwrites |
| Return only changed keys from nodes | Cleaner, less error-prone |
| Use `state.get(key, default)` | Handles uninitialized state |
| Always provide an exit condition for loops | Prevents infinite loops |
| Use `graph.compile()` before invoking | Validates the graph structure |
| Visualize with `get_graph().draw_ascii()` | Catch routing issues early |
| Stream steps for complex pipelines | Visibility into processing progress |
| Keep nodes focused (single responsibility) | Easier to test and debug |

---

## Interview Preparation

### Easy
**Q: What is LangGraph and how does it differ from LCEL chains?**

> LangGraph is a framework for building stateful, multi-step AI workflows as directed graphs. Unlike LCEL chains (which are linear: A → B → C), LangGraph supports conditional routing, loops, parallel execution, and shared state. You define a State (TypedDict), Nodes (functions that process state), and Edges (connections between nodes, potentially conditional). This enables complex patterns like agents, human-in-the-loop, retries, and multi-branch processing that linear chains can't express.

### Medium
**Q: What are reducers in LangGraph and why are they important?**

> Reducers control how state updates are applied. Without a reducer, returning a state key overwrites its value. With a reducer like `add_messages`, values are appended instead. You specify reducers using `Annotated[type, reducer_function]` in the state TypedDict. Common reducers: `add_messages` (appends messages), `operator.add` (concatenates lists), or custom functions. Reducers are critical for lists and accumulators — without them, each node would overwrite previous results instead of building on them.

### Hard
**Q: How would you build a custom agent using LangGraph's StateGraph?**

> Define a state with `Annotated[list, add_messages]` for messages. Create two nodes: (1) **agent node** — calls `llm.bind_tools(tools).invoke(state["messages"])` and returns the response. (2) **tool node** — reads `tool_calls` from the last AI message, executes each tool, and returns `ToolMessage` results. Add a conditional edge from agent: if `tool_calls` exist → go to tools node; otherwise → END. Add an edge from tools → agent (loop back). Compile and run. This is essentially what `create_react_agent` does internally, but building it manually gives you full control to add custom logic like human approval steps, logging, or conditional tool execution.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **StateGraph** | Graph builder — define state, nodes, and edges |
| **State (TypedDict)** | Shared data structure flowing through the graph |
| **Node** | Function that reads state and returns updates |
| **Edge** | Connection between nodes |
| **Conditional Edge** | Edge that routes based on state (via routing function) |
| **Reducer** | Controls how state updates merge (append vs overwrite) |
| **`add_messages`** | Built-in reducer that appends messages |
| **`START` / `END`** | Special nodes marking graph entry and exit |
| **`graph.compile()`** | Validates and creates an executable runnable |
| **Loop** | Conditional edge that points back to an earlier node |

---

> [← Previous: ReAct Agents](chapter-36-react-agents.md) | [Next: Custom Agents →](chapter-38-custom-agents.md)

# Chapter 9.2: Built-in Tools — Search, Wikipedia, Calculator & More

> **Phase 9 — Tools & Tool Calling** | [← Previous: What Are Tools?](chapter-30-what-are-tools.md) | [Next: Custom Tools →](chapter-32-custom-tools.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Install and use LangChain's most popular built-in tools
- ✅ Use DuckDuckGo Search to give LLMs real-time web access
- ✅ Query Wikipedia for factual knowledge
- ✅ Execute Python code safely with the Python REPL tool
- ✅ Combine multiple tools with an LLM using `bind_tools`
- ✅ Build a **multi-tool research assistant** that picks the right tool per query

| | |
|---|---|
| **Prerequisites** | Chapter 9.1 (What Are Tools?), Phase 6 (Chains & Runnables) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

In Chapter 9.1, you learned *what* tools are and *why* LLMs need them. Now it's time to get your hands dirty.

LangChain ships with a rich ecosystem of **pre-built tools** — ready-to-use functions that you can plug into any chain or agent. No need to build everything from scratch.

```
┌──────────────────────────────────────────────────────┐
│            LangChain Built-in Tools                   │
│                                                       │
│  🔍 Search        │  📖 Wikipedia     │  🐍 Python    │
│  DuckDuckGo       │  Query articles   │  REPL / exec  │
│  Tavily           │  Full summaries   │  Math, data    │
│  Google Search    │                   │                │
│                   │                   │                │
│  📐 Math          │  🌐 Requests      │  📁 File I/O  │
│  Calculator       │  HTTP GET/POST    │  Read / write  │
│  Wolfram Alpha    │  API calls        │  List files    │
│                   │                   │                │
│  🗂️ JSON          │  💻 Shell         │  🧩 And more  │
│  Parse & query    │  Bash commands    │  SQL, Arxiv,   │
│  JSON data        │  (careful!)       │  YouTube, etc. │
└──────────────────────────────────────────────────────┘
```

**In this chapter, we'll focus on the four most useful ones:**
1. **DuckDuckGo Search** — real-time web search
2. **Wikipedia** — factual knowledge retrieval
3. **Python REPL** — code execution (math, data processing)
4. **LLM Math** — reliable calculations

---

## Part 1: Setup & Installation

```bash
# Core LangChain packages (you should already have these)
pip install langchain langchain-openai

# Built-in tools — install what you need
pip install duckduckgo-search          # DuckDuckGo Search
pip install wikipedia                   # Wikipedia
pip install langchain-experimental      # Python REPL tool
```

### LLM Setup (Used Throughout)

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)
```

---

## Part 2: DuckDuckGo Search — Real-Time Web Access

This is probably the **most useful built-in tool**. It gives your LLM access to the live internet — for free, with no API key required.

### Basic Usage (Tool Directly)

```python
from langchain_community.tools import DuckDuckGoSearchRun

# Create the search tool
search = DuckDuckGoSearchRun()

# Use it directly — returns a string of search results
result = search.invoke("Latest LangChain updates 2025")
print(result)
```

**Output** (truncated):
```
LangChain 0.3 was released with major changes including...
The framework now supports native tool calling across all providers...
```

### Inspect the Tool's Metadata

```python
# Every tool has these attributes — this is what the LLM sees
print(f"Name:        {search.name}")
print(f"Description: {search.description}")
print(f"Args Schema: {search.args_schema.schema()}")
```

```
Name:        duckduckgo_search
Description: A wrapper around DuckDuckGo Search. Useful for when you need to 
             answer questions about current events. Input should be a search query.
Args Schema: {'properties': {'query': {'title': 'Query', 'type': 'string'}}, 
              'required': ['query'], 'title': 'DuckDuckGoSearchRunInput', 'type': 'object'}
```

### DuckDuckGoSearchResults — Structured Output

For more control, use `DuckDuckGoSearchResults` which returns structured results with snippets, titles, and links:

```python
from langchain_community.tools import DuckDuckGoSearchResults

# Get structured results
search_results = DuckDuckGoSearchResults(num_results=5)
result = search_results.invoke("Python 3.13 new features")
print(result)
```

**Output:**
```
[snippet: Python 3.13 introduces a new interactive interpreter..., 
 title: What's New in Python 3.13, 
 link: https://docs.python.org/3/whatsnew/3.13.html],
[snippet: The free-threaded build allows Python to run without GIL..., 
 title: Python 3.13 Release Notes, 
 link: https://www.python.org/downloads/release/python-3130/]
```

### Customizing the Search

```python
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# Create a wrapper with custom settings
wrapper = DuckDuckGoSearchAPIWrapper(
    region="in-en",          # Region: India - English
    time="m",                # Time filter: past month (d=day, w=week, m=month, y=year)
    max_results=5,           # Number of results
    safesearch="moderate"    # SafeSearch: on, moderate, off
)

# Use wrapper with the tool
search = DuckDuckGoSearchRun(api_wrapper=wrapper)
result = search.invoke("AI developments in India")
print(result)
```

### ⚠️ DuckDuckGo Rate Limits

```python
# DuckDuckGo has informal rate limits — too many requests too fast = blocked
# Best practice: add a small delay between calls
import time

queries = ["Python 3.13", "LangChain agents", "vector databases"]
for q in queries:
    result = search.invoke(q)
    print(f"🔍 {q}: {result[:100]}...")
    time.sleep(1)  # Be polite — 1 second between requests
```

---

## Part 3: Wikipedia — Factual Knowledge

Wikipedia is ideal for **factual, encyclopedic** queries. It's more reliable than web search for established topics because it returns full article content rather than random web snippets.

### Basic Usage

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Create with custom settings
wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=2,        # Number of articles to fetch
    doc_content_chars_max=2000  # Max characters per article
)

wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Search Wikipedia
result = wiki.invoke("Transformer neural network architecture")
print(result[:500])
```

**Output:**
```
Page: Transformer (deep learning architecture)
Summary: A transformer is a deep learning architecture based on the 
multi-head attention mechanism, proposed in the 2017 paper "Attention 
Is All You Need". Text is converted to numerical representations called 
tokens, and each token is converted into a vector via lookup from a word 
embedding table...
```

### Tool Metadata

```python
print(f"Name:        {wiki.name}")
print(f"Description: {wiki.description}")
```

```
Name:        wikipedia
Description: A wrapper around Wikipedia. Useful for when you need to answer 
             general questions about people, places, companies, facts, 
             historical events, or other subjects. Input should be a search query.
```

### When to Use Wikipedia vs Search

| Use Wikipedia | Use DuckDuckGo |
|--------------|----------------|
| Historical facts, biographies | Current events, latest news |
| Scientific concepts | Real-time data (stock prices, weather) |
| Established topics with Wikipedia pages | Niche/recent topics |
| Need detailed, structured content | Need broad web coverage |
| Want more reliable, cited information | Need diverse sources |

---

## Part 4: Python REPL — Code Execution

The Python REPL tool lets the LLM **write and execute Python code**. This is incredibly powerful for:
- Mathematical calculations
- Data processing
- String manipulation
- Any computation the LLM can't do natively

### Setup

```python
from langchain_experimental.tools import PythonREPLTool

# Create the tool
python_repl = PythonREPLTool()
```

### Basic Usage

```python
# The LLM can generate code and this tool executes it
result = python_repl.invoke("print(7847 * 3291)")
print(result)  # 25820577

# More complex calculations
result = python_repl.invoke("""
import math
# Calculate compound interest
principal = 100000
rate = 0.08
years = 10
amount = principal * (1 + rate) ** years
print(f"After {years} years: ₹{amount:,.2f}")
print(f"Interest earned: ₹{amount - principal:,.2f}")
""")
print(result)
```

**Output:**
```
After 10 years: ₹215,892.50
Interest earned: ₹115,892.50
```

### ⚠️ Security Warning — Python REPL Is Dangerous

```python
# The Python REPL can execute ANY code — including malicious code!
# ❌ NEVER expose this to untrusted users in production

# An LLM could potentially generate:
#   import os; os.system("rm -rf /")      ← Deletes everything
#   import subprocess; subprocess.run(...) ← Runs shell commands
#   open("/etc/passwd").read()             ← Reads sensitive files

# ✅ Safety measures:
# 1. Use in development/testing only
# 2. In production, use sandboxed environments (Docker, E2B, Modal)
# 3. Use human-in-the-loop for code review before execution
# 4. Restrict imports with custom REPL implementations
```

### Tool Metadata

```python
print(f"Name:        {python_repl.name}")
print(f"Description: {python_repl.description}")
```

```
Name:        Python_REPL
Description: A Python shell. Use this to execute python commands. Input should 
             be a valid python command. If you want to see the output of a value, 
             you should print it out with `print(...)`.
```

---

## Part 5: LLM Math Chain — Reliable Calculations

For simpler math where you don't need full Python, the LLM Math tool chains the LLM with a calculator:

```python
from langchain.chains import LLMMathChain
from langchain.tools import Tool

# Create the math chain
llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)

# Wrap it as a tool
calculator = Tool(
    name="Calculator",
    description="Useful for when you need to answer questions about math. "
                "Input should be a mathematical expression.",
    func=llm_math.run
)

# Use it
result = calculator.invoke("What is the square root of 144 plus 13 squared?")
print(result)  # Answer: 181.0  (√144 = 12, 13² = 169, 12 + 169 = 181)
```

### How It Works Internally

```
User: "What is 15% of 340?"
         │
         ↓
LLM translates to math expression: "0.15 * 340"
         │
         ↓
Python eval() computes: 51.0
         │
         ↓
Result: "Answer: 51.0"
```

---

## Part 6: Binding Tools to LLMs

Now the real magic — give an LLM **multiple tools** and let it **decide** which one to use.

### `bind_tools()` — The Key Method

```python
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool

# Create tools
search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(
    top_k_results=1, doc_content_chars_max=1000
))
python_repl = PythonREPLTool()

# List of tools
tools = [search, wiki, python_repl]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)
```

### What `bind_tools` Does

```python
# When you bind tools, the LLM gets tool schemas in every request:
# {
#   "tools": [
#     {"name": "duckduckgo_search", "description": "...", "parameters": {...}},
#     {"name": "wikipedia", "description": "...", "parameters": {...}},
#     {"name": "Python_REPL", "description": "...", "parameters": {...}}
#   ]
# }
#
# The LLM can now CHOOSE which tool to call based on the user's question.
```

### Testing Tool Selection

```python
# The LLM decides which tool to call — it returns an AIMessage with tool_calls

# Question 1: Current events → should pick Search
response = llm_with_tools.invoke("What happened in tech news today?")
print(f"Tool calls: {response.tool_calls}")
# [{'name': 'duckduckgo_search', 'args': {'query': 'tech news today'}, 'id': '...'}]

# Question 2: Factual knowledge → should pick Wikipedia
response = llm_with_tools.invoke("Who invented the transformer architecture?")
print(f"Tool calls: {response.tool_calls}")
# [{'name': 'wikipedia', 'args': {'query': 'Transformer architecture deep learning'}, 'id': '...'}]

# Question 3: Math → should pick Python REPL
response = llm_with_tools.invoke("What is 2 to the power of 32?")
print(f"Tool calls: {response.tool_calls}")
# [{'name': 'Python_REPL', 'args': {'command': 'print(2 ** 32)'}, 'id': '...'}]

# Question 4: General chat → NO tool call
response = llm_with_tools.invoke("Hello! How are you?")
print(f"Tool calls: {response.tool_calls}")  # []
print(f"Content:    {response.content}")       # "Hello! I'm doing well..."
```

### Understanding the Response

```python
response = llm_with_tools.invoke("Search for the latest Python release")

# The response is an AIMessage
print(type(response))  # <class 'langchain_core.messages.AIMessage'>

# If the LLM decides to call a tool:
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"Tool:  {tool_call['name']}")
        print(f"Args:  {tool_call['args']}")
        print(f"ID:    {tool_call['id']}")
else:
    # No tool needed — LLM answered directly
    print(f"Answer: {response.content}")
```

### ⚠️ `bind_tools` Does NOT Execute Tools

```python
# Common misconception: bind_tools makes the LLM call the tool
# Reality: The LLM only DECIDES which tool to call and with what args
# YOU still need to execute the tool yourself!

response = llm_with_tools.invoke("What's the weather in Mumbai?")

# The LLM outputs: "Call duckduckgo_search with query='weather Mumbai'"
# But it hasn't actually searched! You need to:
# 1. Parse the tool call from the response
# 2. Execute the corresponding tool
# 3. Send the result back to the LLM

# We'll automate this in Part 7 and in the Agents chapter (Phase 10).
```

---

## Part 7: Executing Tool Calls — The Complete Loop

Here's how to manually execute the full tool-calling loop:

```python
from langchain_core.messages import HumanMessage, ToolMessage

# Step 1: Create the tool map (name → tool object)
tool_map = {tool.name: tool for tool in tools}

def ask_with_tools(question: str) -> str:
    """Complete tool-calling loop: ask → LLM decides → execute → LLM responds."""
    
    messages = [HumanMessage(content=question)]
    
    # Step 2: Ask the LLM (it may return a tool call or a direct answer)
    response = llm_with_tools.invoke(messages)
    messages.append(response)
    
    # Step 3: If no tool calls, return the direct answer
    if not response.tool_calls:
        return response.content
    
    # Step 4: Execute each tool call
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]
        
        print(f"🔧 Calling tool: {tool_name}({tool_args})")
        
        # Execute the tool
        tool = tool_map[tool_name]
        result = tool.invoke(tool_args)
        
        print(f"📋 Result: {str(result)[:200]}...")
        
        # Add the tool result to messages
        messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_id
        ))
    
    # Step 5: Send everything back to the LLM for a final answer
    final_response = llm_with_tools.invoke(messages)
    return final_response.content


# --- Test it! ---

# Current events
print("=" * 60)
answer = ask_with_tools("What are the latest developments in AI?")
print(f"\n💬 Answer: {answer}")

# Factual knowledge
print("=" * 60)
answer = ask_with_tools("Tell me about the history of Python programming language")
print(f"\n💬 Answer: {answer}")

# Math
print("=" * 60)
answer = ask_with_tools("Calculate the compound interest on $10,000 at 7% for 5 years")
print(f"\n💬 Answer: {answer}")

# No tool needed
print("=" * 60)
answer = ask_with_tools("Write a haiku about programming")
print(f"\n💬 Answer: {answer}")
```

### Handling Multiple Tool Calls

```python
def ask_with_tools_multi(question: str, max_iterations: int = 5) -> str:
    """Handle multiple rounds of tool calling (LLM may need several tools)."""
    
    messages = [HumanMessage(content=question)]
    
    for i in range(max_iterations):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # No more tool calls — we have the final answer
        if not response.tool_calls:
            return response.content
        
        # Execute all tool calls in this round
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"🔧 [{i+1}] {tool_name}({tool_args})")
            
            try:
                tool = tool_map[tool_name]
                result = tool.invoke(tool_args)
            except Exception as e:
                result = f"Error: {str(e)}"
            
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))
    
    return "Max iterations reached. Last response: " + response.content


# Test: Question that might need multiple tools
answer = ask_with_tools_multi(
    "Search for who won the latest Cricket World Cup, "
    "then look up that team's captain on Wikipedia"
)
print(f"\n💬 Answer: {answer}")
```

---

## Part 8: More Built-in Tools (Quick Reference)

### Tavily Search (Better Quality, Needs API Key)

```python
# pip install tavily-python
from langchain_community.tools.tavily_search import TavilySearchResults

# Requires TAVILY_API_KEY environment variable
# Get free API key at https://tavily.com
tavily = TavilySearchResults(max_results=3)
result = tavily.invoke("LangChain vs LlamaIndex comparison 2025")
print(result)
```

**Why Tavily over DuckDuckGo?**

| Feature | DuckDuckGo | Tavily |
|---------|-----------|--------|
| Cost | Free | Free tier (1000 req/mo) |
| API Key | Not required | Required |
| Result quality | Good | Better (AI-optimized) |
| Rate limits | Informal, can get blocked | Clear limits |
| Response format | Raw text | Structured JSON |
| Best for | Prototyping, free projects | Production apps |

### Arxiv — Research Papers

```python
# pip install arxiv
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=1000
))

result = arxiv.invoke("Retrieval Augmented Generation")
print(result[:500])
```

### Requests — HTTP API Calls

```python
from langchain_community.tools.requests.tool import (
    RequestsGetTool,
    RequestsPostTool,
)
from langchain_community.utilities.requests import TextRequestsWrapper

# Create a GET tool
requests_wrapper = TextRequestsWrapper()
get_tool = RequestsGetTool(
    requests_wrapper=requests_wrapper,
    description="Make HTTP GET requests to URLs. Input is the URL."
)

# Fetch a public API
result = get_tool.invoke("https://api.github.com/repos/langchain-ai/langchain")
print(result[:300])
```

### Shell Tool (Use with Extreme Caution)

```python
from langchain_community.tools import ShellTool

# ⚠️ DANGEROUS — can execute ANY shell command
shell = ShellTool()
result = shell.invoke("echo 'Hello from shell!' && date")
print(result)

# NEVER expose this to untrusted users or autonomous agents without safeguards!
```

### JSON Tools

```python
# For working with JSON data
from langchain_community.tools.json.tool import JsonSpec, JsonListKeysTool, JsonGetValueTool

import json
data = {
    "users": [
        {"name": "Alice", "age": 30, "role": "engineer"},
        {"name": "Bob", "age": 25, "role": "designer"}
    ],
    "metadata": {"version": "1.0", "total": 2}
}

spec = JsonSpec(dict_=data, max_value_length=500)
list_keys = JsonListKeysTool(spec=spec)
get_value = JsonGetValueTool(spec=spec)

print(list_keys.invoke(""))          # users, metadata
print(get_value.invoke("users/0"))   # {"name": "Alice", ...}
```

---

## Part 9: Complete Project — Multi-Tool Research Assistant

Let's build a research assistant that uses multiple tools to answer complex questions:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()


class ResearchAssistant:
    """A multi-tool research assistant that uses Search, Wikipedia, and Python."""
    
    def __init__(self):
        # LLM
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
            PythonREPLTool(),
        ]
        
        # Tool map for execution
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # System prompt
        self.system_message = SystemMessage(content="""You are a research assistant with access to tools.

Guidelines:
- Use DuckDuckGo Search for current events, recent news, and real-time information.
- Use Wikipedia for established facts, historical events, biographies, and scientific concepts.
- Use Python REPL for calculations, data processing, and analysis. Always use print() to show results.
- If a question doesn't need any tool, answer directly from your knowledge.
- Always cite which tool you used and provide comprehensive answers.
- If one tool fails, try another approach.
""")
        
        # Conversation history
        self.messages = [self.system_message]
    
    def ask(self, question: str, max_tool_rounds: int = 3) -> str:
        """Ask a question. The assistant will use tools as needed."""
        
        print(f"\n{'='*60}")
        print(f"❓ Question: {question}")
        print(f"{'='*60}")
        
        # Add user message
        self.messages.append(HumanMessage(content=question))
        
        for round_num in range(max_tool_rounds):
            # Get LLM response
            response = self.llm_with_tools.invoke(self.messages)
            self.messages.append(response)
            
            # If no tool calls, we have the final answer
            if not response.tool_calls:
                print(f"\n💬 Answer:\n{response.content}")
                return response.content
            
            # Execute tool calls
            for tool_call in response.tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                tc_id = tool_call["id"]
                
                print(f"\n🔧 Round {round_num + 1}: Using {name}")
                print(f"   Args: {args}")
                
                try:
                    tool = self.tool_map[name]
                    result = tool.invoke(args)
                    result_str = str(result)
                    
                    # Truncate very long results
                    if len(result_str) > 3000:
                        result_str = result_str[:3000] + "\n... [truncated]"
                    
                    print(f"   ✅ Got {len(result_str)} chars of results")
                    
                except Exception as e:
                    result_str = f"Tool error: {str(e)}"
                    print(f"   ❌ Error: {e}")
                
                self.messages.append(ToolMessage(
                    content=result_str,
                    tool_call_id=tc_id
                ))
        
        # If we exhausted rounds, get final response
        final = self.llm_with_tools.invoke(self.messages)
        self.messages.append(final)
        print(f"\n💬 Answer:\n{final.content}")
        return final.content
    
    def reset(self):
        """Clear conversation history."""
        self.messages = [self.system_message]
        print("🔄 Conversation reset.")


# --- Usage ---
assistant = ResearchAssistant()

# Query 1: Current events (should use DuckDuckGo)
assistant.ask("What are the latest developments in large language models?")

# Query 2: Factual knowledge (should use Wikipedia)
assistant.ask("Explain the history and architecture of GPT models")

# Query 3: Calculation (should use Python REPL)
assistant.ask(
    "If I invest $50,000 at 8% annual compound interest, "
    "how much will I have after 15 years? Show me year by year."
)

# Query 4: Multi-tool (might use search + python)
assistant.ask(
    "Search for the current populations of India and China, "
    "then calculate the percentage difference between them."
)

# Query 5: No tool needed
assistant.ask("What are the SOLID principles in software engineering?")
```

### Expected Output Flow

```
============================================================
❓ Question: What are the latest developments in large language models?
============================================================

🔧 Round 1: Using duckduckgo_search
   Args: {'query': 'latest developments large language models 2025'}
   ✅ Got 1247 chars of results

💬 Answer:
Based on my research, here are the latest developments in LLMs:
1. GPT-5 has been announced with improved reasoning...
2. Claude 3.5 Opus introduced extended thinking...
...

============================================================
❓ Question: If I invest $50,000 at 8% annual compound interest...
============================================================

🔧 Round 1: Using Python_REPL
   Args: {'command': 'principal = 50000\nrate = 0.08\nfor year in range(1, 16):\n    ...'}
   ✅ Got 423 chars of results

💬 Answer:
Here's your investment growth year by year:
Year  1: $54,000.00
Year  2: $58,320.00
...
Year 15: $158,608.45

Your $50,000 would grow to $158,608.45 — a gain of $108,608.45!
```

---

## Common Mistakes

### Mistake 1: Thinking `bind_tools` executes tools automatically
```python
# ❌ Expecting the LLM to return the search results directly
llm_with_tools = llm.bind_tools([search])
response = llm_with_tools.invoke("Search for Python news")
print(response.content)  # Often empty! The LLM returned a tool_call, not content

# ✅ Check for tool_calls and execute them manually
if response.tool_calls:
    result = search.invoke(response.tool_calls[0]["args"])
    # Then send result back to LLM for final answer
```

### Mistake 2: Not sending tool results back to the LLM
```python
# ❌ Executing the tool but not letting the LLM interpret the result
result = search.invoke("Python news")
print(result)  # Raw search results — not a nice answer

# ✅ Send the result back as a ToolMessage
messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
final = llm_with_tools.invoke(messages)
print(final.content)  # Natural language answer based on search results
```

### Mistake 3: Forgetting the tool_call_id
```python
# ❌ ToolMessage without matching ID — LLM gets confused
messages.append(ToolMessage(content=result))  # Missing tool_call_id!

# ✅ Always include the tool_call_id from the original tool call
messages.append(ToolMessage(
    content=result,
    tool_call_id=tool_call["id"]  # Must match!
))
```

### Mistake 4: Using Python REPL in production without sandboxing
```python
# ❌ Exposing Python REPL to users directly
# A malicious user could ask: "Run os.system('rm -rf /')"

# ✅ For production:
# - Use sandboxed execution (Docker, E2B, Modal)
# - Implement code review / human-in-the-loop
# - Restrict allowed imports
# - Set execution timeouts
```

### Mistake 5: Too many tools confuse the LLM
```python
# ❌ Giving the LLM 15 tools — it picks the wrong one frequently
tools = [search, wiki, python, calc, shell, json, arxiv, requests, ...]

# ✅ Start with 3-5 focused tools
# Use tool routing to select relevant tools based on query type
tools = [search, wiki, python_repl]  # Covers 90% of use cases
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Start with DuckDuckGo + Wikipedia + Python REPL | Covers search, knowledge, and computation |
| Use `bind_tools()` for LLM-driven tool selection | LLM picks the best tool per query |
| Always handle tool errors gracefully | Return error message, don't crash |
| Truncate long tool results | Saves tokens, prevents context overflow |
| Add a system prompt explaining when to use each tool | Improves tool selection accuracy |
| Implement max iterations for multi-tool loops | Prevents infinite loops |
| Log all tool calls and results | Debugging and audit trail |
| Use Tavily over DuckDuckGo in production | Better quality, reliable rate limits |

---

## Interview Preparation

### Easy
**Q: What built-in tools does LangChain provide?**

> LangChain provides a wide ecosystem of built-in tools including: **DuckDuckGo Search** for real-time web search (no API key needed), **Wikipedia** for factual knowledge retrieval, **Python REPL** for code execution and calculations, **Tavily Search** for AI-optimized search results, **Arxiv** for research papers, **Shell** for system commands, and **JSON tools** for structured data querying. These tools can be bound to any LLM using `bind_tools()`, allowing the LLM to decide which tool to call based on the user's query.

### Medium
**Q: Walk through the complete tool execution loop when using `bind_tools`.**

> The loop has five steps: (1) **Bind** tools to the LLM using `llm.bind_tools(tools)` — this gives the LLM tool schemas. (2) **Invoke** the LLM with a user question — the LLM returns an `AIMessage` that either has `content` (direct answer) or `tool_calls` (structured tool call requests). (3) **Check** for `tool_calls` — if empty, the LLM answered directly. (4) **Execute** each tool call by looking up the tool by name and calling `tool.invoke(args)`. (5) **Send back** the results as `ToolMessage` objects (with matching `tool_call_id`) and invoke the LLM again for a final natural language answer. This loop may repeat if the LLM needs multiple tools.

### Hard
**Q: How would you make the Python REPL tool production-safe?**

> Multiple layers of protection: (1) **Sandboxed execution** — run code in isolated Docker containers or cloud sandboxes like E2B or Modal, with no network access and limited filesystem. (2) **Import restrictions** — whitelist safe modules (math, statistics, datetime) and block dangerous ones (os, subprocess, shutil, socket). (3) **Execution timeouts** — kill any code running longer than N seconds to prevent infinite loops. (4) **Resource limits** — cap memory usage and CPU time. (5) **Code review** — implement human-in-the-loop where generated code is shown to a human before execution. (6) **Output sanitization** — limit output length, strip sensitive data. (7) **Audit logging** — log all generated and executed code for review. In most production cases, it's better to restrict to specific computation tools (calculator, data analysis) rather than a general-purpose REPL.

### Senior
**Q: You're building a production AI assistant with 20+ tools. Users report the LLM frequently picks the wrong tool. How do you fix this?**

> Systematic approach: (1) **Tool routing** — don't give all 20 tools at once. First classify the user's intent (using a fast LLM or keyword matching), then present only 3-5 relevant tools. This dramatically improves selection accuracy. (2) **Improve descriptions** — make each tool's description clearly state WHEN to use it and WHEN NOT to use it. Include examples of matching queries. (3) **Remove ambiguity** — if two tools overlap (e.g., "web_search" and "news_search"), merge them or add explicit disambiguation in descriptions. (4) **Few-shot examples** — include 2-3 examples in the system prompt showing correct tool selection for similar queries. (5) **Evaluation suite** — build a test set of 100+ queries with expected tool selections, measure accuracy, and iterate on descriptions. (6) **Fallback strategy** — if a tool returns empty/error, automatically retry with an alternative tool. (7) **Use a stronger model** — tool selection accuracy varies significantly by model; GPT-4o is much better than GPT-3.5 at multi-tool selection.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Built-in tools** | Pre-built, ready-to-use tools in LangChain's ecosystem |
| **DuckDuckGo Search** | Free web search, no API key needed, great for current events |
| **Wikipedia** | Factual knowledge retrieval from Wikipedia articles |
| **Python REPL** | Execute Python code — powerful but dangerous without sandboxing |
| **LLM Math** | LLM translates natural language → math expression → calculates |
| **`bind_tools()`** | Gives the LLM tool schemas so it can decide which to call |
| **Tool execution loop** | LLM decides → your code executes → result back to LLM → final answer |
| **`ToolMessage`** | How you send tool results back to the LLM (must include `tool_call_id`) |
| **Tool routing** | Present only relevant tools to the LLM based on query classification |

---

> [← Previous: What Are Tools?](chapter-30-what-are-tools.md) | [Next: Custom Tools →](chapter-32-custom-tools.md)

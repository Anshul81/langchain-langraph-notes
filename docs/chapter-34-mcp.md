# Chapter 9.5: MCP — Model Context Protocol

> **Phase 9 — Tools & Tool Calling** | [← Previous: Tool Calling Deep Dive](chapter-33-tool-calling-deep-dive.md) | [Next: Phase 10 — Agents →](../phase-09-agents/chapter-35-intro-to-agents.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand what MCP is and why it was created
- ✅ Know the MCP architecture (hosts, clients, servers)
- ✅ Understand MCP's three primitives: tools, resources, and prompts
- ✅ Use MCP servers with LangChain via `langchain-mcp-adapters`
- ✅ Build a custom MCP server that exposes tools
- ✅ Connect to existing MCP servers (filesystem, GitHub, databases)
- ✅ Understand the relationship between MCP and traditional tool calling

| | |
|---|---|
| **Prerequisites** | Chapter 9.3 (Custom Tools), Chapter 9.4 (Tool Calling Deep Dive) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction — The Problem MCP Solves

Before MCP, every AI application had to build **custom integrations** for every tool, every API, every data source:

```
                     WITHOUT MCP
                     ──────────
App 1 ──custom code──→ GitHub API
App 1 ──custom code──→ Database
App 1 ──custom code──→ File System
App 1 ──custom code──→ Slack API

App 2 ──custom code──→ GitHub API     ← Same work, done again!
App 2 ──custom code──→ Database       ← Same work, done again!
App 2 ──custom code──→ File System    ← Same work, done again!

N apps × M integrations = N×M custom integrations 😱
```

**MCP fixes this** by creating a **universal standard** — like USB-C for AI:

```
                      WITH MCP
                      ────────
App 1 ──MCP──→ ┌─────────────────┐
App 2 ──MCP──→ │  MCP Server:    │──→ GitHub API
App 3 ──MCP──→ │  GitHub         │
               └─────────────────┘

App 1 ──MCP──→ ┌─────────────────┐
App 2 ──MCP──→ │  MCP Server:    │──→ PostgreSQL
App 3 ──MCP──→ │  Database       │
               └─────────────────┘

N apps × 1 protocol = N+M integrations ✅
```

**MCP (Model Context Protocol)** is an open standard created by **Anthropic** (November 2024) that defines how AI applications connect to external tools and data sources.

---

## Part 1: MCP Architecture

### The Three Roles

```
┌──────────────────────────────────────────────────────────┐
│                    MCP Architecture                       │
│                                                           │
│  ┌─────────────┐     ┌─────────────┐     ┌────────────┐  │
│  │    HOST      │     │   CLIENT    │     │   SERVER   │  │
│  │             │     │             │     │            │  │
│  │ Claude      │     │ Maintains   │     │ Exposes:   │  │
│  │ Desktop,    │────→│ 1:1 session │────→│ • Tools    │  │
│  │ Your App,   │     │ with server │     │ • Resources│  │
│  │ IDE Plugin  │     │             │     │ • Prompts  │  │
│  └─────────────┘     └─────────────┘     └────────────┘  │
│                                                           │
│  "The AI app"    "The connection"     "The integration"   │
└──────────────────────────────────────────────────────────┘
```

| Role | What It Does | Examples |
|------|-------------|----------|
| **Host** | The AI application the user interacts with | Claude Desktop, Cursor, your LangChain app |
| **Client** | Manages the connection to an MCP server | Built into the host (1 client per server) |
| **Server** | Exposes capabilities (tools, resources, prompts) | GitHub MCP server, Filesystem server, your custom server |

### The Three Primitives

MCP servers can expose three types of capabilities:

| Primitive | What It Is | Direction | Example |
|-----------|-----------|-----------|---------|
| **Tools** | Functions the LLM can call | LLM → Server → Action | `create_issue()`, `query_db()` |
| **Resources** | Data the LLM can read | Server → LLM | Files, database schemas, API docs |
| **Prompts** | Pre-built prompt templates | Server → LLM | "Summarize this PR", "Review this code" |

```python
# Tools   = "What the AI can DO"     (actions, like LangChain tools)
# Resources = "What the AI can READ" (data, like context/RAG)  
# Prompts = "What the AI can SAY"    (templates, like prompt templates)
```

### Transport Protocols

MCP supports two transport mechanisms:

| Transport | How It Works | Best For |
|-----------|-------------|----------|
| **stdio** | Server runs as a subprocess, communicates via stdin/stdout | Local tools (filesystem, CLI) |
| **SSE** | Server runs over HTTP with Server-Sent Events | Remote/cloud services |

---

## Part 2: MCP vs Traditional Tool Calling

### What Changes?

```
TRADITIONAL (LangChain @tool):
─────────────────────────────
1. You write the tool function
2. You bind it to your LLM with bind_tools()
3. Your app executes the tool
4. The tool code lives IN your app

MCP:
────
1. Someone writes an MCP server (tool code lives OUTSIDE your app)
2. Your app connects to the MCP server
3. MCP server exposes tools via the standard protocol
4. Your app discovers and uses those tools
5. The tool code lives in the MCP server (separate process)
```

### Key Differences

| Aspect | Traditional @tool | MCP Server |
|--------|-------------------|------------|
| Where tool code lives | Inside your app | Separate process/service |
| Discovery | You manually bind tools | Auto-discovered from server |
| Reusability | Copy-paste between apps | Connect any app to same server |
| Language | Same as your app (Python) | Any language (Python, TypeScript, etc.) |
| Isolation | Runs in your process | Runs in separate process (safer) |
| Standard | LangChain-specific | Universal open standard |
| Ecosystem | LangChain tools | Growing ecosystem (GitHub, Slack, etc.) |

### When to Use Which

```
Use @tool when:
├── Building a quick prototype
├── Tool logic is tightly coupled to your app
├── You don't need cross-app reuse
└── Simple tools with no external dependencies

Use MCP when:
├── Building production integrations
├── Multiple apps need the same tools
├── You want process isolation (security)
├── Using existing MCP servers from the ecosystem
└── Building tools that others will consume
```

---

## Part 3: Using MCP Servers with LangChain

### Setup

```bash
pip install langchain-mcp-adapters
pip install mcp
```

### Connecting to an MCP Server (stdio transport)

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import os
from dotenv import load_dotenv
load_dotenv()


async def main():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
        openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
    )
    
    # Connect to MCP servers
    async with MultiServerMCPClient(
        {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y", "@modelcontextprotocol/server-filesystem",
                    "./data"  # Directory to expose
                ],
                "transport": "stdio",
            }
        }
    ) as client:
        # Get tools from the MCP server
        tools = client.get_tools()
        
        print(f"Discovered {len(tools)} tools from MCP server:")
        for tool in tools:
            print(f"  🔧 {tool.name}: {tool.description[:60]}...")
        
        # Create an agent with MCP tools
        agent = create_react_agent(llm, tools)
        
        # Use the agent
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "List all files in the data directory"}]
        })
        
        print(f"\n💬 {result['messages'][-1].content}")


asyncio.run(main())
```

### What Happens Under the Hood

```
Your App                  MCP Client              MCP Server (subprocess)
   │                          │                          │
   │──"List files"──────────→│                          │
   │                          │──initialize──────────→  │
   │                          │←─capabilities─────────  │
   │                          │  (tools list)           │
   │                          │                          │
   │←─tools: [list_dir,      │                          │
   │   read_file, ...]       │                          │
   │                          │                          │
   │──invoke list_dir()──→   │                          │
   │                          │──call tool───────────→  │
   │                          │←─result──────────────   │
   │←─["file1.txt", ...]     │                          │
```

### Multiple MCP Servers

```python
async with MultiServerMCPClient(
    {
        # Server 1: Filesystem access
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"],
            "transport": "stdio",
        },
        # Server 2: GitHub integration
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "transport": "stdio",
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")
            }
        },
        # Server 3: Remote server via SSE
        "my_api": {
            "url": "http://localhost:8080/sse",
            "transport": "sse",
        }
    }
) as client:
    # All tools from ALL servers are available
    tools = client.get_tools()
    print(f"Total tools from all servers: {len(tools)}")
    
    # Create agent with all tools
    agent = create_react_agent(llm, tools)
```

---

## Part 4: Building a Custom MCP Server

### A Simple MCP Server (Python)

```python
# file: my_mcp_server.py
"""A custom MCP server that exposes weather and calculation tools."""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json
import math


# Create the MCP server
server = Server("my-tools-server")


# --- Define available tools ---

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of tools this server provides."""
    return [
        Tool(
            name="get_weather",
            description="Get the current weather for a city. "
                       "Use when the user asks about weather or temperature.",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name, e.g., 'Mumbai', 'London'"
                    }
                },
                "required": ["city"]
            }
        ),
        Tool(
            name="calculate",
            description="Evaluate a mathematical expression. "
                       "Use when the user needs math calculations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g., '2 ** 10 + math.sqrt(144)'"
                    }
                },
                "required": ["expression"]
            }
        ),
        Tool(
            name="get_time",
            description="Get the current date and time. "
                       "Use when the user asks what time or date it is.",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
    ]


# --- Implement tool execution ---

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool and return its result."""
    
    if name == "get_weather":
        city = arguments.get("city", "Unknown")
        # Simulated weather data
        weather_data = {
            "Mumbai": {"temp": 32, "condition": "Partly Cloudy", "humidity": 78},
            "Delhi": {"temp": 38, "condition": "Sunny", "humidity": 45},
            "London": {"temp": 18, "condition": "Overcast", "humidity": 72},
            "New York": {"temp": 25, "condition": "Clear", "humidity": 55},
        }
        
        data = weather_data.get(city, {"temp": "N/A", "condition": "Unknown", "humidity": "N/A"})
        result = (
            f"Weather in {city}:\n"
            f"  Temperature: {data['temp']}°C\n"
            f"  Condition: {data['condition']}\n"
            f"  Humidity: {data['humidity']}%"
        )
        return [TextContent(type="text", text=result)]
    
    elif name == "calculate":
        expression = arguments.get("expression", "")
        try:
            result = eval(expression, {"__builtins__": {}, "math": math})
            return [TextContent(type="text", text=f"{expression} = {result}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    elif name == "get_time":
        from datetime import datetime
        now = datetime.now()
        return [TextContent(
            type="text", 
            text=f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
        )]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


# --- Run the server ---

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Using Your Custom MCP Server with LangChain

```python
# file: use_my_server.py
import asyncio
import os
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()


async def main():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
        openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
    )
    
    # Connect to YOUR custom MCP server
    async with MultiServerMCPClient(
        {
            "my_tools": {
                "command": "python",
                "args": ["my_mcp_server.py"],
                "transport": "stdio",
            }
        }
    ) as client:
        tools = client.get_tools()
        print(f"📦 Discovered {len(tools)} tools:")
        for t in tools:
            print(f"   🔧 {t.name}: {t.description[:50]}...")
        
        # Create agent
        agent = create_react_agent(llm, tools)
        
        # Test questions
        questions = [
            "What's the weather in Mumbai?",
            "Calculate 2 to the power of 16",
            "What time is it right now?",
            "What's the weather in London and what's 100 * 3.14159?",
        ]
        
        for q in questions:
            print(f"\n❓ {q}")
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": q}]
            })
            print(f"💬 {result['messages'][-1].content}")


asyncio.run(main())
```

---

## Part 5: MCP Server with SSE Transport (Remote)

For remote/cloud deployment, use SSE (Server-Sent Events):

```python
# file: mcp_sse_server.py
"""MCP server with SSE transport — can be hosted remotely."""

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
import uvicorn
import json


server = Server("remote-tools-server")

# Define tools (same as before)
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="lookup_product",
            description="Look up a product by name or ID in our catalog.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Product name or ID"}
                },
                "required": ["query"]
            }
        ),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "lookup_product":
        products = {
            "laptop": {"name": "ProBook Laptop", "price": 59999, "stock": 12},
            "headphones": {"name": "SoundMax Pro", "price": 2499, "stock": 45},
        }
        query = arguments.get("query", "").lower()
        for key, product in products.items():
            if query in key or query in product["name"].lower():
                return [TextContent(
                    type="text",
                    text=json.dumps(product, indent=2)
                )]
        return [TextContent(type="text", text=f"Product '{query}' not found")]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


# Create SSE transport
sse = SseServerTransport("/messages/")

async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(streams[0], streams[1])

# Create Starlette app
app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Connecting to the SSE Server

```python
# From your LangChain app:
async with MultiServerMCPClient(
    {
        "remote_catalog": {
            "url": "http://localhost:8080/sse",
            "transport": "sse",
        }
    }
) as client:
    tools = client.get_tools()
    agent = create_react_agent(llm, tools)
    # Use agent...
```

---

## Part 6: Popular MCP Servers

The MCP ecosystem is growing rapidly. Here are the most popular servers:

### Official Servers (by Anthropic/Community)

| Server | What It Does | Install |
|--------|-------------|---------|
| **Filesystem** | Read/write files, list directories | `npx @modelcontextprotocol/server-filesystem` |
| **GitHub** | Create issues, read repos, manage PRs | `npx @modelcontextprotocol/server-github` |
| **PostgreSQL** | Query and inspect databases | `npx @modelcontextprotocol/server-postgres` |
| **SQLite** | Local database operations | `npx @modelcontextprotocol/server-sqlite` |
| **Brave Search** | Web search via Brave | `npx @modelcontextprotocol/server-brave-search` |
| **Google Maps** | Geocoding, directions, places | `npx @modelcontextprotocol/server-google-maps` |
| **Slack** | Send messages, read channels | `npx @modelcontextprotocol/server-slack` |
| **Memory** | Persistent knowledge graph | `npx @modelcontextprotocol/server-memory` |

### Using the Filesystem Server

```python
async with MultiServerMCPClient(
    {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "./my_project"  # Directory to expose
            ],
            "transport": "stdio",
        }
    }
) as client:
    tools = client.get_tools()
    # Available tools: read_file, write_file, list_directory, 
    #                  create_directory, move_file, search_files, etc.
    
    agent = create_react_agent(llm, tools)
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Read the README.md file"}]
    })
```

### Using the GitHub Server

```python
async with MultiServerMCPClient(
    {
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "transport": "stdio",
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")
            }
        }
    }
) as client:
    tools = client.get_tools()
    # Available tools: create_issue, list_issues, create_pull_request,
    #                  search_repositories, get_file_contents, etc.
    
    agent = create_react_agent(llm, tools)
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": "List open issues in the langchain-ai/langchain repository"
        }]
    })
```

---

## Part 7: MCP Resources and Prompts

### Resources — Exposing Data

Resources let your server expose data that the LLM can read:

```python
from mcp.server import Server
from mcp.types import Resource, TextContent

server = Server("data-server")

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="config://app-settings",
            name="Application Settings",
            description="Current application configuration",
            mimeType="application/json"
        ),
        Resource(
            uri="docs://api-reference",
            name="API Reference",
            description="API documentation for our service",
            mimeType="text/markdown"
        ),
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "config://app-settings":
        import json
        config = {
            "app_name": "MyApp",
            "version": "2.1.0",
            "features": ["search", "analytics", "notifications"],
            "max_users": 1000,
        }
        return json.dumps(config, indent=2)
    
    elif uri == "docs://api-reference":
        return """# API Reference
        
## GET /api/users
Returns a list of all users.

## POST /api/users
Create a new user. Body: {"name": "string", "email": "string"}

## GET /api/users/:id
Get a specific user by ID.
"""
    
    return f"Resource not found: {uri}"
```

### Prompts — Reusable Templates

```python
from mcp.types import Prompt, PromptArgument, PromptMessage, TextContent

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="code_review",
            description="Review code for best practices, bugs, and improvements",
            arguments=[
                PromptArgument(
                    name="code",
                    description="The code to review",
                    required=True
                ),
                PromptArgument(
                    name="language",
                    description="Programming language",
                    required=False
                )
            ]
        ),
    ]

@server.get_prompt()
async def get_prompt(name: str, arguments: dict) -> list[PromptMessage]:
    if name == "code_review":
        code = arguments.get("code", "")
        language = arguments.get("language", "unknown")
        
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Please review this {language} code:

```{language}
{code}
```

Check for:
1. Bugs and logical errors
2. Security vulnerabilities
3. Performance issues
4. Code style and best practices
5. Suggestions for improvement
"""
                )
            )
        ]
    
    return []
```

---

## Part 8: MCP in Production

### Security Considerations

```
⚠️ MCP Security Checklist:
├── 🔐 Validate all tool inputs on the server side
├── 🔐 Use least-privilege access (filesystem: read-only if possible)
├── 🔐 Don't expose sensitive environment variables to MCP servers
├── 🔐 Use SSE transport + authentication for remote servers
├── 🔐 Log all tool calls for audit trail
├── 🔐 Set timeouts on tool execution
├── 🔐 Sandbox filesystem access to specific directories
└── 🔐 Review third-party MCP servers before using them
```

### Error Handling Pattern

```python
from mcp.types import TextContent

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "query_database":
            # Validate arguments
            query = arguments.get("query", "")
            if not query:
                return [TextContent(
                    type="text", 
                    text="Error: 'query' argument is required"
                )]
            
            # Prevent SQL injection
            if any(keyword in query.upper() for keyword in ["DROP", "DELETE", "TRUNCATE"]):
                return [TextContent(
                    type="text",
                    text="Error: Destructive queries are not allowed"
                )]
            
            # Execute with timeout
            import asyncio
            result = await asyncio.wait_for(
                execute_query(query),
                timeout=30.0
            )
            return [TextContent(type="text", text=str(result))]
        
    except asyncio.TimeoutError:
        return [TextContent(type="text", text="Error: Query timed out (30s limit)")]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]
```

### Monitoring MCP Servers

```python
import logging
from datetime import datetime

# Set up logging for your MCP server
logging.basicConfig(
    filename="mcp_server.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("mcp-server")

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    start_time = datetime.now()
    logger.info(f"Tool call: {name} | Args: {arguments}")
    
    try:
        result = await execute_tool(name, arguments)
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Tool success: {name} | Duration: {duration:.2f}s")
        return result
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Tool error: {name} | Duration: {duration:.2f}s | Error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]
```

---

## Common Mistakes

### Mistake 1: Confusing MCP tools with LangChain @tool
```python
# ❌ Trying to use @tool syntax in an MCP server
from langchain_core.tools import tool

@tool
def my_mcp_tool(x: str) -> str:  # This is a LangChain tool, NOT an MCP tool!
    """..."""
    ...

# ✅ MCP uses its own Tool type and server decorators
from mcp.types import Tool

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [Tool(name="my_tool", description="...", inputSchema={...})]
```

### Mistake 2: Not using async with MCP
```python
# ❌ MCP is async — can't use sync code
client = MultiServerMCPClient({...})  # Can't use without async context!

# ✅ Always use async context manager
async with MultiServerMCPClient({...}) as client:
    tools = client.get_tools()
```

### Mistake 3: Forgetting environment variables for MCP servers
```python
# ❌ GitHub server fails silently without token
"github": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "transport": "stdio",
    # Missing env!
}

# ✅ Pass required environment variables
"github": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "transport": "stdio",
    "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN")
    }
}
```

### Mistake 4: Exposing unrestricted filesystem access
```python
# ❌ DANGEROUS — exposes entire filesystem
"filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/"],
    "transport": "stdio",
}

# ✅ Restrict to specific directories
"filesystem": {
    "command": "npx",
    "args": [
        "-y", "@modelcontextprotocol/server-filesystem",
        "./data",          # Only the data directory
        "./config"         # And config directory
    ],
    "transport": "stdio",
}
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use MCP for reusable integrations, `@tool` for app-specific logic | Right tool for the right job |
| Start with official MCP servers | Well-tested, maintained, secure |
| Always use async context managers (`async with`) | Proper cleanup of connections |
| Restrict filesystem access to specific directories | Security — prevent unauthorized access |
| Pass environment variables via `env` parameter | Don't leak secrets |
| Set timeouts on tool execution | Prevent hanging MCP servers |
| Log all tool calls in production | Debugging, audit trail, monitoring |
| Review third-party MCP server code before use | Security — untrusted code risk |
| Use SSE transport for remote servers, stdio for local | Performance and reliability |
| Handle errors gracefully in MCP server tools | Return error messages, don't crash |

---

## Interview Preparation

### Easy
**Q: What is MCP (Model Context Protocol)?**

> MCP is an open standard created by Anthropic that defines how AI applications connect to external tools and data sources. It's like "USB-C for AI" — instead of building custom integrations for every tool in every app, MCP provides a universal protocol. An MCP server exposes tools, resources, and prompts through a standard interface. Any MCP-compatible host (Claude Desktop, LangChain apps, IDE plugins) can connect to any MCP server. This means integrations are built once and used everywhere.

### Medium
**Q: What are MCP's three primitives (tools, resources, prompts) and when would you use each?**

> **Tools** are functions the LLM can call to take actions — like `create_issue()` or `query_database()`. The LLM decides when to call them based on user queries. Use tools when the AI needs to DO something. **Resources** are data the LLM can read — like files, database schemas, or configuration. They're exposed via URIs and provide context. Use resources when the AI needs to READ structured data. **Prompts** are pre-built prompt templates that encode best practices — like "Review this code" or "Summarize this PR". Use prompts to standardize common interactions with the LLM.

### Hard
**Q: How does MCP differ from traditional LangChain tool calling, and when would you choose one over the other?**

> Key differences: (1) **Location** — `@tool` code lives inside your app process; MCP tool code runs in a separate server process. (2) **Reusability** — `@tool` functions are app-specific; MCP servers can be shared across any MCP-compatible app. (3) **Isolation** — MCP provides process-level isolation (security benefit). (4) **Discovery** — MCP tools are auto-discovered from the server; `@tool` requires manual binding. (5) **Ecosystem** — MCP has a growing ecosystem of pre-built servers. Choose `@tool` for simple, app-specific logic during development. Choose MCP for production integrations that multiple apps need, when using existing ecosystem servers, or when you need process isolation for security.

### Senior
**Q: How would you architect a production AI system using MCP for a company with 5 different AI applications?**

> Architecture: (1) **Shared MCP servers** — Build company-wide MCP servers for common integrations (internal database, Jira, Slack, document store). Deploy as services with SSE transport behind authentication. (2) **App-specific tools** — Each AI app has its own `@tool` functions for app-specific logic that doesn't need sharing. (3) **Gateway pattern** — Use an MCP gateway/proxy that handles authentication, rate limiting, logging, and routing to backend MCP servers. (4) **Security layers** — Each MCP server has least-privilege access; filesystem servers restricted to specific directories; database servers use read-only connections by default. (5) **Monitoring** — Centralized logging of all MCP tool calls across all apps for debugging and compliance. (6) **Versioning** — MCP servers versioned independently; clients negotiate capabilities during initialization. (7) **Fallback** — If an MCP server is down, the AI app degrades gracefully with an error message rather than crashing.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **MCP** | Model Context Protocol — universal standard for AI tool integrations |
| **Host** | The AI application (Claude Desktop, your LangChain app) |
| **Client** | Manages the connection to an MCP server |
| **Server** | Exposes tools, resources, and prompts |
| **Tools** | Functions the LLM can call (actions) |
| **Resources** | Data the LLM can read (context) |
| **Prompts** | Pre-built prompt templates |
| **stdio transport** | Server runs as subprocess (local tools) |
| **SSE transport** | Server runs over HTTP (remote/cloud) |
| **`langchain-mcp-adapters`** | LangChain package to connect to MCP servers |
| **Ecosystem** | Growing library of pre-built MCP servers (GitHub, DB, filesystem) |

---

> [← Previous: Tool Calling Deep Dive](chapter-33-tool-calling-deep-dive.md) | [Next: Phase 10 — Agents →](../phase-09-agents/chapter-35-intro-to-agents.md)

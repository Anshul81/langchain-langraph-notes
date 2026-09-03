# Chapter 10.4: Custom Agent Architectures with LangGraph

> **Phase 10 — Agents** | [← Previous: LangGraph Fundamentals](chapter-37-langgraph-fundamentals.md) | [Next: Human-in-the-Loop →](chapter-39-human-in-the-loop.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Build custom agent architectures beyond the standard ReAct pattern
- ✅ Implement a **Plan-and-Execute** agent
- ✅ Build a **multi-agent** system with supervisor routing
- ✅ Create agents with **subgraphs** (nested workflows)
- ✅ Implement **self-reflective** agents that critique and improve their output
- ✅ Build a **research team** with specialized sub-agents

| | |
|---|---|
| **Prerequisites** | Chapter 10.3 (LangGraph Fundamentals) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction

`create_react_agent` gives you one pattern: LLM → Tools → Loop. But real-world problems need custom architectures:

```
Standard ReAct:     LLM ↔ Tools (loop)

Custom architectures you can build with LangGraph:
├── Plan-and-Execute:    Planner → Executor → Replanner → ...
├── Multi-Agent:         Supervisor → Worker1/Worker2/Worker3
├── Self-Reflective:     Generate → Critique → Revise → ...
├── Map-Reduce:          Split → Process (parallel) → Merge
└── Hierarchical:        Manager → Sub-Managers → Workers
```

---

## Part 1: Plan-and-Execute Agent

The LLM first creates a plan, then executes each step:

```python
import os
from typing import TypedDict, Annotated, Literal
from operator import add
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)


# --- State ---
class PlanExecuteState(TypedDict):
    question: str
    plan: list[str]
    current_step: int
    step_results: Annotated[list, add]
    final_answer: str


# --- Tools ---
@tool
def search(query: str) -> str:
    """Search for information."""
    data = {
        "india population": "India's population is approximately 1.44 billion (2024).",
        "china population": "China's population is approximately 1.43 billion (2024).",
        "usa population": "The USA population is approximately 334 million (2024).",
        "india gdp": "India's GDP is approximately $3.7 trillion (2024).",
        "usa gdp": "USA's GDP is approximately $25.5 trillion (2024).",
    }
    for key, value in data.items():
        if key in query.lower():
            return value
    return f"No data found for: {query}"

tool_map = {"search": search}


# --- Nodes ---

def planner(state: PlanExecuteState) -> dict:
    """Create a step-by-step plan to answer the question."""
    response = llm.invoke([
        SystemMessage(content="""You are a planner. Given a question, create a step-by-step plan.
Each step should be a single, actionable instruction.
Return ONLY the steps, one per line, numbered.
Keep it to 2-5 steps maximum."""),
        HumanMessage(content=f"Question: {state['question']}")
    ])
    
    steps = [
        line.strip().lstrip("0123456789.").strip()
        for line in response.content.split("\n")
        if line.strip() and any(c.isalpha() for c in line)
    ]
    
    print(f"📋 Plan created with {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"   {i}. {step}")
    
    return {"plan": steps, "current_step": 0}


def executor(state: PlanExecuteState) -> dict:
    """Execute the current step of the plan."""
    step_idx = state["current_step"]
    step = state["plan"][step_idx]
    
    print(f"\n🔧 Executing step {step_idx + 1}: {step}")
    
    # Use LLM to decide how to execute this step
    response = llm.invoke([
        SystemMessage(content=f"""Execute this step of a research plan.
Previous results: {state.get('step_results', [])}

If you need to search for information, describe what you found.
If you need to calculate something, do the calculation.
Be concise and factual."""),
        HumanMessage(content=f"Step to execute: {step}")
    ])
    
    result = f"Step {step_idx + 1}: {response.content}"
    print(f"   ✅ {response.content[:100]}...")
    
    return {
        "step_results": [result],
        "current_step": step_idx + 1
    }


def synthesizer(state: PlanExecuteState) -> dict:
    """Synthesize all step results into a final answer."""
    print(f"\n📝 Synthesizing final answer...")
    
    results_text = "\n".join(state["step_results"])
    
    response = llm.invoke([
        SystemMessage(content="Synthesize the research results into a clear, comprehensive answer."),
        HumanMessage(content=f"Question: {state['question']}\n\nResearch Results:\n{results_text}")
    ])
    
    return {"final_answer": response.content}


# --- Routing ---

def should_continue_executing(state: PlanExecuteState) -> Literal["executor", "synthesizer"]:
    """Check if there are more steps to execute."""
    if state["current_step"] < len(state["plan"]):
        return "executor"
    return "synthesizer"


# --- Build Graph ---

graph = StateGraph(PlanExecuteState)

graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("synthesizer", synthesizer)

graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", should_continue_executing)
graph.add_edge("synthesizer", END)

plan_execute_agent = graph.compile()

# --- Test ---
result = plan_execute_agent.invoke({
    "question": "Compare the populations and GDPs of India and the USA. Which country has a higher GDP per capita?",
    "plan": [],
    "current_step": 0,
    "step_results": [],
    "final_answer": ""
})

print(f"\n{'='*60}")
print(f"💬 Final Answer:\n{result['final_answer']}")
```

---

## Part 2: Multi-Agent Supervisor Pattern

A supervisor agent routes tasks to specialized worker agents:

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, START, END


class SupervisorState(TypedDict):
    query: str
    assigned_to: str
    worker_results: Annotated[list, add]
    final_answer: str


# --- Supervisor ---

def supervisor(state: SupervisorState) -> dict:
    """Route the query to the right specialist."""
    response = llm.invoke([
        SystemMessage(content="""You are a supervisor. Route the user's query to the right specialist.
Available specialists:
- 'researcher': For factual questions, history, science
- 'analyst': For data analysis, comparisons, calculations
- 'writer': For content creation, summaries, explanations

Respond with ONLY the specialist name."""),
        HumanMessage(content=state["query"])
    ])
    
    assigned = response.content.strip().lower()
    valid = ["researcher", "analyst", "writer"]
    assigned = assigned if assigned in valid else "researcher"
    
    print(f"👔 Supervisor assigned to: {assigned}")
    return {"assigned_to": assigned}


# --- Workers ---

def researcher(state: SupervisorState) -> dict:
    """Research specialist — handles factual queries."""
    response = llm.invoke([
        SystemMessage(content="You are a research specialist. Provide accurate, well-sourced factual answers. Be thorough."),
        HumanMessage(content=state["query"])
    ])
    print(f"🔬 Researcher completed")
    return {"worker_results": [f"[Researcher] {response.content}"]}


def analyst(state: SupervisorState) -> dict:
    """Analysis specialist — handles data and comparisons."""
    response = llm.invoke([
        SystemMessage(content="You are a data analyst. Provide detailed analysis with numbers, comparisons, and insights. Use structured formats."),
        HumanMessage(content=state["query"])
    ])
    print(f"📊 Analyst completed")
    return {"worker_results": [f"[Analyst] {response.content}"]}


def writer(state: SupervisorState) -> dict:
    """Writing specialist — handles content creation."""
    response = llm.invoke([
        SystemMessage(content="You are a professional writer. Create clear, engaging, well-structured content."),
        HumanMessage(content=state["query"])
    ])
    print(f"✍️ Writer completed")
    return {"worker_results": [f"[Writer] {response.content}"]}


def compile_answer(state: SupervisorState) -> dict:
    """Compile worker results into a final answer."""
    response = llm.invoke([
        SystemMessage(content="Compile the specialist's work into a polished final answer."),
        HumanMessage(content=f"Query: {state['query']}\n\nSpecialist Output:\n{state['worker_results'][-1]}")
    ])
    return {"final_answer": response.content}


# --- Routing ---

def route_to_worker(state: SupervisorState) -> Literal["researcher", "analyst", "writer"]:
    return state["assigned_to"]


# --- Build Graph ---

graph = StateGraph(SupervisorState)

graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("analyst", analyst)
graph.add_node("writer", writer)
graph.add_node("compile", compile_answer)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_to_worker)
graph.add_edge("researcher", "compile")
graph.add_edge("analyst", "compile")
graph.add_edge("writer", "compile")
graph.add_edge("compile", END)

supervisor_agent = graph.compile()

# Test
queries = [
    "What is quantum computing and how does it work?",     # → researcher
    "Compare Python vs JavaScript job markets in 2024",     # → analyst
    "Write a product description for a smart water bottle",  # → writer
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"❓ {q}")
    result = supervisor_agent.invoke({
        "query": q, "assigned_to": "", "worker_results": [], "final_answer": ""
    })
    print(f"\n💬 {result['final_answer'][:200]}...")
```

---

## Part 3: Self-Reflective Agent

An agent that generates, critiques, and improves its output:

```python
from typing import TypedDict, Annotated, Literal
from operator import add
from langgraph.graph import StateGraph, START, END


class ReflectionState(TypedDict):
    task: str
    draft: str
    critique: str
    revision_count: int
    max_revisions: int
    is_approved: bool
    revision_log: Annotated[list, add]


def generate(state: ReflectionState) -> dict:
    """Generate an initial draft or revision."""
    if state.get("draft"):
        # Revision based on critique
        response = llm.invoke([
            SystemMessage(content="You are an expert writer. Revise the draft based on the critique provided."),
            HumanMessage(content=f"""
Task: {state['task']}

Current Draft:
{state['draft']}

Critique:
{state['critique']}

Please revise the draft to address the critique while maintaining quality.""")
        ])
    else:
        # Initial generation
        response = llm.invoke([
            SystemMessage(content="You are an expert writer. Create high-quality content."),
            HumanMessage(content=f"Task: {state['task']}")
        ])
    
    revision = state.get("revision_count", 0) + (1 if state.get("draft") else 0)
    print(f"✍️ {'Revision' if state.get('draft') else 'Initial draft'} #{revision}")
    
    return {
        "draft": response.content,
        "revision_count": revision,
        "revision_log": [f"Draft v{revision}: {len(response.content)} chars"]
    }


def critique(state: ReflectionState) -> dict:
    """Critique the current draft."""
    response = llm.invoke([
        SystemMessage(content="""You are a critical reviewer. Evaluate the draft on:
1. Clarity and coherence
2. Completeness
3. Accuracy
4. Engagement

If the draft is excellent and needs no changes, respond with exactly: "APPROVED"
Otherwise, provide specific, actionable feedback for improvement."""),
        HumanMessage(content=f"Task: {state['task']}\n\nDraft:\n{state['draft']}")
    ])
    
    is_approved = "APPROVED" in response.content.upper()
    print(f"🔍 Critique: {'APPROVED ✅' if is_approved else 'Needs revision ⚠️'}")
    
    return {
        "critique": response.content,
        "is_approved": is_approved,
        "revision_log": [f"Critique: {'Approved' if is_approved else 'Revision needed'}"]
    }


def should_revise(state: ReflectionState) -> Literal["generate", "__end__"]:
    """Decide if another revision is needed."""
    if state.get("is_approved", False):
        return END
    if state.get("revision_count", 0) >= state.get("max_revisions", 3):
        print(f"⚠️ Max revisions reached")
        return END
    return "generate"


# Build graph
graph = StateGraph(ReflectionState)

graph.add_node("generate", generate)
graph.add_node("critique", critique)

graph.add_edge(START, "generate")
graph.add_edge("generate", "critique")
graph.add_conditional_edges("critique", should_revise)

reflection_agent = graph.compile()

# Test
result = reflection_agent.invoke({
    "task": "Write a compelling 100-word introduction for a blog post about why Python is the best first programming language",
    "draft": "",
    "critique": "",
    "revision_count": 0,
    "max_revisions": 3,
    "is_approved": False,
    "revision_log": [],
})

print(f"\n{'='*60}")
print(f"📄 Final Draft (after {result['revision_count']} revision(s)):")
print(f"\n{result['draft']}")
print(f"\n📋 Revision Log:")
for log in result["revision_log"]:
    print(f"   {log}")
```

---

## Part 4: Map-Reduce Pattern

Process multiple items in parallel, then merge results:

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END


class MapReduceState(TypedDict):
    documents: list[str]
    summaries: Annotated[list, add]
    final_summary: str
    current_doc_index: int


def map_summarize(state: MapReduceState) -> dict:
    """Summarize the current document."""
    idx = state["current_doc_index"]
    doc = state["documents"][idx]
    
    response = llm.invoke([
        SystemMessage(content="Summarize this text in 1-2 sentences. Be concise."),
        HumanMessage(content=doc)
    ])
    
    print(f"📄 Summarized document {idx + 1}/{len(state['documents'])}")
    
    return {
        "summaries": [f"Doc {idx + 1}: {response.content}"],
        "current_doc_index": idx + 1
    }


def reduce_combine(state: MapReduceState) -> dict:
    """Combine all summaries into a final comprehensive summary."""
    all_summaries = "\n".join(state["summaries"])
    
    response = llm.invoke([
        SystemMessage(content="Combine these individual summaries into one comprehensive summary. Highlight common themes and key differences."),
        HumanMessage(content=all_summaries)
    ])
    
    print(f"🔗 Combined {len(state['summaries'])} summaries")
    return {"final_summary": response.content}


def should_continue_mapping(state: MapReduceState):
    if state["current_doc_index"] < len(state["documents"]):
        return "map"
    return "reduce"


graph = StateGraph(MapReduceState)
graph.add_node("map", map_summarize)
graph.add_node("reduce", reduce_combine)

graph.add_edge(START, "map")
graph.add_conditional_edges("map", should_continue_mapping)
graph.add_edge("reduce", END)

map_reduce = graph.compile()

# Test
documents = [
    "Machine learning is transforming healthcare with predictive diagnostics and personalized medicine. AI models can now detect diseases earlier than human doctors.",
    "Self-driving cars use computer vision and deep learning to navigate roads. Companies like Tesla and Waymo are leading the autonomous vehicle revolution.",
    "Natural language processing has enabled chatbots and virtual assistants to understand human language. GPT and BERT models have set new benchmarks in NLP tasks.",
    "Reinforcement learning has achieved superhuman performance in games like Go and chess. These techniques are now being applied to robotics and resource optimization.",
]

result = map_reduce.invoke({
    "documents": documents,
    "summaries": [],
    "final_summary": "",
    "current_doc_index": 0,
})

print(f"\n{'='*60}")
print(f"📝 Combined Summary:\n{result['final_summary']}")
```

---

## Common Mistakes

### Mistake 1: Over-engineering with agents when a chain would suffice
```python
# ❌ Building a complex multi-agent system for a simple task
# "Summarize this document" doesn't need 3 specialized agents!

# ✅ Start simple, add complexity only when needed
chain = prompt | llm | parser  # Start here
# Only build an agent when the LLM needs to make decisions
```

### Mistake 2: Not setting max iterations on reflection loops
```python
# ❌ Reflection can loop forever
graph.add_conditional_edges("critique", should_revise)
# What if the critic NEVER approves?

# ✅ Always cap iterations
def should_revise(state):
    if state["revision_count"] >= state["max_revisions"]:
        return END  # Safety valve
    if state["is_approved"]:
        return END
    return "generate"
```

### Mistake 3: Not tracking state properly in multi-step workflows
```python
# ❌ No way to debug what happened
def my_node(state):
    return {"result": do_stuff()}

# ✅ Add logging to state
def my_node(state):
    result = do_stuff()
    return {
        "result": result,
        "processing_log": [f"Step completed: {len(result)} chars"]
    }
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Start with `create_react_agent`, customize when needed | Don't over-engineer |
| Use Plan-and-Execute for complex multi-step tasks | Better planning = better results |
| Cap reflection loops with `max_revisions` | Prevent infinite loops |
| Add processing logs to state | Debugging and monitoring |
| Use the supervisor pattern for multi-domain problems | Specialization improves quality |
| Keep state minimal | Large state = higher token costs |
| Test each node independently before building the graph | Easier debugging |
| Use streaming to show progress | Critical for long-running workflows |

---

## Interview Preparation

### Easy
**Q: What is the Plan-and-Execute agent pattern?**

> Plan-and-Execute separates planning from execution. First, a planner LLM creates a step-by-step plan to answer the question. Then, an executor processes each step sequentially, gathering information and performing calculations. Finally, a synthesizer combines all results into a final answer. This is better than ReAct for complex, multi-step tasks because it creates a coherent plan upfront rather than deciding step-by-step.

### Medium
**Q: How does the multi-agent supervisor pattern work?**

> A supervisor agent classifies the user's query and routes it to specialized worker agents. Each worker (researcher, analyst, writer, etc.) has its own system prompt and expertise. The supervisor decides which worker is best suited for the task, the worker processes it, and the result is compiled into a final answer. This pattern improves quality through specialization — each worker can be optimized for its domain without the complexity of being a generalist.

### Hard
**Q: Design a self-reflective agent that improves its output. What are the trade-offs?**

> Architecture: A generator node creates initial content, a critic node evaluates it against criteria (clarity, accuracy, completeness), and if not approved, the generator revises based on feedback. This loops until approval or max iterations. **Trade-offs**: (1) **Quality vs Cost** — each revision adds 2 LLM calls; 3 revisions = 6 extra calls. (2) **Convergence** — some outputs never satisfy the critic, so max_revisions is essential. (3) **Critic quality** — if the critic is poor, revisions degrade quality. (4) **Latency** — each revision adds 2-5 seconds. Mitigations: use a strong model for the critic, set max_revisions to 2-3, and cache the final result.

---

## Summary

| Architecture | Pattern | Best For |
|-------------|---------|----------|
| **ReAct** | LLM ↔ Tools (loop) | General-purpose tool use |
| **Plan-and-Execute** | Plan → Execute steps → Synthesize | Complex multi-step research |
| **Supervisor** | Route → Specialized workers | Multi-domain problems |
| **Self-Reflective** | Generate → Critique → Revise (loop) | Quality-critical content |
| **Map-Reduce** | Process each → Combine all | Batch document processing |

---

> [← Previous: LangGraph Fundamentals](chapter-37-langgraph-fundamentals.md) | [Next: Human-in-the-Loop →](chapter-39-human-in-the-loop.md)

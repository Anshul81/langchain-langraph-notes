# 🦜⛓️ LangChain Mastery: From Zero to Production AI Applications

> **Stop copy-pasting LLM snippets. Start shipping AI systems that actually work.**

49 chapters. 13 phases. One goal: turn you into the engineer who *builds* the AI features everyone else is demoing.

This isn't a "watch and nod" course. Every chapter has code you run, a project you build, and interview questions you'll actually get asked. By the end, you'll have a portfolio of real AI applications and the mental models to design your own.

---

## 📋 Table of Contents

- [Who This Course Is For](#-who-this-course-is-for)
- [What You'll Build](#-what-youll-build)
- [Visual Course Map](#-visual-course-map)
- [Phase-by-Phase Breakdown](#-phase-by-phase-breakdown)
- [Learning Path Recommendations](#-learning-path-recommendations)
- [Time Estimates](#-time-estimates)
- [Tech Stack & Setup](#-tech-stack--setup)
- [How to Use This Course](#-how-to-use-this-course)
- [Progress Tracker](#-progress-tracker)

---

## 👥 Who This Course Is For

| You are... | This course will... |
|---|---|
| 🐍 **A Python developer** who wants to add AI to your toolkit | Take you from `import langchain` to production agents |
| 🧠 **An AI enthusiast** tired of toy notebooks | Show you how the pieces fit together into real systems |
| 📊 **An ML engineer** moving from models to applications | Bridge the gap between "the model works" and "the product works" |
| 🔧 **A backend dev** asked to "add a chatbot" | Give you the architecture patterns so you don't rebuild it three times |

### ✅ Prerequisites

**You need:**
- Comfortable Python (functions, classes, dicts, list comprehensions — you can write a 100-line script without googling syntax)
- Basic terminal skills (`cd`, `pip install`, running scripts)
- Willingness to read error messages instead of panicking

**You do NOT need:**
- Machine learning background
- Math beyond high-school algebra (we'll teach the vector math you need in Phase 6)
- Prior LangChain experience (that's the whole point)

> 💡 **Not sure if your Python is strong enough?** Phase 0 exists exactly for you. If you can already explain `*args`, context managers, generators, and `async/await` from memory — skim it and jump to Phase 1.

---

## 🏗️ What You'll Build

These aren't throwaway exercises. Each one is portfolio-worthy and maps to a real job requirement.

| # | Project | What It Does | Built In |
|---|---|---|---|
| 1 | 🧾 **Structured Data Extractor** | Feed it messy text (emails, invoices, resumes) → get validated Pydantic objects out | Phase 3 |
| 2 | 💬 **Terminal Chatbot with Memory** | A CLI assistant that remembers your conversation, streams responses, and survives API failures | Phase 4–5 |
| 3 | 🔎 **Semantic Search Engine** | Search your own documents by *meaning*, not keywords. Local (Chroma/FAISS) and cloud | Phase 6–7 |
| 4 | 🧰 **Multi-Tool Research Assistant** | An LLM that searches the web, reads Wikipedia, does math, and calls your custom APIs — including via MCP | Phase 8 |
| 5 | 🕵️ **Autonomous Agent with Human Approval** | A LangGraph agent that plans, acts, and *pauses for your sign-off* before doing anything risky | Phase 9 |
| 6 | 📚 **Production RAG Chatbot** | "Chat with your docs" done right: loaders, smart chunking, hybrid retrieval, source citations | Phase 10–11 |
| 7 | 🧪 **RAG Evaluation Harness** | Measure faithfulness, relevance, and hallucination rate — so you can prove your RAG actually works | Phase 11 |
| 8 | 🚀 **Capstone: Deployed Agentic Application** | Everything together: agents + RAG + tools + memory, traced with LangSmith, ready for real users | Phase 12 |

---

## 🗺️ Visual Course Map

Six milestones. Each one unlocks the next. Here's the whole journey in one screen:

```
                        🦜⛓️  LANGCHAIN MASTERY — THE JOURNEY
 ═══════════════════════════════════════════════════════════════════════════════

  🏗️ MILESTONE 1: FOUNDATION                                    (Ch 0–13, ~23h)
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Phase 0              Phase 1               Phase 2                         │
  │  🐍 Python Power-Up ──▶ 🤖 LLM Fundamentals ──▶ ✍️ Prompt Engineering       │
  │  *args, async, OOP     tokens, API, params    messages, few-shot, parsing   │
  └─────────────────────────────────────────┬───────────────────────────────────┘
                                            │
                                            ▼
  ⚙️ MILESTONE 2: CORE LANGCHAIN                                (Ch 14–22, ~21h)
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Phase 3              Phase 4                    Phase 5                    │
  │  🔗 LangChain Core ──▶ ⛓️ Chains & Runnables ──▶ 🧠 Memory                  │
  │  LCEL, structured      retry, fallback,           conversation state        │
  │  output                streaming, async                                     │
  └───────────────┬────────────────────────────────────────────┬────────────────┘
                  │                                            │
                  ▼                                            ▼
  🧠 MILESTONE 3: KNOWLEDGE & RETRIEVAL      🛠️ MILESTONE 4: TOOLS & AGENTS
  (Ch 23–29, ~14h)                           (Ch 30–39, ~27h)
  ┌─────────────────────────────────┐        ┌──────────────────────────────────┐
  │  Phase 6          Phase 7       │        │  Phase 8           Phase 9       │
  │  📐 Embeddings ──▶ 🗄️ Vector DBs │        │  🔧 Tools ────────▶ 🤖 Agents     │
  │  vectors,          Chroma,      │        │  built-in, custom,  ReAct,       │
  │  similarity        FAISS, cloud │        │  tool calling, MCP  LangGraph,   │
  │                                 │        │                     HITL         │
  └───────────────┬─────────────────┘        └────────────────┬─────────────────┘
                  │                                           │
                  ▼                                           │
  📚 MILESTONE 5: RAG MASTERY                 (Ch 40–45, ~17h)│
  ┌─────────────────────────────────────────┐                 │
  │  Phase 10           Phase 11            │                 │
  │  📄 RAG ──────────▶ 🎯 Advanced RAG      │                 │
  │  loaders, splitting  hybrid retrieval,  │                 │
  │  full pipeline       reranking, eval    │                 │
  └────────────────────┬────────────────────┘                 │
                       │                                      │
                       └──────────────┬───────────────────────┘
                                      ▼
  🚀 MILESTONE 6: PRODUCTION                                   (Ch 46–48, ~18h)
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  Phase 12                                                                   │
  │  🏭 Production LangChain ──▶ 🔍 LangSmith ──▶ 🏆 CAPSTONE PROJECT            │
  │  caching, cost, security     tracing, evals    ship something real          │
  └─────────────────────────────────────────────────────────────────────────────┘

 ═══════════════════════════════════════════════════════════════════════════════
  Legend:  ──▶ builds directly on     │ Milestones 3 & 4 can be done in either order
```

> 🔑 **Key insight:** Milestones 3 (Retrieval) and 4 (Agents) are independent branches off Core LangChain. Do them in either order. But you need *both* before RAG Mastery gets interesting, and you need *everything* before Production.

---

## 📖 Phase-by-Phase Breakdown

Difficulty key: 🟢 Beginner · 🟡 Intermediate · 🔴 Advanced
Time = reading + running code + doing the exercises. Your mileage will vary.

---

### 🐍 Phase 0 — Python Power-Up

LangChain's codebase leans *hard* on modern Python: everything is a Runnable (OOP), streaming uses generators, async is everywhere, and Pydantic models define every schema. If these feel fuzzy, LangChain will feel like magic — and magic you can't debug. This phase makes sure the Python never gets in your way.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 0 | `*args` & `**kwargs` | 1h | 🟢 | Why every LangChain signature has `**kwargs` and how to read them |
| 1 | Context Managers | 1.5h | 🟢 | `with` blocks, `__enter__`/`__exit__`, and why callbacks/tracing use them |
| 2 | Generators | 1.5h | 🟢 | `yield`, lazy evaluation — the engine behind `.stream()` |
| 3 | Type Hints & Pydantic | 2h | 🟡 | Model validation, `Field`, schemas — the backbone of structured output and tools |
| 4 | Async Python | 2h | 🟡 | `async/await`, `asyncio.gather` — how you'll run 50 LLM calls in parallel |
| 5 | OOP Patterns | 1.5h | 🟡 | Composition, protocols, `__or__` overloading — why `prompt | llm | parser` works |

**Total: ~9.5 hours**

**✅ After this phase you'll be able to:**
- Read any LangChain source file without getting lost in syntax
- Write Pydantic models confidently (you'll do this in nearly every later chapter)
- Understand what `async def astream()` means before you ever call it

**🎯 Checkpoint:** Build a mini "pipe" framework — classes that can be composed with `|`, support `.invoke()` and `.stream()` (via generators), and validate their inputs with Pydantic. You'll basically reinvent a baby LCEL. Then Phase 3 will feel like coming home.

---

### 🤖 Phase 1 — LLM Fundamentals

Before you abstract something, you should understand what you're abstracting. This phase strips away all frameworks and talks to an LLM raw: what it is, how it eats text, what the API actually returns, and which knobs change its behavior. Skip this and you'll spend months cargo-culting `temperature=0.7` without knowing why.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 6 | What is an LLM? | 1h | 🟢 | Next-token prediction, why models hallucinate, what "context window" really means |
| 7 | Tokens & Tokenization | 1.5h | 🟢 | `tiktoken`, why "strawberry" is 3 tokens, and how tokens = money |
| 8 | First API Call | 1.5h | 🟢 | Raw OpenAI SDK call, reading the response object, handling API keys safely |
| 9 | Model Parameters | 2h | 🟡 | `temperature`, `top_p`, `max_tokens`, stop sequences — with experiments, not hand-waving |

**Total: ~6 hours**

**✅ After this phase you'll be able to:**
- Estimate the cost of any LLM feature before building it
- Explain to a stakeholder why the model "made something up"
- Pick sane parameter defaults for chatbots vs. extraction vs. creative tasks

**🎯 Checkpoint:** Build a **token cost calculator CLI** — paste in text, choose a model, get token count and estimated price. Then run the same prompt at `temperature` 0, 0.7, and 1.5 ten times each and document what changes.

---

### ✍️ Phase 2 — Prompt Engineering

Prompting isn't "asking nicely." It's the interface layer of your application, and it deserves the same rigor as an API design. This phase covers message roles, the techniques that reliably improve output quality, templating for reuse, and — critically — getting *structured* data back out instead of prose you have to regex.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 10 | Messages | 1.5h | 🟢 | System / Human / AI roles, why the system prompt is your config file |
| 11 | Few-Shot & Chain-of-Thought | 2h | 🟡 | Examples that steer behavior, "think step by step" and when it actually helps |
| 12 | Prompt Templates | 2h | 🟢 | Variables, partials, composing templates — prompts as code, not strings |
| 13 | Output Parsing | 2.5h | 🟡 | JSON mode, parsing to Pydantic, handling malformed output gracefully |

**Total: ~8 hours**

**✅ After this phase you'll be able to:**
- Write prompts that produce consistent, parseable output 95%+ of the time
- Version and test your prompts like any other code
- Know when few-shot examples help and when they just burn tokens

**🎯 Checkpoint:** Build a **sentiment + entity extractor** using pure prompting: input a customer review, output a validated JSON object with sentiment, mentioned products, and issues. Make it robust to the model occasionally returning junk.

> 💡 **Motivation check:** After Phase 2 you can already build useful things without LangChain. That's intentional — you'll appreciate the framework more when you know what it's saving you from.

---

### 🔗 Phase 3 — LangChain Core

Now the framework. LangChain's modern core is LCEL (LangChain Expression Language) — a way to compose components with `|` that gives you streaming, async, batching, and tracing for free. This phase gets you set up, teaches the Runnable mental model, and shows the two things you'll use daily: chat templates and structured output.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 14 | LangChain Setup | 1h | 🟢 | Package structure (`langchain-core`, `langchain-openai`, etc.), env config, version pinning |
| 15 | Runnables & LCEL | 2.5h | 🟡 | The Runnable protocol, `|` composition, `invoke/batch/stream` — the heart of everything |
| 16 | Chat Prompt Templates | 2h | 🟢 | `ChatPromptTemplate`, `MessagesPlaceholder`, few-shot chat templates |
| 17 | Structured Output | 2.5h | 🟡 | `.with_structured_output()`, Pydantic schemas, when to use JSON mode vs. tool calling |

**Total: ~8 hours**

**✅ After this phase you'll be able to:**
- Build `prompt | llm | parser` chains and explain exactly what each `|` does
- Get typed Python objects back from an LLM in one line
- Navigate LangChain's package ecosystem without import errors

**🎯 Checkpoint:** Rebuild your Phase 2 extractor in LangChain. It should be ~5x less code. Then extend it: **Project 1 — Structured Data Extractor** that handles resumes, invoices, and emails with different Pydantic schemas selected at runtime.

---

### ⛓️ Phase 4 — Chains & Runnables

Real applications aren't one LLM call. They're pipelines: call A feeds call B, some steps run in parallel, things fail and need retries, and users expect tokens to stream in real time. This phase is where LangChain earns its keep — you'll learn the composition patterns that make complex flows readable and resilient.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 18 | Runnables Deep Dive | 2.5h | 🟡 | `RunnableLambda`, `RunnableParallel`, `RunnablePassthrough`, `RunnableBranch` |
| 19 | Sequential Chains | 2h | 🟡 | Multi-step pipelines, passing state between steps, `itemgetter` tricks |
| 20 | Retry, Fallback & Error Handling | 2.5h | 🔴 | `.with_retry()`, `.with_fallbacks()`, model fallback chains, graceful degradation |
| 21 | Streaming & Async Chains | 3h | 🔴 | `.stream()`, `.astream()`, `astream_events`, batching, concurrency limits |

**Total: ~10 hours**

**✅ After this phase you'll be able to:**
- Design multi-step LLM pipelines that read like a flowchart
- Build chains that fall back from GPT-4 to a cheaper model when rate-limited
- Stream tokens to a UI while running background steps concurrently

**🎯 Checkpoint:** Build a **blog post generator pipeline**: outline → parallel section drafting → critique → revision. Must stream the final output, retry on failure, and fall back to a different model. Time the sync vs. async versions.

> 💪 By the end of Phase 4, you'll be chaining LLM calls like a pro — and you'll wince every time you see someone do it with nested `if` statements.

---

### 🧠 Phase 5 — Memory

LLMs are stateless. Every call starts from zero. "Memory" is just you deciding what history to inject into the next prompt — and that decision involves real tradeoffs between cost, context limits, and coherence. One dense chapter, because getting this right matters more than the number of chapters suggests.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 22 | Conversation Memory | 3h | 🟡 | `RunnableWithMessageHistory`, buffer vs. window vs. summary memory, session management, persistence |

**Total: ~3 hours**

**✅ After this phase you'll be able to:**
- Build multi-turn chatbots that remember context across sessions
- Choose the right memory strategy for your context budget
- Persist conversation history to a real store (SQLite, Redis) instead of a dict

**🎯 Checkpoint:** **Project 2 — Terminal Chatbot with Memory.** Streaming responses, persistent sessions (quit and resume), summary memory that kicks in after N turns, and a `/history` command. This is your first "I'd actually use this" project.

---

### 📐 Phase 6 — Embeddings & Vector Math

Here's the idea that unlocks everything in retrieval: you can turn text into a list of numbers such that *similar meanings produce nearby numbers*. This phase builds that intuition from scratch, compares real embedding models, and teaches the similarity math — with just enough NumPy to make it concrete.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 23 | What Are Embeddings? | 1.5h | 🟢 | Vectors as meaning, visualizing embeddings, why "king − man + woman ≈ queen" |
| 24 | Embedding Models | 2h | 🟡 | OpenAI vs. open-source (HuggingFace, sentence-transformers), dimensions, cost, benchmarks |
| 25 | Similarity Search | 2.5h | 🟡 | Cosine vs. dot product vs. Euclidean, brute-force search in NumPy, why it doesn't scale |

**Total: ~6 hours**

**✅ After this phase you'll be able to:**
- Explain embeddings to a non-technical person in two sentences
- Pick an embedding model based on cost/quality/latency tradeoffs
- Implement semantic search from scratch in 30 lines of NumPy

**🎯 Checkpoint:** Build a **semantic FAQ matcher**: embed 100 FAQ entries, take a user question, return the top-3 matches by cosine similarity. No vector DB yet — pure NumPy. You'll feel exactly why Phase 7 exists.

---

### 🗄️ Phase 7 — Vector Databases

Your NumPy search works for 100 documents. At 10 million, it doesn't. Vector databases solve this with approximate nearest-neighbor indexes, metadata filtering, and persistence. This phase covers the two local options you'll use constantly (Chroma, FAISS) and when to graduate to a managed cloud service.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 26 | Why Vector Databases? | 1.5h | 🟢 | ANN indexes (HNSW, IVF), the speed/accuracy tradeoff, the vector DB landscape |
| 27 | ChromaDB | 2h | 🟡 | Collections, persistence, metadata filtering, the LangChain `Chroma` integration |
| 28 | FAISS | 2h | 🟡 | Index types, saving/loading, when FAISS beats Chroma (and vice versa) |
| 29 | Cloud Vector DBs | 2.5h | 🟡 | Pinecone / Weaviate / Qdrant / pgvector — namespaces, hybrid search, cost models |

**Total: ~8 hours**

**✅ After this phase you'll be able to:**
- Stand up a persistent vector store with metadata filtering in 20 lines
- Choose between Chroma, FAISS, pgvector, and managed services with actual reasons
- Explain HNSW to an interviewer without breaking a sweat

**🎯 Checkpoint:** **Project 3 — Semantic Search Engine.** Index a folder of your own documents (notes, PDFs, code). Support metadata filters (`source`, `date`), swap between Chroma and FAISS backends via config, and benchmark query latency at 1k vs. 100k chunks.

---

### 🔧 Phase 8 — Tools & Tool Calling

An LLM that can only talk is a very expensive autocomplete. Tools let it *act*: search the web, run code, query your database, call your APIs. This phase covers how tool calling works under the hood, how to build your own tools, and MCP — the emerging standard for plugging tools into any LLM app.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 30 | What Are Tools? | 1.5h | 🟢 | Function calling explained, the tool schema, the LLM-decides/you-execute loop |
| 31 | Built-in Tools | 2h | 🟢 | Tavily/DuckDuckGo search, Wikipedia, calculator, Python REPL — and their gotchas |
| 32 | Custom Tools | 2.5h | 🟡 | `@tool` decorator, `StructuredTool`, Pydantic arg schemas, writing docstrings the LLM understands |
| 33 | Tool Calling Deep Dive | 3h | 🔴 | `bind_tools`, parallel tool calls, `ToolMessage`, error handling, forcing tool choice |
| 34 | MCP (Model Context Protocol) | 3h | 🔴 | MCP servers/clients, `langchain-mcp-adapters`, building and consuming an MCP server |

**Total: ~12 hours**

**✅ After this phase you'll be able to:**
- Wrap any Python function or API as an LLM-callable tool
- Debug the full tool-calling loop: request → tool call → result → final answer
- Expose your tools via MCP so Claude Desktop, Cursor, or any client can use them

**🎯 Checkpoint:** **Project 4 — Multi-Tool Research Assistant.** Web search + Wikipedia + calculator + a custom tool hitting a real API (weather, stocks, your choice). Then expose your custom tool as an MCP server and consume it from LangChain.

---

### 🤖 Phase 9 — Agents

Chains follow a fixed path. Agents *decide* the path — they reason about what to do, use tools, observe results, and loop until done. This is the most powerful and most dangerous thing you'll build. This phase teaches the ReAct pattern, then LangGraph (the production way to build agents), then how to keep humans in control.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 35 | Intro to Agents | 2h | 🟡 | Agent vs. chain, the reason-act-observe loop, when *not* to use an agent |
| 36 | ReAct Agents | 3h | 🟡 | Building ReAct from scratch, then with `create_react_agent`, reading agent traces |
| 37 | LangGraph Fundamentals | 4h | 🔴 | State, nodes, edges, conditional routing, checkpointers — graphs as agent runtimes |
| 38 | Custom Agents | 3h | 🔴 | Plan-and-execute, multi-agent supervisor patterns, tool-error recovery |
| 39 | Human-in-the-Loop | 3h | 🔴 | `interrupt()`, approval gates, editing agent state mid-run, resumable workflows |

**Total: ~15 hours**

**✅ After this phase you'll be able to:**
- Build agents that reliably complete multi-step tasks (and know their failure modes)
- Model any agent workflow as a LangGraph state machine
- Add approval checkpoints so your agent never sends that email without asking

**🎯 Checkpoint:** **Project 5 — Autonomous Agent with Human Approval.** A LangGraph agent that researches a topic, drafts a report, and pauses for your review before "publishing" (writing to a file / sending a message). Must be resumable after a restart via checkpointer.

> ⚠️ **Opinionated take:** 80% of "we need an agent" requests are actually "we need a chain with a branch." Learn agents deeply so you know when *not* to reach for them.

---

### 📄 Phase 10 — RAG

Retrieval-Augmented Generation is the killer app of LLMs: give the model your documents so it answers from *your* knowledge, not its training data. You already have the pieces (embeddings, vector DBs, chains). This phase assembles them into a real pipeline and covers the unglamorous parts that actually determine quality — loading and chunking.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 40 | Intro to RAG | 1.5h | 🟢 | Why RAG beats fine-tuning for knowledge, the retrieve → augment → generate loop |
| 41 | Document Loaders | 2.5h | 🟡 | PDFs, web pages, Notion, GitHub, CSVs — and cleaning the garbage they produce |
| 42 | Text Splitting | 3h | 🟡 | Recursive character, token-based, semantic, code-aware splitting — chunk size experiments |
| 43 | Complete RAG Pipeline | 3h | 🟡 | Retriever → prompt → LLM with source citations, conversational RAG with history |

**Total: ~10 hours**

**✅ After this phase you'll be able to:**
- Build a "chat with your docs" app end-to-end in an afternoon
- Diagnose bad answers: is it retrieval, chunking, or prompting?
- Cite sources so users can verify every claim

**🎯 Checkpoint:** **Project 6 (v1) — RAG Chatbot** over a real corpus (your company docs, a textbook, a GitHub repo). Conversational, streaming, with clickable source citations. Keep it — you'll upgrade it in Phase 11.

---

### 🎯 Phase 11 — Advanced RAG

Basic RAG demos well and disappoints in production. Questions get rephrased wrong, the right chunk ranks 7th, and nobody knows if the new chunking strategy helped or hurt. This phase covers the retrieval techniques that separate toy RAG from production RAG — and how to *measure* the difference.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 44 | Advanced Retrieval | 3.5h | 🔴 | Hybrid search (BM25 + vectors), multi-query, HyDE, parent-document, reranking, contextual compression |
| 45 | RAG Evaluation | 3.5h | 🔴 | Faithfulness, answer relevance, context precision/recall — building golden datasets, RAGAS, LLM-as-judge |

**Total: ~7 hours**

**✅ After this phase you'll be able to:**
- Boost retrieval quality 20–40% with hybrid search and reranking
- Build an eval suite that catches regressions before users do
- Answer "is it better?" with numbers instead of vibes

**🎯 Checkpoint:** **Project 6 (v2) + Project 7 — RAG Eval Harness.** Create a 50-question golden dataset for your Phase 10 chatbot. Measure baseline. Add hybrid search + reranking. Measure again. Write up the results like you'd present them to your team.

---

### 🚀 Phase 12 — Production

Everything until now ran on your laptop with your API key. Production means: other people's money, other people's data, and 3 AM pages. This phase covers the hardening (caching, rate limits, security, cost control), the observability (LangSmith tracing and evals), and then the capstone where you put it all together.

| Ch | Title | Time | Difficulty | What You'll Learn |
|---|---|---|---|---|
| 46 | Production LangChain | 3h | 🔴 | LLM caching, rate limiting, prompt injection defense, cost tracking, config management, serving with FastAPI |
| 47 | LangSmith | 3h | 🟡 | Tracing every call, debugging agent runs, datasets & evaluators, monitoring in prod |
| 48 | Capstone Project | 12h+ | 🔴 | Design, build, evaluate, and deploy a complete agentic application — your choice of domain |

**Total: ~18 hours**

**✅ After this phase you'll be able to:**
- Ship an LLM application you'd be comfortable putting your name on
- Trace a bad output back to the exact prompt, retrieval, or tool call that caused it
- Talk credibly about LLM cost, latency, and safety in an architecture review

**🎯 Checkpoint:** **Project 8 — The Capstone.** Combine agents + RAG + tools + memory into one deployed application with LangSmith tracing, an eval suite, and a README that explains your architecture decisions. Suggested ideas: customer support agent over your docs, codebase Q&A with GitHub tools, research assistant with citation verification.

> 🏆 **Finish the capstone and you're not "learning LangChain" anymore. You're an AI engineer with a shipped product.**

---

## 🛤️ Learning Path Recommendations

Not everyone needs every chapter. Pick your track:

### 🚀 Fast Track — Experienced Python Devs (~75 hours)

You already know async, Pydantic, and OOP cold. You've called an LLM API before.

| Action | Chapters |
|---|---|
| ⏭️ **Skip** | Ch 0, 1, 2, 5 (Python basics) · Ch 6, 8 (LLM basics) |
| 👀 **Skim** (15 min each) | Ch 3, 4 (verify Pydantic v2 + asyncio knowledge) · Ch 7, 9, 10 · Ch 14, 16 · Ch 23, 26, 30, 35, 40 (intro chapters) |
| ✅ **Do fully** | Everything else — especially Ch 15, 17, 18, 20, 21, 33, 34, 37–39, 43–48 |

### 📚 Complete Track — Recommended (~120 hours)

Do every chapter in order. Build every checkpoint. This is the path with the fewest gaps and the strongest portfolio.

**Why we recommend it:** The "boring" chapters (tokens, context managers, similarity math) are the ones that make you the person on the team who can actually *debug* the AI system instead of just restarting it.

### 🎯 RAG-Focused Track — "I just want to chat with my docs" (~65 hours)

You have a specific RAG project and want the shortest path to production quality.

```
Phase 1 (Ch 7, 8, 9) ──▶ Phase 2 (Ch 10, 12, 13) ──▶ Phase 3 (all)
        ──▶ Phase 4 (Ch 18, 20, 21) ──▶ Phase 5 (Ch 22)
        ──▶ Phase 6 (all) ──▶ Phase 7 (all)
        ──▶ Phase 10 (all) ──▶ Phase 11 (all)
        ──▶ Phase 12 (Ch 46, 47) ──▶ Capstone as a RAG app
```

**Skip:** Phase 0 (unless needed), Phase 8, Phase 9.
**Come back for:** Phase 8–9 when your users start asking for the chatbot to "just do the thing" instead of explaining it. (They will.)

---

## ⏱️ Time Estimates

### By Phase

| Phase | Name | Chapters | Hours | Cumulative |
|---|---|---|---|---|
| 0 | 🐍 Python Power-Up | 6 | 9.5 | 9.5 |
| 1 | 🤖 LLM Fundamentals | 4 | 6 | 15.5 |
| 2 | ✍️ Prompt Engineering | 4 | 8 | 23.5 |
| 3 | 🔗 LangChain Core | 4 | 8 | 31.5 |
| 4 | ⛓️ Chains & Runnables | 4 | 10 | 41.5 |
| 5 | 🧠 Memory | 1 | 3 | 44.5 |
| 6 | 📐 Embeddings & Vector Math | 3 | 6 | 50.5 |
| 7 | 🗄️ Vector Databases | 4 | 8 | 58.5 |
| 8 | 🔧 Tools & Tool Calling | 5 | 12 | 70.5 |
| 9 | 🤖 Agents | 5 | 15 | 85.5 |
| 10 | 📄 RAG | 4 | 10 | 95.5 |
| 11 | 🎯 Advanced RAG | 2 | 7 | 102.5 |
| 12 | 🚀 Production | 3 | 18 | **120.5** |

**Total: ~120 hours** (Complete Track, including all checkpoints)

### Suggested Schedules

| Pace | Weekly Commitment | Duration | Best For |
|---|---|---|---|
| 🐢 **Steady** | 5–6 hrs/week | ~22 weeks | Full-time job, learning on the side |
| 🚶 **Recommended** | 8–10 hrs/week | **~12 weeks** | Serious upskilling, 1–2 hrs on weekdays + a weekend block |
| 🏃 **Intensive** | 20+ hrs/week | ~6 weeks | Between jobs, bootcamp mode, deadline-driven |

**Sample 12-week plan (Recommended pace):**

```
Week 1   │ Phase 0 + Phase 1          │ 🏗️ Foundation
Week 2   │ Phase 2 + Phase 3 (start)  │
Week 3   │ Phase 3 + Phase 4 (start)  │ ⚙️ Core LangChain
Week 4   │ Phase 4 + Phase 5          │  → Project 2 done
Week 5   │ Phase 6 + Phase 7          │ 🧠 Knowledge & Retrieval → Project 3 done
Week 6   │ Phase 8                    │ 🛠️ Tools & Agents → Project 4 done
Week 7   │ Phase 9 (Ch 35–37)         │
Week 8   │ Phase 9 (Ch 38–39)         │  → Project 5 done
Week 9   │ Phase 10                   │ 📚 RAG Mastery → Project 6 v1 done
Week 10  │ Phase 11                   │  → Project 7 done
Week 11  │ Phase 12 (Ch 46–47)        │ 🚀 Production
Week 12  │ Capstone                   │  → Project 8 SHIPPED 🏆
```

> 💡 **Real talk:** The estimates assume you actually type the code, not just read it. Reading-only takes a third of the time and teaches you a tenth as much.

---

## 🧰 Tech Stack & Setup

### Requirements

| Component | Requirement | Notes |
|---|---|---|
| **Python** | 3.10+ | 3.11 or 3.12 recommended. 3.10 is the floor for modern type hints |
| **Package manager** | `uv` (recommended) or `pip` + `venv` | `uv` is 10–100x faster; worth the 2-minute install |
| **LLM access** | OpenAI API key **or** LiteLLM proxy | See below |
| **Git** | Any recent version | For cloning and for the GitHub loader chapters |
| **RAM** | 8 GB minimum, 16 GB comfortable | FAISS and local embedding models eat memory |

### LLM Access Options

**Option A — OpenAI API key (simplest)**
```bash
export OPENAI_API_KEY="sk-..."
```
Budget ~$10–20 for the full course if you use `gpt-4o-mini` for most exercises. Set a hard spending limit in your OpenAI dashboard *before* Phase 9 — agents in a loop can burn money fast.

**Option B — LiteLLM proxy (for teams / multi-provider)**
```bash
export OPENAI_API_BASE="[http://your-litellm-proxy:4000"](http://your-litellm-proxy:4000")
export OPENAI_API_KEY="your-proxy-key"
```
Lets you swap providers (Anthropic, Azure, Bedrock, local Ollama) without changing course code. Great if your company already runs one.

**Option C — Local models (free, slower)**
Ollama + `langchain-ollama` works for most chapters. Tool calling and structured output quality varies by model — expect some friction in Phases 8–9.

### Core Packages

```bash
# Create and activate environment
uv venv && source .venv/bin/activate      # or: python -m venv .venv

# Core LangChain
uv pip install langchain langchain-core langchain-openai langchain-community

# Agents & orchestration
uv pip install langgraph langsmith

# Vector stores & embeddings
uv pip install chromadb faiss-cpu sentence-transformers

# Tools & MCP
uv pip install langchain-mcp-adapters tavily-python wikipedia duckduckgo-search

# Document loading & splitting
uv pip install pypdf beautifulsoup4 unstructured tiktoken

# RAG evaluation
uv pip install ragas

# Utilities
uv pip install pydantic python-dotenv rich numpy fastapi uvicorn
```

Each chapter folder has its own `requirements.txt` with pinned versions — **use those** when you hit a version conflict. LangChain moves fast; pinning saves you from "it worked yesterday" bugs.

### Recommended IDE Setup

**VS Code** (or Cursor) with:
- **Python** + **Pylance** extensions — type checking will catch half your Pydantic mistakes before you run anything
- **Jupyter** extension — for exploratory chapters (Phases 1, 6, 11) where you want to poke at outputs
- **Ruff** — fast linting/formatting, zero config
- **Even Better TOML** — for `pyproject.toml`

Settings worth enabling:
```json
{
  "python.analysis.typeCheckingMode": "basic",
  "editor.formatOnSave": true,
  "[python]": { "editor.defaultFormatter": "charliermarsh.ruff" }
}
```

**Environment file** — every chapter loads from a `.env` in the repo root:
```bash
OPENAI_API_KEY=sk-...
LANGCHAIN_TRACING_V2=true          # enable from Phase 12 (or earlier if curious)
LANGCHAIN_API_KEY=lsv2_...
TAVILY_API_KEY=tvly-...            # Phase 8
```

> ⚠️ Add `.env` to `.gitignore` before your first commit. This course has a chapter on security; don't become its case study.

---

## 📘 How to Use This Course

### Chapter Format

Every chapter follows the same 9-part structure, so you always know where you are:

```
📍 Learning Objectives   →  What you'll be able to do (not just "understand")
📖 Introduction           →  Why this matters, real-world context, the problem we're solving
🧠 Theory                 →  The concepts, with diagrams — as short as possible, no shorter
💻 Code Examples          →  Runnable, annotated, progressively complex
🔨 Project                →  Build something that uses the chapter's ideas
✏️ Exercises              →  3–5 tasks, from "modify the example" to "build it differently"
🎤 Interview Prep         →  Questions you'll actually get asked, with model answers
📝 Summary                →  The 5 things to remember
🃏 Flashcards             →  Spaced-repetition-ready Q&A for review
```

### How to Get the Most Out of It

1. **Type the code. Don't paste it.** Muscle memory is real. You'll learn the API by fumbling with it.
2. **Do the checkpoint project before moving to the next phase.** It's the only way to know you actually got it.
3. **Break things on purpose.** Change a parameter. Remove a step. Feed it garbage. Watch what happens. This is how you build intuition.
4. **Keep a `notes.md`.** Every time something surprises you, write

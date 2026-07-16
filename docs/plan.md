# 🗺️ AI Engineering Masterclass — Your Personalized Roadmap

## Student Profile

| Attribute | Value |
|-----------|-------|
| **Python Experience** | 2 years (Beginner) |
| **Target Role** | AI Engineer / Backend Engineer |
| **Interview Timeline** | 3 months (~12 weeks) |
| **Daily Study Hours** | 2 hours |
| **Total Available Hours** | ~180 hours |
| **Learning Style** | Theory → Code |
| **Pace** | Moderate |
| **Strengths** | Flask APIs, Git, Docker, SQL, Debugger |
| **Key Gaps** | Advanced Python, LLM internals, Embeddings, Async, Pydantic, Vector DBs, LangChain, LangGraph |

---

## Roadmap Overview

```
Week 1-2    ████░░░░░░░░  Phase 0-2: Foundations (Python + LLM + Prompts)
Week 3-4    ████████░░░░  Phase 3-7: LangChain Core (Chains, LCEL, Parsers)
Week 5-6    ████████░░░░  Phase 8-10: Memory, Tools, Agents
Week 7-8    ████████░░░░  Phase 11-12: RAG + Advanced RAG
Week 9-10   ████████░░░░  Phase 13-20: LangGraph Deep Dive
Week 11     ████████████  Phase 21-24: Multi-Agent + Production
Week 12     ████████████  Phase 25: Capstone Project
```

---

## Phase 0 — Python Power-Up (Week 1, ~10 hours)

> **Why:** Your Python gaps (context managers, generators, async, Pydantic, type hints) will block you in LangChain. We fix this FIRST.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 0.1 | `*args`, `**kwargs`, Unpacking | 1.5 | Config builder function |
| 0.2 | Context Managers (`with` statement) | 1.5 | File processor utility |
| 0.3 | Generators & Iterators | 1.5 | Streaming data simulator |
| 0.4 | Type Hints & Pydantic Models | 2.0 | API request/response validator |
| 0.5 | Async Python (`asyncio`, `await`) | 2.0 | Async API caller |
| 0.6 | OOP Patterns for AI Apps | 1.5 | Base Agent class design |

**Milestone:** You can write production-style Python with type hints, Pydantic, async, and proper structure.

---

## Phase 1 — LLM Fundamentals (Week 1-2, ~8 hours)

> **Why:** You use ChatGPT but don't know HOW it works. Understanding internals = understanding why LangChain exists.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 1.1 | What is an LLM? (Transformers, Training, Inference) | 2.0 | — |
| 1.2 | Tokens, Tokenization, Context Windows | 2.0 | Token counter tool |
| 1.3 | OpenAI API — Your First LLM Call | 2.0 | CLI chatbot (raw API) |
| 1.4 | Chat Models vs Completion Models, Temperature, Parameters | 2.0 | Parameter playground |

**Milestone:** You can call OpenAI/Gemini APIs directly and understand tokens, costs, and parameters.

---

## Phase 2 — Prompt Engineering (Week 2, ~6 hours)

> **Why:** Garbage prompts = garbage output. This is the most underrated skill in AI engineering.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 2.1 | System / User / Assistant Messages | 1.5 | Role-based chatbot |
| 2.2 | Zero-shot, Few-shot, Chain-of-Thought | 2.0 | Classification engine |
| 2.3 | Prompt Templates & Variables | 1.5 | Dynamic prompt builder |
| 2.4 | Output Formatting (JSON mode, Structured Output) | 1.0 | Structured data extractor |

**Milestone:** You can craft expert-level prompts and get consistent, structured outputs.

---

## Phase 3 — Embeddings & Vector Math (Week 3, ~6 hours)

> **Why:** Embeddings are the foundation of RAG, search, and recommendation. No embeddings = no RAG.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 3.1 | What are Embeddings? (Intuition + Math) | 2.0 | Word similarity finder |
| 3.2 | Embedding Models (OpenAI, HuggingFace, Sentence Transformers) | 2.0 | Embedding comparison tool |
| 3.3 | Similarity Search (Cosine, Dot Product, Euclidean) | 2.0 | Semantic search engine (no DB) |

**Milestone:** You understand how machines "understand" meaning and can build basic semantic search.

---

## Phase 4 — Vector Databases (Week 3, ~6 hours)

> **Why:** You can't do RAG without storing and retrieving embeddings efficiently.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 4.1 | Why Vector Databases? (vs SQL, vs NoSQL) | 1.5 | — |
| 4.2 | ChromaDB — Local Vector Store | 2.0 | Document store |
| 4.3 | FAISS — Facebook AI Similarity Search | 1.5 | Fast similarity search |
| 4.4 | Pinecone / Weaviate — Cloud Vector DBs (Overview) | 1.0 | — |

**Milestone:** You can store, index, and retrieve documents by meaning.

---

## Phase 5 — LangChain Core (Week 4, ~8 hours)

> **Why:** LangChain is the framework that connects LLMs + Tools + Memory + Data. This is where it all begins.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 5.1 | Why LangChain? Architecture & Philosophy | 1.5 | — |
| 5.2 | Installation, Environment Setup, Project Structure | 1.5 | Project scaffold |
| 5.3 | Chat Models (OpenAI, Gemini, Ollama) | 2.0 | Multi-model chatbot |
| 5.4 | LCEL — LangChain Expression Language | 3.0 | Chain composition playground |

**Milestone:** You can build LangChain apps with proper structure and use LCEL fluently.

---

## Phase 6 — Chains & Runnables (Week 4-5, ~8 hours)

> **Why:** Chains are how you compose complex LLM workflows. LCEL makes them powerful.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 6.1 | Runnables (invoke, batch, stream, astream) | 2.0 | Streaming translator |
| 6.2 | RunnablePassthrough, RunnableLambda, RunnableParallel | 2.0 | Data pipeline |
| 6.3 | Sequential Chains | 2.0 | Blog writer chain |
| 6.4 | Retry, Fallback, Error Handling in Chains | 2.0 | Resilient API chain |

**Milestone:** You can build robust, composable, production-ready chains.

---

## Phase 7 — Prompts & Output Parsers (Week 5, ~6 hours)

> **Why:** LangChain's prompt system and parsers give you structured, reliable LLM outputs.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 7.1 | PromptTemplate, ChatPromptTemplate, MessagesPlaceholder | 2.0 | Dynamic prompt system |
| 7.2 | Output Parsers (StrOutputParser, JsonOutputParser, PydanticOutputParser) | 2.0 | Resume parser |
| 7.3 | Structured Output with `with_structured_output()` | 2.0 | Data extraction pipeline |

**Milestone:** You can build type-safe, structured LLM pipelines.

---

## Phase 8 — Memory Systems (Week 5-6, ~8 hours)

> **Why:** Without memory, your chatbot forgets everything after each message. Memory = conversations.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 8.1 | Why Memory? Stateless vs Stateful LLMs | 1.5 | — |
| 8.2 | ConversationBufferMemory, WindowMemory, SummaryMemory | 3.0 | Chatbot with memory |
| 8.3 | Chat Message History (In-memory, Redis, PostgreSQL) | 2.0 | Persistent chatbot |
| 8.4 | Memory in LCEL Chains | 1.5 | Conversational chain |

**Milestone:** You can build chatbots that remember context across sessions.

---

## Phase 9 — Tools & Tool Calling (Week 6, ~8 hours)

> **Why:** Tools let LLMs interact with the real world — search, calculate, call APIs, read files.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 9.1 | What are Tools? Why LLMs Need Them | 1.5 | — |
| 9.2 | Built-in Tools (Search, Wikipedia, Calculator) | 2.0 | Wikipedia agent |
| 9.3 | Custom Tools with `@tool` decorator | 2.0 | Weather tool |
| 9.4 | Tool Calling Protocol (OpenAI function calling) | 1.5 | Multi-tool caller |
| 9.5 | MCP — Model Context Protocol | 1.0 | MCP overview |

**Milestone:** You can create custom tools and let LLMs use them autonomously.

---

## Phase 10 — Agents (Week 6-7, ~10 hours)

> **Why:** Agents = LLMs that DECIDE which tools to use and in what order. This is where AI gets powerful.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 10.1 | What is an Agent? (ReAct, Planning, Execution) | 2.0 | — |
| 10.2 | AgentExecutor (Legacy but important) | 2.0 | Calculator agent |
| 10.3 | Tool-Calling Agents | 2.0 | Research agent |
| 10.4 | Agent with Memory | 2.0 | Conversational agent |
| 10.5 | Error Handling & Agent Debugging | 2.0 | Robust agent patterns |

**Milestone:** You can build autonomous agents that reason, plan, and act.

---

## Phase 11 — RAG Fundamentals (Week 7-8, ~10 hours)

> **Why:** RAG = Retrieval Augmented Generation. The #1 enterprise AI pattern. This is where jobs are.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 11.1 | RAG Architecture (Indexing → Retrieval → Generation) | 2.0 | — |
| 11.2 | Document Loaders (PDF, CSV, Web, YouTube) | 2.0 | Multi-source loader |
| 11.3 | Text Splitters (Recursive, Semantic, Token-based) | 2.0 | Chunking analyzer |
| 11.4 | Retrievers & RAG Chain | 2.0 | **📦 PDF Chat App** |
| 11.5 | Evaluation & Quality Metrics | 2.0 | RAG evaluator |

**Milestone:** You can build a complete PDF/document chat application.

---

## Phase 12 — Advanced RAG (Week 8, ~10 hours)

> **Why:** Basic RAG fails in production. Advanced techniques are what separate juniors from seniors.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 12.1 | Parent Document Retriever | 2.0 | Hierarchical retriever |
| 12.2 | Contextual Compression & Re-ranking | 2.0 | Compressed RAG |
| 12.3 | Hybrid Search (Dense + Sparse) | 2.0 | Hybrid search engine |
| 12.4 | Multi-Query & RAG Fusion | 2.0 | Advanced research bot |
| 12.5 | Conversational RAG (Chat + Retrieval) | 2.0 | **📦 Enterprise Document Chat** |

**Milestone:** You can build production-grade RAG that handles edge cases.

---

## Phase 13 — LangGraph Fundamentals (Week 9, ~8 hours)

> **Why:** LangGraph replaces AgentExecutor. It gives you full control over agent workflows with graphs.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 13.1 | Why LangGraph? (Problems with AgentExecutor) | 1.5 | — |
| 13.2 | StateGraph, START, END | 2.0 | Simple graph |
| 13.3 | State & Reducers | 2.0 | Stateful workflow |
| 13.4 | Nodes & Edges | 2.5 | **📦 Customer Support Router** |

**Milestone:** You can build stateful, graph-based AI workflows.

---

## Phase 14 — LangGraph Routing & Control Flow (Week 9, ~8 hours)

> **Why:** Real agents need conditional logic, loops, and parallel execution.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 14.1 | Conditional Edges & Dynamic Routing | 2.0 | Intent router |
| 14.2 | Cycles & Loops (Iterative Agents) | 2.0 | Self-correcting agent |
| 14.3 | Parallel Branches | 2.0 | Parallel research agent |
| 14.4 | Error Handling, Retries, Recovery | 2.0 | Fault-tolerant graph |

**Milestone:** You can build complex agent workflows with branching, loops, and error recovery.

---

## Phase 15 — LangGraph Persistence & Memory (Week 10, ~8 hours)

> **Why:** Production agents need to save state, resume, and remember across sessions.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 15.1 | Checkpointing (SQLite, PostgreSQL) | 2.0 | Persistent agent |
| 15.2 | Interrupt & Resume (Human-in-the-Loop) | 2.0 | Approval workflow |
| 15.3 | Long-term Memory (Cross-session) | 2.0 | Learning agent |
| 15.4 | Streaming in LangGraph | 2.0 | Real-time streaming agent |

**Milestone:** You can build agents that pause, wait for humans, and remember everything.

---

## Phase 16 — Multi-Agent Systems (Week 10-11, ~12 hours)

> **Why:** Complex tasks need multiple specialized agents working together. This is the cutting edge.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 16.1 | Supervisor Architecture | 2.0 | Supervisor agent |
| 16.2 | Swarm Architecture | 2.0 | Agent swarm |
| 16.3 | Hierarchical Multi-Agent | 2.0 | Research team |
| 16.4 | Agent Communication & Handoff | 2.0 | Collaborative agents |
| 16.5 | Reflection & Planning Patterns | 2.0 | Self-improving agent |
| 16.6 | **📦 Multi-Agent Research System** | 2.0 | Full project |

**Milestone:** You can architect and build multi-agent systems for complex real-world tasks.

---

## Phase 17 — Production & Deployment (Week 11, ~10 hours)

> **Why:** Building locally is one thing. Deploying reliably is what companies pay for.

| Chapter | Topic | Hours | Project |
|---------|-------|-------|---------|
| 17.1 | FastAPI + LangChain/LangGraph Integration | 2.0 | API server |
| 17.2 | Streaming Responses (SSE, WebSockets) | 2.0 | Streaming API |
| 17.3 | Observability (LangSmith, Callbacks, Logging) | 2.0 | Monitored app |
| 17.4 | Cost Optimization & Caching | 2.0 | Cached LLM system |
| 17.5 | Docker + Docker Compose Deployment | 2.0 | Containerized app |

**Milestone:** You can deploy, monitor, and optimize AI applications in production.

---

## Phase 18 — Capstone Project (Week 11-12, ~20 hours)

> **Why:** Everything comes together. This becomes your portfolio piece and proves you're job-ready.

### 🏗️ Project: Enterprise AI Platform

A production-grade AI platform that includes:

| Component | Technology |
|-----------|-----------|
| **Backend** | FastAPI |
| **AI Framework** | LangChain + LangGraph |
| **Multi-Agent** | Supervisor + Specialized Agents |
| **RAG** | Advanced RAG with hybrid search |
| **Database** | PostgreSQL |
| **Vector Store** | ChromaDB / FAISS |
| **Cache** | Redis |
| **Memory** | Long-term + Short-term |
| **Auth** | JWT Authentication |
| **Streaming** | Server-Sent Events |
| **Observability** | LangSmith + Custom logging |
| **Deployment** | Docker + Docker Compose |
| **Testing** | Pytest |
| **CI/CD** | GitHub Actions |

**Milestone:** You have a production-grade AI platform in your portfolio.

---

## 📅 Weekly Schedule Overview

| Week | Phase | Focus | Key Deliverable |
|------|-------|-------|-----------------|
| **1** | 0-1 | Python + LLM Fundamentals | First API call to OpenAI |
| **2** | 1-2 | LLM Deep Dive + Prompt Engineering | Structured data extractor |
| **3** | 3-4 | Embeddings + Vector Databases | Semantic search engine |
| **4** | 5-6 | LangChain Core + Chains | Blog writer chain |
| **5** | 7-8 | Parsers + Memory | Chatbot with persistent memory |
| **6** | 9-10 | Tools + Agents | Autonomous research agent |
| **7** | 11 | RAG Fundamentals | PDF Chat App |
| **8** | 12 | Advanced RAG | Enterprise Document Chat |
| **9** | 13-14 | LangGraph Core + Routing | Customer Support Router |
| **10** | 15-16 | Persistence + Multi-Agent | Multi-Agent Research System |
| **11** | 17 | Production + Deployment | Deployed AI API |
| **12** | 18 | Capstone Project | Enterprise AI Platform |

---

## 📚 GitHub Pages Course Structure

Every completed chapter will produce a publication-ready markdown file:

```
langchain-mastery/
├── README.md
├── docs/
│   ├── phase-00-python-powerup/
│   │   ├── chapter-00-args-kwargs.md
│   │   ├── chapter-01-context-managers.md
│   │   ├── chapter-02-generators.md
│   │   ├── chapter-03-type-hints-pydantic.md
│   │   ├── chapter-04-async-python.md
│   │   └── chapter-05-oop-patterns.md
│   ├── phase-01-llm-fundamentals/
│   │   ├── chapter-06-what-is-an-llm.md
│   │   ├── chapter-07-tokens-tokenization.md
│   │   ├── chapter-08-first-api-call.md
│   │   └── chapter-09-model-parameters.md
│   ├── phase-02-prompt-engineering/
│   │   └── ...
│   └── ...
├── projects/
│   ├── 01-calculator-agent/
│   ├── 02-wikipedia-agent/
│   ├── 03-pdf-chat/
│   └── ...
└── capstone/
    └── enterprise-ai-platform/
```

---

## ⚠️ Important Notes

> [!IMPORTANT]
> **API Keys Required:** You will need API keys for OpenAI (or Gemini/Anthropic). I'll guide you through setup. Free tiers and local alternatives (Ollama) will be used wherever possible to minimize cost.

> [!NOTE]
> **Adaptive Roadmap:** This roadmap is a living document. If a topic needs more time, we adjust. If you grasp something quickly, we accelerate. The goal is mastery, not speed.

> [!TIP]
> **Interview Prep is Built-in:** Every chapter includes interview questions. By Phase 18, you'll have covered 200+ interview questions across all difficulty levels.

---

## Open Questions

1. **Which LLM provider do you want to use primarily?** OpenAI (GPT-4), Google (Gemini), Anthropic (Claude), or local (Ollama)? This affects setup in Phase 1.

2. **Do you already have any API keys?** (OpenAI, Google AI Studio, etc.)

3. **Which code editor do you use?** (VS Code, PyCharm, etc.)

4. **Do you want to set up the GitHub Pages repository now or after a few chapters?**

---

**Ready to begin? Once you approve this roadmap, we start Phase 0, Chapter 0.1: `*args` and `**kwargs` — your first step toward becoming an AI Engineer.** 🚀

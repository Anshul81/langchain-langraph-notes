# Chapter 13.3: Capstone Project — Full-Stack AI Assistant

> **Phase 13 — Production** | [← Previous: LangSmith](chapter-47-langsmith.md) | [Course Complete! 🎓](#congratulations)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Build a **full-stack AI assistant** that combines everything you've learned
- ✅ Implement RAG + Agents + Tools + Memory in one system
- ✅ Create a conversational knowledge base with tool augmentation
- ✅ Add production features: error handling, logging, cost tracking
- ✅ Deploy as a working API with FastAPI
- ✅ Evaluate and iterate on your system

| | |
|---|---|
| **Prerequisites** | All previous chapters |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 2-3 hours |

---

## Introduction — Bringing It All Together

This capstone project combines **every major concept** from the course into one application:

```
COURSE CONCEPTS USED:
├── Phase 1-2:  LLM fundamentals + prompt engineering
├── Phase 3-4:  LangChain core + LCEL chains
├── Phase 5:    Conversation memory
├── Phase 6-7:  Embeddings + vector databases
├── Phase 8-9:  Tools + tool calling + MCP
├── Phase 10:   Agents (ReAct, LangGraph)
├── Phase 11:   RAG (load, split, embed, retrieve, generate)
├── Phase 12:   Advanced RAG (hybrid search, evaluation)
└── Phase 13:   Production (error handling, deployment, monitoring)
```

### What You'll Build

**An AI-powered company assistant** that can:
1. 📚 **Answer questions** from a knowledge base (RAG)
2. 🔧 **Use tools** for calculations, lookups, and actions
3. 💬 **Remember conversations** across multiple turns
4. 🧠 **Reason and decide** which tools to use (Agent)
5. 🛡️ **Handle errors gracefully** with fallbacks
6. 📊 **Track usage** and costs

---

## Part 1: Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   AI Company Assistant                        │
│                                                              │
│  ┌──────────┐                                                │
│  │   USER   │                                                │
│  └────┬─────┘                                                │
│       │                                                      │
│       ▼                                                      │
│  ┌──────────┐     ┌───────────────────────────────────────┐  │
│  │ FastAPI  │────→│         LangGraph Agent               │  │
│  │ /ask     │     │                                       │  │
│  └──────────┘     │  ┌─────────────────┐                  │  │
│                   │  │  Agent Node     │                  │  │
│                   │  │  (GPT-4o-mini)  │                  │  │
│                   │  │  Decides: RAG?  │                  │  │
│                   │  │  Tool? Answer?  │                  │  │
│                   │  └───────┬─────────┘                  │  │
│                   │          │                             │  │
│                   │    ┌─────┴──────┐                     │  │
│                   │    ▼            ▼                     │  │
│                   │  ┌──────┐  ┌────────┐                │  │
│                   │  │ RAG  │  │ Tools  │                │  │
│                   │  │ Tool │  │        │                │  │
│                   │  │      │  │ calc   │                │  │
│                   │  │Vector│  │ lookup │                │  │
│                   │  │Search│  │ time   │                │  │
│                   │  └──────┘  └────────┘                │  │
│                   │                                       │  │
│                   │  ┌─────────────────┐                  │  │
│                   │  │    Memory       │                  │  │
│                   │  │  (MemorySaver)  │                  │  │
│                   │  └─────────────────┘                  │  │
│                   └───────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Infrastructure                                        │  │
│  │  • ChromaDB (vector store)                             │  │
│  │  • Logging (JSON structured)                           │  │
│  │  • Cost tracking                                       │  │
│  │  • Error handling + fallbacks                          │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 2: Complete Implementation

### Project Setup

```bash
mkdir ai-assistant && cd ai-assistant
pip install langchain langchain-openai langchain-chroma langgraph
pip install fastapi uvicorn python-dotenv pydantic-settings
pip install tiktoken chromadb
```

### `config.py` — Configuration

```python
# config.py
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API
    openai_api_key: str
    openai_api_base: str = "https://api.openai.com/v1"
    
    # Model
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    
    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 5
    chroma_dir: str = "./chroma_db"
    collection_name: str = "company_kb"
    
    # Budget
    daily_budget: float = 10.0
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()
```

### `knowledge_base.py` — Document Ingestion & RAG

```python
# knowledge_base.py
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import get_settings

load_dotenv()


class KnowledgeBase:
    """Manages the company knowledge base."""
    
    def __init__(self):
        settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.vectorstore = None
        self.settings = settings
    
    def ingest(self, documents: list[Document]):
        """Ingest documents into the knowledge base."""
        chunks = self.splitter.split_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.settings.collection_name,
            persist_directory=self.settings.chroma_dir,
        )
        
        return len(chunks)
    
    def load_existing(self):
        """Load existing knowledge base."""
        self.vectorstore = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.settings.chroma_dir,
        )
    
    def search(self, query: str, k: int = None) -> list[Document]:
        """Search the knowledge base."""
        if not self.vectorstore:
            return []
        
        k = k or self.settings.retrieval_k
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_retriever(self):
        """Get a retriever for use with chains."""
        if not self.vectorstore:
            raise ValueError("Knowledge base not initialized")
        
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.settings.retrieval_k, "fetch_k": 10}
        )


# --- Company Data ---
COMPANY_DOCUMENTS = [
    Document(
        page_content="""CodeAssist - Product Overview
CodeAssist is an AI-powered code review tool built for modern development teams. It analyzes pull requests in real-time, providing instant feedback on code quality, security vulnerabilities, and best practices.

Key Features:
- Automated code review for Python, JavaScript, TypeScript, Java, and Go
- Security vulnerability detection (SAST)
- Style guide enforcement customizable per team
- Integration with GitHub, GitLab, and Bitbucket
- AI-powered suggestions with explanations

Pricing:
- Starter: $29/month per developer (up to 10 developers)
- Professional: $49/month per developer (up to 50 developers, priority support)
- Enterprise: Custom pricing (unlimited, SSO, dedicated support, on-premises)
All plans include a 14-day free trial.""",
        metadata={"source": "products/codeassist.md", "category": "product"}
    ),
    Document(
        page_content="""Company History
TechAssist was founded in 2020 by Aarav Patel in Bangalore, India. Starting as a 3-person startup, the company has grown to 45 employees.

Key Milestones:
- 2020 Q1: Founded, initial prototype
- 2021 Q1: Seed funding ₹3 crore from Indian Angel Network
- 2021 Q3: CodeAssist v1.0 launched
- 2022 Q3: Series A ₹25 crore from Sequoia India
- 2024 Q1: 45 employees, international expansion
- 2024 Q3: Revenue ₹4.5 crore/quarter, ARR ₹18 crore

Mission: "Make every developer's code better, automatically."
Headquarters: Koramangala, Bangalore""",
        metadata={"source": "about/history.md", "category": "company"}
    ),
    Document(
        page_content="""Team & Tech Stack
45 employees: Engineering (25), Sales & Marketing (10), Operations (5), Leadership (5).

Tech Stack:
- Backend: Python (FastAPI), Go
- Frontend: React, TypeScript
- ML/AI: PyTorch, LangChain, Hugging Face
- Database: PostgreSQL, Redis, ChromaDB
- Infrastructure: AWS (ap-south-1, us-east-1), Kubernetes
- CI/CD: GitHub Actions, ArgoCD""",
        metadata={"source": "team/overview.md", "category": "team"}
    ),
    Document(
        page_content="""Q3 2024 Financial Highlights
Revenue: ₹4.5 crore ($540K USD), 35% QoQ growth, 120% YoY growth.
ARR: ₹18 crore ($2.16M USD).
Revenue Breakdown: SaaS 70%, Consulting 20%, Training 10%.
Paying Customers: 280 total, 15 enterprise.
Top Enterprise: TechCorp (500 seats), DataFlow (200), BuildFast (150).
Net Revenue Retention: 135%. Monthly Churn: 2.1%.""",
        metadata={"source": "finance/q3-2024.md", "category": "finance"}
    ),
    Document(
        page_content="""Support & Billing Policies
Support Hours: 9 AM - 6 PM IST Mon-Fri (Standard). 24/7 (Enterprise).
SLA: Critical 1h (Enterprise), 4h (Pro), 24h (Starter).

Refund Policy:
- 14-day free trial, no charges
- Monthly: Cancel anytime, access until period ends, no partial refunds
- Annual: Full refund within 30 days. After 30 days, credit only.
- Enterprise: Per contract terms.

Security: SOC 2 Type II, GDPR compliant, AES-256 encryption, 99.9% uptime SLA.""",
        metadata={"source": "policies/support-billing.md", "category": "policy"}
    ),
]
```

### `assistant.py` — The AI Agent

```python
# assistant.py
import os
import math
import logging
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from knowledge_base import KnowledgeBase, COMPANY_DOCUMENTS
from config import get_settings

load_dotenv()
logger = logging.getLogger("assistant")


class AIAssistant:
    """Full-stack AI assistant combining RAG + Agents + Tools + Memory."""
    
    def __init__(self):
        settings = get_settings()
        
        # LLM
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=settings.temperature,
            openai_api_key=settings.openai_api_key,
            openai_api_base=settings.openai_api_base,
            max_retries=3,
            request_timeout=30,
        )
        
        # Knowledge base
        self.kb = KnowledgeBase()
        self._init_kb()
        
        # Tools
        self.tools = self._create_tools()
        
        # Memory
        self.memory = MemorySaver()
        
        # Agent
        self.agent = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=self.memory,
            state_modifier=SystemMessage(content=self._system_prompt()),
        )
        
        # Stats
        self.query_count = 0
        self.start_time = datetime.now()
        
        logger.info("AI Assistant initialized")
    
    def _init_kb(self):
        """Initialize the knowledge base."""
        try:
            self.kb.load_existing()
            count = self.kb.vectorstore._collection.count()
            if count == 0:
                raise ValueError("Empty KB")
            logger.info(f"Loaded existing KB: {count} chunks")
        except Exception:
            chunks = self.kb.ingest(COMPANY_DOCUMENTS)
            logger.info(f"Created new KB: {chunks} chunks")
    
    def _system_prompt(self) -> str:
        return f"""You are a helpful AI assistant for TechAssist, the company behind CodeAssist.

Today's date: {datetime.now().strftime('%Y-%m-%d')}

You have access to these tools:
1. **search_knowledge_base**: Search the company knowledge base for information about products, policies, team, finances, etc. Use this for ANY question about the company.
2. **calculate**: Do math calculations, conversions, and comparisons.
3. **get_current_time**: Get the current date and time.

Guidelines:
- ALWAYS search the knowledge base before answering company questions
- Be specific — include numbers, dates, and details
- If info isn't in the knowledge base, say "I don't have that information"
- Be friendly, professional, and concise
- When mentioning pricing, always mention the 14-day free trial
- For follow-up questions, use conversation context
"""
    
    def _create_tools(self):
        """Create the tools for the agent."""
        kb = self.kb  # Capture reference
        
        @tool
        def search_knowledge_base(query: str) -> str:
            """Search the company knowledge base for information.
            
            Use for questions about: products, pricing, company history, team, 
            financials, policies, support, security, or anything company-related.
            
            Args:
                query: The search query describing what you're looking for
            """
            try:
                docs = kb.search(query, k=5)
                if not docs:
                    return "No relevant information found in the knowledge base."
                
                results = []
                for doc in docs:
                    source = doc.metadata.get("source", "unknown")
                    results.append(f"[{source}]\n{doc.page_content}")
                
                return "\n\n---\n\n".join(results)
            except Exception as e:
                return f"Error searching knowledge base: {str(e)}"
        
        @tool
        def calculate(expression: str) -> str:
            """Evaluate a mathematical expression.
            
            Use for: arithmetic, percentages, comparisons, unit conversions.
            
            Args:
                expression: A Python math expression, e.g., '4500000 * 0.7', 'math.sqrt(144)'
            """
            try:
                result = eval(expression, {"__builtins__": {}, "math": math})
                return f"{expression} = {result}"
            except Exception as e:
                return f"Error: {str(e)}"
        
        @tool
        def get_current_time() -> str:
            """Get the current date and time. Use when the user asks about today's date or time."""
            now = datetime.now()
            return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"
        
        return [search_knowledge_base, calculate, get_current_time]
    
    def ask(self, question: str, thread_id: str = "default") -> dict:
        """Ask the assistant a question."""
        self.query_count += 1
        start = datetime.now()
        
        try:
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 15
            }
            
            # Invoke agent
            result = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config
            )
            
            answer = result["messages"][-1].content
            duration = (datetime.now() - start).total_seconds()
            
            # Count tool calls
            tool_calls = sum(
                len(msg.tool_calls) 
                for msg in result["messages"] 
                if hasattr(msg, 'tool_calls') and msg.tool_calls
            )
            
            logger.info(
                f"Query #{self.query_count} | "
                f"Duration: {duration:.1f}s | "
                f"Tools: {tool_calls} | "
                f"Thread: {thread_id} | "
                f"Q: {question[:60]}..."
            )
            
            return {
                "answer": answer,
                "duration": round(duration, 2),
                "tool_calls": tool_calls,
                "thread_id": thread_id,
            }
        
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            logger.error(f"Error on query #{self.query_count}: {e}")
            
            return {
                "answer": "I'm sorry, I encountered an error processing your request. Please try again.",
                "duration": round(duration, 2),
                "tool_calls": 0,
                "thread_id": thread_id,
                "error": str(e),
            }
    
    def get_stats(self) -> dict:
        """Get assistant statistics."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "queries_handled": self.query_count,
            "uptime_seconds": round(uptime, 0),
            "kb_chunks": self.kb.vectorstore._collection.count() if self.kb.vectorstore else 0,
        }


# --- Interactive Demo ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    
    assistant = AIAssistant()
    
    print("\n" + "="*60)
    print("🤖 AI Company Assistant — Interactive Demo")
    print("="*60)
    
    demo_conversations = [
        # Conversation 1: Product questions
        ("user-1", [
            "What product does TechAssist make?",
            "How much does it cost for a team of 20?",
            "Do they have a free trial?",
        ]),
        # Conversation 2: Finance questions
        ("user-2", [
            "What was Q3 2024 revenue?",
            "What percentage comes from SaaS?",
            "Calculate the SaaS revenue amount",
        ]),
        # Conversation 3: Mixed questions
        ("user-3", [
            "Who founded the company and when?",
            "How many employees do they have now?",
            "What's the refund policy for annual plans?",
            "What time is it?",
        ]),
    ]
    
    for thread_id, questions in demo_conversations:
        print(f"\n{'─'*60}")
        print(f"👤 Conversation: {thread_id}")
        print(f"{'─'*60}")
        
        for q in questions:
            result = assistant.ask(q, thread_id=thread_id)
            print(f"\n  ❓ {q}")
            print(f"  💬 {result['answer']}")
            print(f"     ⏱ {result['duration']}s | 🔧 {result['tool_calls']} tools")
    
    print(f"\n{'='*60}")
    print(f"📊 Stats: {assistant.get_stats()}")
```

### `api.py` — FastAPI Service

```python
# api.py
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from assistant import AIAssistant

# --- Globals ---
assistant = None
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global assistant
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")
    assistant = AIAssistant()
    logger.info("API started")
    yield
    logger.info("API stopped")


app = FastAPI(
    title="AI Company Assistant API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --- Models ---
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field(default="default", max_length=100)

class AskResponse(BaseModel):
    answer: str
    duration: float
    tool_calls: int
    thread_id: str
    timestamp: str


# --- Endpoints ---
@app.get("/health")
async def health():
    stats = assistant.get_stats() if assistant else {}
    return {"status": "healthy", "stats": stats}

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not assistant:
        raise HTTPException(500, "Assistant not initialized")
    
    result = assistant.ask(request.question, request.thread_id)
    
    if "error" in result:
        raise HTTPException(500, result["error"])
    
    return AskResponse(
        answer=result["answer"],
        duration=result["duration"],
        tool_calls=result["tool_calls"],
        thread_id=result["thread_id"],
        timestamp=datetime.now().isoformat(),
    )

# Run: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Part 3: Testing Your Assistant

```python
# test_assistant.py
"""Quick test suite for the AI Assistant."""

from assistant import AIAssistant


def test_assistant():
    assistant = AIAssistant()
    
    test_cases = [
        {
            "question": "What is CodeAssist?",
            "should_contain": ["code review", "AI"],
            "category": "product"
        },
        {
            "question": "How much does the Starter plan cost?",
            "should_contain": ["29", "month"],
            "category": "pricing"
        },
        {
            "question": "When was the company founded?",
            "should_contain": ["2020"],
            "category": "company"
        },
        {
            "question": "What is the refund policy?",
            "should_contain": ["30 days", "refund"],
            "category": "policy"
        },
        {
            "question": "Calculate 4500000 * 0.7",
            "should_contain": ["3150000"],
            "category": "calculation"
        },
        {
            "question": "What's the weather in Mumbai?",
            "should_contain": ["don't have", "not"],
            "category": "out-of-scope"
        },
    ]
    
    passed = 0
    failed = 0
    
    for tc in test_cases:
        result = assistant.ask(tc["question"])
        answer_lower = result["answer"].lower()
        
        found = any(keyword.lower() in answer_lower for keyword in tc["should_contain"])
        status = "✅" if found else "❌"
        
        if found:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} [{tc['category']}] {tc['question']}")
        if not found:
            print(f"   Expected keywords: {tc['should_contain']}")
            print(f"   Got: {result['answer'][:150]}...")
    
    print(f"\n📊 Results: {passed}/{passed+failed} passed ({passed/(passed+failed)*100:.0f}%)")


if __name__ == "__main__":
    test_assistant()
```

---

## Part 4: Evaluation Checklist

Before considering your capstone complete, verify:

```
FUNCTIONALITY:
☐ Agent correctly answers company questions using RAG
☐ Agent uses calculator for math questions
☐ Agent uses time tool when asked about the date/time
☐ Agent says "I don't know" for out-of-scope questions
☐ Agent remembers context in multi-turn conversations
☐ Different thread_ids have separate memories

QUALITY:
☐ Answers are accurate and specific (include numbers/dates)
☐ Sources are used correctly (no hallucination)
☐ Follow-up questions work ("How much does IT cost?" → uses context)
☐ At least 80% of test cases pass

PRODUCTION READINESS:
☐ Error handling — doesn't crash on bad input
☐ Timeouts — doesn't hang forever
☐ Logging — all queries and errors logged
☐ Health check — /health endpoint works
☐ Input validation — rejects empty/too-long queries
```

---

## Part 5: Extension Ideas

Once your capstone is working, try these extensions:

| Extension | Difficulty | What to Build |
|-----------|-----------|---------------|
| **Web UI** | Easy | Streamlit or Gradio frontend for the chatbot |
| **More data sources** | Easy | Load PDFs, web pages into the knowledge base |
| **Human-in-the-loop** | Medium | Approval flow before certain actions |
| **Multi-model routing** | Medium | GPT-4o for complex, GPT-4o-mini for simple |
| **User authentication** | Medium | JWT auth, per-user conversation history |
| **Streaming responses** | Medium | Server-Sent Events for real-time typing |
| **MCP integration** | Medium | Connect to GitHub/Slack MCP servers |
| **Agentic RAG** | Hard | Agent evaluates retrieval quality and re-searches |
| **Multi-agent system** | Hard | Specialized agents for support, sales, engineering |
| **LangSmith integration** | Medium | Tracing, evaluation datasets, monitoring |

---

## Congratulations! 🎓

You've completed the **LangChain Mastery** course!

### What You've Learned

```
Phase 0:   Python fundamentals for AI development
Phase 1:   LLM fundamentals — tokens, APIs, models
Phase 2:   Prompt engineering — templates, few-shot, chain-of-thought
Phase 3:   LangChain core — models, messages, output parsers
Phase 4:   Chains & LCEL — composing components with the pipe operator
Phase 5:   Memory — conversation history, buffer, summary, window
Phase 6:   Embeddings — vector representations, similarity, models
Phase 7:   Vector databases — ChromaDB, FAISS, search strategies
Phase 8:   Tools — built-in tools, custom tools, tool calling, MCP
Phase 9:   Agents — ReAct, LangGraph, custom architectures, HITL
Phase 10:  RAG — document loading, splitting, retrieval, generation
Phase 11:  Advanced RAG — hybrid search, reranking, evaluation
Phase 12:  Production — deployment, LangSmith, monitoring
Phase 13:  Capstone — everything combined into one working system
```

### What's Next

```
YOUR JOURNEY CONTINUES:
├── 🔨 Build real projects — the best way to learn
├── 📚 Read the LangChain docs — they update frequently
├── 🌐 Explore the ecosystem — LangGraph, LangSmith, LangServe
├── 🤝 Join the community — Discord, GitHub Discussions
├── 📰 Stay current — LLM field moves FAST
├── 🎯 Specialize — pick one area (RAG, agents, evaluation) and go deep
└── 💼 Build your portfolio — contribute to open source, blog about your projects
```

### Key Resources

| Resource | URL |
|----------|-----|
| LangChain Docs | https://python.langchain.com/ |
| LangGraph Docs | https://langchain-ai.github.io/langgraph/ |
| LangSmith | https://smith.langchain.com/ |
| LangChain GitHub | https://github.com/langchain-ai/langchain |
| LangChain Discord | https://discord.gg/langchain |

---

> **You started as a beginner. You're now a LangChain practitioner.**
> 
> **Go build something amazing.** 🚀

---

> [← Previous: LangSmith](chapter-47-langsmith.md) | **Course Complete!** 🎓

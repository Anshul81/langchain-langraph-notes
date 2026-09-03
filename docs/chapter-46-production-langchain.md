# Chapter 13.1: Production LangChain — Deployment, Observability & Best Practices

> **Phase 13 — Production** | [← Previous: RAG Evaluation](../phase-11-advanced-rag/chapter-45-rag-evaluation.md) | [Next: LangSmith →](chapter-47-langsmith.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Structure a LangChain project for production readiness
- ✅ Implement proper error handling, retries, and fallbacks
- ✅ Add logging, tracing, and observability
- ✅ Manage API keys and configuration securely
- ✅ Implement rate limiting and cost controls
- ✅ Deploy a LangChain app as a FastAPI service
- ✅ Build a **production-ready RAG API** with health checks, logging, and error handling

| | |
|---|---|
| **Prerequisites** | All previous phases |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction — From Prototype to Production

Most LangChain tutorials show the happy path. Production is the **unhappy path**:

```
PROTOTYPE:
  llm.invoke("Hello")  →  "Hi there!"  ✅

PRODUCTION:
  llm.invoke("Hello")  →  API timeout  ❌
  llm.invoke("Hello")  →  Rate limited  ❌
  llm.invoke("Hello")  →  Invalid API key  ❌
  llm.invoke("Hello")  →  Model overloaded  ❌
  llm.invoke("Hello")  →  Response too long  ❌
  llm.invoke("Hello")  →  "Hi there!"  ✅ (finally!)
```

---

## Part 1: Project Structure

```
my_langchain_app/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py             # Configuration management
│   ├── models.py             # Pydantic request/response models
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── rag_chain.py      # RAG chain implementation
│   │   └── chat_chain.py     # Chat chain implementation
│   ├── tools/
│   │   ├── __init__.py
│   │   └── custom_tools.py   # Custom tool definitions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py        # Logging configuration
│   │   └── callbacks.py      # Custom callbacks for monitoring
│   └── vectorstore/
│       ├── __init__.py
│       └── manager.py        # Vector store management
├── tests/
│   ├── test_rag_chain.py
│   └── test_tools.py
├── data/                     # Document data for ingestion
├── .env                      # Environment variables (NEVER commit!)
├── .env.example              # Template for env vars
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Part 2: Configuration Management

```python
# app/config.py
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    openai_api_key: str
    openai_api_base: str = "https://api.openai.com/v1"
    
    # Model configuration
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1000
    
    # RAG configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 5
    
    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "knowledge_base"
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_minute: int = 100000
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "app.log"
    
    # App
    app_name: str = "LangChain RAG API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
```

---

## Part 3: Error Handling & Retries

### Retry with Fallback

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithFallbacks
from tenacity import retry, stop_after_attempt, wait_exponential


def create_llm_with_fallback(settings):
    """Create LLM with automatic retry and fallback."""
    
    # Primary model
    primary = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        max_retries=3,       # Built-in retry for transient errors
        request_timeout=30,  # 30-second timeout
    )
    
    # Fallback model (cheaper, always available)
    fallback = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        max_retries=3,
        request_timeout=30,
    )
    
    # Chain with fallback: try primary, fall back if it fails
    return primary.with_fallbacks([fallback])
```

### LCEL Error Handling

```python
from langchain_core.runnables import RunnableLambda

# Wrap any step with error handling
def safe_invoke(func):
    """Wrap a function with error handling."""
    async def wrapper(input_data):
        try:
            return await func(input_data)
        except Exception as e:
            return {"error": str(e), "input": str(input_data)[:200]}
    return wrapper

# Or use RunnableLambda with try/except
def safe_retriever(query: str):
    """Retrieve with error handling."""
    try:
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant documents found."
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        return f"Retrieval error: {str(e)}. Answering without context."
```

---

## Part 4: Logging & Observability

### Custom Callback Handler

```python
# app/utils/callbacks.py
import time
import logging
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage

logger = logging.getLogger("langchain_app")


class ProductionCallbackHandler(BaseCallbackHandler):
    """Callback handler for production monitoring."""
    
    def __init__(self):
        self.start_time = None
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        logger.info(f"LLM call started | Model: {serialized.get('name', 'unknown')}")
    
    def on_llm_end(self, response, **kwargs):
        duration = time.time() - self.start_time
        
        # Extract token usage
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            # Estimate cost (GPT-4o-mini pricing)
            cost = (prompt_tokens * 0.15 + completion_tokens * 0.6) / 1_000_000
            self.total_tokens += total_tokens
            self.total_cost += cost
            
            logger.info(
                f"LLM call completed | Duration: {duration:.2f}s | "
                f"Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens}) | "
                f"Cost: ${cost:.6f}"
            )
        else:
            logger.info(f"LLM call completed | Duration: {duration:.2f}s")
    
    def on_llm_error(self, error, **kwargs):
        logger.error(f"LLM error: {error}")
    
    def on_retriever_start(self, serialized, query, **kwargs):
        logger.info(f"Retrieval started | Query: {query[:100]}...")
    
    def on_retriever_end(self, documents, **kwargs):
        logger.info(f"Retrieved {len(documents)} documents")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        logger.info(f"Tool call: {serialized.get('name', 'unknown')} | Input: {input_str[:100]}...")
    
    def on_tool_end(self, output, **kwargs):
        logger.info(f"Tool result: {str(output)[:200]}...")
    
    def on_tool_error(self, error, **kwargs):
        logger.error(f"Tool error: {error}")


# Usage
callback = ProductionCallbackHandler()
result = llm.invoke("Hello", config={"callbacks": [callback]})
```

### Structured Logging Setup

```python
# app/utils/logging.py
import logging
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO", log_file: str = "app.log"):
    """Configure production logging."""
    
    # JSON formatter for file output
    json_formatter = JSONFormatter()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(json_formatter)
    
    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger("langchain_app")
```

---

## Part 5: Rate Limiting & Cost Controls

```python
import time
import asyncio
from collections import deque


class RateLimiter:
    """Token bucket rate limiter for API calls."""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_rpm = max_requests_per_minute
        self.request_times: deque = deque()
    
    async def acquire(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove requests older than 1 minute
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # Wait if at limit
        if len(self.request_times) >= self.max_rpm:
            wait_time = 60 - (now - self.request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self.request_times.append(time.time())


class CostTracker:
    """Track API costs and enforce budgets."""
    
    # Approximate costs per 1M tokens
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    }
    
    def __init__(self, daily_budget: float = 10.0):
        self.daily_budget = daily_budget
        self.daily_cost = 0.0
        self.last_reset = time.time()
    
    def track(self, model: str, input_tokens: int, output_tokens: int):
        """Track cost for an API call."""
        # Reset daily counter
        if time.time() - self.last_reset > 86400:
            self.daily_cost = 0.0
            self.last_reset = time.time()
        
        pricing = self.PRICING.get(model, {"input": 1.0, "output": 1.0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        self.daily_cost += cost
        
        return cost
    
    def check_budget(self) -> bool:
        """Check if within daily budget."""
        return self.daily_cost < self.daily_budget
    
    def get_remaining_budget(self) -> float:
        return max(0, self.daily_budget - self.daily_cost)
```

---

## Part 6: FastAPI Deployment

```python
# app/main.py
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime

from app.config import get_settings, Settings
from app.utils.logging import setup_logging
from app.utils.callbacks import ProductionCallbackHandler

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --- Globals ---
logger = None
vectorstore = None
rag_chain = None
cost_tracker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, cleanup on shutdown."""
    global logger, vectorstore, rag_chain, cost_tracker
    
    settings = get_settings()
    logger = setup_logging(settings.log_level, settings.log_file)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
        max_retries=3,
        request_timeout=30,
    )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )
    
    # Load vector store
    vectorstore = Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retrieval_k}
    )
    
    # Create RAG chain
    prompt = ChatPromptTemplate.from_template(
        """Answer based on the context. If unsure, say "I don't know."

Context: {context}
Question: {question}

Answer:"""
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    cost_tracker = CostTracker(daily_budget=50.0)
    
    logger.info("Application initialized successfully")
    
    yield  # App runs here
    
    # Cleanup
    logger.info("Shutting down...")


# --- FastAPI App ---
app = FastAPI(
    title="LangChain RAG API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models ---
class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to answer")

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = []
    latency_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    version: str
    vectorstore_docs: int
    daily_cost: float
    budget_remaining: float


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    doc_count = vectorstore._collection.count() if vectorstore else 0
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        vectorstore_docs=doc_count,
        daily_cost=round(cost_tracker.daily_cost, 4) if cost_tracker else 0,
        budget_remaining=round(cost_tracker.get_remaining_budget(), 2) if cost_tracker else 0,
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question against the knowledge base."""
    start_time = datetime.now()
    
    # Budget check
    if cost_tracker and not cost_tracker.check_budget():
        raise HTTPException(status_code=429, detail="Daily budget exceeded")
    
    try:
        # Invoke RAG chain
        callback = ProductionCallbackHandler()
        answer = rag_chain.invoke(
            request.question,
            config={"callbacks": [callback]}
        )
        
        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Question answered | Latency: {latency:.0f}ms | "
                    f"Q: {request.question[:80]}...")
        
        return AnswerResponse(
            answer=answer,
            sources=[],
            latency_ms=round(latency, 1),
            timestamp=datetime.now().isoformat(),
        )
    
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/ingest")
async def ingest_documents(texts: list[str]):
    """Ingest new documents into the knowledge base."""
    try:
        from langchain_core.documents import Document
        
        settings = get_settings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        docs = [Document(page_content=t) for t in texts]
        chunks = splitter.split_documents(docs)
        vectorstore.add_documents(chunks)
        
        logger.info(f"Ingested {len(texts)} documents ({len(chunks)} chunks)")
        
        return {"status": "success", "documents": len(texts), "chunks": len(chunks)}
    
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Part 7: Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.8"

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - ./chroma_db:/app/chroma_db  # Persist vector store
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
```

---

## Common Mistakes

### Mistake 1: Hardcoded API keys
```python
# ❌ NEVER do this
llm = ChatOpenAI(openai_api_key="sk-abc123...")

# ✅ Use environment variables
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
```

### Mistake 2: No error handling on LLM calls
```python
# ❌ Crashes on API timeout
answer = rag_chain.invoke(question)

# ✅ Handle errors gracefully
try:
    answer = rag_chain.invoke(question)
except Exception as e:
    answer = "I'm sorry, I encountered an error. Please try again."
    logger.error(f"RAG error: {e}")
```

### Mistake 3: No timeout on LLM calls
```python
# ❌ Waits forever if API hangs
llm = ChatOpenAI(model="gpt-4o-mini")

# ✅ Set explicit timeout
llm = ChatOpenAI(model="gpt-4o-mini", request_timeout=30)
```

### Mistake 4: No cost monitoring
```python
# ❌ Surprise $500 bill at end of month

# ✅ Track and limit spending
if cost_tracker.daily_cost > daily_budget:
    raise HTTPException(429, "Budget exceeded")
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use environment variables for secrets | Security — never hardcode API keys |
| Set `max_retries=3` and `request_timeout=30` | Handle transient API failures |
| Add fallback models | Resilience — don't crash if primary is unavailable |
| Log every LLM call with tokens and cost | Monitoring and cost control |
| Implement health check endpoints | DevOps — know when the service is healthy |
| Set daily budget limits | Prevent cost explosions |
| Use structured JSON logging | Easier to parse in log aggregators |
| Dockerize the application | Reproducible deployments |
| Add input validation (min/max length) | Prevent abuse and edge cases |
| Version your API | Backward compatibility |

---

## Interview Preparation

### Easy
**Q: What are the key considerations for deploying LangChain to production?**

> Key areas: (1) **Error handling** — retries, timeouts, and fallback models for API failures. (2) **Security** — API keys in environment variables, input validation, rate limiting. (3) **Observability** — logging every LLM call with latency, tokens, and cost. (4) **Cost control** — daily budgets, token tracking, using appropriate model sizes. (5) **Performance** — caching, async operations, connection pooling. (6) **Deployment** — Docker, health checks, graceful shutdown.

### Hard
**Q: Design a production RAG system that handles 1000 requests per minute.**

> Architecture: (1) **Load balancing** — multiple API server instances behind a load balancer. (2) **Async processing** — FastAPI with async LLM calls, non-blocking I/O. (3) **Caching** — Redis cache for frequent queries (80/20 rule: 20% of queries are repeated). (4) **Vector store** — managed service like Pinecone or Weaviate (not local ChromaDB). (5) **LLM** — connection pool with rate limiting per instance; fallback to cheaper model under load. (6) **Queue** — for non-real-time requests, use a message queue (Celery/RabbitMQ). (7) **Monitoring** — Prometheus/Grafana for metrics, alerts on latency P99 > 5s or error rate > 1%. (8) **Scaling** — auto-scale API servers based on request queue length. (9) **Cost** — at 1000 RPM, ~1.44M requests/day; use GPT-4o-mini ($0.15/1M input tokens) and cache aggressively.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Configuration** | Pydantic Settings for type-safe env var management |
| **Error handling** | Retries, timeouts, fallback models |
| **Callbacks** | Custom handlers for logging LLM calls |
| **Rate limiting** | Prevent exceeding API quotas |
| **Cost tracking** | Monitor spending, enforce budgets |
| **FastAPI deployment** | REST API with health checks |
| **Docker** | Containerized, reproducible deployment |

---

> [← Previous: RAG Evaluation](../phase-11-advanced-rag/chapter-45-rag-evaluation.md) | [Next: LangSmith →](chapter-47-langsmith.md)

# Chapter 11.4: Complete RAG Pipeline — From Documents to Answers

> **Phase 11 — RAG** | [← Previous: Text Splitting](chapter-42-text-splitting.md) | [Next: Phase 12 — Advanced RAG →](../phase-11-advanced-rag/chapter-44-advanced-retrieval.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Build a **complete, production-quality RAG system** end-to-end
- ✅ Combine document loading, splitting, embedding, storing, and retrieval
- ✅ Create a RAG chain using LCEL with proper prompt engineering
- ✅ Add conversation history (multi-turn RAG)
- ✅ Implement source citations in answers
- ✅ Build a **company knowledge base chatbot** as your capstone project

| | |
|---|---|
| **Prerequisites** | All previous chapters in Phase 11 |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction

You've learned the individual pieces. Now let's put them all together into a **complete, polished RAG application**:

```
Documents → Load → Split → Embed → Store → Retrieve → Generate → Answer
```

---

## Part 1: The Complete Pipeline

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ============================================================
# STEP 1: LLM & Embeddings
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

# ============================================================
# STEP 2: Load Documents
# ============================================================

# Company knowledge base documents
raw_documents = [
    Document(
        page_content="""CodeAssist - Product Overview

CodeAssist is an AI-powered code review tool built for modern development teams. It analyzes pull requests in real-time, providing instant feedback on code quality, security vulnerabilities, and best practices.

Key Features:
- Automated code review for Python, JavaScript, TypeScript, Java, and Go
- Security vulnerability detection using SAST (Static Application Security Testing)
- Style guide enforcement customizable per team
- Integration with GitHub, GitLab, and Bitbucket
- AI-powered suggestions with explanations, not just flags
- Dashboard with team analytics and improvement trends

Pricing:
- Starter: $29/month per developer (up to 10 developers)
- Professional: $49/month per developer (up to 50 developers, priority support)
- Enterprise: Custom pricing (unlimited developers, SSO, dedicated support, on-premises option)

All plans include a 14-day free trial with no credit card required.""",
        metadata={"source": "products/codeassist.md", "category": "product"}
    ),
    Document(
        page_content="""Company History

TechAssist was founded in 2020 by Aarav Patel in Bangalore, India. Starting as a 3-person team working from a co-working space, the company was born from Aarav's frustration with slow, manual code review processes at his previous job.

Timeline:
- 2020 Q1: Company founded, initial prototype built
- 2020 Q3: First beta users (15 teams)
- 2021 Q1: Seed funding of ₹3 crore from Indian Angel Network
- 2021 Q3: CodeAssist v1.0 launched publicly
- 2022 Q1: 100th paying customer milestone
- 2022 Q3: Series A funding of ₹25 crore from Sequoia India
- 2023 Q1: Team grew to 30 employees
- 2023 Q3: International expansion - first US customers
- 2024 Q1: Team reached 45 employees
- 2024 Q3: Revenue hit ₹4.5 crore/quarter (ARR: ₹18 crore)

Mission: "Make every developer's code better, automatically."
Vision: "A world where code quality is never a bottleneck."

Headquarters: Koramangala, Bangalore, India
US Office: San Francisco, California (3 employees)""",
        metadata={"source": "about/history.md", "category": "company"}
    ),
    Document(
        page_content="""Engineering Team & Tech Stack

Team Composition (45 total):
- Engineering: 25 (Frontend: 8, Backend: 10, ML/AI: 5, DevOps: 2)
- Sales & Marketing: 10
- Operations & HR: 5
- Leadership: 5

Engineering Tech Stack:
- Backend: Python (FastAPI), Go (high-performance services)
- Frontend: React, TypeScript
- ML/AI: Python, PyTorch, LangChain, Hugging Face Transformers
- Database: PostgreSQL (primary), Redis (caching), ChromaDB (vectors)
- Infrastructure: AWS (ap-south-1 and us-east-1), Docker, Kubernetes
- CI/CD: GitHub Actions, ArgoCD
- Monitoring: Datadog, PagerDuty

Development Practices:
- Agile/Scrum with 2-week sprints
- Code review required for all PRs (yes, we use our own product!)
- 90% test coverage target
- Weekly tech talks and knowledge sharing sessions
- Quarterly hackathons""",
        metadata={"source": "team/engineering.md", "category": "team"}
    ),
    Document(
        page_content="""Customer Support Policy

Support Channels:
- Email: support@techassist.io (all plans)
- Live chat: Available on Professional and Enterprise plans
- Phone: Enterprise plan only
- Slack: Enterprise plan gets a dedicated Slack channel

Support Hours:
- Standard: 9 AM - 6 PM IST, Monday through Friday
- Enterprise: 24/7 support with guaranteed 1-hour response time

Response Time SLAs:
- Critical (system down): 1 hour (Enterprise), 4 hours (Professional), 24 hours (Starter)
- High (feature broken): 4 hours (Enterprise), 8 hours (Professional), 48 hours (Starter)
- Medium (question): 8 hours (Enterprise), 24 hours (Professional), 72 hours (Starter)
- Low (feature request): 24 hours (Enterprise), 72 hours (Professional), 1 week (Starter)

Escalation Process:
1. Support agent attempts resolution
2. If unresolved in 2 hours, escalate to senior support
3. If unresolved in 4 hours, escalate to engineering team
4. Critical issues page the on-call engineer immediately""",
        metadata={"source": "policies/support.md", "category": "support"}
    ),
    Document(
        page_content="""Billing & Refund Policy

Payment Methods:
- Credit/Debit Cards (Visa, Mastercard, Amex)
- Bank Transfer / Wire Transfer (Enterprise only, net 30 terms)
- UPI (for Indian customers)

Billing Cycles:
- Monthly plans: Billed on the same date each month
- Annual plans: Billed once per year (20% discount vs monthly)

Refund Policy:
- 14-day free trial: No charges during trial. Cancel anytime.
- Monthly plans: Cancel anytime, access continues until end of billing period. No partial refunds.
- Annual plans: Full refund within 30 days of purchase. After 30 days, remaining months are credited toward future use. No cash refunds after 30 days.
- Enterprise contracts: Subject to contract terms. Contact your account manager.

How to cancel:
1. Go to Settings → Billing → Cancel Subscription
2. Or email billing@techassist.io
3. Confirmation sent within 24 hours""",
        metadata={"source": "policies/billing.md", "category": "billing"}
    ),
    Document(
        page_content="""Security & Compliance

Data Security:
- All data encrypted at rest (AES-256) and in transit (TLS 1.3)
- SOC 2 Type II certified
- GDPR compliant (data processing agreement available)
- No customer code is stored — we analyze in real-time and discard
- API keys and secrets are never logged or stored

Infrastructure Security:
- AWS with VPC isolation per customer (Enterprise)
- Regular penetration testing by third-party security firm
- Bug bounty program (report to security@techassist.io)
- 99.9% uptime SLA (99.99% for Enterprise)

Compliance:
- SOC 2 Type II
- GDPR
- ISO 27001 (in progress, expected Q2 2025)
- HIPAA (Enterprise only, available upon request)""",
        metadata={"source": "security/overview.md", "category": "security"}
    ),
    Document(
        page_content="""Q3 2024 Financial Highlights

Revenue:
- Q3 2024 Revenue: ₹4.5 crore ($540K USD)
- Year-over-year growth: 120%
- Quarter-over-quarter growth: 35%
- Annual Recurring Revenue (ARR): ₹18 crore ($2.16M USD)

Revenue Breakdown:
- SaaS subscriptions: 70% (₹3.15 crore)
- Consulting & implementation: 20% (₹0.9 crore)
- Training & workshops: 10% (₹0.45 crore)

Customer Metrics:
- Total paying customers: 280
- Enterprise customers: 15
- Average contract value (Enterprise): ₹45 lakhs/year
- Net revenue retention: 135%
- Monthly churn rate: 2.1%

Top Enterprise Customers:
- TechCorp: 500 developer seats
- DataFlow Inc: 200 developer seats
- BuildFast: 150 developer seats
- CloudNine Systems: 120 developer seats
- SecureCode Labs: 100 developer seats""",
        metadata={"source": "finance/q3-2024.md", "category": "finance"}
    ),
    Document(
        page_content="""Integration Guide

Supported Integrations:
1. GitHub: Full integration — PR reviews, issue creation, CI checks
2. GitLab: Full integration — MR reviews, pipeline integration
3. Bitbucket: Basic integration — PR reviews only
4. VS Code Extension: Real-time code analysis in editor
5. JetBrains Plugin: IntelliJ, PyCharm, WebStorm support
6. Slack: Notifications and daily digest
7. Jira: Create tickets from code review findings
8. API: RESTful API for custom integrations

Setup Steps (GitHub):
1. Install the CodeAssist GitHub App from the marketplace
2. Grant repository access (specific repos or all repos)
3. Configure rules in .codeassist.yaml in your repo root
4. Push your first PR — CodeAssist reviews automatically!

Configuration (.codeassist.yaml):
```yaml
version: 1
languages: [python, javascript]
severity_threshold: medium
auto_approve: false
exclude_paths: [tests/, docs/, vendor/]
custom_rules:
  - no-print-statements
  - require-type-hints
```""",
        metadata={"source": "docs/integrations.md", "category": "integration"}
    ),
]

print(f"📄 Loaded {len(raw_documents)} documents")

# ============================================================
# STEP 3: Split Documents
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(raw_documents)
print(f"✂️ Split into {len(chunks)} chunks")

# Inspect
for i, chunk in enumerate(chunks[:3]):
    print(f"\n  Chunk {i+1} ({len(chunk.page_content)} chars):")
    print(f"  Source: {chunk.metadata['source']}")
    print(f"  Content: {chunk.page_content[:100]}...")

# ============================================================
# STEP 4: Embed & Store
# ============================================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="company_knowledge",
    persist_directory="./chroma_company_db"
)

print(f"\n💾 Stored {vectorstore._collection.count()} chunks in ChromaDB")

# ============================================================
# STEP 5: Create Retriever
# ============================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",        # Maximum Marginal Relevance for diversity
    search_kwargs={
        "k": 5,               # Return top 5 chunks
        "fetch_k": 10,        # Fetch 10, then pick 5 most diverse
        "lambda_mult": 0.7    # 0=max diversity, 1=max relevance
    }
)

# ============================================================
# STEP 6: RAG Chain
# ============================================================

rag_prompt = ChatPromptTemplate.from_template("""You are a helpful assistant for TechAssist, the company behind CodeAssist.
Answer the question based ONLY on the following context. If the context doesn't contain enough information, say "I don't have that information in our knowledge base."

Be specific — include numbers, dates, and details when available.
If the user asks about pricing, always mention the 14-day free trial.

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs):
    """Format retrieved documents into a string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# TEST
# ============================================================

questions = [
    "What is CodeAssist and how much does it cost?",
    "When was the company founded and by whom?",
    "What programming languages does the engineering team use?",
    "What is the refund policy for annual plans?",
    "What was Q3 2024 revenue?",
    "How do I set up the GitHub integration?",
    "Is the company GDPR compliant?",
    "What are the support hours?",
    "Who are the largest enterprise customers?",
    "What's the weather in Mumbai?",  # Not in our data
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"❓ {q}")
    answer = rag_chain.invoke(q)
    print(f"💬 {answer}")
```

---

## Part 2: RAG with Source Citations

```python
from langchain_core.runnables import RunnableParallel

def rag_with_sources(question: str) -> dict:
    """RAG that returns the answer with source citations."""
    
    # Retrieve relevant documents
    docs = retriever.invoke(question)
    
    # Format context
    context = format_docs(docs)
    
    # Generate answer
    answer = (rag_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question
    })
    
    # Extract unique sources
    sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))
    
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "num_chunks_retrieved": len(docs)
    }


# Test
result = rag_with_sources("What is the refund policy?")
print(f"💬 {result['answer']}")
print(f"📚 Sources: {result['sources']}")
print(f"📄 Chunks retrieved: {result['num_chunks_retrieved']}")
```

---

## Part 3: Conversational RAG (Multi-Turn)

RAG that remembers previous conversation turns:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# Conversational RAG prompt
conversational_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant for TechAssist. 
Answer questions based on the provided context. 
Use conversation history for follow-up questions.
If you don't know, say so."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """Context:
{context}

Question: {question}"""),
])


class ConversationalRAG:
    """RAG with conversation memory."""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.chat_history: list = []
        
        self.chain = (
            conversational_rag_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def ask(self, question: str) -> str:
        """Ask a question with conversation context."""
        # Reformulate the question using chat history for better retrieval
        search_query = self._reformulate_query(question)
        
        # Retrieve
        docs = self.retriever.invoke(search_query)
        context = format_docs(docs)
        
        # Generate
        answer = self.chain.invoke({
            "context": context,
            "question": question,
            "chat_history": self.chat_history
        })
        
        # Update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        
        # Keep history manageable (last 10 exchanges)
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        return answer
    
    def _reformulate_query(self, question: str) -> str:
        """Use chat history to create a standalone search query."""
        if not self.chat_history:
            return question
        
        reformulate_prompt = ChatPromptTemplate.from_template(
            """Given the chat history and a follow-up question, reformulate the question 
to be a standalone search query (no pronouns, include full context).

Chat History:
{history}

Follow-up Question: {question}

Standalone Question:"""
        )
        
        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in self.chat_history[-6:]  # Last 3 exchanges
        )
        
        result = (reformulate_prompt | self.llm | StrOutputParser()).invoke({
            "history": history_text,
            "question": question
        })
        
        return result
    
    def reset(self):
        """Clear conversation history."""
        self.chat_history = []


# --- Demo ---
chatbot = ConversationalRAG(retriever, llm)

# Multi-turn conversation
print(chatbot.ask("What product does the company make?"))
print(f"\n{'='*60}")
print(chatbot.ask("How much does it cost?"))  # "it" → CodeAssist
print(f"\n{'='*60}")
print(chatbot.ask("Do they offer a free trial?"))
print(f"\n{'='*60}")
print(chatbot.ask("What about their refund policy?"))
print(f"\n{'='*60}")
print(chatbot.ask("When was the company founded?"))
```

---

## Part 4: Complete Capstone — Company Knowledge Base Chatbot

```python
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class KnowledgeBaseChatbot:
    """Complete RAG chatbot with conversation memory and source tracking."""
    
    def __init__(self, collection_name: str = "kb", persist_dir: str = "./kb_db"):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0,
            openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
            openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
            openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.retriever = None
        self.chat_history = []
        self.query_log = []
    
    def ingest(self, documents: list[Document]):
        """Ingest documents into the knowledge base."""
        print(f"📥 Ingesting {len(documents)} documents...")
        
        # Split
        chunks = self.splitter.split_documents(documents)
        print(f"✂️ Split into {len(chunks)} chunks")
        
        # Embed & store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
        )
        
        print(f"✅ Knowledge base ready with {self.vectorstore._collection.count()} chunks")
    
    def load_existing(self):
        """Load an existing knowledge base."""
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.7}
        )
        print(f"📂 Loaded knowledge base: {self.vectorstore._collection.count()} chunks")
    
    def ask(self, question: str) -> dict:
        """Ask a question and get an answer with sources."""
        if not self.retriever:
            return {"answer": "Knowledge base not initialized. Call ingest() first.", "sources": []}
        
        start_time = datetime.now()
        
        # Reformulate for better retrieval (if there's history)
        search_query = self._reformulate(question)
        
        # Retrieve
        docs = self.retriever.invoke(search_query)
        
        # Format context with source labels
        context = "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        )
        
        # Generate answer
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant. Answer based on the provided context.
If the context doesn't contain the answer, say "I don't have that information."
Be specific — include numbers, dates, and details. Keep answers concise but thorough."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])
        
        answer = (prompt | self.llm | StrOutputParser()).invoke({
            "context": context,
            "question": question,
            "chat_history": self.chat_history[-10:]  # Last 5 exchanges
        })
        
        # Update history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        
        # Collect sources
        sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))
        duration = (datetime.now() - start_time).total_seconds()
        
        # Log
        self.query_log.append({
            "question": question,
            "search_query": search_query,
            "num_docs": len(docs),
            "sources": sources,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "answer": answer,
            "sources": sources,
            "num_docs": len(docs),
            "duration": duration
        }
    
    def _reformulate(self, question: str) -> str:
        """Reformulate question using chat history."""
        if not self.chat_history:
            return question
        
        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content[:200]}"
            for m in self.chat_history[-6:]
        )
        
        result = self.llm.invoke(
            f"Rewrite as standalone search query:\nHistory:\n{history_text}\nQuestion: {question}\nStandalone query:"
        )
        return result.content.strip()
    
    def reset_conversation(self):
        """Clear conversation history."""
        self.chat_history = []
    
    def get_stats(self):
        """Get chatbot usage statistics."""
        if not self.query_log:
            return {"queries": 0}
        return {
            "total_queries": len(self.query_log),
            "avg_duration": sum(q["duration"] for q in self.query_log) / len(self.query_log),
            "avg_docs_retrieved": sum(q["num_docs"] for q in self.query_log) / len(self.query_log),
        }


# --- Demo ---
bot = KnowledgeBaseChatbot()

# Ingest the company documents (from Part 1)
bot.ingest(raw_documents)

# Interactive demo
conversations = [
    "What product does TechAssist make?",
    "How much does the Professional plan cost?",
    "Is there a free trial?",
    "What languages does CodeAssist support?",
    "Tell me about the engineering team",
    "What was the Q3 revenue?",
    "Are they SOC 2 compliant?",
    "How do I get support?",
]

for q in conversations:
    result = bot.ask(q)
    print(f"\n❓ {q}")
    print(f"💬 {result['answer']}")
    print(f"📚 Sources: {result['sources']} ({result['duration']:.1f}s)")

# Stats
print(f"\n📊 {json.dumps(bot.get_stats(), indent=2)}")
```

---

## Common Mistakes

### Mistake 1: Not testing retrieval quality separately
```python
# ❌ Only testing the final answer — can't tell if retrieval or generation is the problem

# ✅ Test retrieval independently
docs = retriever.invoke("What is the pricing?")
for doc in docs:
    print(f"[{doc.metadata['source']}] {doc.page_content[:100]}...")
# Ask: Are these the RIGHT documents? If not, fix chunking/embeddings.
```

### Mistake 2: Sending full conversation history for retrieval
```python
# ❌ Searching with "How much does it cost?" without context
docs = retriever.invoke("How much does it cost?")  # Cost of WHAT?

# ✅ Reformulate with context: "How much does CodeAssist cost?"
search_query = reformulate(question, chat_history)
docs = retriever.invoke(search_query)
```

### Mistake 3: Not handling the "I don't know" case
```python
# ❌ LLM hallucinates an answer when the context doesn't have it

# ✅ Explicitly tell the LLM to say "I don't know"
prompt = """Answer based ONLY on the context. 
If the answer is NOT in the context, say "I don't have that information."

Context: {context}
Question: {question}"""
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Test retrieval quality independently | Find the bottleneck (retrieval vs generation) |
| Use MMR retrieval for diversity | Avoid redundant similar chunks |
| Reformulate follow-up questions | Better retrieval for conversational RAG |
| Include "I don't know" in prompt | Reduce hallucination |
| Track sources in metadata | Enable citations |
| Persist the vector store | Don't re-embed on every restart |
| Log queries and sources | Debugging and analytics |
| Limit conversation history | Prevent token overflow |

---

## Interview Preparation

### Easy
**Q: Walk through the complete RAG pipeline.**

> The RAG pipeline has two phases: **Indexing** (done once): Load documents → Split into chunks (300-1000 tokens) → Embed chunks into vectors → Store in a vector database. **Querying** (per question): Embed the user's question → Search the vector DB for similar chunks → Format the top-k chunks as context → Send context + question to the LLM → Generate an answer grounded in the retrieved documents.

### Medium
**Q: How do you handle multi-turn conversations in RAG?**

> The key challenge is that follow-up questions ("How much does it cost?") lack context. Solution: **query reformulation** — before retrieval, use the LLM to rewrite the question as a standalone query using conversation history. "How much does it cost?" becomes "What is the pricing for CodeAssist?" This reformulated query is used for vector search, while the original question is used in the final prompt. Keep conversation history to the last 5-10 exchanges to avoid token overflow.

### Hard
**Q: You built a RAG system but answers are often wrong. How do you debug it?**

> Systematic debugging: (1) **Test retrieval separately** — for failing queries, check if the right chunks are being retrieved. If not, the problem is chunking, embeddings, or query formulation. (2) **Check chunk quality** — inspect chunks manually. Are they too large (noise), too small (no context), or split at bad boundaries? (3) **Check embedding quality** — try different embedding models or adjust chunk size. (4) **Check the prompt** — is the LLM being instructed to use only the context? Does it say "I don't know" when appropriate? (5) **Add reranking** — a cross-encoder reranker can dramatically improve which chunks reach the LLM. (6) **Try hybrid search** — combine vector search with keyword/BM25 search for better coverage. (7) **Evaluate quantitatively** — build a test set with question-answer pairs and measure retrieval precision, recall, and answer accuracy.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Complete RAG pipeline** | Load → Split → Embed → Store → Retrieve → Generate |
| **Source citations** | Track document sources in metadata for verifiability |
| **Conversational RAG** | Multi-turn with query reformulation for better retrieval |
| **MMR retrieval** | Diverse results, avoids redundant similar chunks |
| **Query reformulation** | Rewrite follow-ups as standalone queries |
| **"I don't know"** | Explicit prompt instruction to reduce hallucination |
| **Retrieval testing** | Debug retrieval independently from generation |

---

> [← Previous: Text Splitting](chapter-42-text-splitting.md) | [Next: Phase 12 — Advanced RAG →](../phase-11-advanced-rag/chapter-44-advanced-retrieval.md)

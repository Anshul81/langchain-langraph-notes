# Chapter 11.1: Introduction to RAG — Retrieval-Augmented Generation

> **Phase 11 — RAG** | [← Previous: Human-in-the-Loop](../phase-09-agents/chapter-39-human-in-the-loop.md) | [Next: Document Loaders →](chapter-41-document-loaders.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand what RAG is and why it's the most important LLM pattern
- ✅ Know the RAG pipeline: Load → Split → Embed → Store → Retrieve → Generate
- ✅ Understand when to use RAG vs fine-tuning vs prompt engineering
- ✅ Know the different types of RAG (naive, advanced, agentic)
- ✅ Build a **minimal working RAG** system from scratch

| | |
|---|---|
| **Prerequisites** | Phase 7 (Vector Databases), Phase 9 (Tools & Tool Calling) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 35 minutes |

---

## Introduction — The Most Important LLM Pattern

RAG is the **#1 most deployed LLM pattern in production**. If you learn one thing from this course, let it be RAG.

### The Problem

LLMs know what they were trained on — but they don't know **your data**:

```
❌ "What's our Q3 revenue?"        → LLM doesn't know your company's financials
❌ "Summarize the latest policy doc" → LLM doesn't have your documents
❌ "What did the customer say?"     → LLM doesn't have your tickets
❌ "Answer from our FAQ"            → LLM doesn't have your knowledge base
```

### The Solution: RAG

**Retrieval-Augmented Generation (RAG)** = "Look up relevant information first, then answer."

```
WITHOUT RAG:
User Question → LLM → Answer (from training data, often wrong/outdated)

WITH RAG:
User Question → Search Your Data → Find Relevant Docs → LLM + Docs → Accurate Answer
```

### The Analogy

```
LLM without RAG = Student taking an exam with NO notes
    → Relies on memory, might hallucinate answers

LLM with RAG = Student taking an OPEN-BOOK exam
    → Looks up the answer in relevant documents
    → Much more accurate and verifiable
```

---

## Part 1: How RAG Works

### The RAG Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
│                                                                  │
│  INDEXING (done once)                                            │
│  ─────────────────                                               │
│  ┌──────┐    ┌───────┐    ┌────────┐    ┌──────────────┐        │
│  │ LOAD │───→│ SPLIT │───→│ EMBED  │───→│    STORE     │        │
│  │      │    │       │    │        │    │              │        │
│  │ PDFs │    │ Into  │    │ Convert│    │ Vector       │        │
│  │ Docs │    │ chunks│    │ to     │    │ Database     │        │
│  │ Web  │    │       │    │ vectors│    │ (ChromaDB,   │        │
│  │ APIs │    │       │    │        │    │  FAISS, etc.)│        │
│  └──────┘    └───────┘    └────────┘    └──────┬───────┘        │
│                                                 │                │
│  QUERYING (every user question)                 │                │
│  ──────────────────────────────                 │                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐   │               │
│  │ QUESTION │───→│ RETRIEVE │───→│ GENERATE │   │               │
│  │          │    │          │    │          │   │               │
│  │ "What is │    │ Search   │◄───┘          │               │
│  │ our Q3   │    │ vector DB│    │ LLM +    │               │
│  │ revenue?"│    │ for top-k│    │ context  │               │
│  └──────────┘    │ relevant │    │ = answer │               │
│                  │ chunks   │    └──────────┘               │
│                  └──────────┘                                │
└──────────────────────────────────────────────────────────────────┘
```

### The Six Steps

| Step | What | Tools |
|------|------|-------|
| **1. Load** | Import documents from various sources | Document loaders (PDF, Web, DB) |
| **2. Split** | Break documents into smaller chunks | Text splitters (RecursiveCharacter, etc.) |
| **3. Embed** | Convert text chunks to vectors | Embedding models (OpenAI, Sentence Transformers) |
| **4. Store** | Save vectors in a searchable database | Vector stores (ChromaDB, FAISS, Pinecone) |
| **5. Retrieve** | Find chunks most similar to the user's question | Similarity search, MMR |
| **6. Generate** | LLM answers using the retrieved context | ChatOpenAI with retrieved docs |

---

## Part 2: RAG vs Alternatives

### When to Use RAG

| Approach | When | Cost | Accuracy |
|----------|------|------|----------|
| **Prompt Engineering** | Small context, few docs | Free | Medium |
| **RAG** | Large knowledge base, up-to-date info | Low-Medium | High |
| **Fine-Tuning** | Change model behavior/style | High | Medium-High |
| **RAG + Fine-Tuning** | Best of both | Highest | Highest |

### Decision Framework

```
Is your data < 100K tokens (roughly 50 pages)?
├── YES → Just stuff it into the prompt (prompt engineering)
│
└── NO  → Is your data changing frequently?
          ├── YES → Use RAG (re-index as needed)
          │
          └── NO  → Do you need to change the model's behavior?
                    ├── YES → Fine-tune + RAG
                    └── NO  → RAG is sufficient
```

### RAG Advantages

```
✅ No model retraining needed
✅ Data can be updated without retraining
✅ Sources are traceable (citations!)
✅ Works with any LLM (GPT-4, Claude, Llama)
✅ Keeps data private (not in model weights)
✅ Cost-effective compared to fine-tuning
✅ Reduces hallucinations significantly
```

---

## Part 3: Types of RAG

### Naive RAG (Basic)

```
Question → Embed → Search → Top-K docs → LLM → Answer

Simple, fast, but limited:
❌ Chunks might not contain the answer
❌ Irrelevant chunks dilute quality
❌ No query reformulation
❌ Fixed retrieval strategy
```

### Advanced RAG

```
Question → Query Rewriting → Hybrid Search → Reranking → LLM → Answer

Better retrieval through:
✅ Query expansion and reformulation
✅ Hybrid search (vector + keyword)
✅ Cross-encoder reranking
✅ Contextual compression
✅ Multi-query retrieval
```

### Agentic RAG

```
Question → Agent → [Search → Evaluate → Re-search → ...] → Answer

Full agent control:
✅ Agent decides WHEN to retrieve
✅ Agent evaluates retrieval quality
✅ Agent can re-query with better terms
✅ Agent combines retrieval with other tools
✅ Self-correcting retrieval
```

---

## Part 4: Minimal Working RAG

Let's build RAG from scratch in the simplest way possible:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --- Step 1: LLM and Embeddings ---
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

# --- Step 2: Create Documents ---
# In real apps, these come from PDFs, web pages, databases, etc.
documents = [
    Document(
        page_content="Our company was founded in 2020 by Aarav Patel in Bangalore. We started as a 3-person startup building developer tools.",
        metadata={"source": "about.md", "section": "history"}
    ),
    Document(
        page_content="Q3 2024 revenue was ₹4.5 crore, a 35% increase from Q2. Our primary revenue comes from SaaS subscriptions (70%) and consulting (30%).",
        metadata={"source": "financials.md", "section": "revenue"}
    ),
    Document(
        page_content="We have 45 employees across engineering (25), sales (10), marketing (5), and operations (5). Our engineering team works in Python, TypeScript, and Go.",
        metadata={"source": "team.md", "section": "headcount"}
    ),
    Document(
        page_content="Our main product is CodeAssist — an AI-powered code review tool. It supports Python, JavaScript, TypeScript, Java, and Go. Pricing starts at $29/month per developer.",
        metadata={"source": "products.md", "section": "codeassist"}
    ),
    Document(
        page_content="Our return policy allows full refunds within 30 days of purchase. After 30 days, we offer 50% credit for annual plans. Monthly plans can be cancelled anytime.",
        metadata={"source": "policies.md", "section": "refunds"}
    ),
    Document(
        page_content="We use AWS for cloud infrastructure, with primary regions in Mumbai (ap-south-1) and US East (us-east-1). Our uptime SLA is 99.9%.",
        metadata={"source": "infrastructure.md", "section": "cloud"}
    ),
    Document(
        page_content="Customer support hours are 9 AM to 6 PM IST, Monday through Friday. Enterprise customers get 24/7 support with a dedicated account manager.",
        metadata={"source": "support.md", "section": "hours"}
    ),
    Document(
        page_content="Our biggest customers include TechCorp (500 seats), DataFlow Inc (200 seats), and BuildFast (150 seats). Total ARR is ₹18 crore.",
        metadata={"source": "customers.md", "section": "enterprise"}
    ),
]

# --- Step 3: Store in Vector DB ---
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="company_kb"
)

# --- Step 4: Create Retriever ---
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Return top 3 most relevant chunks
)

# --- Step 5: RAG Prompt ---
rag_prompt = ChatPromptTemplate.from_template("""Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't have that information in my knowledge base."

Context:
{context}

Question: {question}

Answer:""")

# --- Step 6: Build the RAG Chain ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# --- Test! ---
questions = [
    "What was our Q3 revenue?",
    "Who founded the company?",
    "What programming languages does our engineering team use?",
    "What is the refund policy?",
    "What's the weather in Mumbai?",  # Not in our data
    "How many employees do we have?",
    "What are our support hours?",
]

for q in questions:
    print(f"\n❓ {q}")
    answer = rag_chain.invoke(q)
    print(f"💬 {answer}")
```

### How It Works Internally

```
User: "What was our Q3 revenue?"
         │
         ↓
[1] EMBED the question → vector
         │
         ↓
[2] SEARCH vectorstore → find similar chunks
    → Found: "Q3 2024 revenue was ₹4.5 crore..."
    → Found: "Our biggest customers include..."
    → Found: "Total ARR is ₹18 crore"
         │
         ↓
[3] FORMAT into prompt:
    "Context: Q3 2024 revenue was ₹4.5 crore...
     Question: What was our Q3 revenue?
     Answer:"
         │
         ↓
[4] LLM generates answer from context:
    "Q3 2024 revenue was ₹4.5 crore, a 35% increase from Q2."
```

---

## Part 5: Adding Source Citations

```python
from langchain_core.runnables import RunnableParallel

# Retrieve docs AND pass the question through
rag_with_sources = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()
)

def answer_with_sources(input_dict):
    """Generate answer with source citations."""
    docs = input_dict["context"]
    question = input_dict["question"]
    
    # Format context with source labels
    context_parts = []
    sources = set()
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        sources.add(source)
        context_parts.append(f"[{source}] {doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    # Generate answer
    prompt = rag_prompt.format(context=context, question=question)
    answer = llm.invoke(prompt).content
    
    return {
        "answer": answer,
        "sources": list(sources),
        "num_docs": len(docs)
    }


# Test with sources
result = answer_with_sources(
    rag_with_sources.invoke("What is our refund policy?")
)
print(f"💬 {result['answer']}")
print(f"📚 Sources: {result['sources']}")
print(f"📄 Retrieved {result['num_docs']} documents")
```

---

## Part 6: What's Coming in This Phase

| Chapter | What You'll Build |
|---------|------------------|
| **11.1** (this) | Understanding RAG conceptually + minimal RAG |
| **11.2** | Document Loaders (PDF, Web, CSV, Notion, etc.) |
| **11.3** | Text Splitting strategies (chunking is CRITICAL) |
| **11.4** | Complete RAG pipeline with real documents |

After this phase, **Phase 12 — Advanced RAG** covers query rewriting, re-ranking, hybrid search, and evaluation.

---

## Common Mistakes

### Mistake 1: Not chunking documents properly
```python
# ❌ Feeding entire documents to the vector store
# A 100-page PDF as one chunk → retrieval finds the whole doc, not the relevant paragraph

# ✅ Split into meaningful chunks (300-1000 tokens each)
# We'll cover this in Chapter 11.3
```

### Mistake 2: Not limiting the LLM to use only the context
```python
# ❌ LLM makes up answers not in the context
prompt = "Answer this question: {question}"

# ✅ Explicitly restrict to context
prompt = """Answer ONLY based on the following context.
If the answer is not in the context, say "I don't know."

Context: {context}
Question: {question}"""
```

### Mistake 3: Retrieving too few or too many chunks
```python
# ❌ k=1 → might miss the answer
# ❌ k=20 → too much noise, wastes tokens

# ✅ k=3-5 is a good starting point
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

### Mistake 4: Same embedding model for indexing and querying
```python
# ❌ Indexed with model A, querying with model B
# Vectors from different models are INCOMPATIBLE

# ✅ Always use the same embedding model for both
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use the same embedding model for indexing and querying | Vectors must be comparable |
| Chunk documents into 300-1000 token pieces | Too small = no context, too large = too noisy |
| Include "I don't know" instruction in the prompt | Reduces hallucination |
| Retrieve 3-5 chunks (start with k=4) | Balance between coverage and noise |
| Include source metadata in documents | Enables citations |
| Use persistent vector stores | Don't re-embed on every restart |
| Evaluate retrieval quality separately from generation | Find the bottleneck |
| Use MMR for diverse retrieval | Avoids redundant similar chunks |

---

## Interview Preparation

### Easy
**Q: What is RAG and why is it important?**

> RAG (Retrieval-Augmented Generation) is a pattern where relevant documents are retrieved from a knowledge base and provided as context to an LLM before generating an answer. It's the most important LLM pattern because it solves the core limitation of LLMs — they don't know your private data. RAG enables accurate, up-to-date, and verifiable answers without retraining the model. The pipeline is: Load documents → Split into chunks → Embed as vectors → Store in vector DB → Retrieve relevant chunks → Generate answer with context.

### Medium
**Q: When would you use RAG vs fine-tuning?**

> **RAG** when: you need the model to answer from specific documents, data changes frequently (re-index vs retrain), you need source citations, or cost is a concern. **Fine-tuning** when: you need to change the model's behavior/style (e.g., always respond in a specific format), teach domain-specific terminology, or optimize for a specific task. **Both together** when you need the model to speak in a domain-specific way AND answer from documents. RAG is almost always the right first choice — it's cheaper, faster to implement, and data can be updated without retraining.

### Hard
**Q: What are the three types of RAG and when would you use each?**

> (1) **Naive RAG**: Simple embed-search-generate. Fast to build, works for many use cases. Use for: prototyping, simple Q&A, small knowledge bases. Limitation: retrieval quality depends heavily on chunk quality and query-document similarity. (2) **Advanced RAG**: Adds query rewriting, hybrid search (vector + keyword), re-ranking, and contextual compression. Use when: naive RAG retrieval quality is insufficient, queries are complex or ambiguous, or the knowledge base is large. (3) **Agentic RAG**: An agent decides when and how to retrieve — can reformulate queries, evaluate results, and re-search. Use when: questions require multi-step reasoning, the agent needs to combine retrieval with other tools, or retrieval quality varies and self-correction is needed.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **RAG** | Retrieve relevant docs, then generate answer with context |
| **Indexing** | One-time: Load → Split → Embed → Store |
| **Querying** | Per question: Embed → Retrieve → Generate |
| **Vector store** | Database for similarity search (ChromaDB, FAISS) |
| **Retriever** | Component that finds relevant chunks (k=3-5) |
| **Context window** | Retrieved chunks are injected into the LLM prompt |
| **Naive RAG** | Basic embed → search → generate |
| **Advanced RAG** | Adds rewriting, reranking, hybrid search |
| **Agentic RAG** | Agent controls when/how to retrieve |

---

> [← Previous: Human-in-the-Loop](../phase-09-agents/chapter-39-human-in-the-loop.md) | [Next: Document Loaders →](chapter-41-document-loaders.md)

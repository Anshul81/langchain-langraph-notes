# Chapter 12.1: Advanced Retrieval — Query Rewriting, Hybrid Search & Reranking

> **Phase 12 — Advanced RAG** | [← Previous: Complete RAG Pipeline](../phase-10-rag/chapter-43-complete-rag-pipeline.md) | [Next: RAG Evaluation →](chapter-45-rag-evaluation.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Implement **multi-query retrieval** for better coverage
- ✅ Use **query decomposition** to handle complex questions
- ✅ Implement **hybrid search** (vector + keyword/BM25)
- ✅ Add **cross-encoder reranking** for precision
- ✅ Use **contextual compression** to remove noise from retrieved chunks
- ✅ Build an **advanced RAG pipeline** that dramatically outperforms naive RAG

| | |
|---|---|
| **Prerequisites** | Phase 11 (Complete RAG Pipeline) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 55 minutes |

---

## Introduction — Why Naive RAG Isn't Enough

Naive RAG fails in predictable ways:

```
❌ Problem 1: Query-document mismatch
   User: "What's the cost?"          → Embedding search finds "pricing plan details"
   But the BEST chunk says:          "Our Starter plan is $29/month per developer."
   The word "cost" never appears in the best chunk!

❌ Problem 2: Complex questions need decomposition
   User: "Compare our revenue growth with the team size growth"
   → Single vector search can't find BOTH revenue AND team data in one query

❌ Problem 3: Top-k retrieves duplicates
   → 3 of 5 retrieved chunks say the same thing (wasted context!)

❌ Problem 4: Noisy chunks
   → Retrieved chunk has 500 chars but only 1 sentence is relevant
```

**Advanced retrieval techniques fix all of these.**

---

## Part 1: Multi-Query Retrieval

Instead of one search query, generate **multiple** queries and merge results:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini", temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

# Assuming vectorstore is already set up from Chapter 11.4
# vectorstore = Chroma(persist_directory="./chroma_company_db", ...)

# Create a multi-query retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=llm
)

# What happens internally:
# User: "What's the pricing?"
# LLM generates 3 alternative queries:
#   1. "How much does CodeAssist cost?"
#   2. "What are the subscription plans and their prices?"
#   3. "Pricing tiers and features for CodeAssist"
# 
# All 3 queries search the vector store → results are merged and deduplicated!

results = multi_query_retriever.invoke("What's the pricing?")
print(f"Retrieved {len(results)} unique documents")
for doc in results:
    print(f"  [{doc.metadata.get('source', '?')}] {doc.page_content[:80]}...")
```

### Custom Multi-Query Prompt

```python
# Customize the query generation prompt
multi_query_prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant helping to improve document retrieval.
Generate 3 different search queries that would help find information to answer this question.
Each query should approach the question from a different angle.

Question: {question}

Provide 3 alternative queries, one per line:"""
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=llm,
    prompt=multi_query_prompt
)
```

---

## Part 2: Query Decomposition

For complex questions, break them into sub-questions:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def decompose_and_retrieve(question: str, retriever, llm) -> list[Document]:
    """Break a complex question into sub-questions and retrieve for each."""
    
    # Step 1: Decompose
    decompose_prompt = ChatPromptTemplate.from_template(
        """Break this question into 2-4 simpler sub-questions that can be answered independently.
Each sub-question should be a standalone search query.

Question: {question}

Sub-questions (one per line):"""
    )
    
    sub_questions_text = (decompose_prompt | llm | StrOutputParser()).invoke(
        {"question": question}
    )
    
    sub_questions = [q.strip().lstrip("0123456789.-) ") 
                     for q in sub_questions_text.split("\n") 
                     if q.strip()]
    
    print(f"📋 Decomposed into {len(sub_questions)} sub-questions:")
    for i, sq in enumerate(sub_questions, 1):
        print(f"   {i}. {sq}")
    
    # Step 2: Retrieve for each sub-question
    all_docs = []
    seen_contents = set()
    
    for sq in sub_questions:
        docs = retriever.invoke(sq)
        for doc in docs:
            # Deduplicate
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                all_docs.append(doc)
    
    print(f"📄 Retrieved {len(all_docs)} unique documents")
    return all_docs


# Test with a complex question
docs = decompose_and_retrieve(
    "Compare the company's revenue growth with their team size growth and tech stack",
    retriever, llm
)
```

---

## Part 3: Hybrid Search (Vector + Keyword)

Combine semantic search with traditional keyword matching:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 (keyword-based) retriever
# BM25 excels at exact keyword matching
bm25_retriever = BM25Retriever.from_documents(
    chunks,  # Your split documents
    k=4
)

# Vector (semantic) retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Ensemble: combine both with weights
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # 40% keyword, 60% semantic
)

# Test
results = hybrid_retriever.invoke("SOC 2 Type II compliance")
# BM25 finds exact keyword match: "SOC 2 Type II certified"
# Vector finds semantically similar: "security certifications and compliance standards"
# Both contribute to better retrieval!

print(f"Hybrid search found {len(results)} documents")
for doc in results:
    print(f"  [{doc.metadata.get('source', '?')}] {doc.page_content[:80]}...")
```

### When Hybrid Beats Vector-Only

| Scenario | Vector Only | Hybrid (Vector + BM25) |
|----------|------------|----------------------|
| "SOC 2 Type II" | ⚠️ Might miss exact term | ✅ BM25 finds exact match |
| "How does security work?" | ✅ Semantic match | ✅ Also works |
| "ORD-12345" | ❌ IDs don't embed well | ✅ BM25 exact match |
| "Python FastAPI" | ⚠️ May miss exact stack | ✅ BM25 + semantic |
| "Tell me about the team" | ✅ Good semantic match | ✅ Also works |

---

## Part 4: Cross-Encoder Reranking

Rerankers are **much more accurate** than embedding similarity — they compare query and document directly:

```python
# pip install sentence-transformers

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Create cross-encoder reranker
cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=cross_encoder, top_n=3)

# Wrap your retriever with reranking
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 10})  # Fetch 10, rerank to top 3
)

# Test
results = reranking_retriever.invoke("What is the refund policy for annual subscriptions?")
print(f"Reranked to {len(results)} documents")
for doc in results:
    print(f"  [{doc.metadata.get('source', '?')}] {doc.page_content[:100]}...")
```

### How Reranking Works

```
WITHOUT RERANKING:
  Query: "annual plan refund"
  Vector search returns top 5 (by embedding similarity):
    1. [0.89] "Annual plans billed once per year..." (partially relevant)
    2. [0.87] "14-day free trial..." (not about refunds!)
    3. [0.85] "Full refund within 30 days..." ← THIS IS THE BEST!
    4. [0.84] "Payment methods: Credit cards..." (irrelevant)
    5. [0.82] "Monthly plans: Cancel anytime..." (wrong plan type)

WITH RERANKING:
  Step 1: Vector search retrieves top 10 (cast a wide net)
  Step 2: Cross-encoder scores each (query, document) pair
  Step 3: Re-sort by cross-encoder score:
    1. [0.95] "Full refund within 30 days..." ← Now #1!
    2. [0.72] "Annual plans billed once per year..."
    3. [0.45] "Monthly plans: Cancel anytime..."
```

### Using LLM-Based Reranking (No Local Model)

```python
from langchain.retrievers.document_compressors import LLMChainFilter

# Use the LLM to filter out irrelevant documents
llm_filter = LLMChainFilter.from_llm(llm)

filtering_retriever = ContextualCompressionRetriever(
    base_compressor=llm_filter,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 8})
)

# The LLM evaluates each document: "Is this relevant to the query?"
# Only documents the LLM deems relevant are returned
results = filtering_retriever.invoke("What are the enterprise pricing options?")
```

---

## Part 5: Contextual Compression

Extract only the relevant parts from each chunk:

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

# LLM extracts only the relevant sentences from each chunk
extractor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=extractor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 5})
)

results = compression_retriever.invoke("What is the uptime SLA?")

# Without compression:
#   "All data encrypted at rest (AES-256)... SOC 2 Type II certified...
#    GDPR compliant... 99.9% uptime SLA (99.99% for Enterprise)..."
#   → 500 chars, but only the SLA part matters

# With compression:
#   "99.9% uptime SLA (99.99% for Enterprise)"
#   → Only the relevant extract!

for doc in results:
    print(f"  📄 {doc.page_content}")  # Much shorter, more focused
```

---

## Part 6: Complete Advanced RAG Pipeline

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()


class AdvancedRAG:
    """Advanced RAG with multi-query, hybrid search, and contextual compression."""
    
    def __init__(self, chunks, vectorstore):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0,
            openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
            openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
        )
        
        # Layer 1: Hybrid retriever (vector + BM25)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
        
        hybrid = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7]
        )
        
        # Layer 2: Multi-query on top of hybrid
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=hybrid,
            llm=self.llm
        )
        
        # RAG prompt
        self.prompt = ChatPromptTemplate.from_template(
            """Answer based ONLY on the context below. Be specific with numbers and dates.
If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""
        )
    
    def ask(self, question: str) -> dict:
        """Ask with advanced retrieval."""
        # Retrieve with multi-query + hybrid
        docs = self.retriever.invoke(question)
        
        # Format
        context = "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', '?')}] {doc.page_content}"
            for doc in docs
        )
        
        # Generate
        answer = (self.prompt | self.llm | StrOutputParser()).invoke({
            "context": context,
            "question": question
        })
        
        sources = list(set(doc.metadata.get("source", "?") for doc in docs))
        
        return {
            "answer": answer,
            "sources": sources,
            "num_docs": len(docs)
        }


# Usage (assumes chunks and vectorstore from previous chapter)
# advanced_rag = AdvancedRAG(chunks, vectorstore)
# result = advanced_rag.ask("What is the pricing for enterprise plans?")
# print(f"💬 {result['answer']}")
# print(f"📚 Sources: {result['sources']}")
```

---

## Common Mistakes

### Mistake 1: Multi-query without deduplication
```python
# ❌ Three queries might return the same document 3 times!
# MultiQueryRetriever handles this automatically, but custom implementations might not.

# ✅ Always deduplicate by content hash
seen = set()
unique_docs = []
for doc in all_docs:
    h = hash(doc.page_content[:100])
    if h not in seen:
        seen.add(h)
        unique_docs.append(doc)
```

### Mistake 2: Over-relying on reranking without fixing chunking
```python
# ❌ "My retrieval is bad, let me add reranking"
# If your chunks are bad (too large, split mid-sentence), reranking can't fix that

# ✅ Fix chunking FIRST, then add reranking as a polish step
```

### Mistake 3: Using too many retrieval layers (over-engineering)
```python
# ❌ Multi-query → Hybrid → Reranking → Compression → Filtering → ...
# Each layer adds latency and cost!

# ✅ Start simple, add layers only when measurably needed:
# Step 1: Vector search (baseline)
# Step 2: + Hybrid search (if keyword matches are important)
# Step 3: + Reranking (if top-k precision matters)
# Step 4: + Multi-query (if queries are diverse/ambiguous)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Start with naive RAG, add complexity incrementally | Measure improvement at each step |
| Use hybrid search when exact terms matter | IDs, acronyms, technical terms |
| Use multi-query for ambiguous questions | Different phrasings find different chunks |
| Rerank with cross-encoder for precision | Much more accurate than embedding similarity |
| Use contextual compression to reduce noise | Only send relevant text to the LLM |
| Always deduplicate merged results | Multiple retrievers may return the same doc |
| Measure retrieval quality with test queries | Can't improve what you don't measure |

---

## Interview Preparation

### Medium
**Q: What is hybrid search and when would you use it?**

> Hybrid search combines vector (semantic) search with keyword (BM25) search using an EnsembleRetriever. Vector search finds semantically similar content ("cost" matches "pricing"), while BM25 finds exact keyword matches ("SOC 2 Type II" matches exactly). Use hybrid when your data contains technical terms, product codes, acronyms, or proper nouns that don't embed well semantically. Typical weights: 60-70% vector, 30-40% keyword.

### Hard
**Q: Explain how cross-encoder reranking improves RAG and its trade-offs.**

> Cross-encoders score a (query, document) pair together, giving much more accurate relevance scores than separate embedding comparison. The workflow: retrieve top-20 with fast vector search, then rerank with a cross-encoder to get top-5. **Pros**: dramatically better precision, catches relevant documents that vector search ranks low. **Cons**: much slower (can't pre-compute — must score each pair), increases latency by 200-500ms, requires loading a model. For production: batch the reranking, use a fast model (ms-marco-MiniLM), and only rerank the top 10-20 candidates. Alternative: use LLM-based filtering/reranking (more accurate but slower and costlier).

---

## Summary

| Technique | What It Does | When to Use |
|-----------|-------------|------------|
| **Multi-Query** | Generate 3+ query variations, merge results | Ambiguous or broad questions |
| **Query Decomposition** | Break complex questions into sub-questions | Multi-part questions |
| **Hybrid Search** | Vector + BM25 keyword search | Technical terms, IDs, acronyms |
| **Cross-Encoder Reranking** | Re-score documents with query-document model | When top-k precision matters |
| **Contextual Compression** | Extract only relevant parts from chunks | Long chunks with mixed content |
| **LLM Filtering** | LLM decides if each doc is relevant | When precision > cost |

---

> [← Previous: Complete RAG Pipeline](../phase-10-rag/chapter-43-complete-rag-pipeline.md) | [Next: RAG Evaluation →](chapter-45-rag-evaluation.md)

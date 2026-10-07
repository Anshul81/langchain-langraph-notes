# Chapter 12.4: Multi-Query & RAG Fusion — Better Recall Through Query Diversity

> **Phase 12 — Advanced RAG** | [← Previous: Hybrid Search](chapter-53-hybrid-search.md) | [Next: Conversational RAG →](chapter-55-conversational-rag.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Use **`MultiQueryRetriever`** to generate paraphrased search queries
- ✅ Implement **Reciprocal Rank Fusion (RRF)** to merge ranked lists
- ✅ Combine multi-query with **hybrid** retrievers from Chapter 12.3
- ✅ Control cost/latency when running multiple retrievals per user question
- ✅ Evaluate multi-query gains on ambiguous questions
- ✅ Build a **research assistant retriever** that covers broad product questions

| | |
|---|---|
| **Prerequisites** | Chapters 12.2–12.3, Phase 11 RAG |
| **Estimated Reading Time** | 28 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction — One Query, One Blind Spot

Users ask questions **once**, but relevant documents use **many phrasings**. A single embedding query retrieves only one neighborhood in vector space.

### The Problem

```
User: "How do I upgrade and what does it cost?"

Indexed chunks use:
  - "Changing subscription tiers"
  - "Pro plan billing FAQ"
  - "Proration on mid-cycle upgrades"

Single query embedding may miss "proration" chunk entirely
```

### The Solution — Multi-Query + Fusion

```
User question
     │
     ▼
 LLM generates Q1, Q2, Q3  (multi-query)
     │
     ├── retrieve(Q1) → ranked list L1
     ├── retrieve(Q2) → ranked list L2
     └── retrieve(Q3) → ranked list L3
     │
     ▼
 RRF merge(L1, L2, L3) → unified ranking → top-k to LLM
```

**Multi-query improves recall; RRF merges without fragile score normalization.**

---

## Part 1: Setup

```bash
pip install langchain langchain-openai langchain-chroma python-dotenv
```

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)
```

---

## Part 2: Knowledge Base

```python
docs = [
    Document(page_content="Upgrade from Starter to Pro anytime from billing settings.", metadata={"id": "1"}),
    Document(page_content="Pro plan costs $79 per developer per month billed monthly.", metadata={"id": "2"}),
    Document(page_content="Mid-cycle upgrades are prorated: you pay the difference for remaining days.", metadata={"id": "3"}),
    Document(page_content="Downgrades take effect at the next renewal date.", metadata={"id": "4"}),
    Document(page_content="Enterprise pricing is custom; contact sales for volume discounts.", metadata={"id": "5"}),
]

vectorstore = Chroma.from_documents(docs, embeddings, collection_name="mq_fusion")
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

---

## Part 3: `MultiQueryRetriever`

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)

question = "How do I upgrade and what will I pay?"
results = multi_retriever.invoke(question)

print(f"Retrieved {len(results)} documents")
for d in results:
    print(d.metadata.get("id"), d.page_content[:65])
```

Internally the LLM emits several queries, runs each against `base_retriever`, dedupes.

### Custom Prompt (Domain-Tuned)

```python
from langchain_core.prompts import ChatPromptTemplate

multi_prompt = ChatPromptTemplate.from_template(
    """You improve search for a SaaS billing knowledge base.
Generate 3 diverse search queries to answer the user question.
Include synonyms for upgrade, pricing, proration, and plan names.

Question: {question}

One query per line:"""
)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
    prompt=multi_prompt,
)
```

---

## Part 4: Reciprocal Rank Fusion (RRF)

When merging lists from **different retrievers** or **multiple queries**, raw scores are incomparable. RRF uses ranks:

\[
\text{score}(d) = \sum \frac{1}{k + \text{rank}(d)}
\]

Common `k=60` (constant smoothing).

```python
from collections import defaultdict
from langchain_core.documents import Document

def doc_key(doc: Document) -> str:
    return doc.metadata.get("id") or doc.page_content[:120]

def reciprocal_rank_fusion(list_of_lists: list[list[Document]], k: int = 60) -> list[Document]:
    scores = defaultdict(float)
    doc_map = {}
    for ranked in list_of_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc_key(doc)
            doc_map[key] = doc
            scores[key] += 1.0 / (k + rank)
    ordered_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in ordered_keys]
```

### Manual Multi-Query With RRF

```python
from langchain_core.output_parsers import StrOutputParser

query_gen = ChatPromptTemplate.from_template(
    "Generate 3 search queries (one per line) for:\n{question}"
) | llm | StrOutputParser()

def generate_queries(question: str) -> list[str]:
    text = query_gen.invoke({"question": question})
    lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
    return lines[:3] or [question]

def retrieve_with_rrf(question: str, top_n: int = 4) -> list[Document]:
    queries = generate_queries(question)
    lists = [base_retriever.invoke(q) for q in queries]
    fused = reciprocal_rank_fusion(lists)
    return fused[:top_n]

for d in retrieve_with_rrf(question):
    print(d.metadata["id"], d.page_content[:60])
```

Expect doc `3` (proration) to rise when multi-query includes billing angles.

---

## Part 5: Multi-Query + Hybrid (Chapter 12.3)

```python
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

bm25 = BM25Retriever.from_documents(docs)
bm25.k = 4
hybrid = EnsembleRetriever(retrievers=[base_retriever, bm25], weights=[0.6, 0.4])

multi_hybrid = MultiQueryRetriever.from_llm(retriever=hybrid, llm=llm)

hits = multi_hybrid.invoke("SOC-style audit: prorated upgrade charges")
```

Stack layers only when evals prove each step helps — multi-query adds **LLM + N retrievals** per request.

---

## Part 6: Latency and Cost Controls

| Technique | Effect |
|-----------|--------|
| Limit to 2–3 sub-queries | Cuts retrieval multiplier |
| Cache sub-query results per session | Helps follow-ups |
| Run multi-query only if confidence low | Router classifies "ambiguous" |
| Smaller model for query generation | Cheap paraphrases |
| Parallel retrievals | asyncio / thread pool |

```python
def smart_retrieve(question: str) -> list[Document]:
    if len(question.split()) <= 4:
        return base_retriever.invoke(question)
    return multi_retriever.invoke(question)
```

---

## Part 7: End-to-End RAG With Fusion

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

rag_prompt = ChatPromptTemplate.from_template(
    """Use the context to answer. If unsure, say so.

Context:
{context}

Question: {question}
"""
)

def build_context(question: str) -> str:
    docs = retrieve_with_rrf(question, top_n=4)
    return "\n\n".join(f"- {d.page_content}" for d in docs)

rag = (
    {"context": lambda x: build_context(x["question"]), "question": lambda x: x["question"]}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print(rag.invoke({"question": question}))
```

---

## Part 8: Query Decomposition vs Multi-Query

| Pattern | Input example | Generated outputs |
|---------|---------------|-------------------|
| **Multi-query** | "upgrade cost" | 3 paraphrases of same intent |
| **Decomposition** | "Compare Starter vs Pro SSO" | "Starter SSO?", "Pro SSO?" |

```python
decompose_prompt = ChatPromptTemplate.from_template(
    """Split this question into 2 independent search questions (one per line):
{question}"""
)

decompose_chain = decompose_prompt | llm | StrOutputParser()

def decompose_retrieve(question: str) -> list[Document]:
    text = decompose_chain.invoke({"question": question})
    subqs = [ln.strip() for ln in text.splitlines() if ln.strip()][:2]
    lists = [base_retriever.invoke(q) for q in subqs]
    return reciprocal_rank_fusion(lists)[:4]
```

Use decomposition when the user combines **two facets**; use multi-query when vocabulary mismatch is the main issue.

---

## Part 9: RAG-Fusion Paper Intuition

**RAG-Fusion** (popular pattern name) ≈ multi-query retrieval + fusion ranker. The LLM generates diverse queries; each query produces a ranking; fusion boosts documents that appear near the top in **multiple** lists — a signal of robust relevance.

```
Doc A: ranks #1, #2, #3 across three queries → high RRF score
Doc B: ranks #1, #20, #18 → lower fused score
```

You do not need a separate library — `MultiQueryRetriever` plus RRF captures the core idea.

---

## Part 10: Failure Telemetry

```python
def retrieve_with_logging(question: str) -> list[Document]:
    subqs = generate_queries(question)
    print("sub_queries:", subqs)
    lists = []
    for q in subqs:
        hits = base_retriever.invoke(q)
        print(f"  {q!r} -> {[d.metadata.get('id') for d in hits]}")
        lists.append(hits)
    fused = reciprocal_rank_fusion(lists)
    return fused[:4]
```

When users report "it couldn't find the doc," compare sub-queries to how the doc is written — often a prompt tweak fixes systematic gaps.

---

## Common Mistakes

### Mistake 1: Too many sub-queries (5+)
```python
# ❌ 5 queries × hybrid × rerank = slow, expensive
# ✅ 2–3 diverse queries, measure lift
```

### Mistake 2: Averaging incompatible scores
```python
# ❌ Add BM25 score + cosine similarity directly
# ✅ Use RRF or learned reranker
```

### Mistake 3: Identical paraphrases
```python
# ❌ Prompt allows "upgrade plan" three times
# ✅ Prompt demands different angles: how-to, price, policy
```

### Mistake 4: No deduplication before LLM
```python
# Multi-query already dedupes — still dedupe when merging with other pipelines
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Log generated sub-queries | Debug weird retrieval |
| A/B multi-query on eval set | Justify extra LLM call |
| Use RRF when combining lists | Robust across retrievers |
| Pair with compression (12.2) | Multi-query increases chunk count |
| Cap total context tokens | Fusion can over-fetch |

---

## Interview Preparation

### Easy
**Q: What does MultiQueryRetriever do?**

> It uses an LLM to generate several alternative search queries from the user's question, runs each against a base retriever, and merges the results (with deduplication). This improves recall when documents use different wording than the user.

### Medium
**Q: What is Reciprocal Rank Fusion?**

> RRF merges multiple ranked document lists by summing `1/(k+rank)` for each appearance of a document across lists. It avoids comparing incompatible scores from BM25 and vector search. Documents that rank highly in several lists rise to the top.

### Hard
**Q: How do you decide between multi-query and query decomposition?**

> **Multi-query** generates paraphrases of the same intent — good for vocabulary mismatch. **Query decomposition** splits multi-part questions ("compare A and B") into sub-questions — good for compositional QA. Use decomposition when failure mode is missing a second facet; use multi-query when relevant docs exist but lexical/semantic gap blocks one embedding.

### Hard
**Q: Does MultiQueryRetriever guarantee better answers?**

> It improves **recall** — more relevant chunks enter the pool. Generation can still fail if chunks are noisy or contradictory. Always measure end-to-end answer quality (Chapter 11.5). If recall is already high, multi-query adds latency without lift.

### Senior
**Q: Design a retrieval router for an enterprise RAG platform.**

> Classify queries (keyword-heavy, ambiguous, multi-hop, conversational). Route: exact ID → BM25-only; short factual → vector k=4; ambiguous → multi-query+RRF; complex → decompose then fuse. Enforce budgets (max sub-queries, max tokens). Emit telemetry per route. Continuous eval per route with regression gates on deploy. Allow tenant-specific prompts for query generation.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Multi-query** | LLM paraphrases → multiple retrievals |
| **RRF** | Rank-based fusion across lists |
| **Dedup** | Same doc from many queries appears once |
| **Hybrid + multi-query** | Recall boost; watch latency |
| **Router** | Skip multi-query when unnecessary |

---

## Hands-on Exercise

Create 8 golden questions for the billing KB (include one two-part question). Compare **base retriever** vs **multi-query** vs **manual RRF** on recall@4 (did the right doc IDs appear?). Document when multi-query did not help.

**Stretch goal:** Plot (on paper is fine) RRF scores for the top 5 doc IDs for one question — verify the winner appears in at least two sub-query lists.

---

## Part 11: Cost Model (Back-of-Envelope)

```
Cost per user question ≈
  (1 LLM call for sub-queries) × prompt tokens
+ N × vector search latency
+ (1 LLM call for answer) × (context + answer) tokens
```

If sub-queries = 3 and vector search is 40ms each, you add ~120ms before generation. Only enable multi-query on tiers or routes where recall gains justify **~1 extra LLM call** per turn.

---

## What's Next

Retrieval gets smarter; users also **chat**. **Chapter 12.5** wires **conversation history** into retrieval for enterprise document chat.

---

> [← Previous: Hybrid Search](chapter-53-hybrid-search.md) | [Next: Conversational RAG →](chapter-55-conversational-rag.md)

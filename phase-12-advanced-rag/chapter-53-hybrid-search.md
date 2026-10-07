# Chapter 12.3: Hybrid Search — Dense + Sparse (BM25)

> **Phase 12 — Advanced RAG** | [← Previous: Contextual Compression](chapter-52-compression-reranking.md) | [Next: Multi-Query & RAG Fusion →](chapter-54-multi-query-rag-fusion.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Explain **semantic (dense)** vs **keyword (sparse)** retrieval
- ✅ Implement **BM25** with LangChain's `BM25Retriever`
- ✅ Combine retrievers with **`EnsembleRetriever`**
- ✅ Tune vector vs keyword weights for your corpus
- ✅ Know when hybrid beats pure vector search
- ✅ Build **hybrid search** over product docs with SKUs and acronyms

| | |
|---|---|
| **Prerequisites** | Phase 11 RAG, Chapter 12.2 (optional reranking) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction — When Embeddings Miss the Exact Token

Vector search excels at **meaning**: "cost" ≈ "pricing". It struggles when the user needs **exact tokens**: product IDs, error codes, legal citations, acronyms.

### The Problem

```
User: "SOC 2 Type II report availability"

Best chunk contains exact string "SOC 2 Type II"
Embedding query may rank a vague "security compliance overview" higher
→ Wrong context → vague LLM answer
```

### The Solution — Hybrid Search

```
                    ┌─────────────────┐
               ┌───→│ Vector retriever │───┐
Query ─────────┤    │ (semantic)       │   │
               │    └─────────────────┘   │
               │                          ├──→ Merge / RRF ──→ Top-k docs
               │    ┌─────────────────┐   │
               └───→│ BM25 retriever   │───┘
                    │ (keyword)        │
                    └─────────────────┘
```

**Combine recall of keywords with semantic generalization.**

---

## Part 1: Setup

```bash
pip install langchain langchain-openai langchain-chroma rank_bm25 python-dotenv
```

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)
```

---

## Part 2: Corpus With IDs and Acronyms

```python
docs = [
    Document(
        page_content="CodeAssist Pro plan includes SSO, audit logs, and SOC 2 Type II report on request.",
        metadata={"sku": "CA-PRO", "source": "pricing.md"},
    ),
    Document(
        page_content="Error E-4421: embedding quota exceeded. Retry after 60 seconds or upgrade plan.",
        metadata={"sku": "RUNTIME", "source": "errors.md"},
    ),
    Document(
        page_content="Our security program covers encryption at rest, pen tests, and compliance documentation.",
        metadata={"sku": "SEC-OVERVIEW", "source": "security.md"},
    ),
    Document(
        page_content="Starter plan CA-START: no SSO, community support, 1M tokens/month cap.",
        metadata={"sku": "CA-START", "source": "pricing.md"},
    ),
]
```

---

## Part 3: Vector Retriever

```python
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="hybrid_demo",
    persist_directory="./chroma_hybrid_demo",
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
```

---

## Part 4: BM25 Retriever

```python
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4
```

BM25 scores term frequency and inverse document frequency — strong on rare tokens like `E-4421`, `SOC`, `CA-PRO`.

---

## Part 5: `EnsembleRetriever`

```python
from langchain.retrievers import EnsembleRetriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.6, 0.4],
)

query = "SOC 2 Type II report"
results = hybrid_retriever.invoke(query)

for i, doc in enumerate(results, 1):
    print(i, doc.metadata.get("source"), doc.page_content[:70])
```

### Weight Tuning

| Corpus profile | Suggested weights (vector, BM25) |
|----------------|----------------------------------|
| Marketing prose | 0.75 / 0.25 |
| Mixed docs + codes | 0.55 / 0.45 |
| Logs, tickets, SKUs | 0.4 / 0.6 |

```python
# Experiment helper
def compare(query: str):
    print("=== VECTOR ===")
    for d in vector_retriever.invoke(query)[:3]:
        print(" ", d.page_content[:60])
    print("=== BM25 ===")
    for d in bm25_retriever.invoke(query)[:3]:
        print(" ", d.page_content[:60])
    print("=== HYBRID ===")
    for d in hybrid_retriever.invoke(query)[:3]:
        print(" ", d.page_content[:60])

compare("error E-4421 embedding quota")
```

---

## Part 6: Deduplication After Ensemble

Ensemble may return duplicates from both retrievers:

```python
def dedupe_documents(documents: list[Document]) -> list[Document]:
    seen = set()
    unique = []
    for doc in documents:
        key = (doc.metadata.get("source"), doc.page_content[:200])
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique

hybrid_deduped = dedupe_documents(hybrid_retriever.invoke("CA-PRO SSO"))
```

For production, prefer **Reciprocal Rank Fusion (RRF)** — covered in Chapter 12.4.

---

## Part 7: Hybrid + LLM Generation (Quick RAG)

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)

prompt = ChatPromptTemplate.from_template(
    "Answer from context only.\n\nContext:\n{context}\n\nQuestion: {q}"
)

def get_context(q: str) -> str:
    hits = dedupe_documents(hybrid_retriever.invoke(q))[:4]
    return "\n".join(d.page_content for d in hits)

chain = (
    {"context": lambda x: get_context(x["q"]), "q": lambda x: x["q"]}
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke({"q": "Do we have SOC 2 Type II for Pro?"}))
```

---

## Part 8: Architecture in Production

```
Offline:
  Documents → chunk → embed → vector DB
                     └→ BM25 index (in-memory or Elasticsearch/OpenSearch)

Online:
  Query → (optional rewrite) → hybrid retrieve → rerank → compress → LLM
```

Hosted vector DBs (Pinecone, Weaviate) often offer **hybrid built-in** — same idea, one API.

---

## Part 9: Side-by-Side Recall Experiment

```python
EVAL = [
    ("SOC 2 Type II report", {"pricing.md"}),
    ("error E-4421 quota", {"errors.md"}),
    ("remote work VPN policy", {"security.md"}),  # may need semantic vector
]

def sources(docs):
    return {d.metadata.get("source") for d in docs}

def eval_retriever(name, retriever_fn):
    hits = 0
    for q, expected in EVAL:
        got = sources(retriever_fn(q)[:3])
        if expected & got:
            hits += 1
        print(f"{name} q={q!r} expected={expected} got={got}")
    print(f"{name} recall@3: {hits}/{len(EVAL)}")

eval_retriever("vector", vector_retriever.invoke)
eval_retriever("bm25", bm25_retriever.invoke)
eval_retriever("hybrid", hybrid_retriever.invoke)
```

Run this table whenever you change chunk size or weights — hybrid should win on code/ID queries without hurting semantic ones.

---

## Part 10: Elasticsearch / OpenSearch Mental Model

Even if you use LangChain's in-memory BM25 for learning, production stacks often look like:

```
Ingest pipeline:
  text → tokenize → inverted index (BM25)
      → embedding model → vector field (HNSW)

Query:
  bool query { should: [match BM25, knn vector] }
```

LangChain's `EnsembleRetriever` teaches the **logic**; your platform team may implement the same fusion inside the search engine rather than in Python.

---

## Part 11: Token Budget After Hybrid

Hybrid increases **unique** chunks — cap before LLM:

```python
MAX_CHARS = 6000

def hybrid_context(query: str) -> str:
    docs = dedupe_documents(hybrid_retriever.invoke(query))
    chunks, total = [], 0
    for d in docs:
        if total + len(d.page_content) > MAX_CHARS:
            break
        chunks.append(d.page_content)
        total += len(d.page_content)
    return "\n\n".join(chunks)
```

Pair with contextual compression (Chapter 12.2) if parents/hybrid still overflow the window.

---

## Common Mistakes

### Mistake 1: Hybrid on tiny corpora (<50 chunks)
```python
# ❌ BM25 adds complexity with no gain
# ✅ Pure vector often enough for demos
```

### Mistake 2: Equal weights without evaluation
```python
# ❌ weights=[0.5, 0.5] because it "feels fair"
# ✅ Measure recall@k on labeled questions
```

### Mistake 3: BM25 on unnormalized HTML noise
```python
# ❌ Massive boilerplate dominates term stats
# ✅ Clean nav/footer, split by logical sections first
```

### Mistake 4: Skipping deduplication before LLM
```python
# ❌ Same doc twice wastes context window
# ✅ Dedupe by content hash or source+offset
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Evaluate hybrid vs vector alone | Prove BM25 helps your queries |
| Boost BM25 for support/engineering KBs | Codes and IDs matter |
| Keep k moderate per retriever before merge | 4+4 → dedupe → top 5 |
| Pair hybrid with reranking (Ch. 12.2) | Better ordering after recall |
| Log which retriever contributed each doc | Tune weights with data |

---

## Interview Preparation

### Easy
**Q: What is hybrid search in RAG?**

> Hybrid search combines dense vector retrieval with sparse keyword retrieval (often BM25). Vector search finds semantically similar text; BM25 finds exact term matches. Merging results improves recall when queries include rare tokens, acronyms, or IDs while still handling paraphrases.

### Medium
**Q: How does LangChain's EnsembleRetriever work?**

> It runs multiple retrievers on the same query and merges their ranked lists using configured weights. You typically pass a vector retriever and a BM25Retriever with weights like 0.6 and 0.4. Results are combined and ranked; you should deduplicate documents that appear in both lists before sending context to the LLM.

### Hard
**Q: When would hybrid search hurt performance?**

> When the corpus is small, highly semantic, and lacks exact-match requirements — BM25 adds latency and noise. Poorly preprocessed text (HTML, templates) skews BM25. Extremely short queries can behave oddly on BM25. If vector recall is already near-perfect on evals, hybrid increases index maintenance (second index) for marginal gains.

### Hard
**Q: Why not always set BM25 weight to 1.0 for technical docs?**

> Pure keyword search misses paraphrases and conceptual questions ("How do I authenticate users?" vs "OAuth flow"). Zero vector weight removes semantic generalization. Technical corpora usually still need **both** signals — tune toward BM25, rarely eliminate vectors entirely unless evals prove semantic search adds noise.

### Senior
**Q: How do you implement hybrid search at scale?**

> Use OpenSearch/Elasticsearch with BM25 plus dense vectors in one cluster, or managed hybrid from your vector vendor. Shard by tenant, refresh indexes on ingest, monitor query latency split by retriever. Use RRF instead of manual weights when lists differ in scale. Cache hot queries. A/B test weights per product line. Fall back to vector-only if sparse index stale.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Dense retrieval** | Embedding similarity (semantic) |
| **Sparse / BM25** | Keyword relevance (exact tokens) |
| **EnsembleRetriever** | Weighted merge of retrievers |
| **Weights** | Balance semantic vs keyword bias |
| **Deduping** | Required after ensemble merge |

---

## Hands-on Exercise

Extend the corpus with 5 error codes (`E-1001`…`E-1005`). Build eval queries that mention codes vs descriptions ("quota exceeded" without code). Report which retriever wins per query and pick weights that maximize recall@3 on your mini eval set.

**Stretch goal:** Implement a tiny grid search over weights `[0.5,0.5], [0.6,0.4], [0.7,0.3]` printing recall@3 for your eval list — automate picking the best pair.

---

## Part 12: Senior Interview Follow-Up

**Q: How is hybrid search different from metadata filtering?**

> Metadata filtering restricts the candidate set (`department=legal`). Hybrid scoring ranks **text relevance** within candidates using semantic + lexical signals. Use filters for authorization and tenancy; use hybrid for ranking. They compose: filter ACL → hybrid retrieve top-k.

---

## What's Next

Hybrid search improves **recall**; **multi-query and RAG fusion** widen query coverage. **Chapter 12.4** generates diverse queries and merges ranked lists with RRF.

---

> [← Previous: Contextual Compression](chapter-52-compression-reranking.md) | [Next: Multi-Query & RAG Fusion →](chapter-54-multi-query-rag-fusion.md)

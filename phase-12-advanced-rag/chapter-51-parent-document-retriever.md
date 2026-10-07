# Chapter 12.1: Parent Document Retriever — Small Chunks Search, Big Chunks Answer

> **Phase 12 — Advanced RAG** | [← Previous: RAG Evaluation Basics](../phase-11-rag/chapter-50-rag-evaluation-basics.md) | [Next: Contextual Compression →](chapter-52-compression-reranking.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Explain the **small-to-big** retrieval pattern and why naive chunking hurts answers
- ✅ Implement **`ParentDocumentRetriever`** with child and parent splitters
- ✅ Store child embeddings while returning **parent context** to the LLM
- ✅ Tune child vs parent chunk sizes for your document types
- ✅ Combine parent retrieval with metadata for citations
- ✅ Build a **policy manual Q&A** system that needs surrounding context

| | |
|---|---|
| **Prerequisites** | Phase 11 (RAG pipeline, text splitting, evaluation) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction — The Chunk Size Dilemma

RAG retrieval wants **small chunks** (precise vector match). Generation wants **large context** (complete paragraphs, tables, lists).

### The Problem

```
Document section (800 words):
  "## Refund Policy
   Items returned within 30 days...
   Exceptions: digital goods, customized items...
   Process: contact support@..."

Split into 200-token chunks:
  Chunk A: "## Refund Policy Items returned within"
  Chunk B: "30 days... Exceptions: digital goods"
  
User: "Are digital downloads refundable?"
  → Retrieves Chunk B (partial)
  → LLM misses "Exceptions" header context → wrong answer
```

### The Solution — Parent Document Retriever

```
INDEX:
  Parent doc (full section)     ── stored in docstore (by ID)
        │
        ├── child chunk 1  ── embedded in vector store
        ├── child chunk 2  ── embedded
        └── child chunk 3  ── embedded

QUERY:
  Search vector store on CHILD chunks (high precision)
  Map hits → PARENT IDs → return FULL parent to LLM
```

**Search small, read big.**

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
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

## Part 2: Sample Corpus — HR & Policy Manual

```python
raw_docs = [
    Document(
        page_content=(
            "## Refund Policy\n"
            "Physical products may be returned within 30 days with proof of purchase. "
            "Restocking fee of 15% applies after 7 days.\n"
            "Exceptions: digital downloads, gift cards, and customized merchandise "
            "are non-refundable once delivered.\n"
            "Process: email support@example.com with order ID ORD-XXXX."
        ),
        metadata={"source": "policy/refunds.md", "section": "refunds"},
    ),
    Document(
        page_content=(
            "## Remote Work\n"
            "Employees may work remotely up to 3 days per week with manager approval. "
            "Core hours 10:00–16:00 local time apply for meetings.\n"
            "Security: VPN required, company laptop only, no personal cloud storage "
            "for confidential documents."
        ),
        metadata={"source": "policy/remote.md", "section": "remote"},
    ),
    Document(
        page_content=(
            "## Health Benefits\n"
            "Medical, dental, and vision coverage starts on day 1 for full-time staff. "
            "Dependents enrolled within 30 days of hire receive same effective date.\n"
            "Open enrollment: November 1–15 annually. Life insurance: 1x salary default."
        ),
        metadata={"source": "policy/benefits.md", "section": "benefits"},
    ),
]
```

---

## Part 3: Build `ParentDocumentRetriever`

```python
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever

# Child chunks: what we embed (search)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

# Parent chunks: what we send to the LLM (context)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)

vectorstore = Chroma(
    collection_name="parent_child_policy",
    embedding_function=embeddings,
    persist_directory="./chroma_parent_child",
)

docstore = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

retriever.add_documents(raw_docs)
```

### What Happens on `add_documents`

1. Each input `Document` is split into **parents** (up to 800 chars).
2. Each parent is split into **children** (200 chars).
3. Children are embedded into **Chroma**.
4. Full parent documents live in **`docstore`** keyed by ID.
5. Child metadata links back to parent ID.

---

## Part 4: Retrieve and Inspect

```python
query = "Can I get a refund on a digital download?"
parents = retriever.invoke(query)

print(f"Retrieved {len(parents)} parent document(s)")
for doc in parents:
    print("---")
    print(doc.metadata)
    print(doc.page_content[:400])
```

You should see the **full Refund Policy section**, not a tiny fragment — even though matching happened on a child mentioning "digital".

Compare with naive retriever:

```python
from langchain_chroma import Chroma

naive_vs = Chroma.from_documents(
    child_splitter.split_documents(raw_docs),
    embeddings,
    collection_name="naive_small_only",
)
naive_hits = naive_vs.similarity_search(query, k=2)
print("Naive chunk sizes:", [len(d.page_content) for d in naive_hits])
print("Parent chunk sizes:", [len(d.page_content) for d in parents])
```

---

## Part 5: RAG Chain With Parent Context

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    blocks = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        blocks.append(f"[{src}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)

rag_prompt = ChatPromptTemplate.from_template(
    """Answer using ONLY the context below. Cite the source file in brackets.
If the answer is not in context, say "I don't know."

Context:
{context}

Question: {question}
"""
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("Are digital downloads refundable?"))
```

Expected: cites non-refundable exception with full policy wording.

---

## Part 6: Tuning Guidelines

| Document type | Child size | Parent size | Notes |
|---------------|------------|-------------|-------|
| Dense legal/HR | 150–250 | 800–1500 | Parents preserve section headers |
| API docs | 300–400 | 1200+ | Keep code blocks intact in parent |
| FAQs | 100–200 | whole Q&A pair | Parent = one FAQ item |
| PDFs with tables | split by headers | full subsection | Avoid splitting tables in child |

```
Too small parent  → still missing cross-paragraph context
Too large parent  → noise, higher token cost
Too large child   → weak retrieval precision
Too small child   → many vectors, slower index
```

---

## Part 7: Persistence Notes

- **Vector store**: persist Chroma directory (children).
- **Docstore**: `InMemoryStore` is lost on restart — for production use **`LocalFileStore`**, Redis, or embed parent content in vector metadata (with size limits).

```python
# Pattern: re-add parents on startup if docstore empty but vector exists
# Or use langchain_community.storage RedisStore for parent docs
```

Re-index when parent splitter settings change — child/parent links must stay consistent.

---

## Part 8: When to Choose Parent vs Other Advanced Patterns

```
Question needs full section context?  → ParentDocumentRetriever
Question needs exact SKU/code?        → Hybrid BM25 (Ch. 12.3)
Question is vague / broad?            → Multi-query (Ch. 12.4)
Chunk is huge but one sentence matters? → Compression (Ch. 12.2)
```

Parent document retrieval does **not** replace reranking — if many parents match, add a cross-encoder or LLM filter on parent summaries before stuffing context.

```python
# Optional: retrieve parents, then take top 2 by simple keyword overlap
def rank_parents(query: str, parents: list[Document], k: int = 2) -> list[Document]:
    q_tokens = set(query.lower().split())
    scored = []
    for p in parents:
        overlap = len(q_tokens & set(p.page_content.lower().split()))
        scored.append((overlap, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]
```

---

## Part 9: Markdown-Aware Splitting (Recommended)

```python
parent_splitter_md = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=40,
    separators=["\n## ", "\n### ", "\n\n", "\n", " "],
)
```

Heading-first separators keep **parent boundaries aligned with sections**, so a child hit on "digital goods" still maps to a parent that includes the `## Refund Policy` header — critical for compliance Q&A.

---

## Common Mistakes

### Mistake 1: Same splitter for parent and child
```python
# ❌ No benefit — identical chunks
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# ✅ Child noticeably smaller than parent
```

### Mistake 2: Forgetting to persist docstore
```python
# ❌ InMemoryStore in production → parents missing after restart
# ✅ Durable docstore or rebuild pipeline on deploy
```

### Mistake 3: Duplicate parents in multi-chunk hits
```python
# Parent retriever dedupes by parent ID for a query — good.
# If you merge with other retrievers, dedupe by source+section manually.
```

### Mistake 4: Massive parent sizes
```python
# ❌ parent chunk_size=8000 → blows context window
# ✅ Target 1–3 parents per query, each fits budget
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Split on headings first when possible | Parents align with document structure |
| Keep `source` metadata on parents | Citations survive retrieval |
| Evaluate retrieval AND answer quality | Parent pattern fixes context, not all hallucinations |
| Log which parent IDs were used | Debug wrong-section retrieval |
| Cap `k` at parent level | Usually 2–4 parents enough |

---

## Interview Preparation

### Easy
**Q: What problem does Parent Document Retriever solve?**

> It decouples search granularity from generation context. Small child chunks improve embedding search precision, but the retriever returns larger parent documents so the LLM sees complete sections. This reduces answers that are technically retrieved but missing critical nearby sentences.

### Medium
**Q: Walk through the data flow of ParentDocumentRetriever.**

> Documents are split into parents stored in a docstore. Parents are further split into children that are embedded in a vector store. At query time, similarity search runs on children. Matching child documents reference parent IDs; those parents are fetched from the docstore and returned as context. The LLM never sees only the tiny child unless you choose to.

### Hard
**Q: When would you NOT use parent-document retrieval?**

> When documents are already atomic (single FAQ entries, short tweets), when entire files fit in context cheaply, or when you need sentence-level citation with minimal fluff. Also skip if your bottleneck is generation quality, not missing context — fix prompts/models first. For highly structured data (SQL, APIs), direct lookup beats chunking.

### Hard
**Q: Can you use ParentDocumentRetriever with a hosted vector DB only?**

> Yes for **child** vectors. You still need a **docstore** (or equivalent) for parent bodies — the retriever resolves child hits to parent IDs and fetches full text. Some teams store parent text in object storage and cache hot parents in Redis; the vector DB alone is not enough unless you duplicate parent content in metadata fields (size-limited).

### Senior
**Q: How do you operate parent-child RAG in production?**

> Version index and docstore together; automate rebuild on content or splitter changes. Monitor parent token sizes P95, dedupe across retrievers, enforce max context budget before LLM. Use evaluation sets with questions requiring cross-sentence reasoning within sections. For multi-tenant, partition collections and docstores by tenant. Backup docstore — losing it breaks child→parent mapping even if vectors remain.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Child chunks** | Embedded for search (small) |
| **Parent documents** | Returned to LLM (large) |
| **Docstore** | Holds parent bodies by ID |
| **Small-to-big** | Precision at index, completeness at generation |
| **ParentDocumentRetriever** | LangChain orchestration of the pattern |

---

## Hands-on Exercise

Add a fourth document with a **table-like** benefits comparison (plain text rows). Tune child/parent sizes so a question about *"dental vs vision enrollment"* retrieves the correct parent including row headers. Compare answers against naive 200-token-only RAG and record one metric from Chapter 11.5 (e.g. answer correctness on 3 questions).

**Stretch goal:** Store parent documents in a JSON file keyed by UUID at index time; on retrieval, log `parent_id` alongside the user question for audit trails in regulated industries.

---

## Part 10: Interview Quick Reference

| Symptom | Likely fix |
|---------|------------|
| Right section, wrong sentence | Smaller child chunks |
| Right child hit, missing header | Markdown-aware parent split |
| Too much irrelevant text in parent | Add compression (Ch. 12.2) |
| Slow indexing | Fewer/larger children, batch embed |

---

## What's Next

Parent retrieval widens context; **compression and re-ranking** refine it. **Chapter 12.2** removes noise and re-orders hits for maximum precision.

---

> [← Previous: RAG Evaluation Basics](../phase-11-rag/chapter-50-rag-evaluation-basics.md) | [Next: Contextual Compression →](chapter-52-compression-reranking.md)

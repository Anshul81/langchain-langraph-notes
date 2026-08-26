# Chapter 3.2: Embedding Models

> **Phase 3 — Embeddings & Vector Math** | [← Previous: What Are Embeddings?](chapter-23-what-are-embeddings.md) | [Next: Similarity Search →](chapter-25-similarity-search.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand the landscape of embedding models (cloud vs local)
- ✅ Use OpenAI embedding models (`text-embedding-3-small`, `text-embedding-3-large`)
- ✅ Use HuggingFace / Sentence Transformers (free, local, no API key)
- ✅ Compare models on quality, speed, cost, and dimensions
- ✅ Choose the right model for your use case
- ✅ Build an embedding comparison tool

| | |
|---|---|
| **Prerequisites** | Chapter 3.1 (What Are Embeddings?) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

Not all embedding models are equal. They differ in:

```
                    Quality
                      ↑
   text-embedding-3-large ●
                          |   Cohere embed-v3 ●
                          |
   text-embedding-3-small ●
                          |        BGE-large ●
                          |
         all-MiniLM-L6 ● |
                          |
                ──────────┼──────────→ Speed / Cost
                 Expensive       Cheap / Free
```

You need to know when to use which.

---

## Part 1: The Embedding Model Landscape

| Category | Models | Cost | Speed | Quality | Privacy |
|----------|--------|------|-------|---------|---------|
| **Cloud APIs** | OpenAI, Cohere, Google, Voyage | Pay per token | Fast (GPU servers) | Highest | ❌ Data sent to cloud |
| **Local (Sentence Transformers)** | all-MiniLM, BGE, E5 | Free | Medium (your hardware) | Good to Great | ✅ Data stays local |
| **Local (Ollama)** | nomic-embed, mxbai-embed | Free | Medium | Good | ✅ Data stays local |

### When to Use What

| Scenario | Best Choice | Why |
|----------|-------------|-----|
| Production app, quality matters | OpenAI `text-embedding-3-small` | Best balance of quality/cost |
| Maximum accuracy needed | OpenAI `text-embedding-3-large` or Cohere `embed-v3` | Highest quality |
| Sensitive data (HIPAA, finance) | Sentence Transformers (local) | Data never leaves your machine |
| Learning / experimenting | Sentence Transformers (local) | Free, no API key needed |
| Budget constrained | Sentence Transformers or Ollama | Zero cost |
| Very high throughput | Sentence Transformers on GPU | No rate limits, no API latency |

---

## Part 2: OpenAI Embedding Models

### Available Models

| Model | Dimensions | Max Tokens | Cost (per 1M tokens) | Quality |
|-------|-----------|------------|---------------------|---------|
| `text-embedding-3-small` | 1536 | 8191 | ~$0.02 | ★★★★☆ |
| `text-embedding-3-large` | 3072 | 8191 | ~$0.13 | ★★★★★ |
| `text-embedding-ada-002` | 1536 | 8191 | ~$0.10 | ★★★☆☆ (legacy) |

### Using OpenAI Directly

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE")
)

# Single text
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="LangChain is a framework for building AI applications."
)
embedding = response.data[0].embedding
print(f"Dimensions: {len(embedding)}")  # 1536
print(f"First 5: {embedding[:5]}")

# Batch (multiple texts in one call — cheaper and faster)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "I love eating pizza."
    ]
)
embeddings = [item.embedding for item in response.data]
print(f"Got {len(embeddings)} embeddings, each {len(embeddings[0])}D")
```

### Using OpenAI via LangChain

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# LangChain wrapper — same API, integrates with vector stores
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

# Single text
vector = embeddings_model.embed_query("What is machine learning?")
print(f"Query embedding: {len(vector)} dimensions")

# Multiple documents (batch)
vectors = embeddings_model.embed_documents([
    "Machine learning is a branch of AI.",
    "Neural networks mimic the brain.",
    "Pizza is my favorite food."
])
print(f"Document embeddings: {len(vectors)} vectors, each {len(vectors[0])}D")
```

### ⚠️ `embed_query` vs `embed_documents`

```python
# embed_query() — for the USER'S QUESTION (single text)
query_vec = embeddings_model.embed_query("What is AI?")

# embed_documents() — for the DOCUMENTS to search through (batch)
doc_vecs = embeddings_model.embed_documents(["Doc 1...", "Doc 2..."])

# Some models (like Cohere) use different strategies for queries vs documents.
# OpenAI treats them the same, but always use the correct method
# for compatibility with all providers.
```

### Dimensionality Reduction (text-embedding-3 Feature)

OpenAI's v3 models support **Matryoshka embeddings** — you can request fewer dimensions:

```python
# Full dimensions: 1536
embeddings_full = OpenAIEmbeddings(model="text-embedding-3-small")

# Reduced dimensions: 512 (faster search, less storage, slightly lower quality)
embeddings_small = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512  # Only v3 models support this!
)

vec_full = embeddings_full.embed_query("Hello world")
vec_small = embeddings_small.embed_query("Hello world")

print(f"Full: {len(vec_full)}D")    # 1536
print(f"Small: {len(vec_small)}D")  # 512
```

**Tradeoff**: Fewer dimensions = faster search + less storage, but slightly reduced accuracy.

| Dimensions | Storage per 1M docs | Search Speed | Quality Loss |
|-----------|-------------------|-------------|-------------|
| 3072 | ~12 GB | Slowest | None (best) |
| 1536 | ~6 GB | Medium | Negligible |
| 512 | ~2 GB | Fast | Small (~2%) |
| 256 | ~1 GB | Fastest | Moderate (~5%) |

---

## Part 3: Sentence Transformers (Free, Local)

The open-source alternative. No API key, no cost, runs on your machine.

### Installation

```bash
pip install sentence-transformers
```

### Basic Usage

```python
from sentence_transformers import SentenceTransformer

# Download model (~90 MB, cached after first download)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Single text
embedding = model.encode("LangChain is amazing")
print(f"Dimensions: {len(embedding)}")  # 384
print(f"Type: {type(embedding)}")       # numpy.ndarray

# Batch (very fast — runs locally)
texts = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "I love eating pizza."
]
embeddings = model.encode(texts)
print(f"Shape: {embeddings.shape}")  # (3, 384)
```

### Popular Sentence Transformer Models

| Model | Dimensions | Size | Quality | Speed |
|-------|-----------|------|---------|-------|
| `all-MiniLM-L6-v2` | 384 | 80 MB | ★★★☆☆ | ★★★★★ (fastest) |
| `all-mpnet-base-v2` | 768 | 420 MB | ★★★★☆ | ★★★☆☆ |
| `BAAI/bge-small-en-v1.5` | 384 | 130 MB | ★★★★☆ | ★★★★☆ |
| `BAAI/bge-large-en-v1.5` | 1024 | 1.3 GB | ★★★★★ | ★★☆☆☆ |
| `intfloat/e5-large-v2` | 1024 | 1.3 GB | ★★★★★ | ★★☆☆☆ |

### Using Sentence Transformers via LangChain

```python
from langchain_huggingface import HuggingFaceEmbeddings

# LangChain wrapper — plugs into vector stores seamlessly
embeddings_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},          # or "cuda" for GPU
    encode_kwargs={"normalize_embeddings": True}  # Normalize for cosine similarity
)

# Same API as OpenAIEmbeddings!
query_vec = embeddings_model.embed_query("What is machine learning?")
doc_vecs = embeddings_model.embed_documents(["Doc 1", "Doc 2", "Doc 3"])

print(f"Query: {len(query_vec)}D")        # 384
print(f"Docs: {len(doc_vecs)} vectors")   # 3
```

### GPU Acceleration

```python
# If you have an NVIDIA GPU with CUDA:
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},  # 10-50x faster than CPU!
    encode_kwargs={"normalize_embeddings": True, "batch_size": 64}
)
```

---

## Part 4: Ollama Embeddings (Local, Easy Setup)

If you already use Ollama for local LLMs, it also serves embeddings:

```bash
# Pull an embedding model
ollama pull nomic-embed-text
```

```python
from langchain_ollama import OllamaEmbeddings

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

query_vec = embeddings_model.embed_query("What is machine learning?")
print(f"Dimensions: {len(query_vec)}")  # 768
```

| Model | Dimensions | Size | Quality |
|-------|-----------|------|---------|
| `nomic-embed-text` | 768 | 274 MB | ★★★★☆ |
| `mxbai-embed-large` | 1024 | 670 MB | ★★★★☆ |
| `snowflake-arctic-embed` | 1024 | 670 MB | ★★★★☆ |

---

## Part 5: Comparing Models Head-to-Head

```python
import os
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- Setup models ---

# Model 1: OpenAI (cloud)
from langchain_openai import OpenAIEmbeddings
openai_emb = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

# Model 2: Sentence Transformers (local)
from langchain_huggingface import HuggingFaceEmbeddings
local_emb = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- Test Data ---
test_pairs = [
    ("The cat sat on the mat.", "A feline rested on the rug."),           # Similar
    ("Machine learning is powerful.", "AI algorithms are very capable."),  # Similar
    ("The cat sat on the mat.", "Stock prices rose sharply today."),       # Different
    ("I love programming in Python.", "Python is my favorite language."),  # Very similar
    ("The weather is sunny.", "Financial markets are volatile."),          # Different
]


# --- Compare ---
print(f"{'Pair':<60} {'OpenAI':>8} {'Local':>8}")
print("─" * 80)

for text1, text2 in test_pairs:
    # OpenAI
    start = time.time()
    oai_v1 = openai_emb.embed_query(text1)
    oai_v2 = openai_emb.embed_query(text2)
    oai_time = time.time() - start
    oai_sim = cosine_sim(oai_v1, oai_v2)

    # Local
    start = time.time()
    loc_v1 = local_emb.embed_query(text1)
    loc_v2 = local_emb.embed_query(text2)
    loc_time = time.time() - start
    loc_sim = cosine_sim(loc_v1, loc_v2)

    label = f"{text1[:25]}... ↔ {text2[:25]}..."
    print(f"{label:<60} {oai_sim:>8.4f} {loc_sim:>8.4f}")

print(f"\n📐 OpenAI dimensions: {len(oai_v1)}")
print(f"📐 Local dimensions: {len(loc_v1)}")
```

### Expected Results

```
Both models should agree on the ranking:
  ✅ "Python programming" ↔ "Python favorite" → highest similarity
  ✅ "cat on mat" ↔ "feline on rug" → high similarity
  ✅ "cat on mat" ↔ "stock prices" → low similarity

But absolute values differ:
  OpenAI: similar pairs → 0.85-0.95, different → 0.15-0.35
  Local:  similar pairs → 0.70-0.90, different → 0.05-0.25
```

---

## Part 6: Cost Analysis

### OpenAI Pricing

```python
# text-embedding-3-small: ~$0.02 per 1M tokens
# Average document chunk: ~200 tokens

# Embedding 100K document chunks:
chunks = 100_000
avg_tokens = 200
total_tokens = chunks * avg_tokens  # 20M tokens
cost = (total_tokens / 1_000_000) * 0.02  # $0.40

print(f"Embedding 100K chunks: ${cost:.2f}")  # $0.40

# Query cost (one search):
query_tokens = 20
query_cost = (query_tokens / 1_000_000) * 0.02
print(f"Per query: ${query_cost:.6f}")  # $0.0000004 (basically free)
```

### Sentence Transformers Cost

```python
# Cost: $0.00 (runs locally)
# But: uses your CPU/GPU and RAM
# Speed: ~100-500 texts/second on CPU, ~5000-20000 on GPU
```

### Total Cost Comparison (100K documents, 1000 queries/day, 30 days)

| Model | Embedding Cost | Query Cost (30 days) | Total |
|-------|---------------|---------------------|-------|
| OpenAI `3-small` | $0.40 | $0.012 | **$0.41** |
| OpenAI `3-large` | $2.60 | $0.078 | **$2.68** |
| Sentence Transformers | $0 | $0 | **$0** |
| Ollama | $0 | $0 | **$0** |

**Conclusion**: OpenAI embeddings are **very cheap**. The main reason to use local models is **privacy** or **zero-latency**, not cost.

---

## Common Mistakes

### Mistake 1: Mixing embedding models in the same vector store
```python
# ❌ NEVER do this — vectors live in different spaces!
store.add(openai_emb.embed_documents(["doc1"]))      # 1536D
store.add(huggingface_emb.embed_documents(["doc2"]))  # 384D — CRASH or garbage results

# ✅ One model per vector store, always
store.add(openai_emb.embed_documents(["doc1", "doc2"]))
```

### Mistake 2: Not normalizing local embeddings
```python
# ❌ Some local models output non-normalized vectors
model = HuggingFaceEmbeddings(model_name="some-model")
# Cosine similarity might give unexpected results!

# ✅ Always normalize
model = HuggingFaceEmbeddings(
    model_name="some-model",
    encode_kwargs={"normalize_embeddings": True}  # ← important!
)
```

### Mistake 3: Embedding one text at a time in a loop
```python
# ❌ 1000 API calls — slow and expensive
embeddings = []
for doc in documents:
    emb = model.embed_query(doc)  # One API call per doc!
    embeddings.append(emb)

# ✅ Batch — one API call for all
embeddings = model.embed_documents(documents)
```

### Mistake 4: Using `embed_query` for documents
```python
# ❌ Technically works for OpenAI, but breaks with other providers
doc_vecs = [model.embed_query(doc) for doc in docs]

# ✅ Use the correct method
doc_vecs = model.embed_documents(docs)  # Batch + correct strategy
query_vec = model.embed_query(question)  # Single + query strategy
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Start with `text-embedding-3-small` for cloud apps | Best quality/cost balance |
| Start with `all-MiniLM-L6-v2` for local/free apps | Fastest, decent quality |
| Always use `embed_documents()` for docs, `embed_query()` for queries | Some providers optimize differently |
| Batch your embedding calls | Faster and cheaper |
| Normalize local model outputs | Required for cosine similarity |
| Choose ONE model per project and stick with it | Switching requires re-embedding everything |
| Store model name in metadata | Know which model generated your vectors |

---

## Interview Preparation

### Easy
**Q: Name three embedding model providers.**

> (1) **OpenAI** — `text-embedding-3-small/large`, cloud API, highest quality. (2) **Sentence Transformers / HuggingFace** — open-source models like `all-MiniLM-L6-v2` and `BGE`, run locally for free. (3) **Cohere** — `embed-v3`, cloud API with separate query/document embedding modes for better retrieval.

### Medium
**Q: Why does LangChain distinguish between `embed_query()` and `embed_documents()`?**

> Some embedding models (notably Cohere) use **different strategies** for queries vs documents. For example, Cohere prefixes queries with "search_query:" and documents with "search_document:" to optimize the vector space for retrieval. OpenAI doesn't differentiate, but using the correct method ensures your code works with any provider. `embed_documents()` is also designed for batch processing.

### Hard
**Q: You have 10 million documents. Should you use OpenAI or local embeddings?**

> **Depends on the constraints.** OpenAI cost: 10M × 200 tokens × $0.02/1M = ~$40 — very reasonable. But you're sending 10M documents over the internet, which raises privacy concerns and takes time (rate limits). Local Sentence Transformers on a GPU can process ~10K docs/second, finishing in ~17 minutes with zero cost and full privacy. **Recommendation**: Use local for initial bulk indexing, OpenAI for queries (higher quality for the thing that matters most — matching). Or use local for everything if privacy is critical.

### Senior
**Q: What are Matryoshka embeddings and why are they useful?**

> Matryoshka embeddings (supported by OpenAI v3 and some open-source models like `nomic-embed-text`) are trained so that the **first N dimensions** of a vector carry the most important information. You can truncate a 3072D vector to 512D or 256D with graceful quality degradation. This enables: (1) **adaptive retrieval** — do a fast first-pass search with 256D, then re-rank with full 3072D, (2) **storage savings** — 6x less storage with 512D vs 3072D, (3) **flexible deployment** — use fewer dimensions on edge devices, full dimensions on servers.

---

## Summary

| Component | What It Does |
|-----------|-------------|
| `OpenAIEmbeddings(model="text-embedding-3-small")` | Cloud embeddings via OpenAI (1536D, ~$0.02/1M tokens) |
| `OpenAIEmbeddings(dimensions=512)` | Reduced dimensionality (Matryoshka, v3 only) |
| `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` | Free local embeddings (384D, fast) |
| `HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")` | Free local, high quality (1024D) |
| `OllamaEmbeddings(model="nomic-embed-text")` | Local via Ollama (768D) |
| `embed_query(text)` | Embed a single query (optimized for search) |
| `embed_documents(texts)` | Embed a batch of documents (bulk processing) |

---

> [← Previous: What Are Embeddings?](chapter-23-what-are-embeddings.md) | [Next: Similarity Search →](chapter-25-similarity-search.md)

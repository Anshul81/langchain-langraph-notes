# Chapter 4.3: FAISS — Facebook AI Similarity Search

> **Phase 4 — Vector Databases** | [← Previous: ChromaDB](chapter-27-chromadb.md) | [Next: Cloud Vector DBs →](chapter-29-cloud-vector-dbs.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand FAISS's architecture and when to choose it over ChromaDB
- ✅ Build indexes with different algorithms (Flat, IVF, HNSW, PQ)
- ✅ Choose the right index type for your scale and accuracy needs
- ✅ Integrate FAISS with LangChain via the `FAISS` wrapper
- ✅ Save and load indexes for persistence
- ✅ Build a fast similarity search system handling large datasets

| | |
|---|---|
| **Prerequisites** | Chapter 4.1 (Why Vector DBs?), Chapter 4.2 (ChromaDB) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction — FAISS vs ChromaDB

| Aspect | ChromaDB | FAISS |
|--------|----------|-------|
| **Built by** | Chroma Inc. | Meta (Facebook AI Research) |
| **Type** | Full vector database | Vector search **library** |
| **Metadata storage** | ✅ Built-in | ❌ None — you manage it yourself |
| **Document storage** | ✅ Built-in | ❌ None — you manage it yourself |
| **CRUD operations** | ✅ Full CRUD | ❌ Add + Search only (no delete/update by default) |
| **Index algorithms** | HNSW only | Flat, IVF, HNSW, PQ, and combinations |
| **Speed at scale** | Good (<5M) | **Excellent** (100M+) |
| **GPU support** | ❌ | ✅ (faiss-gpu) |
| **Memory efficiency** | Medium | **Excellent** (PQ compression) |
| **Ease of use** | ★★★★★ | ★★★☆☆ |
| **Best for** | Prototyping, small apps | Large-scale, performance-critical |

**TL;DR**: ChromaDB is a **database** (stores everything). FAISS is a **search engine** (just the index — you store docs/metadata separately).

```bash
pip install faiss-cpu          # CPU version
# pip install faiss-gpu        # GPU version (requires CUDA)
```

---

## Part 1: FAISS Native API — Flat Index (Brute Force)

The simplest index — stores all vectors and searches by exhaustive comparison. 100% accurate but O(n).

```python
import numpy as np
import faiss

# --- Create some sample data ---
dimension = 128          # Embedding dimensions
num_vectors = 10000      # Number of documents

# Random vectors (simulating embeddings)
np.random.seed(42)
data = np.random.random((num_vectors, dimension)).astype("float32")

# Normalize (required for cosine similarity via inner product)
faiss.normalize_L2(data)

# --- Build a Flat Index ---
# IndexFlatIP = Inner Product (= cosine similarity for normalized vectors)
# IndexFlatL2 = Euclidean distance
index = faiss.IndexFlatIP(dimension)

# Add vectors
index.add(data)
print(f"Index contains {index.ntotal} vectors")  # 10000

# --- Search ---
query = np.random.random((1, dimension)).astype("float32")
faiss.normalize_L2(query)

k = 5  # Top 5 results
distances, indices = index.search(query, k)

print(f"Top {k} results:")
print(f"  Indices:   {indices[0]}")     # [4821, 7293, 1044, ...]
print(f"  Distances: {distances[0]}")   # [0.82, 0.79, 0.77, ...] (similarity scores)
```

### Understanding the Return Values

```python
# distances: (num_queries, k) — similarity/distance scores
# indices:   (num_queries, k) — positions in the original data array

# For IndexFlatIP (inner product): higher = more similar
# For IndexFlatL2 (euclidean):     lower  = more similar

# Batch search — multiple queries at once
queries = np.random.random((5, dimension)).astype("float32")
faiss.normalize_L2(queries)
distances, indices = index.search(queries, k)
# distances.shape = (5, 5) — 5 queries × 5 results each
```

---

## Part 2: IVF Index — Fast Approximate Search

Partitions vectors into clusters (Voronoi cells). Searches only nearby clusters instead of everything.

```python
# --- IVF Index ---
dimension = 128
num_vectors = 100000

data = np.random.random((num_vectors, dimension)).astype("float32")
faiss.normalize_L2(data)

# Number of clusters (rule of thumb: sqrt(n) to 4*sqrt(n))
nlist = 100  # 100 clusters for 100K vectors

# IVF requires a quantizer (sub-index for cluster centroids)
quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# IVF MUST be trained before adding data!
print("Training index...")
index.train(data)        # Learns cluster centroids
print("Adding vectors...")
index.add(data)          # Assigns each vector to its nearest cluster
print(f"Index: {index.ntotal} vectors in {nlist} clusters")

# --- Search ---
query = np.random.random((1, dimension)).astype("float32")
faiss.normalize_L2(query)

# nprobe: how many clusters to search (accuracy vs speed tradeoff)
index.nprobe = 10  # Search 10 of 100 clusters (10%)

distances, indices = index.search(query, 5)
print(f"Results: {indices[0]}")
print(f"Scores:  {distances[0]}")
```

### Tuning nprobe

```python
import time

for nprobe in [1, 5, 10, 25, 50, 100]:
    index.nprobe = nprobe
    
    start = time.time()
    distances, indices = index.search(query, 5)
    elapsed = (time.time() - start) * 1000
    
    print(f"nprobe={nprobe:>3}  time={elapsed:>6.2f}ms  top_score={distances[0][0]:.4f}")

# Expected:
# nprobe=  1  time=  0.12ms  top_score=0.7823   ← Fast, might miss best
# nprobe= 10  time=  0.45ms  top_score=0.8234   ← Good balance
# nprobe=100  time=  4.20ms  top_score=0.8241   ← Exhaustive (same as Flat)
```

---

## Part 3: Product Quantization (PQ) — Memory Compression

PQ compresses vectors to use 4-16x less memory. Critical for 100M+ vector datasets.

```python
# --- IVF + PQ: Fast search + compressed storage ---
dimension = 128
num_vectors = 100000

data = np.random.random((num_vectors, dimension)).astype("float32")
faiss.normalize_L2(data)

nlist = 100
m = 16         # Number of sub-quantizers (dimension must be divisible by m)
nbits = 8      # Bits per sub-quantizer (usually 8)

index = faiss.IndexIVFPQ(
    faiss.IndexFlatIP(dimension),
    dimension,
    nlist,
    m,        # Splits 128D into 16 sub-vectors of 8D each
    nbits     # Each sub-vector quantized to 2^8 = 256 centroids
)

index.train(data)
index.add(data)

# Memory comparison
flat_memory = num_vectors * dimension * 4  # 4 bytes per float32
pq_memory = num_vectors * m * (nbits // 8)  # Much smaller!

print(f"Flat memory:  {flat_memory / 1e6:.1f} MB")   # ~51.2 MB
print(f"PQ memory:    {pq_memory / 1e6:.1f} MB")     # ~1.6 MB  (32x smaller!)

# Search
index.nprobe = 10
query = np.random.random((1, dimension)).astype("float32")
faiss.normalize_L2(query)

distances, indices = index.search(query, 5)
print(f"Results: {indices[0]}")
# Slightly less accurate than Flat, but uses 32x less memory!
```

---

## Part 4: HNSW Index in FAISS

```python
# FAISS also supports HNSW (like ChromaDB uses)
dimension = 128
num_vectors = 100000

data = np.random.random((num_vectors, dimension)).astype("float32")
faiss.normalize_L2(data)

# HNSW parameters
M = 32              # Number of connections per node (higher = better recall, more memory)
ef_construction = 64  # Construction quality (higher = better index, slower build)

index = faiss.IndexHNSWFlat(dimension, M)
index.hnsw.efConstruction = ef_construction
index.hnsw.efSearch = 32  # Search quality (higher = better recall, slower search)

# HNSW doesn't need training!
index.add(data)

# Search
query = np.random.random((1, dimension)).astype("float32")
faiss.normalize_L2(query)

distances, indices = index.search(query, 5)
print(f"Results: {indices[0]}")
# Note: HNSW uses L2 distance by default in FAISS (lower = more similar)
```

---

## Part 5: Index Comparison

```python
import time
import numpy as np
import faiss

dimension = 128
num_vectors = 100000
data = np.random.random((num_vectors, dimension)).astype("float32")
faiss.normalize_L2(data)

query = np.random.random((1, dimension)).astype("float32")
faiss.normalize_L2(query)

results = {}

# --- Flat (Brute Force) ---
idx = faiss.IndexFlatIP(dimension)
idx.add(data)
start = time.time()
d, i = idx.search(query, 5)
results["Flat"] = {
    "time_ms": (time.time() - start) * 1000,
    "top_id": i[0][0],
    "top_score": d[0][0]
}

# --- IVF Flat ---
quantizer = faiss.IndexFlatIP(dimension)
idx = faiss.IndexIVFFlat(quantizer, dimension, 100, faiss.METRIC_INNER_PRODUCT)
idx.train(data)
idx.add(data)
idx.nprobe = 10
start = time.time()
d, i = idx.search(query, 5)
results["IVFFlat"] = {
    "time_ms": (time.time() - start) * 1000,
    "top_id": i[0][0],
    "top_score": d[0][0]
}

# --- IVF + PQ ---
idx = faiss.IndexIVFPQ(faiss.IndexFlatIP(dimension), dimension, 100, 16, 8)
idx.train(data)
idx.add(data)
idx.nprobe = 10
start = time.time()
d, i = idx.search(query, 5)
results["IVFPQ"] = {
    "time_ms": (time.time() - start) * 1000,
    "top_id": i[0][0],
    "top_score": d[0][0]
}

# --- Print ---
print(f"{'Index':<12} {'Time (ms)':<12} {'Top ID':<10} {'Top Score':<12}")
print("─" * 50)
for name, r in results.items():
    print(f"{name:<12} {r['time_ms']:<12.3f} {r['top_id']:<10} {r['top_score']:<12.4f}")
```

### Decision Table

| Index Type | Speed | Accuracy | Memory | Training | Best For |
|-----------|-------|----------|--------|----------|----------|
| `IndexFlatIP/L2` | Slowest | 100% exact | High | No | <100K vectors, ground truth |
| `IndexIVFFlat` | Fast | ~95-99% | High | Yes | 100K-10M vectors |
| `IndexIVFPQ` | Fastest | ~90-95% | **Very low** | Yes | 10M-1B vectors, RAM constrained |
| `IndexHNSWFlat` | Fast | ~97-99% | High | No | 100K-10M, no training needed |

---

## Part 6: Saving and Loading Indexes

```python
import faiss

# --- Save ---
faiss.write_index(index, "my_index.faiss")
print("Index saved to disk")

# --- Load ---
loaded_index = faiss.read_index("my_index.faiss")
print(f"Loaded index with {loaded_index.ntotal} vectors")

# Search works immediately
distances, indices = loaded_index.search(query, 5)
```

### ⚠️ FAISS Only Saves Vectors, Not Documents

```python
import json

# You must save your document mapping separately!
doc_mapping = {
    0: {"content": "Machine learning is...", "source": "textbook.pdf"},
    1: {"content": "Neural networks are...", "source": "dl_guide.pdf"},
    # ...
}

# Save mapping alongside index
with open("doc_mapping.json", "w") as f:
    json.dump(doc_mapping, f)

# Load both
index = faiss.read_index("my_index.faiss")
with open("doc_mapping.json", "r") as f:
    doc_mapping = json.load(f)

# Search and retrieve documents
distances, indices = index.search(query, 5)
for idx in indices[0]:
    doc = doc_mapping[str(idx)]
    print(f"📄 {doc['content'][:60]}... (source: {doc['source']})")
```

---

## Part 7: FAISS with LangChain

The easiest way to use FAISS — LangChain handles document storage and ID mapping for you.

```bash
pip install langchain-community faiss-cpu
```

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

# --- Create from documents ---
docs = [
    Document(page_content="Machine learning enables computers to learn from data.",
             metadata={"source": "ml.pdf", "category": "AI"}),
    Document(page_content="Deep learning uses multi-layered neural networks.",
             metadata={"source": "dl.pdf", "category": "AI"}),
    Document(page_content="Docker packages apps into portable containers.",
             metadata={"source": "docker.pdf", "category": "DevOps"}),
    Document(page_content="Python is the most popular language for data science.",
             metadata={"source": "python.pdf", "category": "Programming"}),
    Document(page_content="Transformers use self-attention for sequence processing.",
             metadata={"source": "dl.pdf", "category": "AI"}),
    Document(page_content="Kubernetes orchestrates containerized applications.",
             metadata={"source": "k8s.pdf", "category": "DevOps"}),
]

vectorstore = FAISS.from_documents(docs, embeddings)
print(f"Created FAISS store with {vectorstore.index.ntotal} vectors")
```

### Searching

```python
# Basic search
results = vectorstore.similarity_search("How do neural networks work?", k=3)
for doc in results:
    print(f"📄 [{doc.metadata['category']}] {doc.page_content[:60]}...")

# Search with scores
results = vectorstore.similarity_search_with_score("machine learning", k=3)
for doc, score in results:
    print(f"[{score:.4f}] {doc.page_content[:60]}...")
    # Note: FAISS LangChain returns L2 distances (lower = more similar)

# Metadata filtering
results = vectorstore.similarity_search(
    "learning algorithms",
    k=3,
    filter={"category": "AI"}
)
```

### MMR Search

```python
results = vectorstore.max_marginal_relevance_search(
    "AI and deep learning",
    k=3,
    fetch_k=10,
    lambda_mult=0.7
)
for doc in results:
    print(f"📄 {doc.page_content[:60]}...")
```

### Save and Load

```python
# Save (creates a directory with index + docstore)
vectorstore.save_local("./faiss_index")

# Load
loaded_store = FAISS.load_local(
    "./faiss_index",
    embeddings,
    allow_dangerous_deserialization=True  # Required — FAISS uses pickle
)

results = loaded_store.similarity_search("neural networks", k=2)
```

### Add More Documents

```python
new_docs = [
    Document(page_content="Redis is an in-memory cache and message broker.",
             metadata={"source": "redis.pdf", "category": "Database"})
]
vectorstore.add_documents(new_docs)
print(f"Now have {vectorstore.index.ntotal} vectors")
```

### Merge Two FAISS Indexes

```python
# Useful for combining indexes built separately
store_1 = FAISS.from_documents(docs[:3], embeddings)
store_2 = FAISS.from_documents(docs[3:], embeddings)

store_1.merge_from(store_2)
print(f"Merged store: {store_1.index.ntotal} vectors")
```

### As a Retriever

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

# Plug directly into RAG chains
relevant_docs = retriever.invoke("How does machine learning work?")
```

---

## Part 8: FAISS vs ChromaDB — When to Use Which

| Scenario | Winner | Why |
|----------|--------|-----|
| Quick prototype | **ChromaDB** | Zero config, built-in doc storage |
| >1M vectors | **FAISS** | Better algorithms, GPU support |
| Need metadata filtering | **ChromaDB** | Native, rich operators |
| Need metadata filtering (LangChain) | **Tie** | Both support it via LangChain |
| Need to merge indexes | **FAISS** | `merge_from()` built-in |
| Need delete/update | **ChromaDB** | Full CRUD native |
| Memory constrained | **FAISS** | PQ compression (32x smaller) |
| GPU available | **FAISS** | faiss-gpu for massive speedup |
| Simple persistence | **ChromaDB** | Just set persist_directory |
| Maximum search speed | **FAISS** | IVF+PQ on GPU is unbeatable |

---

## Common Mistakes

### Mistake 1: Forgetting to normalize vectors for inner product search
```python
# ❌ Non-normalized vectors + IndexFlatIP = not cosine similarity
index = faiss.IndexFlatIP(dim)
index.add(raw_vectors)  # Not normalized!

# ✅ Normalize before adding
faiss.normalize_L2(vectors)
index.add(vectors)
```

### Mistake 2: Forgetting to train IVF indexes
```python
# ❌ Adding without training — will crash!
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
index.add(data)  # RuntimeError: index not trained!

# ✅ Train first, then add
index.train(data)
index.add(data)
```

### Mistake 3: Not saving document mapping alongside FAISS index
```python
# ❌ FAISS only returns integer indices — useless without a mapping
distances, indices = index.search(query, 5)
# indices = [4821, 7293, ...] — what documents are these?!

# ✅ Maintain a mapping (or use LangChain which does it for you)
doc_mapping = {i: doc for i, doc in enumerate(documents)}
```

### Mistake 4: Using float64 instead of float32
```python
# ❌ FAISS requires float32 — float64 will crash
data = np.random.random((1000, 128))  # Default is float64!
index.add(data)  # Error!

# ✅ Always cast to float32
data = np.random.random((1000, 128)).astype("float32")
```

### Mistake 5: Ignoring `allow_dangerous_deserialization`
```python
# ❌ Loading fails without this flag (security feature)
store = FAISS.load_local("./faiss_index", embeddings)

# ✅ Acknowledge the pickle deserialization risk
store = FAISS.load_local("./faiss_index", embeddings,
                          allow_dangerous_deserialization=True)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Start with `IndexFlatIP` for prototypes | Exact results, no training needed |
| Use `IndexIVFFlat` for 100K-10M vectors | Good speed/accuracy balance |
| Use `IndexIVFPQ` for 10M+ vectors | 32x memory savings |
| Always normalize vectors for cosine similarity | Use `faiss.normalize_L2()` |
| Use `float32` dtype | FAISS requirement |
| Save doc mapping alongside index | FAISS doesn't store documents |
| Use LangChain's FAISS wrapper for most cases | Handles doc storage, IDs, metadata |
| Use `merge_from` for distributed indexing | Build indexes in parallel, merge later |
| Tune `nprobe` for IVF indexes | Balance speed vs accuracy |

---

## Interview Preparation

### Easy
**Q: What is FAISS?**

> FAISS (Facebook AI Similarity Search) is an open-source library from Meta for efficient similarity search of dense vectors. Unlike ChromaDB which is a full database, FAISS is a pure search library — it handles indexing and searching vectors but doesn't store documents or metadata. It supports multiple index types (Flat, IVF, HNSW, PQ) and is the fastest option for large-scale vector search, supporting both CPU and GPU.

### Medium
**Q: What is Product Quantization and why does it matter?**

> Product Quantization (PQ) compresses high-dimensional vectors by splitting each vector into sub-vectors and quantizing each sub-vector to its nearest centroid from a learned codebook. A 128D float32 vector (512 bytes) can be compressed to ~16 bytes (32x reduction). This dramatically reduces memory usage — making it possible to search 1 billion vectors on a single machine. The tradeoff is a small accuracy loss (~5-10%) because the compressed representation is approximate. FAISS's `IndexIVFPQ` combines IVF clustering with PQ compression for fast, memory-efficient search at massive scale.

### Hard
**Q: You have 500 million 1536D embeddings. Design the FAISS index.**

> At 500M × 1536D × 4 bytes = ~3TB for raw vectors — won't fit in RAM. Solution: **IndexIVFPQ** with settings: `nlist=10000` (clusters), `m=96` (sub-quantizers, 1536/96=16D each), `nbits=8`. This compresses each vector from 6KB to ~96 bytes — total ~45GB, fits in RAM. Training uses a representative sample (~5M vectors). At search time, set `nprobe=32-64` for good recall. For even more speed, use `faiss-gpu` to run the search on GPU. Add an `IndexIDMap` wrapper to map FAISS indices to your document IDs. Store actual documents in a separate database (PostgreSQL/Redis) and join by ID after search.

### Senior
**Q: ChromaDB vs FAISS vs pgvector — how do you choose for a production RAG system?**

> **pgvector**: Choose when you already have PostgreSQL, need ACID transactions, need to JOIN vector results with relational data (user permissions, document metadata), and have <5M vectors. Simplest infrastructure story. **ChromaDB**: Choose for rapid prototyping, small-to-medium apps (<1M), or when you want a self-contained solution with built-in document storage. Easy to start, hard to scale. **FAISS**: Choose for maximum performance, 10M+ vectors, when you control the infrastructure, or need GPU acceleration. Requires more engineering (separate doc storage, ID mapping). In practice, many production systems use **pgvector** because it avoids adding new infrastructure, with FAISS reserved for cases where pgvector's performance isn't sufficient.

---

## Summary

| Component | What It Does |
|-----------|-------------|
| `IndexFlatIP` / `IndexFlatL2` | Brute-force exact search (baseline) |
| `IndexIVFFlat` | Cluster-based approximate search (fast) |
| `IndexIVFPQ` | Cluster + compression (fast + memory efficient) |
| `IndexHNSWFlat` | Graph-based approximate search (no training) |
| `faiss.normalize_L2(vectors)` | Normalize for cosine similarity |
| `index.train(data)` | Train IVF/PQ indexes on data |
| `index.nprobe = N` | Tune IVF search accuracy vs speed |
| `faiss.write_index()` / `read_index()` | Save/load indexes to disk |
| `FAISS.from_documents(docs, emb)` | LangChain: create FAISS store from documents |
| `vectorstore.merge_from(other)` | LangChain: merge two FAISS stores |

---

> [← Previous: ChromaDB](chapter-27-chromadb.md) | [Next: Cloud Vector DBs →](chapter-29-cloud-vector-dbs.md)

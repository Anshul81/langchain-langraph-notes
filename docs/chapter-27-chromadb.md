# Chapter 4.2: ChromaDB — Local Vector Store

> **Phase 4 — Vector Databases** | [← Previous: Why Vector Databases?](chapter-26-why-vector-databases.md) | [Next: FAISS →](chapter-28-faiss.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Install and configure ChromaDB (in-memory and persistent)
- ✅ Create collections, add documents, and query with similarity search
- ✅ Use metadata filtering with `where` and `where_document` clauses
- ✅ Integrate ChromaDB with LangChain via `Chroma` wrapper
- ✅ Build a complete **document store** with CRUD operations
- ✅ Understand ChromaDB's architecture, limitations, and production readiness

| | |
|---|---|
| **Prerequisites** | Chapter 4.1 (Why Vector Databases?), Chapter 3.2 (Embedding Models) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction

ChromaDB is the **SQLite of vector databases** — simple, embedded, zero-config. It's the default choice for:

- Learning and prototyping
- Small-to-medium apps (<1M vectors)
- Local development before migrating to cloud

```bash
pip install chromadb
```

That's it. No server, no Docker, no signup. One line and you're running.

---

## Part 1: ChromaDB Native API

### Creating a Client

```python
import chromadb

# --- Option 1: In-memory (ephemeral — lost when script ends) ---
client = chromadb.Client()

# --- Option 2: Persistent (saved to disk — survives restarts) ---
client = chromadb.PersistentClient(path="./chroma_data")

# --- Option 3: Client-Server (separate ChromaDB server) ---
# First start server: chroma run --path ./chroma_server_data
# client = chromadb.HttpClient(host="localhost", port=8000)
```

### Collections — Like Tables

```python
# Create a collection (or get it if it already exists)
collection = client.get_or_create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # Distance metric: cosine, l2, or ip
)

# List all collections
print(client.list_collections())

# Delete a collection
# client.delete_collection("my_documents")
```

### Adding Documents

ChromaDB can **auto-embed** text using its built-in default model, or you can provide your own embeddings.

```python
# --- Method 1: Let ChromaDB embed for you (uses all-MiniLM-L6-v2 by default) ---
collection.add(
    ids=["doc_1", "doc_2", "doc_3"],
    documents=[
        "Machine learning is a subset of artificial intelligence.",
        "Python is the most popular language for data science.",
        "React is a JavaScript library for building user interfaces."
    ],
    metadatas=[
        {"category": "AI", "source": "textbook", "page": 1},
        {"category": "Programming", "source": "blog", "page": 1},
        {"category": "Web Dev", "source": "docs", "page": 1}
    ]
)

print(f"Collection has {collection.count()} documents")  # 3
```

```python
# --- Method 2: Provide your own embeddings ---
import numpy as np

collection.add(
    ids=["doc_4"],
    embeddings=[[0.12, -0.34, 0.56, ...]],  # Your pre-computed embedding
    documents=["Docker simplifies deployment."],
    metadatas=[{"category": "DevOps"}]
)
```

### ⚠️ IDs Must Be Unique

```python
# ❌ Adding with duplicate ID silently overwrites!
collection.add(ids=["doc_1"], documents=["New content"])  # Overwrites old doc_1

# ✅ Use meaningful, unique IDs
import hashlib
def make_id(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
```

---

## Part 2: Querying — Similarity Search

### Basic Query

```python
results = collection.query(
    query_texts=["How do computers learn from data?"],
    n_results=3
)

# Results structure
print(results["ids"])         # [["doc_1", "doc_3", "doc_2"]]
print(results["documents"])   # [["Machine learning is...", "React is...", "Python..."]]
print(results["metadatas"])   # [[{"category": "AI"}, ...]]
print(results["distances"])   # [[0.234, 0.567, 0.789]]  ← lower = more similar (for cosine distance)
```

### Understanding Results Structure

```python
# Results are nested lists because you can query MULTIPLE queries at once
# results["ids"][0]       → results for query 0
# results["ids"][0][0]    → best match for query 0
# results["ids"][0][1]    → second best match for query 0

# Single query helper:
query_result = results["ids"][0]
for i, doc_id in enumerate(query_result):
    print(f"#{i+1} [{results['distances'][0][i]:.4f}] {results['documents'][0][i][:60]}...")
```

### ⚠️ Distance vs Similarity

```python
# ChromaDB returns DISTANCES, not similarities!
# For cosine space:
#   distance = 1 - cosine_similarity
#   distance = 0 → identical
#   distance = 1 → orthogonal
#   distance = 2 → opposite

# To convert back to similarity:
similarity = 1 - results["distances"][0][0]
```

---

## Part 3: Metadata Filtering

The real power — combine vector search with structured filters.

### `where` — Filter by Metadata

```python
# Only search AI documents
results = collection.query(
    query_texts=["learning algorithms"],
    n_results=5,
    where={"category": "AI"}  # Metadata filter
)

# Multiple conditions (AND)
results = collection.query(
    query_texts=["learning"],
    n_results=5,
    where={
        "$and": [
            {"category": "AI"},
            {"page": {"$gte": 1}}
        ]
    }
)
```

### Filter Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equals (default) | `{"category": "AI"}` or `{"category": {"$eq": "AI"}}` |
| `$ne` | Not equals | `{"category": {"$ne": "Web Dev"}}` |
| `$gt` | Greater than | `{"page": {"$gt": 5}}` |
| `$gte` | Greater than or equal | `{"page": {"$gte": 1}}` |
| `$lt` | Less than | `{"page": {"$lt": 100}}` |
| `$lte` | Less than or equal | `{"page": {"$lte": 50}}` |
| `$in` | In list | `{"category": {"$in": ["AI", "Programming"]}}` |
| `$nin` | Not in list | `{"category": {"$nin": ["Web Dev"]}}` |

### Logical Operators

```python
# AND: all conditions must match
where={"$and": [{"category": "AI"}, {"source": "textbook"}]}

# OR: any condition matches
where={"$or": [{"category": "AI"}, {"category": "Programming"}]}
```

### `where_document` — Filter by Document Content

```python
# Only search documents containing specific text
results = collection.query(
    query_texts=["programming"],
    n_results=5,
    where_document={"$contains": "Python"}  # Must contain "Python"
)

# Does NOT contain
results = collection.query(
    query_texts=["programming"],
    n_results=5,
    where_document={"$not_contains": "JavaScript"}
)
```

---

## Part 4: CRUD Operations

### Get (Retrieve by ID)

```python
# Get specific documents by ID
docs = collection.get(ids=["doc_1", "doc_2"])
print(docs["documents"])  # ["Machine learning is...", "Python is..."]

# Get with filters (no vector search — just metadata filter)
docs = collection.get(
    where={"category": "AI"},
    include=["documents", "metadatas", "embeddings"]
)

# Get all documents
all_docs = collection.get()
print(f"Total: {len(all_docs['ids'])} documents")
```

### Update

```python
# Update document content and/or metadata
collection.update(
    ids=["doc_1"],
    documents=["Updated: Machine learning enables systems to learn from data automatically."],
    metadatas=[{"category": "AI", "source": "textbook", "page": 1, "updated": True}]
)
```

### Upsert (Insert or Update)

```python
# If ID exists → update. If not → insert. Best of both worlds.
collection.upsert(
    ids=["doc_1", "doc_99"],  # doc_1 exists, doc_99 doesn't
    documents=[
        "Updated ML content.",
        "Brand new document about Kubernetes."
    ],
    metadatas=[
        {"category": "AI"},
        {"category": "DevOps"}
    ]
)
```

### Delete

```python
# Delete by ID
collection.delete(ids=["doc_3"])

# Delete by metadata filter
collection.delete(where={"category": "Web Dev"})

# Verify
print(f"Remaining: {collection.count()} documents")
```

---

## Part 5: ChromaDB with LangChain

This is what you'll use most — LangChain's `Chroma` wrapper integrates seamlessly with chains and RAG.

### Setup

```bash
pip install langchain-chroma
```

### Creating a Vector Store from Documents

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

# Create documents
docs = [
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data.",
        metadata={"source": "ml_textbook.pdf", "page": 1, "category": "AI"}
    ),
    Document(
        page_content="Neural networks are inspired by biological neurons and consist of layers.",
        metadata={"source": "dl_guide.pdf", "page": 5, "category": "AI"}
    ),
    Document(
        page_content="Python's pandas library provides DataFrames for data manipulation.",
        metadata={"source": "python_docs.pdf", "page": 12, "category": "Programming"}
    ),
    Document(
        page_content="Docker containers package applications with their dependencies for consistent deployment.",
        metadata={"source": "devops_handbook.pdf", "page": 3, "category": "DevOps"}
    ),
    Document(
        page_content="REST APIs use HTTP methods like GET, POST, PUT, DELETE for resource operations.",
        metadata={"source": "api_design.pdf", "page": 8, "category": "Backend"}
    ),
    Document(
        page_content="Transformers use self-attention mechanisms to process sequences in parallel.",
        metadata={"source": "dl_guide.pdf", "page": 42, "category": "AI"}
    ),
]

# --- In-Memory ---
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="my_knowledge_base"
)

# --- Persistent (saved to disk) ---
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="my_knowledge_base",
    persist_directory="./chroma_langchain_db"
)

print(f"Stored {vectorstore._collection.count()} documents")
```

### Loading an Existing Persistent Store

```python
# Later: reload without re-embedding
vectorstore = Chroma(
    collection_name="my_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)

print(f"Loaded {vectorstore._collection.count()} documents")
```

### Similarity Search

```python
# Basic search
results = vectorstore.similarity_search(
    query="How do neural networks work?",
    k=3
)

for doc in results:
    print(f"📄 [{doc.metadata['source']}] {doc.page_content[:80]}...")
```

### Similarity Search with Scores

```python
# Get similarity scores alongside results
results_with_scores = vectorstore.similarity_search_with_score(
    query="How do neural networks work?",
    k=3
)

for doc, score in results_with_scores:
    # Score = distance (lower = more similar for cosine)
    similarity = 1 - score  # Convert to similarity
    print(f"[{similarity:.4f}] {doc.page_content[:60]}...")
```

### Metadata Filtering in LangChain

```python
# Filter by metadata
results = vectorstore.similarity_search(
    query="learning from data",
    k=3,
    filter={"category": "AI"}  # Only AI documents
)

# Complex filter
results = vectorstore.similarity_search(
    query="learning",
    k=5,
    filter={
        "$and": [
            {"category": "AI"},
            {"page": {"$gt": 1}}
        ]
    }
)
```

### As a Retriever (For RAG Chains)

```python
# Convert to a Retriever — plugs directly into RAG chains!
retriever = vectorstore.as_retriever(
    search_type="similarity",  # or "mmr" (Maximum Marginal Relevance)
    search_kwargs={
        "k": 5,
        "filter": {"category": "AI"}
    }
)

# Use in a chain
relevant_docs = retriever.invoke("How do transformers work?")
for doc in relevant_docs:
    print(f"📄 {doc.page_content[:80]}...")
```

### MMR (Maximum Marginal Relevance) — Diverse Results

```python
# Problem: top-5 similar results might all say the same thing
# MMR: balance between relevance AND diversity

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,               # Return 5 results
        "fetch_k": 20,        # Fetch 20 candidates first
        "lambda_mult": 0.7    # 0=max diversity, 1=max relevance
    }
)

# Or directly:
results = vectorstore.max_marginal_relevance_search(
    query="machine learning applications",
    k=5,
    fetch_k=20,
    lambda_mult=0.7
)
```

### Adding More Documents Later

```python
# Add documents to an existing vectorstore
new_docs = [
    Document(
        page_content="Kubernetes orchestrates containerized applications at scale.",
        metadata={"source": "k8s_docs.pdf", "page": 1, "category": "DevOps"}
    )
]

vectorstore.add_documents(new_docs)
print(f"Now have {vectorstore._collection.count()} documents")
```

### Deleting Documents

```python
# Delete by ID
vectorstore.delete(ids=["some_document_id"])

# To find IDs, use the underlying collection
collection = vectorstore._collection
all_data = collection.get()
print(f"All IDs: {all_data['ids']}")
```

---

## Part 6: Complete Document Store Project

```python
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()


class DocumentStore:
    """A complete document store backed by ChromaDB."""

    def __init__(self, collection_name: str, persist_dir: str = "./doc_store_db"):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
            openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
        )
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )
        self.collection_name = collection_name

    @property
    def count(self) -> int:
        return self.vectorstore._collection.count()

    def add(self, content: str, metadata: dict = None) -> str:
        """Add a single document. Returns the auto-generated ID."""
        import hashlib
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]

        doc = Document(page_content=content, metadata=metadata or {})
        self.vectorstore.add_documents([doc], ids=[doc_id])
        return doc_id

    def add_batch(self, documents: list[dict]) -> list[str]:
        """Add multiple documents. Each dict has 'content' and optional 'metadata'."""
        import hashlib
        docs = []
        ids = []
        for item in documents:
            doc = Document(
                page_content=item["content"],
                metadata=item.get("metadata", {})
            )
            doc_id = hashlib.md5(item["content"].encode()).hexdigest()[:12]
            docs.append(doc)
            ids.append(doc_id)

        self.vectorstore.add_documents(docs, ids=ids)
        return ids

    def search(self, query: str, k: int = 5, filter: dict = None) -> list[dict]:
        """Search for similar documents."""
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity": round(1 - score, 4)  # Convert distance to similarity
            }
            for doc, score in results
        ]

    def search_mmr(self, query: str, k: int = 5, diversity: float = 0.7) -> list[dict]:
        """Search with diversity (MMR)."""
        results = self.vectorstore.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=k * 4,
            lambda_mult=diversity
        )
        return [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in results
        ]

    def get_retriever(self, k: int = 5, filter: dict = None):
        """Get a LangChain retriever for RAG chains."""
        search_kwargs = {"k": k}
        if filter:
            search_kwargs["filter"] = filter
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def list_all(self) -> list[dict]:
        """List all documents in the store."""
        data = self.vectorstore._collection.get(
            include=["documents", "metadatas"]
        )
        return [
            {"id": id, "content": doc, "metadata": meta}
            for id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]

    def delete(self, ids: list[str]):
        """Delete documents by ID."""
        self.vectorstore.delete(ids=ids)

    def stats(self):
        """Print store statistics."""
        all_data = self.vectorstore._collection.get(include=["metadatas"])
        categories = {}
        sources = {}
        for meta in all_data["metadatas"]:
            cat = meta.get("category", "unknown")
            src = meta.get("source", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            sources[src] = sources.get(src, 0) + 1

        print(f"\n📊 Document Store Stats")
        print(f"   Collection: {self.collection_name}")
        print(f"   Total documents: {self.count}")
        print(f"   Categories: {dict(categories)}")
        print(f"   Sources: {dict(sources)}")


# --- Usage ---
store = DocumentStore("knowledge_base")

# Add documents
store.add_batch([
    {"content": "Machine learning enables computers to learn from data without explicit programming.",
     "metadata": {"category": "AI", "source": "textbook.pdf"}},
    {"content": "Deep learning uses multi-layered neural networks for complex pattern recognition.",
     "metadata": {"category": "AI", "source": "textbook.pdf"}},
    {"content": "Natural language processing allows computers to understand human language.",
     "metadata": {"category": "AI", "source": "nlp_guide.pdf"}},
    {"content": "Python's scikit-learn library provides simple tools for machine learning.",
     "metadata": {"category": "Programming", "source": "python_docs.pdf"}},
    {"content": "Docker packages applications into portable containers.",
     "metadata": {"category": "DevOps", "source": "docker_docs.pdf"}},
    {"content": "Kubernetes orchestrates container deployment at scale.",
     "metadata": {"category": "DevOps", "source": "k8s_docs.pdf"}},
    {"content": "PostgreSQL is a powerful open-source relational database.",
     "metadata": {"category": "Database", "source": "pg_docs.pdf"}},
    {"content": "Redis is an in-memory data store used for caching and message queuing.",
     "metadata": {"category": "Database", "source": "redis_docs.pdf"}},
])

# Stats
store.stats()

# Search
print("\n🔍 Search: 'How do computers learn?'")
for r in store.search("How do computers learn?", k=3):
    print(f"   [{r['similarity']}] {r['content'][:70]}... ({r['metadata']['category']})")

# Filtered search
print("\n🔍 Search: 'data tools' (Programming only)")
for r in store.search("data tools", k=3, filter={"category": "Programming"}):
    print(f"   [{r['similarity']}] {r['content'][:70]}...")

# MMR search (diverse)
print("\n🔍 MMR Search: 'infrastructure and deployment'")
for r in store.search_mmr("infrastructure and deployment", k=3):
    print(f"   {r['content'][:70]}... ({r['metadata']['category']})")

# List all
print(f"\n📋 All documents: {store.count}")
for doc in store.list_all():
    print(f"   [{doc['id']}] {doc['content'][:50]}...")
```

---

## Part 7: ChromaDB Internals & Limits

### Architecture

```
ChromaDB Client
       │
       ├── Collection Manager
       │       │
       │       ├── HNSW Index (vector search)     ← In C++ via hnswlib
       │       ├── Metadata Store (SQLite)         ← Filters, IDs
       │       └── Document Store (SQLite)         ← Raw text storage
       │
       └── Embedding Function
               └── Default: all-MiniLM-L6-v2 (Sentence Transformers)
```

### Limits

| Aspect | Limit |
|--------|-------|
| Max collection size | ~1-5M vectors (depends on RAM) |
| Max embedding dimensions | No hard limit (tested up to 4096) |
| Max metadata value size | 512 characters |
| Concurrent writes | Single-writer (not thread-safe for writes) |
| Concurrent reads | Thread-safe |
| Production readiness | Good for small-to-medium. Not for massive scale. |

### When to Outgrow ChromaDB

| Sign | Alternative |
|------|-------------|
| >5M vectors | FAISS, Qdrant, or Milvus |
| Need multi-node scaling | Pinecone, Weaviate, or Milvus |
| Need real-time updates at high throughput | Qdrant or Milvus |
| Need ACID transactions | pgvector (PostgreSQL) |
| Need managed/serverless | Pinecone |

---

## Common Mistakes

### Mistake 1: Forgetting that ChromaDB returns distances, not similarities
```python
# ❌ Interpreting distances as similarities
score = results["distances"][0][0]  # 0.23
print(f"Similarity: {score}")  # WRONG — 0.23 is LOW distance = HIGH similarity

# ✅ Convert: similarity = 1 - distance (for cosine space)
similarity = 1 - score  # 0.77
```

### Mistake 2: Not using persist_directory
```python
# ❌ Data lost when script ends
client = chromadb.Client()

# ✅ Data saved to disk
client = chromadb.PersistentClient(path="./chroma_data")
```

### Mistake 3: Using the wrong embedding model for queries vs indexing
```python
# ❌ Indexed with OpenAI, querying with default ChromaDB embedder
vectorstore = Chroma.from_documents(docs, embedding=openai_embeddings, ...)

# Later, loading without specifying the same embedding model:
vectorstore = Chroma(persist_directory="./db")  # Uses default — WRONG!

# ✅ Always specify the same model
vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=openai_embeddings  # Same model as indexing!
)
```

### Mistake 4: Not batching adds
```python
# ❌ One API call per document
for doc in documents:
    vectorstore.add_documents([doc])  # Slow!

# ✅ Batch — one call for all
vectorstore.add_documents(documents)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use `PersistentClient` / `persist_directory` always | Don't lose your data |
| Use `get_or_create_collection` | Idempotent — safe to run repeatedly |
| Always specify `embedding_function` when loading | Prevents wrong-model queries |
| Use `upsert` instead of `add` for potentially duplicate data | Avoids errors |
| Use `similarity_search_with_score` to inspect ranking quality | Debug retrieval issues |
| Use MMR search when results are too repetitive | Balances relevance + diversity |
| Batch document additions | Faster embedding and insertion |
| Store source info in metadata | Know where each chunk came from |

---

## Interview Preparation

### Easy
**Q: What is ChromaDB?**

> ChromaDB is an open-source, embedded vector database designed for AI applications. It stores embeddings alongside documents and metadata, supports fast similarity search using HNSW indexing, and integrates directly with LangChain. It runs locally with zero configuration — no separate server needed — making it ideal for prototyping and small-to-medium applications.

### Medium
**Q: What is the difference between `similarity_search` and `max_marginal_relevance_search`?**

> `similarity_search` returns the k documents most similar to the query, which can result in redundant results (top 5 might all say the same thing from different paragraphs). `max_marginal_relevance_search` (MMR) balances **relevance** and **diversity** — it first fetches more candidates (`fetch_k`), then iteratively selects documents that are similar to the query but dissimilar to already-selected documents. The `lambda_mult` parameter controls the tradeoff (0=max diversity, 1=pure relevance). MMR is better for RAG because it gives the LLM more diverse context.

### Hard
**Q: How would you migrate from ChromaDB to a production vector database?**

> The migration path: (1) **Extract** all documents, embeddings, and metadata from ChromaDB using `collection.get(include=["documents", "metadatas", "embeddings"])`. (2) **Load** into the target DB (Pinecone, Qdrant, etc.) using their batch upsert APIs. (3) **Update LangChain code** — swap `Chroma(...)` for `Pinecone(...)` or `Qdrant(...)`. The LangChain `VectorStore` interface is consistent across providers, so the chain code (`retriever = vectorstore.as_retriever()`) stays unchanged. (4) **Re-embed** if changing embedding models, otherwise transfer embeddings directly. Key gotcha: ensure the distance metric matches (cosine vs L2).

---

## Summary

| Component | What It Does |
|-----------|-------------|
| `chromadb.PersistentClient(path=...)` | Create a persistent ChromaDB instance |
| `client.get_or_create_collection(name)` | Create or load a collection |
| `collection.add(ids, documents, metadatas)` | Add documents (auto-embeds) |
| `collection.query(query_texts, n_results, where)` | Similarity search with optional filters |
| `collection.upsert(...)` | Insert or update documents |
| `Chroma.from_documents(docs, embedding)` | LangChain: create vectorstore from documents |
| `vectorstore.similarity_search(query, k, filter)` | LangChain: search with metadata filter |
| `vectorstore.as_retriever(search_kwargs={...})` | LangChain: convert to retriever for RAG chains |
| `vectorstore.max_marginal_relevance_search(...)` | Diverse search results (relevance + diversity) |

---

> [← Previous: Why Vector Databases?](chapter-26-why-vector-databases.md) | [Next: FAISS →](chapter-28-faiss.md)

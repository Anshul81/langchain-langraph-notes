# Chapter 3.3: Similarity Search

> **Phase 3 — Embeddings & Vector Math** | [← Previous: Embedding Models](chapter-24-embedding-models.md) | [Next: Phase 4 — Why Vector Databases? →](../phase-07-vector-databases/chapter-26-why-vector-databases.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Master the three distance metrics: cosine similarity, dot product, Euclidean distance
- ✅ Understand the math behind each metric (with intuition, not just formulas)
- ✅ Know when to use which metric and why
- ✅ Build a complete **semantic search engine** from scratch (no vector DB)
- ✅ Implement top-k retrieval, thresholding, and result ranking

| | |
|---|---|
| **Prerequisites** | Chapter 3.1 (What Are Embeddings?), Chapter 3.2 (Embedding Models) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction

You have embeddings. Now what? You need to **compare** them to find which documents match a query. This is the core of every RAG system, every search engine, every recommendation system.

```
User asks: "How does photosynthesis work?"
                    ↓
           Embed the question → [0.12, -0.34, 0.56, ...]
                    ↓
           Compare against 10,000 document embeddings
                    ↓
           Return the 5 most similar → Feed to LLM
```

The comparison step uses **distance/similarity metrics**. There are three you need to know.

---

## Part 1: Cosine Similarity — The Gold Standard for Text

### The Intuition

Cosine similarity measures the **angle** between two vectors, ignoring their length. Think of it as asking: "Are these two arrows pointing in the same direction?"

```
        ↑ B (short document about AI)
       /
      / θ ← small angle = high similarity
     /
    /
   ────────────────→ A (long document about AI)

Both point roughly the same direction → high cosine similarity!
The fact that B is "shorter" (fewer words) doesn't matter.
```

### The Math

```
                    A · B           Σ(aᵢ × bᵢ)
cos(θ) = ─────────────────── = ─────────────────────
              ‖A‖ × ‖B‖       √Σ(aᵢ²) × √Σ(bᵢ²)
```

- **Numerator** (dot product): Sum of element-wise multiplications
- **Denominator** (norms): Product of vector lengths — this normalizes out magnitude

### Range and Interpretation

| Value | Meaning | Example |
|-------|---------|---------|
| `+1.0` | Identical direction (same meaning) | "dog" ↔ "canine" |
| `+0.7 to +0.9` | Very similar | "car" ↔ "automobile" |
| `+0.3 to +0.6` | Somewhat related | "car" ↔ "road" |
| `~0.0` | Unrelated (orthogonal) | "car" ↔ "philosophy" |
| `-1.0` | Opposite direction | Rare in practice for text |

### Implementation

```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Returns: -1.0 (opposite) to +1.0 (identical)
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0  # Avoid division by zero

    return float(dot / (norm_a * norm_b))


# Example
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.1])  # Very similar
c = np.array([-1.0, -2.0, -3.0])  # Opposite

print(f"a ↔ b: {cosine_similarity(a, b):.4f}")  # ~0.9999 (nearly identical)
print(f"a ↔ c: {cosine_similarity(a, c):.4f}")  # -1.0 (opposite)
```

### Why Cosine Is Best for Text

**Document length doesn't affect similarity.** A 2-sentence summary about AI and a 50-page thesis about AI should score similarly against a query about AI. Cosine only cares about **direction**, not **magnitude**.

```python
# Same direction, different magnitudes
short_doc = np.array([0.5, 0.3, 0.8])   # Short doc about ML
long_doc = np.array([5.0, 3.0, 8.0])    # Long doc about ML (10x longer)

print(f"Cosine: {cosine_similarity(short_doc, long_doc):.4f}")  # 1.0 — identical!
# Cosine sees them as identical because they point the same direction
```

---

## Part 2: Dot Product — Fast and Simple

### The Intuition

The dot product measures how much two vectors "agree" in each dimension, then sums it up. Unlike cosine, it **does** care about magnitude.

```
If both vectors are normalized (length = 1):
    dot product == cosine similarity

If vectors are NOT normalized:
    dot product favors longer (more "confident") vectors
```

### The Math

```
A · B = Σ(aᵢ × bᵢ) = a₁b₁ + a₂b₂ + ... + aₙbₙ
```

That's it — no division by norms. Just multiply and sum.

### Range

- **No fixed range** — can be any real number
- For normalized vectors: -1.0 to +1.0 (same as cosine)
- Higher = more similar

### Implementation

```python
def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dot product similarity.
    Higher = more similar. No fixed range unless normalized.
    """
    return float(np.dot(a, b))


# For normalized vectors, identical to cosine
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)

print(f"Dot product (normalized): {dot_product(a_norm, b_norm):.4f}")
print(f"Cosine similarity:        {cosine_similarity(a, b):.4f}")
# Same value!
```

### When Dot Product Differs from Cosine

```python
# Non-normalized vectors
confident = np.array([10.0, 10.0, 10.0])  # High magnitude = "confident"
uncertain = np.array([0.1, 0.1, 0.1])     # Low magnitude = "uncertain"
query = np.array([1.0, 1.0, 1.0])

print(f"Cosine (confident): {cosine_similarity(query, confident):.4f}")  # 1.0
print(f"Cosine (uncertain): {cosine_similarity(query, uncertain):.4f}")  # 1.0

print(f"Dot (confident): {dot_product(query, confident):.4f}")  # 30.0
print(f"Dot (uncertain): {dot_product(query, uncertain):.4f}")  # 0.3

# Cosine says both are equally similar (same direction).
# Dot product prefers the "confident" vector.
```

### When to Use Dot Product

- When embeddings are **already normalized** (OpenAI, most models normalize)
- When you want to **reward confidence/magnitude**
- When speed matters — dot product is faster (no norm computation)

---

## Part 3: Euclidean Distance — Physical Distance

### The Intuition

Euclidean distance measures the **straight-line distance** between two points in space. Think of it as literally how far apart two points are on a map.

```
        ● B (0.9, 0.8)
        |
        |  d = 0.14 (very close)
        |
        ● A (0.8, 0.7)


                        ● C (0.1, 0.1)
                        
                        d = 0.99 (far away from A)
```

### The Math

```
d(A, B) = √Σ(aᵢ - bᵢ)² = √((a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)²)
```

### Range

- `0` = identical vectors
- `> 0` = some difference
- **Lower = more similar** (opposite of cosine/dot!)

### Implementation

```python
def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean (L2) distance between two vectors.
    Returns: 0 (identical) to +∞. LOWER = more similar.
    """
    return float(np.linalg.norm(a - b))


a = np.array([1.0, 2.0, 3.0])
b = np.array([1.1, 2.1, 3.1])  # Very close
c = np.array([10.0, 20.0, 30.0])  # Far away

print(f"a ↔ b: {euclidean_distance(a, b):.4f}")  # ~0.17 (close = similar)
print(f"a ↔ c: {euclidean_distance(a, c):.4f}")  # ~32.08 (far = different)
```

### The Length Problem

```python
# Same direction, different magnitudes
short_doc = np.array([0.5, 0.3, 0.8])
long_doc = np.array([5.0, 3.0, 8.0])

print(f"Euclidean: {euclidean_distance(short_doc, long_doc):.4f}")  # 9.59 — far apart!
print(f"Cosine:    {cosine_similarity(short_doc, long_doc):.4f}")   # 1.0 — identical!

# Euclidean thinks they're very different because the POINTS are far apart.
# Cosine knows they're identical because they point the same DIRECTION.
```

**This is why Euclidean is less common for text** — document length artificially inflates distance.

---

## Part 4: The Three Metrics Compared

### Side-by-Side

| Metric | Measures | Range | More Similar = | Cares About Length? |
|--------|----------|-------|----------------|-------------------|
| **Cosine similarity** | Angle between vectors | -1 to +1 | Higher | ❌ No |
| **Dot product** | Aligned magnitude | -∞ to +∞ | Higher | ✅ Yes |
| **Euclidean distance** | Straight-line distance | 0 to +∞ | **Lower** | ✅ Yes |

### Mathematical Relationship

For **normalized** vectors (‖A‖ = ‖B‖ = 1):

```python
# All three are mathematically related:
cosine = np.dot(a_norm, b_norm)                         # Cosine
dot = np.dot(a_norm, b_norm)                             # Dot = same as cosine
euclidean = np.sqrt(2 * (1 - np.dot(a_norm, b_norm)))    # Euclidean = f(cosine)

# So for normalized vectors, all three give equivalent rankings!
# The choice only matters for NON-normalized vectors.
```

### Decision Matrix

| Use Case | Best Metric | Why |
|----------|-------------|-----|
| Text search / RAG | **Cosine similarity** | Length-invariant, standard for text |
| Already-normalized embeddings | **Dot product** | Equivalent to cosine, slightly faster |
| Clustering (k-means) | **Euclidean distance** | k-means minimizes squared distances |
| Image embeddings | **Cosine or Euclidean** | Depends on normalization |
| Anomaly detection | **Euclidean distance** | "How far from normal?" |

**Rule of thumb**: For LangChain/RAG work, use **cosine similarity**. Always.

---

## Part 5: Building a Semantic Search Engine (No Vector DB)

The capstone project for Phase 3 — a complete semantic search engine from scratch:

```python
import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# --- 1. Setup Embedding Model ---
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)


# --- 2. Document Corpus ---
documents = [
    {
        "id": 1,
        "title": "Introduction to Machine Learning",
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning.",
        "category": "AI"
    },
    {
        "id": 2,
        "title": "Python for Data Science",
        "content": "Python is the most popular programming language for data science. Libraries like pandas, numpy, and scikit-learn make it easy to analyze and model data.",
        "category": "Programming"
    },
    {
        "id": 3,
        "title": "Deep Learning and Neural Networks",
        "content": "Deep learning uses multi-layered neural networks to learn complex patterns. Convolutional neural networks excel at image recognition while transformers dominate NLP.",
        "category": "AI"
    },
    {
        "id": 4,
        "title": "Web Development with React",
        "content": "React is a JavaScript library for building user interfaces. It uses a virtual DOM and component-based architecture to create fast, interactive web applications.",
        "category": "Web Dev"
    },
    {
        "id": 5,
        "title": "Natural Language Processing",
        "content": "NLP enables computers to understand human language. Key tasks include sentiment analysis, named entity recognition, text summarization, and machine translation.",
        "category": "AI"
    },
    {
        "id": 6,
        "title": "Docker and Containerization",
        "content": "Docker packages applications into containers that run consistently across environments. It solves the 'it works on my machine' problem and simplifies deployment.",
        "category": "DevOps"
    },
    {
        "id": 7,
        "title": "SQL Database Fundamentals",
        "content": "SQL databases store data in structured tables with rows and columns. Key concepts include JOINs, indexes, transactions, and normalization for efficient data storage.",
        "category": "Database"
    },
    {
        "id": 8,
        "title": "Reinforcement Learning",
        "content": "Reinforcement learning trains agents to make decisions by rewarding desired behaviors. Applications include game playing, robotics, and autonomous vehicles.",
        "category": "AI"
    },
    {
        "id": 9,
        "title": "REST API Design Best Practices",
        "content": "REST APIs use HTTP methods to perform CRUD operations. Good API design includes proper status codes, versioning, pagination, and clear resource naming.",
        "category": "Backend"
    },
    {
        "id": 10,
        "title": "Cloud Computing with AWS",
        "content": "AWS offers compute (EC2), storage (S3), databases (RDS), and AI services. Cloud computing enables scalable, pay-as-you-go infrastructure for modern applications.",
        "category": "Cloud"
    },
]


# --- 3. Semantic Search Engine Class ---
class SemanticSearchEngine:
    """A simple semantic search engine using embeddings and cosine similarity."""

    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        self.documents = []
        self.doc_embeddings = []

    def index(self, documents: list[dict], text_field: str = "content"):
        """Embed and store all documents."""
        self.documents = documents
        texts = [doc[text_field] for doc in documents]

        print(f"📊 Indexing {len(texts)} documents...")
        self.doc_embeddings = self.embeddings_model.embed_documents(texts)
        self.doc_embeddings = [np.array(emb) for emb in self.doc_embeddings]
        print(f"✅ Indexed! Each embedding: {len(self.doc_embeddings[0])}D")

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        category_filter: str = None
    ) -> list[dict]:
        """
        Search for documents most similar to the query.

        Args:
            query: Search query text
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)
            category_filter: Optional category to filter by

        Returns:
            List of {document, score} dicts, sorted by similarity
        """
        # Embed the query
        query_embedding = np.array(self.embeddings_model.embed_query(query))

        # Compute similarities
        results = []
        for i, doc_emb in enumerate(self.doc_embeddings):
            # Cosine similarity
            score = float(
                np.dot(query_embedding, doc_emb)
                / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            )

            # Apply filters
            if score < threshold:
                continue
            if category_filter and self.documents[i].get("category") != category_filter:
                continue

            results.append({
                "document": self.documents[i],
                "score": score,
                "rank": 0  # Will be set after sorting
            })

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Assign ranks
        for i, result in enumerate(results[:top_k]):
            result["rank"] = i + 1

        return results[:top_k]

    def display_results(self, query: str, results: list[dict]):
        """Pretty-print search results."""
        print(f"\n{'='*60}")
        print(f"🔍 Query: \"{query}\"")
        print(f"{'='*60}")

        if not results:
            print("   No results found.")
            return

        for r in results:
            doc = r["document"]
            score = r["score"]
            bar = "█" * int(score * 30)
            print(f"\n   #{r['rank']} [{score:.4f}] {bar}")
            print(f"   📄 {doc['title']} [{doc.get('category', 'N/A')}]")
            print(f"   {doc['content'][:100]}...")


# --- 4. Build and Query! ---
engine = SemanticSearchEngine(embeddings_model)
engine.index(documents)

# Query 1: AI-related
results = engine.search("How do machines learn from data?", top_k=3)
engine.display_results("How do machines learn from data?", results)

# Query 2: Web development
results = engine.search("building interactive websites with JavaScript", top_k=3)
engine.display_results("building interactive websites with JavaScript", results)

# Query 3: With category filter
results = engine.search("learning algorithms", top_k=3, category_filter="AI")
engine.display_results("learning algorithms (AI only)", results)

# Query 4: With threshold
results = engine.search("cooking recipes", top_k=5, threshold=0.3)
engine.display_results("cooking recipes (threshold=0.3)", results)

# Query 5: Unrelated — should return low scores
results = engine.search("How to play guitar?", top_k=3)
engine.display_results("How to play guitar?", results)
```

### Expected Output

```
📊 Indexing 10 documents...
✅ Indexed! Each embedding: 1536D

============================================================
🔍 Query: "How do machines learn from data?"
============================================================

   #1 [0.8734] █████████████████████████▏
   📄 Introduction to Machine Learning [AI]
   Machine learning is a subset of artificial intelligence that enables systems to learn from data...

   #2 [0.7521] ██████████████████████▏
   📄 Deep Learning and Neural Networks [AI]
   Deep learning uses multi-layered neural networks to learn complex patterns...

   #3 [0.7103] █████████████████████▏
   📄 Reinforcement Learning [AI]
   Reinforcement learning trains agents to make decisions by rewarding desired behaviors...

============================================================
🔍 Query: "cooking recipes (threshold=0.3)"
============================================================
   No results found.
```

---

## Part 6: Performance Optimization

### Numpy Batch Computation

```python
def batch_cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity of one query against many documents at once.
    Much faster than looping!

    Args:
        query_vec: (D,) query embedding
        doc_vecs: (N, D) matrix of document embeddings

    Returns:
        (N,) array of similarity scores
    """
    # Normalize
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    # Matrix multiplication: (N, D) @ (D,) → (N,)
    return doc_norms @ query_norm


# Usage: compute 10K similarities in one operation
doc_matrix = np.array(engine.doc_embeddings)  # (10, 1536)
query_vec = np.array(embeddings_model.embed_query("machine learning"))

scores = batch_cosine_similarity(query_vec, doc_matrix)
print(f"Computed {len(scores)} similarities at once")
print(f"Top score: {scores.max():.4f} at index {scores.argmax()}")
```

### Scipy Alternative

```python
from scipy.spatial.distance import cosine as cosine_dist

# Note: scipy returns DISTANCE (1 - similarity), not similarity!
similarity = 1 - cosine_dist(vec_a, vec_b)
```

### Scikit-learn for Large Batches

```python
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# Compute all-pairs similarity in one call
# Input: (N, D) matrix → Output: (N, N) similarity matrix
all_pairs = sklearn_cosine(doc_matrix)
print(f"Similarity matrix shape: {all_pairs.shape}")  # (10, 10)
```

---

## Part 7: Similarity Gotchas

### Gotcha 1: High Baseline Similarity

Real embedding spaces often have a "baseline" similarity even for unrelated texts:

```python
# You might expect unrelated texts to score ~0.0
# But in practice:
sim("machine learning", "cooking recipes") ≈ 0.15-0.25  # Not zero!

# This happens because all English texts share common linguistic features.
# Don't use absolute thresholds blindly — use relative ranking instead.
```

### Gotcha 2: Similarity ≠ Relevance

```python
# High similarity doesn't always mean relevant:
query = "What is Python?"
doc1 = "Python is a programming language."          # Relevant ✅
doc2 = "Python is a species of snake."               # Not relevant ❌
# Both might score similarly high because they're about "Python"!

# This is why RAG uses the LLM to generate the final answer —
# it can distinguish between the snake and the language given context.
```

### Gotcha 3: Symmetric vs Asymmetric

```python
# Cosine similarity is SYMMETRIC:
cosine("A", "B") == cosine("B", "A")  # Always true

# But relevance is NOT symmetric:
# "What is Python?" → "Python is a language" (relevant)
# "Python is a language" → "What is Python?" (less natural as a "search")

# Some models (Cohere) handle this with separate query/doc modes
```

---

## Common Mistakes

### Mistake 1: Using Euclidean distance for text without normalizing
```python
# ❌ Non-normalized embeddings + Euclidean = length bias
distance = euclidean_distance(short_doc_emb, long_doc_emb)

# ✅ Either normalize first, or use cosine similarity
normalized = emb / np.linalg.norm(emb)
# Or just use cosine
```

### Mistake 2: Confusing distance (lower=better) with similarity (higher=better)
```python
# ❌ Sorting distances in descending order (wrong!)
results.sort(key=lambda x: x["euclidean_dist"], reverse=True)

# ✅ Distances: sort ascending (closest first)
results.sort(key=lambda x: x["euclidean_dist"])

# ✅ Similarities: sort descending (most similar first)
results.sort(key=lambda x: x["cosine_sim"], reverse=True)
```

### Mistake 3: Using fixed similarity thresholds across models
```python
# ❌ Threshold of 0.8 works for OpenAI but not MiniLM
if similarity > 0.8:  # Too model-specific

# ✅ Use relative ranking (top-k) instead of absolute thresholds
# Or calibrate thresholds per model
```

### Mistake 4: Computing similarity in a Python loop
```python
# ❌ Slow — Python loop over 100K documents
scores = [cosine_similarity(query, doc) for doc in docs]

# ✅ Fast — vectorized numpy operation
scores = batch_cosine_similarity(query_vec, doc_matrix)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use **cosine similarity** for text search | Length-invariant, industry standard |
| Use **top-k retrieval** over fixed thresholds | Thresholds are model-dependent |
| **Vectorize** similarity computation with numpy | 100-1000x faster than Python loops |
| **Pre-normalize** embeddings if doing many comparisons | Avoids recomputing norms every time |
| Use dot product for normalized vectors | Slightly faster, mathematically equivalent |
| Don't trust absolute similarity scores | Use relative ranking instead |
| Always re-embed everything when switching models | Different models = different spaces |

---

## Interview Preparation

### Easy
**Q: What is cosine similarity and what does it measure?**

> Cosine similarity measures the **angle** between two vectors, ranging from -1 to +1. A value of +1 means the vectors point in the same direction (identical meaning), 0 means they're perpendicular (unrelated), and -1 means opposite directions. It's the standard metric for text embeddings because it ignores vector magnitude, so short and long documents about the same topic score equally.

### Medium
**Q: When would you use Euclidean distance instead of cosine similarity?**

> Euclidean distance measures the straight-line distance between two points. Use it for **clustering** (k-means uses squared Euclidean distances), **anomaly detection** (outlier = far from cluster center), and when **magnitude matters** (e.g., how "strongly" a document is about a topic). For standard text retrieval and RAG, cosine similarity is almost always better because it's invariant to document length.

### Hard
**Q: If embeddings are normalized, how are cosine similarity, dot product, and Euclidean distance related mathematically?**

> For unit vectors (‖A‖ = ‖B‖ = 1): **Dot product equals cosine similarity** because the denominator becomes 1. **Euclidean distance** relates as d = √(2 × (1 - cos(θ))). This means all three metrics give the **same ranking** for normalized vectors — they're monotonic transformations of each other. The choice only affects the scale, not the order. This is why in practice, using dot product on normalized OpenAI embeddings is equivalent to cosine similarity but slightly faster.

### Senior
**Q: You're building a search system that retrieves from 10 million documents. How do you make similarity search fast?**

> Brute-force cosine similarity on 10M vectors is too slow (~seconds per query). Solutions: (1) **Approximate Nearest Neighbor (ANN)** algorithms like HNSW (used by ChromaDB, Pinecone) or IVF (used by FAISS) — trade a tiny amount of accuracy for 100-1000x speed. (2) **Dimensionality reduction** — use fewer dimensions (512 instead of 1536) to speed up distance computation. (3) **Pre-filtering** — use metadata filters to narrow the candidate set before vector search. (4) **Quantization** — compress vectors to use less memory and faster computation. This is exactly what **vector databases** (Phase 4) are built to do.

---

## Summary

| Concept | What It Does |
|---------|-------------|
| **Cosine similarity** | Measures angle between vectors; -1 to +1; length-invariant; best for text |
| **Dot product** | Sum of element-wise products; equals cosine for normalized vectors; fastest |
| **Euclidean distance** | Straight-line distance; 0 = identical; good for clustering |
| **Top-k retrieval** | Return the k most similar results regardless of absolute scores |
| **Threshold filtering** | Only return results above a minimum similarity score |
| **Batch computation** | Use numpy matrix operations instead of Python loops for speed |
| **Normalization** | Make all three metrics equivalent by normalizing vectors to unit length |

---

## 🎉 Phase 3 Complete!

You've mastered the **Embeddings & Vector Math** foundation:

| Chapter | What You Learned |
|---------|-----------------|
| 3.1 — What Are Embeddings? | Intuition, vector spaces, semantic arithmetic |
| 3.2 — Embedding Models | OpenAI, Sentence Transformers, Ollama, cost analysis |
| 3.3 — Similarity Search | Cosine, dot product, Euclidean; built a semantic search engine |

**Next: Phase 4 — Vector Databases** — where you'll learn to store and search millions of embeddings efficiently using ChromaDB, FAISS, and cloud solutions.

---

> [← Previous: Embedding Models](chapter-24-embedding-models.md) | [Next: Phase 4 — Why Vector Databases? →](../phase-07-vector-databases/chapter-26-why-vector-databases.md)

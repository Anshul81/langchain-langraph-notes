# Chapter 3.1: What Are Embeddings?

> **Phase 3 — Embeddings & Vector Math** | [← Previous: Output Formatting](../phase-02-prompt-engineering/chapter-13-output-formatting.md) | [Next: Embedding Models →](chapter-24-embedding-models.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand what embeddings are and WHY they exist
- ✅ Grasp the intuition behind representing meaning as numbers
- ✅ Understand embedding spaces, dimensions, and distances
- ✅ Know how embeddings power search, RAG, recommendations, and clustering
- ✅ Build a basic word similarity finder from scratch

| | |
|---|---|
| **Prerequisites** | Phase 2 (Prompt Engineering), basic Python |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction — The Fundamental Problem

Computers understand **numbers**, not **meaning**:

```
Computer sees: "king" → [107, 105, 110, 103]  (ASCII codes)
Computer sees: "queen" → [113, 117, 101, 101, 110]

Are "king" and "queen" similar? ASCII says NO — completely different numbers!
```

But we know "king" and "queen" are semantically related — both are royalty, both are nouns, both relate to monarchy.

**Embeddings solve this.** They convert text into numerical vectors that capture **meaning**:

```
Embedding("king")  → [0.21, 0.83, -0.45, 0.12, ...]   (768 dimensions)
Embedding("queen") → [0.23, 0.81, -0.43, 0.15, ...]   (768 dimensions)

Now: similarity("king", "queen") = 0.95  ← Very similar! ✅
     similarity("king", "pizza") = 0.11  ← Not similar! ✅
```

---

## Part 1: The Intuition — Words as Coordinates

### Simple Example: 2D Embeddings

Imagine we could represent words in just 2 dimensions:
- **X-axis**: Royalty (0 = not royal, 1 = very royal)
- **Y-axis**: Gender (0 = masculine, 1 = feminine)

```
                Feminine
                   ↑
            queen  ●  (0.9, 0.9)
                   |
            princess ● (0.7, 0.85)
                   |
        woman ●    |
       (0.1, 0.8) |
                   |
    ───────────────┼──────────────→ Royalty
                   |
          man ●    |      prince ●  (0.7, 0.15)
       (0.1, 0.2) |
                   |       king ●   (0.9, 0.1)
                   |
              Masculine
```

**Key insight**: Words with similar meaning are **close together** in this space!

- king ↔ queen: Close (both royal)
- king ↔ man: Somewhat close (both masculine)
- king ↔ pizza: Far apart (nothing in common)

### Real Embeddings: 768-3072 Dimensions

Real embedding models use **hundreds to thousands** of dimensions. Each dimension captures some learned aspect of meaning — not as clean as "royalty" or "gender," but the model learns these automatically during training.

```python
# Real embedding dimensions (simplified)
# Dimension 42 might capture "living vs non-living"
# Dimension 187 might capture "positive vs negative sentiment"
# Dimension 503 might capture "abstract vs concrete"
# (We don't choose these — the model learns them from data)
```

---

## Part 2: How Embeddings Are Created

### The Training Process (High-Level)

Embedding models learn from **massive amounts of text**:

```
Training Data: Billions of sentences from the internet

Step 1: Model sees: "The ___ sat on the throne"
        → Learns that "king" and "queen" appear in similar contexts

Step 2: Model sees: "I had ___ for lunch"
        → Learns that "pizza" and "pasta" appear in similar contexts

Step 3: Over billions of examples, the model builds a number system where
        words with similar usage patterns get similar number representations
```

**Core principle**: Words that appear in similar contexts get similar embeddings. This is called the **distributional hypothesis** — "you shall know a word by the company it keeps."

### From Words to Sentences to Documents

Modern embeddings work at multiple levels:

| Level | Example | Use Case |
|-------|---------|----------|
| **Word** embeddings | `"king" → [0.2, 0.8, ...]` | Word similarity, analogies |
| **Sentence** embeddings | `"The king rules wisely" → [0.3, 0.5, ...]` | Semantic search, Q&A |
| **Document** embeddings | `"Full 500-word article..." → [0.1, 0.4, ...]` | Document search, clustering |

For LangChain and RAG, we primarily use **sentence/document embeddings**.

---

## Part 3: What Embeddings Actually Look Like

```python
# Install: pip install openai python-dotenv
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE")
)

# Get embedding for a sentence
response = client.embeddings.create(
    model="text-embedding-3-small",  # OpenAI's embedding model
    input="The king sat on his throne."
)

embedding = response.data[0].embedding

print(f"Dimensions: {len(embedding)}")      # 1536
print(f"First 10 values: {embedding[:10]}")  # [0.023, -0.014, 0.056, ...]
print(f"Type: list of floats")
```

### Key Properties of Embeddings

```python
import numpy as np

vec = np.array(embedding)

print(f"Shape: {vec.shape}")             # (1536,)
print(f"Min value: {vec.min():.4f}")     # ~ -0.05
print(f"Max value: {vec.max():.4f}")     # ~ 0.05
print(f"Mean: {vec.mean():.6f}")         # ~ 0.0 (centered)
print(f"Norm: {np.linalg.norm(vec):.4f}")# ~ 1.0 (normalized)
```

Embeddings are:
- **Fixed-length vectors** — always the same number of dimensions, regardless of input length
- **Dense** — every dimension has a meaningful value (unlike sparse vectors)
- **Normalized** — typically unit length (norm ≈ 1.0)

---

## Part 4: The Magic — Semantic Arithmetic

The most famous embedding property — vector arithmetic captures analogies:

```
king - man + woman ≈ queen
```

```python
import numpy as np

# Pretend these are real embeddings (simplified to 4D)
king  = np.array([0.9, 0.1, 0.8, 0.3])
man   = np.array([0.1, 0.2, 0.8, 0.1])
woman = np.array([0.1, 0.8, 0.8, 0.1])
queen = np.array([0.9, 0.9, 0.8, 0.3])

# king - man + woman = ?
result = king - man + woman
print(f"Computed: {result}")     # [0.9, 0.7, 0.8, 0.3]
print(f"Queen:    {queen}")      # [0.9, 0.9, 0.8, 0.3]
# Very close! The "gender" dimension shifted, "royalty" stayed.
```

### More Analogy Examples

```
Paris - France + Japan ≈ Tokyo
walked - walk + swim ≈ swam
bigger - big + small ≈ smaller
```

This works because embeddings encode **relationships** as **directions** in the vector space.

---

## Part 5: Measuring Similarity

How do you tell if two embeddings are "close"? Three main distance metrics:

### Cosine Similarity (Most Common for Text)

Measures the **angle** between two vectors. Ignores magnitude, focuses on direction:

```python
import numpy as np

def cosine_similarity(a, b):
    """Cosine similarity: -1 (opposite) to +1 (identical)."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example
king  = np.array([0.9, 0.1, 0.8])
queen = np.array([0.85, 0.15, 0.82])
pizza = np.array([0.1, 0.9, 0.05])

print(f"king ↔ queen: {cosine_similarity(king, queen):.4f}")  # ~0.99 (very similar)
print(f"king ↔ pizza: {cosine_similarity(king, pizza):.4f}")  # ~0.25 (not similar)
```

**Scale**:
- `1.0` = identical direction (same meaning)
- `0.0` = orthogonal (unrelated)
- `-1.0` = opposite direction (opposite meaning)

### Dot Product

```python
def dot_product(a, b):
    """Dot product: higher = more similar. No upper bound."""
    return np.dot(a, b)

# For normalized vectors (norm=1), dot product == cosine similarity
```

### Euclidean Distance

```python
def euclidean_distance(a, b):
    """Euclidean distance: 0 = identical. Lower = more similar."""
    return np.linalg.norm(a - b)
```

### Which One to Use?

| Metric | Range | Best For | Used By |
|--------|-------|----------|---------|
| **Cosine similarity** | -1 to +1 | Text embeddings (most common) | OpenAI, ChromaDB default |
| **Dot product** | -∞ to +∞ | Normalized embeddings | Google, Pinecone |
| **Euclidean distance** | 0 to +∞ | Dense embeddings, clustering | FAISS, scikit-learn |

**Rule of thumb**: Use **cosine similarity** for text. It's the standard.

---

## Part 6: Why Embeddings Matter for AI Engineering

### 1. Semantic Search (Foundation of RAG)

```python
# Traditional keyword search:
query = "how to fix a broken heart"
# Finds: documents containing "fix", "broken", "heart"
# Returns: plumbing articles, cardiac surgery papers ❌

# Embedding-based semantic search:
query_embedding = embed("how to fix a broken heart")
# Finds: documents with SIMILAR MEANING
# Returns: relationship advice, emotional healing articles ✅
```

### 2. RAG (Retrieval Augmented Generation)

```
Your PDFs → Split into chunks → Embed each chunk → Store in vector DB
                                                          ↓
User asks question → Embed question → Find similar chunks → Inject into prompt → LLM answers
```

### 3. Clustering & Classification

```python
# Group similar customer support tickets
ticket_embeddings = [embed(ticket) for ticket in tickets]
clusters = kmeans(ticket_embeddings, n_clusters=5)
# Cluster 0: Billing issues
# Cluster 1: Login problems
# Cluster 2: Feature requests
# ...
```

### 4. Recommendation Systems

```python
# "Users who liked X also liked Y"
# Because embed(X) is close to embed(Y) in embedding space
```

### 5. Anomaly Detection

```python
# Find emails that are "unusual" compared to normal patterns
# Outliers in embedding space = potential spam or phishing
```

---

## Part 7: Hands-On — Build a Word Similarity Finder

```python
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE")
)


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts in one API call."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- Experiment 1: Word Similarity ---
words = ["king", "queen", "man", "woman", "prince", "princess",
         "car", "bicycle", "airplane", "pizza", "hamburger", "sushi"]

print("📊 Getting embeddings for", len(words), "words...")
embeddings = get_embeddings_batch(words)

print("\n📐 Similarity Matrix (selected pairs):")
pairs = [
    ("king", "queen"), ("king", "man"), ("king", "pizza"),
    ("car", "bicycle"), ("car", "queen"), ("pizza", "hamburger"),
    ("man", "woman"), ("prince", "princess"), ("airplane", "car")
]

for w1, w2 in pairs:
    i1, i2 = words.index(w1), words.index(w2)
    sim = cosine_similarity(embeddings[i1], embeddings[i2])
    bar = "█" * int(sim * 20)
    print(f"  {w1:>10} ↔ {w2:<10}  {sim:.4f}  {bar}")


# --- Experiment 2: Sentence Similarity ---
print("\n\n📝 Sentence Similarity:")
sentences = [
    "The cat sat on the mat.",
    "A feline was resting on the rug.",
    "The stock market crashed today.",
    "Financial markets experienced a downturn.",
    "I love eating Italian food.",
]

sent_embeddings = get_embeddings_batch(sentences)

for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(sent_embeddings[i], sent_embeddings[j])
        if sim > 0.5:  # Only show high-similarity pairs
            print(f"  [{sim:.4f}] \"{sentences[i][:40]}...\"")
            print(f"           ↔ \"{sentences[j][:40]}...\"")
            print()


# --- Experiment 3: Find Most Similar ---
def find_most_similar(query: str, candidates: list[str], top_k: int = 3):
    """Find the most similar candidates to a query."""
    query_emb = get_embedding(query)
    cand_embs = get_embeddings_batch(candidates)

    similarities = [
        (cand, cosine_similarity(query_emb, emb))
        for cand, emb in zip(candidates, cand_embs)
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔍 Query: \"{query}\"")
    print(f"   Top {top_k} matches:")
    for cand, sim in similarities[:top_k]:
        print(f"   [{sim:.4f}] {cand}")

    return similarities[:top_k]


# Test it!
candidates = [
    "How to train a neural network",
    "Best pizza recipes in Italy",
    "Introduction to machine learning",
    "Python programming for beginners",
    "Deep learning with PyTorch",
    "How to bake chocolate cake",
    "Natural language processing basics",
    "Gardening tips for spring",
]

find_most_similar("I want to learn about AI", candidates)
find_most_similar("cooking dinner tonight", candidates)
```

---

## Common Mistakes

### Mistake 1: Comparing embeddings from different models
```python
# ❌ These live in DIFFERENT vector spaces — comparison is meaningless!
emb_a = openai_model.embed("hello")      # 1536 dimensions
emb_b = huggingface_model.embed("hello")  # 384 dimensions
similarity(emb_a, emb_b)  # ← WRONG!

# ✅ Always use the SAME model for all embeddings you compare
emb_a = openai_model.embed("hello")
emb_b = openai_model.embed("world")
similarity(emb_a, emb_b)  # ← Correct!
```

### Mistake 2: Thinking embeddings are deterministic translations
```python
# ❌ Embeddings don't have "meaning" per dimension
# Dimension 42 does NOT always mean "royalty"
# The meaning is distributed across ALL dimensions

# ✅ Compare embeddings as wholes, not individual dimensions
```

### Mistake 3: Embedding very long text without splitting
```python
# ❌ Each model has a max token limit for input
emb = embed(entire_500_page_book)  # Will fail or truncate!

# ✅ Split into chunks first, embed each chunk
chunks = split(book, chunk_size=500)
embeddings = [embed(chunk) for chunk in chunks]
```

### Mistake 4: Re-embedding documents on every query
```python
# ❌ Expensive! Embedding 10K documents on every search
for doc in documents:
    emb = embed(doc)  # API call for each doc, every time!

# ✅ Embed once, store in vector DB, query with just the question embedding
# (This is what Phase 4 — Vector Databases is for)
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use the **same model** for queries and documents | Different models = different vector spaces |
| **Batch** embedding calls when possible | Faster and cheaper than one-by-one |
| Choose model based on use case | Smaller models (384D) = faster; larger (3072D) = more accurate |
| **Pre-compute** document embeddings, store them | Don't re-embed on every query |
| Use **cosine similarity** for text | Standard metric, works best with normalized embeddings |
| Consider **dimensionality** vs accuracy tradeoff | 1536D is a good balance for most use cases |

---

## Interview Preparation

### Easy
**Q: What is an embedding?**

> A numerical vector representation of text (word, sentence, or document) where semantically similar texts have similar vectors. Embeddings capture meaning in a high-dimensional space, enabling mathematical operations on language — like similarity search, clustering, and analogies.

### Medium
**Q: Why do we use cosine similarity instead of Euclidean distance for text embeddings?**

> Cosine similarity measures the **angle** between vectors, ignoring magnitude. This is better for text because two documents about the same topic but of different lengths would have embeddings pointing in the same **direction** but with different **magnitudes**. Cosine similarity captures that they're about the same thing. Euclidean distance would penalize the length difference.

### Hard
**Q: Can you mix embeddings from different models in the same vector store?**

> **No.** Each embedding model learns its own vector space with its own dimensional meanings. A 1536-dimensional vector from OpenAI and a 1536-dimensional vector from Cohere represent completely different spaces. Comparing them is like comparing Celsius and Fahrenheit numbers directly. If you switch embedding models, you must re-embed ALL documents.

### Senior
**Q: What are the tradeoffs between smaller (384D) and larger (3072D) embedding models?**

> Smaller models: faster inference, less storage, cheaper, but less nuanced — they may conflate subtle differences. Larger models: better at capturing fine-grained semantic distinctions, but more expensive to compute, store, and search. For most production RAG systems, 1536D (like `text-embedding-3-small`) is the sweet spot. Use larger models (3072D) only when retrieval quality is critical and cost isn't a constraint. OpenAI's `text-embedding-3` models also support **Matryoshka embeddings** — you can truncate dimensions (e.g., use only the first 512 of 3072) with graceful degradation.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Embedding** | A fixed-length numerical vector representing text meaning |
| **Embedding space** | The high-dimensional space where all embeddings live |
| **Dimensions** | Number of values in the vector (384, 768, 1536, 3072) |
| **Cosine similarity** | Measures angle between vectors (-1 to +1); standard for text |
| **Semantic similarity** | Texts with similar meaning → similar embeddings → high cosine |
| **Distributional hypothesis** | Words in similar contexts get similar embeddings |
| **Same model rule** | Always embed queries and documents with the same model |

---

> [← Previous: Output Formatting](../phase-02-prompt-engineering/chapter-13-output-formatting.md) | [Next: Embedding Models →](chapter-24-embedding-models.md)

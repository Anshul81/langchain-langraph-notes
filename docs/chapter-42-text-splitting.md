# Chapter 11.3: Text Splitting — The Art of Chunking

> **Phase 11 — RAG** | [← Previous: Document Loaders](chapter-41-document-loaders.md) | [Next: Complete RAG Pipeline →](chapter-43-complete-rag-pipeline.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand why chunking strategy is the **most important RAG decision**
- ✅ Use RecursiveCharacterTextSplitter (the default workhorse)
- ✅ Understand chunk size, overlap, and their impact on retrieval
- ✅ Use specialized splitters for code, Markdown, and HTML
- ✅ Implement semantic chunking for higher quality
- ✅ Know how to choose the right chunking strategy for your data

| | |
|---|---|
| **Prerequisites** | Chapter 11.2 (Document Loaders) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction — Why Chunking Matters More Than Anything

If you get **one thing right** in your RAG pipeline, make it chunking. Here's why:

```
Bad chunks = bad retrieval = bad answers (no matter how good your LLM is)

❌ Chunk too large (5000 tokens):
   → Retrieval returns the whole chapter, but the answer is in one paragraph
   → LLM gets distracted by irrelevant text, wastes tokens

❌ Chunk too small (50 tokens):
   → Chunk lacks context ("He won the award" — who? what award?)
   → Retrieval can't match the question to a meaningful snippet

✅ Chunk just right (300-1000 tokens):
   → Each chunk contains a coherent idea with enough context
   → Retrieval returns focused, relevant information
   → LLM generates accurate answers
```

---

## Part 1: RecursiveCharacterTextSplitter — The Default Choice

This is the splitter you should use **90% of the time**:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,          # Max characters per chunk
    chunk_overlap=50,        # Overlap between adjacent chunks
    length_function=len,     # How to measure length
    separators=["\n\n", "\n", ". ", " ", ""]  # Priority order for splitting
)

text = """
Machine learning is a subset of artificial intelligence that enables systems to learn 
from data without being explicitly programmed. It was first coined by Arthur Samuel in 1959.

There are three main types of machine learning: supervised learning, unsupervised learning, 
and reinforcement learning. Supervised learning uses labeled data to train models, while 
unsupervised learning finds patterns in unlabeled data.

Deep learning is a subset of machine learning that uses neural networks with many layers. 
The breakthrough came in 2012 with AlexNet winning the ImageNet competition. Since then, 
deep learning has revolutionized computer vision, natural language processing, and speech recognition.

Transformers, introduced in 2017 by Vaswani et al., replaced recurrent neural networks 
for most NLP tasks. They use self-attention mechanisms to process sequences in parallel, 
enabling much faster training and better performance.
"""

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
```

### How Recursive Splitting Works

```
The splitter tries separators IN ORDER:

1. "\n\n" (paragraph breaks)     ← Try this first
2. "\n"   (line breaks)          ← Then this
3. ". "   (sentence endings)     ← Then this
4. " "    (word boundaries)      ← Then this
5. ""     (character by character)← Last resort

Algorithm:
1. Try splitting by "\n\n" → if chunks are small enough, done!
2. If a chunk is still too large, split that chunk by "\n"
3. Still too large? Split by ". "
4. Still too large? Split by " "
5. This ensures chunks break at the most natural boundaries possible.
```

### Chunk Size and Overlap Explained

```
Text: "AAAA BBBB CCCC DDDD EEEE FFFF GGGG HHHH"
      |                                           |
      chunk_size = 20 chars, chunk_overlap = 5

Chunk 1: "AAAA BBBB CCCC DDDD"
                          ^^^^
Chunk 2:              "DDDD EEEE FFFF GGGG"
                                      ^^^^
Chunk 3:                          "GGGG HHHH"

Overlap ensures:
✅ Context is preserved across chunk boundaries
✅ A sentence split across chunks appears in both
✅ Retrieval can find information at chunk boundaries
```

---

## Part 2: Choosing Chunk Size

### Guidelines

| Chunk Size | Tokens | Best For | Trade-offs |
|-----------|--------|----------|------------|
| **200-300** chars | ~50-75 | FAQs, definitions | Very focused but may lack context |
| **500-800** chars | ~125-200 | General documents | Good balance (recommended default) |
| **1000-1500** chars | ~250-375 | Technical docs, research | More context but less precise retrieval |
| **2000-4000** chars | ~500-1000 | Summaries, long narratives | Risk: retrieves too much noise |

### Measuring in Tokens (More Accurate)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Use token-based splitting instead of character-based
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # GPT-4's tokenizer
    chunk_size=300,                # 300 tokens per chunk
    chunk_overlap=30               # 30 token overlap
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    tokens = len(tokenizer.encode(chunk))
    print(f"Chunk {i+1}: {tokens} tokens, {len(chunk)} chars")
```

### Choosing Overlap

```python
# Rules of thumb:
# - 10-15% of chunk_size is a good overlap
# - Too little overlap → lose context at boundaries
# - Too much overlap → redundant data, higher cost

# chunk_size=500, overlap=50  (10%) ← Good default
# chunk_size=1000, overlap=100 (10%) ← Good for large chunks
# chunk_size=500, overlap=200 (40%) ← Too much! Wastes space/tokens
```

---

## Part 3: Splitting Documents (Not Just Text)

### Splitting LangChain Documents

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split Document objects — metadata is preserved!
docs = [
    Document(
        page_content="A very long document about machine learning..." * 20,
        metadata={"source": "ml_textbook.pdf", "page": 1, "chapter": "Introduction"}
    )
]

chunks = splitter.split_documents(docs)

print(f"Split 1 document into {len(chunks)} chunks")
for chunk in chunks[:3]:
    print(f"  [{chunk.metadata['source']}, p{chunk.metadata['page']}] {chunk.page_content[:80]}...")
    # Metadata is AUTOMATICALLY inherited by all chunks!
```

---

## Part 4: Specialized Splitters

### Code Splitter

```python
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

# Python code splitter — splits at class/function boundaries
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)

python_code = """
class Calculator:
    \"\"\"A simple calculator class.\"\"\"
    
    def __init__(self):
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        \"\"\"Add two numbers.\"\"\"
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        \"\"\"Multiply two numbers.\"\"\"
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self) -> list:
        \"\"\"Get calculation history.\"\"\"
        return self.history.copy()


def main():
    calc = Calculator()
    print(calc.add(5, 3))
    print(calc.multiply(4, 7))
    print(calc.get_history())
"""

chunks = python_splitter.split_text(python_code)
for i, chunk in enumerate(chunks):
    print(f"\n--- Code Chunk {i+1} ---")
    print(chunk)
```

### Supported Languages

```python
# Available languages for code splitting:
print([lang.value for lang in Language])
# ['cpp', 'go', 'java', 'kotlin', 'js', 'ts', 'php', 'proto', 'python', 
#  'rst', 'ruby', 'rust', 'scala', 'swift', 'markdown', 'latex', 'html', 'sol']
```

### Markdown Header Splitter

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """# Product Documentation

## Getting Started

Install our package using pip:
```
pip install our-product
```

Follow the quickstart guide to set up your first project.

## Configuration

### Database Setup
Configure your database connection in `config.yaml`:
- host: localhost
- port: 5432
- database: myapp

### API Keys
Set your API keys as environment variables.

## Troubleshooting

### Common Errors
If you see "Connection refused", check that the database is running.

### Getting Help
Contact support@example.com for assistance.
"""

headers = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)

for chunk in chunks:
    print(f"Headers: {chunk.metadata}")
    print(f"Content: {chunk.page_content[:100]}...\n")
```

### HTML Splitter

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

html_text = """
<html>
<h1>Product Guide</h1>
<p>Welcome to our product guide.</p>
<h2>Installation</h2>
<p>Run pip install our-product.</p>
<h2>Usage</h2>
<h3>Basic Usage</h3>
<p>Import and create an instance.</p>
<h3>Advanced Usage</h3>
<p>Configure custom settings.</p>
</html>
"""

headers = [
    ("h1", "h1"),
    ("h2", "h2"),
    ("h3", "h3"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(html_text)

for chunk in chunks:
    print(f"Headers: {chunk.metadata}")
    print(f"Content: {chunk.page_content}\n")
```

---

## Part 5: Semantic Chunking (Advanced)

Instead of splitting by character count, split by **semantic meaning**:

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

# Semantic chunker — splits where the meaning changes significantly
semantic_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=70            # Higher = bigger chunks
)

text = """
Machine learning is a subset of artificial intelligence. It enables systems to learn 
from data automatically. The field was pioneered by Arthur Samuel in 1959.

Python is the most popular language for machine learning. Libraries like scikit-learn, 
TensorFlow, and PyTorch make it easy to build ML models. Python's simplicity makes it 
ideal for data science.

The stock market had a volatile day. The S&P 500 dropped 2% while tech stocks rallied. 
Investors were cautious ahead of the Fed meeting. Gold prices remained stable.

Neural networks are inspired by the human brain. They consist of layers of interconnected 
nodes. Deep learning uses many such layers for complex pattern recognition.
"""

chunks = semantic_splitter.split_text(text)

print(f"Semantic splitting produced {len(chunks)} chunks:")
for i, chunk in enumerate(chunks):
    print(f"\n--- Semantic Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
```

### When to Use Semantic Chunking

| Use Semantic Chunking When | Use Character Chunking When |
|---------------------------|----------------------------|
| Documents mix multiple topics | Documents are homogeneous |
| Topic boundaries are important | You need predictable chunk sizes |
| Quality > speed/cost | Speed and cost matter more |
| You have good embeddings | You want zero-cost splitting |

---

## Part 6: Comparing Strategies

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)

text = "A long document..." * 100  # Simulated long text

# Strategy 1: Character splitter (simplest, least intelligent)
char_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=50
)

# Strategy 2: Recursive character splitter (RECOMMENDED)
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Compare results
char_chunks = char_splitter.split_text(text)
recursive_chunks = recursive_splitter.split_text(text)

print(f"CharacterTextSplitter: {len(char_chunks)} chunks")
print(f"RecursiveCharacter:    {len(recursive_chunks)} chunks")
```

### Strategy Comparison Table

| Splitter | How It Works | Best For | Quality |
|----------|-------------|----------|---------|
| **CharacterTextSplitter** | Splits on one separator | Simple text, logs | ⭐ |
| **RecursiveCharacterTextSplitter** | Tries multiple separators in order | General documents | ⭐⭐⭐ |
| **TokenTextSplitter** | Splits by token count | When token budget matters | ⭐⭐ |
| **MarkdownHeaderTextSplitter** | Splits by Markdown headers | Documentation | ⭐⭐⭐ |
| **Language-specific** | Splits by code structure | Source code | ⭐⭐⭐ |
| **SemanticChunker** | Splits by meaning change | Mixed-topic documents | ⭐⭐⭐⭐ |

---

## Common Mistakes

### Mistake 1: Not using any overlap
```python
# ❌ Zero overlap — information at boundaries is lost
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

# ✅ 10-15% overlap preserves boundary context
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
```

### Mistake 2: Chunk size too large
```python
# ❌ Each chunk is a whole page — retrieval returns too much noise
splitter = RecursiveCharacterTextSplitter(chunk_size=5000)

# ✅ 300-1000 characters is the sweet spot
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
```

### Mistake 3: Using character splitter for code
```python
# ❌ Splits in the middle of functions!
splitter = RecursiveCharacterTextSplitter(chunk_size=500)
chunks = splitter.split_text(python_code)

# ✅ Use language-aware splitter
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=500, chunk_overlap=50
)
```

### Mistake 4: Not testing chunk quality
```python
# ❌ Just splitting and hoping for the best

# ✅ Always inspect your chunks!
chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks[:5]):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(chunk[:200])
    print("---")
# Ask: Does each chunk make sense on its own? Contains enough context?
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Start with RecursiveCharacterTextSplitter | Best default for most data |
| Use 500-800 character chunks | Good balance of precision and context |
| Set overlap to 10-15% of chunk_size | Preserve boundary context |
| Use token-based splitting for LLM context management | More predictable than character count |
| Use language-specific splitters for code | Preserves function/class boundaries |
| Use MarkdownHeaderTextSplitter for docs | Natural section boundaries |
| Always inspect chunk quality | Bad chunks = bad retrieval |
| Test retrieval with real queries | Verify chunks contain answers |

---

## Interview Preparation

### Easy
**Q: Why is chunking important in RAG?**

> Chunking determines the quality of retrieval, which determines the quality of answers. Too-large chunks return irrelevant noise alongside the answer. Too-small chunks lack context. The ideal chunk (300-1000 tokens) contains a complete, coherent piece of information that can match a user's query and provide enough context for the LLM to generate an accurate answer.

### Medium
**Q: What is RecursiveCharacterTextSplitter and why is it the recommended default?**

> It splits text by trying multiple separators in priority order: paragraph breaks → line breaks → sentences → words → characters. This means it always splits at the most natural boundary possible for the given chunk size. Unlike simple character or line splitting, it preserves semantic coherence within chunks. It also supports chunk overlap to maintain context across boundaries.

### Hard
**Q: How would you choose the right chunking strategy for a mixed corpus of technical docs, code, and conversational data?**

> Use **different splitters for different content types**: (1) Technical documentation → MarkdownHeaderTextSplitter if Markdown, or RecursiveCharacterTextSplitter with ~800 char chunks. (2) Source code → Language-specific splitters that respect function/class boundaries. (3) Conversational data → Smaller chunks (300-500 chars) since each message is self-contained. (4) For mixed-topic documents → SemanticChunker to split where the topic changes. Add metadata to each chunk identifying its type, then potentially use different retrieval strategies per type. Always validate by testing retrieval quality with representative queries for each content type.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Chunking** | Splitting documents into smaller pieces for retrieval |
| **RecursiveCharacterTextSplitter** | Default splitter — tries natural boundaries in order |
| **chunk_size** | Maximum characters/tokens per chunk (sweet spot: 500-800) |
| **chunk_overlap** | Characters shared between adjacent chunks (10-15%) |
| **Token-based splitting** | More accurate than character-based for LLM context |
| **Language splitter** | Respects code structure (function/class boundaries) |
| **MarkdownHeaderTextSplitter** | Splits by heading hierarchy |
| **SemanticChunker** | Splits by meaning change (embedding-based) |

---

> [← Previous: Document Loaders](chapter-41-document-loaders.md) | [Next: Complete RAG Pipeline →](chapter-43-complete-rag-pipeline.md)

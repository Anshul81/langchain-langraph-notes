# Chapter 11.2: Document Loaders — Ingesting Data from Any Source

> **Phase 11 — RAG** | [← Previous: Intro to RAG](chapter-40-intro-to-rag.md) | [Next: Text Splitting →](chapter-42-text-splitting.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Load documents from PDF, CSV, JSON, Markdown, HTML, and text files
- ✅ Scrape web pages and websites into documents
- ✅ Load data from APIs and databases
- ✅ Understand the `Document` object (page_content + metadata)
- ✅ Handle large document sets with lazy loading
- ✅ Build a **multi-source document ingestion pipeline**

| | |
|---|---|
| **Prerequisites** | Chapter 11.1 (Intro to RAG) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

RAG starts with data. LangChain provides **100+ document loaders** that convert various data sources into a standard format: `Document(page_content, metadata)`.

```
📄 PDFs         → Document(page_content="...", metadata={"source": "report.pdf", "page": 3})
📊 CSVs         → Document(page_content="...", metadata={"source": "data.csv", "row": 42})
🌐 Web pages    → Document(page_content="...", metadata={"source": "https://..."})
📝 Markdown     → Document(page_content="...", metadata={"source": "README.md"})
💾 Databases    → Document(page_content="...", metadata={"table": "products"})
📧 Emails       → Document(page_content="...", metadata={"from": "alice@..."})
```

**Every loader outputs the same format**, so the rest of your RAG pipeline doesn't care where the data came from.

---

## Part 1: The Document Object

```python
from langchain_core.documents import Document

# Every document has two fields:
doc = Document(
    page_content="This is the text content of the document.",
    metadata={
        "source": "report.pdf",
        "page": 1,
        "author": "Aarav",
        "created_at": "2024-09-01"
    }
)

print(doc.page_content)  # The text
print(doc.metadata)       # The metadata dict
```

### Why Metadata Matters

```python
# Metadata enables:
# 1. Filtering during retrieval
results = vectorstore.similarity_search("revenue", filter={"source": "financials.pdf"})

# 2. Source citations in answers
print(f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}")

# 3. Debugging retrieval issues
for doc in results:
    print(f"[{doc.metadata['source']}:{doc.metadata.get('page', '?')}] {doc.page_content[:80]}...")
```

---

## Part 2: Text Files

```python
from langchain_community.document_loaders import TextLoader

# Load a single text file
loader = TextLoader("./data/readme.txt", encoding="utf-8")
docs = loader.load()

print(f"Loaded {len(docs)} document(s)")
print(f"Content: {docs[0].page_content[:200]}...")
print(f"Metadata: {docs[0].metadata}")
# {'source': './data/readme.txt'}
```

### Loading a Directory of Files

```python
from langchain_community.document_loaders import DirectoryLoader

# Load ALL .txt files from a directory
loader = DirectoryLoader(
    "./data/",
    glob="**/*.txt",           # Pattern: all .txt files, including subdirectories
    loader_cls=TextLoader,
    show_progress=True,
    use_multithreading=True    # Parallel loading for speed
)

docs = loader.load()
print(f"Loaded {len(docs)} documents from directory")
```

---

## Part 3: PDF Files

### PyPDF (Most Common)

```bash
pip install pypdf
```

```python
from langchain_community.document_loaders import PyPDFLoader

# Load a PDF — each page becomes a separate Document
loader = PyPDFLoader("./data/annual_report.pdf")
pages = loader.load()

print(f"Loaded {len(pages)} pages")
print(f"Page 1 content: {pages[0].page_content[:200]}...")
print(f"Page 1 metadata: {pages[0].metadata}")
# {'source': './data/annual_report.pdf', 'page': 0}  ← 0-indexed
```

### PyPDFLoader with Options

```python
# Extract text with different strategies
loader = PyPDFLoader(
    "./data/annual_report.pdf",
    extract_images=False  # Skip image extraction for speed
)
pages = loader.load()
```

### Loading Multiple PDFs

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Load all PDFs from a directory
loader = DirectoryLoader(
    "./data/pdfs/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)

all_pages = loader.load()
print(f"Loaded {len(all_pages)} pages from all PDFs")

# Check which PDFs were loaded
sources = set(doc.metadata["source"] for doc in all_pages)
print(f"Sources: {sources}")
```

---

## Part 4: CSV Files

```python
from langchain_community.document_loaders.csv_loader import CSVLoader

# Each row becomes a Document
loader = CSVLoader(
    "./data/products.csv",
    csv_args={"delimiter": ","},
    source_column="product_id"  # Use this column as the 'source' metadata
)

docs = loader.load()

print(f"Loaded {len(docs)} rows")
print(f"Row 1: {docs[0].page_content}")
# "product_id: P001\nname: Wireless Headphones\nprice: 2499\ncategory: Electronics"
print(f"Metadata: {docs[0].metadata}")
# {'source': 'P001', 'row': 0}
```

### Selecting Specific Columns

```python
# Only load specific columns
loader = CSVLoader(
    "./data/products.csv",
    csv_args={"delimiter": ","},
    content_columns=["name", "description", "price"],  # Only these columns in content
    metadata_columns=["category", "product_id"]         # These go to metadata
)
```

---

## Part 5: JSON Files

```python
from langchain_community.document_loaders import JSONLoader

# Load from a JSON file with jq-style path
loader = JSONLoader(
    file_path="./data/products.json",
    jq_schema=".products[]",          # jq expression to extract items
    text_content=False,                # Parse as JSON, not raw text
    content_key="description"          # Which field becomes page_content
)

docs = loader.load()
print(f"Loaded {len(docs)} items from JSON")
```

### JSON Lines (JSONL)

```python
loader = JSONLoader(
    file_path="./data/logs.jsonl",
    jq_schema=".",
    text_content=False,
    json_lines=True  # Each line is a separate JSON object
)

docs = loader.load()
```

---

## Part 6: Markdown Files

```python
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# Load and parse Markdown
loader = UnstructuredMarkdownLoader("./data/README.md")
docs = loader.load()

print(f"Content: {docs[0].page_content[:200]}...")
```

### Split by Headers (Better for RAG)

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

# Split Markdown by headers — each section becomes a document
markdown_text = """
# Introduction
This is the introduction section about our product.

## Features
Our product has these key features:
- Feature A: Fast processing
- Feature B: Easy to use

## Pricing
We offer three tiers: Basic ($9), Pro ($29), Enterprise (custom).

# FAQ
Common questions and answers.

## How do I get started?
Sign up at our website and follow the quickstart guide.
"""

headers_to_split_on = [
    ("#", "h1"),
    ("##", "h2"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = splitter.split_text(markdown_text)

for doc in docs:
    print(f"Metadata: {doc.metadata}")
    print(f"Content: {doc.page_content[:80]}...\n")
```

---

## Part 7: Web Pages

### Single Web Page

```python
from langchain_community.document_loaders import WebBaseLoader

# Load a single web page
loader = WebBaseLoader("https://docs.python.org/3/tutorial/index.html")
docs = loader.load()

print(f"Title: {docs[0].metadata.get('title', 'N/A')}")
print(f"Content: {docs[0].page_content[:300]}...")
```

### Multiple Web Pages

```python
# Load multiple URLs at once
urls = [
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
]

loader = WebBaseLoader(urls)
docs = loader.load()

print(f"Loaded {len(docs)} pages")
for doc in docs:
    print(f"  📄 {doc.metadata.get('source', 'N/A')}: {doc.page_content[:100]}...")
```

### Recursive Web Crawling

```python
from langchain_community.document_loaders import RecursiveUrlLoader

# Crawl a website following links (be careful with depth!)
loader = RecursiveUrlLoader(
    url="https://python.langchain.com/docs/introduction/",
    max_depth=2,         # How many levels of links to follow
    prevent_outside=True  # Stay within the same domain
)

docs = loader.load()
print(f"Crawled {len(docs)} pages")
```

---

## Part 8: HTML Files

```python
from langchain_community.document_loaders import BSHTMLLoader

# Load a local HTML file
loader = BSHTMLLoader("./data/page.html")
docs = loader.load()

print(f"Title: {docs[0].metadata.get('title', 'N/A')}")
print(f"Content: {docs[0].page_content[:200]}...")
```

---

## Part 9: Lazy Loading for Large Datasets

For large document sets, use `lazy_load()` to process one document at a time:

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    "./data/large_corpus/",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

# Lazy loading — processes one document at a time (memory efficient)
doc_count = 0
for doc in loader.lazy_load():
    # Process each document without loading everything into RAM
    doc_count += 1
    
    if doc_count % 100 == 0:
        print(f"Processed {doc_count} documents...")
    
    # Add to vector store one at a time
    # vectorstore.add_documents([doc])

print(f"Total: {doc_count} documents")
```

---

## Part 10: Complete Project — Multi-Source Ingestion Pipeline

```python
import os
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader, WebBaseLoader,
    DirectoryLoader, UnstructuredMarkdownLoader,
)


class DocumentIngestionPipeline:
    """Load documents from multiple sources into a unified format."""
    
    def __init__(self):
        self.documents: list[Document] = []
        self.stats = {"total": 0, "by_source": {}}
    
    def load_text_files(self, directory: str, glob: str = "**/*.txt"):
        """Load text files from a directory."""
        loader = DirectoryLoader(directory, glob=glob, loader_cls=TextLoader)
        docs = loader.load()
        self._add_docs(docs, "text_files")
        return self
    
    def load_pdfs(self, directory: str):
        """Load all PDFs from a directory."""
        loader = DirectoryLoader(
            directory, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
        )
        docs = loader.load()
        self._add_docs(docs, "pdfs")
        return self
    
    def load_csv(self, filepath: str, content_columns: list = None):
        """Load a CSV file."""
        loader = CSVLoader(filepath)
        docs = loader.load()
        self._add_docs(docs, "csv")
        return self
    
    def load_markdown(self, directory: str):
        """Load Markdown files from a directory."""
        loader = DirectoryLoader(
            directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
        )
        docs = loader.load()
        self._add_docs(docs, "markdown")
        return self
    
    def load_web_pages(self, urls: list[str]):
        """Load web pages."""
        loader = WebBaseLoader(urls)
        docs = loader.load()
        self._add_docs(docs, "web")
        return self
    
    def load_custom_data(self, items: list[dict], source_name: str = "custom"):
        """Load custom data as documents."""
        docs = [
            Document(
                page_content=item.get("content", str(item)),
                metadata={
                    "source": source_name,
                    **item.get("metadata", {})
                }
            )
            for item in items
        ]
        self._add_docs(docs, source_name)
        return self
    
    def _add_docs(self, docs: list[Document], source_type: str):
        """Add documents and update stats."""
        # Add source_type to metadata
        for doc in docs:
            doc.metadata["source_type"] = source_type
        
        self.documents.extend(docs)
        self.stats["total"] += len(docs)
        self.stats["by_source"][source_type] = (
            self.stats["by_source"].get(source_type, 0) + len(docs)
        )
        
        print(f"📥 Loaded {len(docs)} documents from {source_type}")
    
    def get_documents(self) -> list[Document]:
        """Get all loaded documents."""
        return self.documents
    
    def print_stats(self):
        """Print ingestion statistics."""
        print(f"\n📊 Ingestion Stats:")
        print(f"   Total documents: {self.stats['total']}")
        for source, count in self.stats["by_source"].items():
            print(f"   {source}: {count}")
        
        # Character count
        total_chars = sum(len(doc.page_content) for doc in self.documents)
        print(f"   Total characters: {total_chars:,}")
        print(f"   Avg chars/doc: {total_chars // max(len(self.documents), 1):,}")


# --- Usage ---
pipeline = DocumentIngestionPipeline()

# Load from multiple sources
pipeline.load_custom_data([
    {"content": "Our company was founded in 2020 in Bangalore.", 
     "metadata": {"category": "about"}},
    {"content": "Q3 revenue was ₹4.5 crore, up 35% from Q2.", 
     "metadata": {"category": "finance"}},
    {"content": "We have 45 employees across 4 departments.", 
     "metadata": {"category": "team"}},
    {"content": "CodeAssist is our main product — AI code review.", 
     "metadata": {"category": "product"}},
], source_name="company_data")

# Print stats
pipeline.print_stats()

# Get all documents ready for the next step (splitting → embedding → storing)
all_docs = pipeline.get_documents()
print(f"\n✅ {len(all_docs)} documents ready for RAG pipeline")
```

---

## Common Mistakes

### Mistake 1: Not preserving metadata
```python
# ❌ Loading without source tracking
docs = [Document(page_content=text) for text in texts]

# ✅ Always include source metadata
docs = [Document(
    page_content=text, 
    metadata={"source": filename, "page": i}
) for i, text in enumerate(texts)]
```

### Mistake 2: Loading binary PDFs without proper loader
```python
# ❌ Trying to load PDF as text
with open("report.pdf", "r") as f:
    text = f.read()  # Garbled binary!

# ✅ Use PyPDFLoader
loader = PyPDFLoader("report.pdf")
docs = loader.load()
```

### Mistake 3: Not handling encoding errors
```python
# ❌ Crashes on non-UTF-8 files
loader = TextLoader("data.txt")

# ✅ Specify encoding or auto-detect
loader = TextLoader("data.txt", encoding="utf-8")
# Or use 'latin-1' as fallback for unknown encodings
```

### Mistake 4: Loading everything into memory at once
```python
# ❌ 10,000 large PDFs → out of memory
docs = loader.load()

# ✅ Use lazy loading for large datasets
for doc in loader.lazy_load():
    vectorstore.add_documents([doc])
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Always include source metadata | Enables citations and debugging |
| Use the right loader for each format | Proper parsing, no garbled text |
| Use `lazy_load()` for large datasets | Memory efficiency |
| Use `DirectoryLoader` for batch processing | Load entire directories at once |
| Handle encoding explicitly | Prevent crashes on non-UTF-8 files |
| Log ingestion stats | Track how much data you're processing |
| Deduplicate before embedding | Save embedding costs |
| Validate loaded content | Check for empty documents |

---

## Interview Preparation

### Easy
**Q: What is a document loader in LangChain?**

> A document loader converts data from various sources (PDFs, web pages, CSVs, databases, etc.) into LangChain's standard `Document` format, which has two fields: `page_content` (the text) and `metadata` (source info, page number, etc.). This standardized format ensures the rest of the RAG pipeline works regardless of the original data source.

### Medium
**Q: How would you load documents from a large corpus of PDFs efficiently?**

> Use `DirectoryLoader` with `PyPDFLoader` as the loader class, enabling `show_progress=True` and `use_multithreading=True` for speed. For very large corpora (10,000+ PDFs), use `lazy_load()` to process one document at a time without loading everything into memory. Add documents to the vector store incrementally. Always preserve metadata (source file, page number) for debugging and citations. Handle errors gracefully — some PDFs may be corrupted or encrypted.

---

## Summary

| Loader | Use Case | Output |
|--------|----------|--------|
| **TextLoader** | Plain text files | 1 doc per file |
| **PyPDFLoader** | PDF files | 1 doc per page |
| **CSVLoader** | CSV/spreadsheet data | 1 doc per row |
| **JSONLoader** | JSON/JSONL data | 1 doc per item |
| **WebBaseLoader** | Web pages | 1 doc per URL |
| **DirectoryLoader** | Batch load from directory | Multiple docs |
| **UnstructuredMarkdownLoader** | Markdown files | 1 doc per file |
| **MarkdownHeaderTextSplitter** | Split Markdown by headers | 1 doc per section |

---

> [← Previous: Intro to RAG](chapter-40-intro-to-rag.md) | [Next: Text Splitting →](chapter-42-text-splitting.md)

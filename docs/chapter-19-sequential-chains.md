# Chapter 4.2: Sequential Chains & Multi-Step Workflows

> **Phase 4 — Chains & Runnables** | [← Previous: Runnables Deep Dive](chapter-18-runnables-deep-dive.md) | [Next: Retry, Fallback & Error Handling →](chapter-20-retry-fallback-error-handling.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Build sequential chains where output of step N feeds into step N+1
- ✅ Transform data between chain steps
- ✅ Build multi-LLM-call workflows (research → draft → review → polish)
- ✅ Debug chains with intermediate inspection
- ✅ Handle different input/output shapes between steps

| | |
|---|---|
| **Prerequisites** | Chapter 4.1 (Runnables Deep Dive) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

Chapter 4.1 taught you `RunnableParallel` — running things **side by side**. But many workflows are **sequential** — each step depends on the previous:

```
User Question → Research → Draft Answer → Review → Polish → Final Output
                  ↓            ↓            ↓         ↓
              (LLM call)   (LLM call)   (LLM call) (LLM call)
```

This is the **most common pattern** in production AI apps:
- Generate → then Review → then Fix
- Extract → then Enrich → then Validate
- Summarize → then Translate → then Format

---

## Part 1: Basic Sequential Chain (String → String)

The simplest sequential chain — output of one step feeds the next:

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "standard"),
    temperature=0.7,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=300
)

# Step 1: Generate a story
story_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative writer. Write a very short story (3-4 sentences)."),
    ("human", "Topic: {topic}")
])

# Step 2: Critique the story
critique_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a literary critic. Give a brief critique."),
    ("human", "Review this story:\n{text}")
])

# Sequential: story → critique
# ⚠️ Step 1 outputs a str, but Step 2 expects {"text": "..."}
# Solution: transform between steps!
chain = (
    story_prompt | llm | StrOutputParser()
    | RunnableLambda(lambda story: {"text": story})  # ← shape transform
    | critique_prompt | llm | StrOutputParser()
)

result = chain.invoke({"topic": "a robot learning to paint"})
print(result)
```

### The Shape Problem

This is the **#1 mistake** with sequential chains:

```python
# ❌ Step 1 outputs str, Step 2 expects dict — will crash!
chain = story_prompt | llm | StrOutputParser() | critique_prompt | llm

# ✅ Add a lambda to transform the shape
chain = (
    story_prompt | llm | StrOutputParser()
    | RunnableLambda(lambda s: {"text": s})  # str → dict
    | critique_prompt | llm | StrOutputParser()
)
```

**Rule**: Always check — does Step N's output shape match Step N+1's input shape?

---

## Part 2: Data Accumulation Pattern (⭐ Most Important)

**Problem**: In a 3-step chain, Step 3 needs data from both Step 1 AND Step 2. But with basic piping, Step 1's output is gone by Step 3.

**Solution**: `RunnablePassthrough.assign()` — keeps ALL previous data and adds new keys:

```python
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

# Step 1: Research key points
research_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "List 3 key points about the topic."),
        ("human", "Topic: {topic}")
    ]) | llm | StrOutputParser()
)

# Step 2: Draft using key points AND audience (from original input)
draft_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Write a short article using these key points for the given audience."),
        ("human", "Key points:\n{key_points}\n\nAudience: {audience}")
    ]) | llm | StrOutputParser()
)

# 🔥 Data Accumulation Pipeline
pipeline = (
    # Start: {"topic": "...", "audience": "..."}
    RunnablePassthrough.assign(
        key_points=research_chain          # adds "key_points" key
    )
    # Now: {"topic": "...", "audience": "...", "key_points": "..."}
    | RunnablePassthrough.assign(
        draft=draft_chain                  # adds "draft" key
    )
    # Now: {"topic": "...", "audience": "...", "key_points": "...", "draft": "..."}
)

result = pipeline.invoke({"topic": "AI in healthcare", "audience": "doctors"})
# result has ALL keys: topic, audience, key_points, draft
```

### How `.assign()` Works Step-by-Step

```
Input:  {"topic": "AI", "audience": "doctors"}
                    │
    ┌───────────────┼───────────────────┐
    │ RunnablePassthrough.assign(       │
    │   key_points=research_chain       │
    │ )                                 │
    └───────────────┼───────────────────┘
                    │
After:  {"topic": "AI", "audience": "doctors", "key_points": "1. Diagnosis..."}
                    │
    ┌───────────────┼───────────────────┐
    │ RunnablePassthrough.assign(       │
    │   draft=draft_chain               │
    │ )                                 │
    └───────────────┼───────────────────┘
                    │
After:  {"topic": "AI", "audience": "doctors", "key_points": "1. ...", "draft": "AI is transforming..."}
```

**This is the pattern you'll use 90% of the time.**

---

## Part 3: Multi-LLM Workflow (Real-World Example)

A production-style 4-step pipeline — research → draft → review → polish:

```python
# Step 1: Research
research_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a researcher. Generate 3-4 key points."),
        ("human", "Research topic: {topic}")
    ]) | llm | StrOutputParser()
)

# Step 2: Draft
draft_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a writer. Write a short article."),
        ("human", "Key points:\n{research}\n\nAudience: {audience}")
    ]) | llm | StrOutputParser()
)

# Step 3: Review
review_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an editor. List 2-3 specific improvements."),
        ("human", "Review this article:\n{draft}")
    ]) | llm | StrOutputParser()
)

# Step 4: Polish
polish_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a senior editor. Apply the feedback to improve the article."),
        ("human", "Original article:\n{draft}\n\nFeedback:\n{review}")
    ]) | llm | StrOutputParser()
)

# Full pipeline with data accumulation
full_pipeline = (
    RunnablePassthrough.assign(research=research_chain)
    | RunnablePassthrough.assign(draft=draft_chain)
    | RunnablePassthrough.assign(review=review_chain)
    | RunnablePassthrough.assign(final_article=polish_chain)
)

result = full_pipeline.invoke({
    "topic": "Benefits of meditation",
    "audience": "busy professionals"
})
# result keys: topic, audience, research, draft, review, final_article
```

---

## Part 4: Debugging Sequential Chains

When chains get complex, inspect intermediate data:

```python
def inspect(data, step_name=""):
    """Print and pass through — for debugging."""
    print(f"\n{'='*40}")
    print(f"🔍 After {step_name}:")
    if isinstance(data, dict):
        for k, v in data.items():
            print(f"  {k}: {str(v)[:100]}...")
    else:
        print(f"  {str(data)[:200]}")
    print(f"{'='*40}\n")
    return data  # ← MUST return data unchanged!

# Insert inspection between steps
pipeline = (
    RunnablePassthrough.assign(research=research_chain)
    | RunnableLambda(lambda x: inspect(x, "Research"))   # ← peek
    | RunnablePassthrough.assign(draft=draft_chain)
    | RunnableLambda(lambda x: inspect(x, "Draft"))      # ← peek
)
```

### Pro Tip: Reusable Inspector

```python
def make_inspector(step_name: str):
    """Factory for inspection functions."""
    @RunnableLambda
    def _inspect(data):
        print(f"🔍 [{step_name}] keys={list(data.keys()) if isinstance(data, dict) else type(data)}")
        return data
    return _inspect

pipeline = (
    RunnablePassthrough.assign(research=research_chain)
    | make_inspector("after_research")
    | RunnablePassthrough.assign(draft=draft_chain)
    | make_inspector("after_draft")
)
```

---

## The 3 Sequential Patterns Compared

| Pattern | When to Use | Data Flow |
|---------|------------|-----------|
| **Basic pipe** `a \| b \| c` | Each step only needs previous output | `str → str → str` |
| **Lambda transform** | Steps have different input shapes | `str → {"key": str} → str` |
| **`RunnablePassthrough.assign()`** | Later steps need earlier data | `dict grows with each step` |

---

## Common Mistakes

### Mistake 1: Shape mismatch between steps
```python
# ❌ prompt expects dict, but previous step outputs str
chain = prompt_a | llm | StrOutputParser() | prompt_b | llm

# ✅ Transform str → dict between steps
chain = (
    prompt_a | llm | StrOutputParser()
    | RunnableLambda(lambda s: {"text": s})
    | prompt_b | llm | StrOutputParser()
)
```

### Mistake 2: Losing data from earlier steps
```python
# ❌ After Step 2, original "topic" is lost
chain = prompt_a | llm | StrOutputParser() | prompt_b | llm

# ✅ Use assign() to accumulate
pipeline = (
    RunnablePassthrough.assign(step1_result=chain_a)
    | RunnablePassthrough.assign(step2_result=chain_b)
)
```

### Mistake 3: Not returning data from inspect functions
```python
# ❌ Returns None — breaks the chain!
def bad_inspect(data):
    print(data)

# ✅ Always return the data
def good_inspect(data):
    print(data)
    return data  # ← critical!
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Use `RunnablePassthrough.assign()` for multi-step pipelines | Preserves all intermediate data |
| Always check output→input shape compatibility | Prevents runtime crashes |
| Use `RunnableLambda` transforms between mismatched steps | Clean shape conversion |
| Add `inspect()` functions during development | Easy debugging |
| Remove inspection functions before production | Performance |
| Keep each chain step focused (single responsibility) | Maintainability |

---

## Interview Preparation

### Easy
**Q: What makes a chain "sequential"?**

> Output of Step N becomes input to Step N+1. The key constraint is that input/output types must be compatible — if Chain A outputs a `str` but Chain B expects a `dict`, you need a transform in between.

### Medium
**Q: How do you pass data from Step 1 to Step 3, skipping Step 2?**

> Use `RunnablePassthrough.assign()` to accumulate all outputs into a single dict. Step 1's output stays in the dict as a key, Step 2 adds another key, and Step 3 can access both. This is the **Data Accumulation Pattern**.

### Hard
**Q: How would you implement a chain that retries Step 2 if Step 3 (a reviewer) says the output is bad?**

> This requires a **loop** — which is beyond basic LCEL. You'd either: (1) use `RunnableBranch` with a manual retry counter in state, (2) use a `while` loop around `chain.invoke()` in plain Python, or (3) use **LangGraph** which natively supports cycles and conditional edges. This is actually one of the main reasons LangGraph exists!

### Senior
**Q: Sequential chains vs LangGraph — when do you switch?**

> Sequential chains (`|` LCEL) are perfect for **linear workflows** with optional branching. Switch to LangGraph when you need: (1) cycles/loops (retry until quality), (2) persistent state across steps, (3) human-in-the-loop pauses, (4) complex conditional routing, or (5) checkpointing/recovery. LangGraph is overkill for simple linear pipelines.

---

## Mini Assignment — Answer

Build a **Blog Post Pipeline** that takes `{"topic": "...", "audience": "..."}` and runs:
Research → Draft → SEO → Final output.

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "standard"),
    temperature=0.7,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    max_tokens=500
)

# Pydantic model for structured SEO output
class SEOData(BaseModel):
    """SEO metadata for a blog post."""
    title: str = Field(description="SEO-optimized blog title")
    meta_description: str = Field(description="Meta description (under 160 chars)")
    keywords: list[str] = Field(description="5 SEO keywords")

# Step 1: Research — generates key points
research_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a researcher. Generate 3-4 key points about the topic. Be specific and factual."),
        ("human", "Topic: {topic}")
    ]) | llm | StrOutputParser()
)

# Step 2: Draft — writes blog post using key points + audience
draft_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a blog writer. Write a short blog post (3-4 paragraphs) using the key points, targeted at the specified audience."),
        ("human", "Key points:\n{key_points}\n\nTarget audience: {audience}")
    ]) | llm | StrOutputParser()
)

# Step 3: SEO — structured output using with_structured_output
seo_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an SEO expert. Generate SEO metadata for this blog post."),
        ("human", "Blog post:\n{blog_post}")
    ]) | llm.with_structured_output(SEOData)
)

# Full pipeline with data accumulation
blog_pipeline = (
    # Start: {"topic": "...", "audience": "..."}
    RunnablePassthrough.assign(
        key_points=research_chain                      # Step 1
    )
    # {"topic", "audience", "key_points"}
    | RunnablePassthrough.assign(
        blog_post=draft_chain                          # Step 2
    )
    # {"topic", "audience", "key_points", "blog_post"}
    | RunnablePassthrough.assign(
        seo=seo_chain                                  # Step 3
    )
    # {"topic", "audience", "key_points", "blog_post", "seo"}
)

# Run it!
result = blog_pipeline.invoke({
    "topic": "Benefits of meditation for mental health",
    "audience": "busy professionals"
})

print(f"Topic: {result['topic']}")
print(f"Audience: {result['audience']}")
print(f"\n📋 Key Points:\n{result['key_points']}")
print(f"\n📝 Blog Post:\n{result['blog_post']}")
print(f"\n🔍 SEO Title: {result['seo'].title}")
print(f"📄 Meta: {result['seo'].meta_description}")
print(f"🏷️ Keywords: {result['seo'].keywords}")
```

### Key Techniques Used:
1. **`RunnablePassthrough.assign()`** — accumulates data through each step
2. **`StrOutputParser()`** — for Steps 1 & 2 (free-text output)
3. **`with_structured_output(SEOData)`** — for Step 3 (Pydantic-structured output)
4. **No `itemgetter` needed** — because `.assign()` passes the full dict, and each prompt template picks the keys it needs automatically

---

## Summary

| Component | What It Does |
|-----------|-------------|
| Basic pipe `a \| b \| c` | Linear chain, each step uses only previous output |
| `RunnableLambda(lambda x: {...})` | Transforms data shape between mismatched steps |
| `RunnablePassthrough.assign(key=chain)` | Keeps all previous data + adds new computed key |
| Chained `.assign()` calls | Data accumulation — the #1 sequential pattern |
| `inspect()` functions | Debug by printing intermediate data |

---

> [← Previous: Runnables Deep Dive](chapter-18-runnables-deep-dive.md) | [Next: Retry, Fallback & Error Handling →](chapter-20-retry-fallback-error-handling.md)

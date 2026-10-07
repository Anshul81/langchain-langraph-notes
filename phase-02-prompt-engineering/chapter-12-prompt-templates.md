# Chapter 2.3: Prompt Templates & Variables

> **Phase 2 — Prompt Engineering** | [← Previous: Few-Shot & CoT](chapter-11-few-shot-cot.md) | [Next: Output Parsing & Structured Output →](chapter-13-output-parsing.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Build reusable prompt templates with dynamic variables
- ✅ Understand why hardcoded prompts are bad for production
- ✅ Create a prompt library for consistent, testable prompts
- ✅ Bridge from raw Python to LangChain's `PromptTemplate`

| | |
|---|---|
| **Prerequisites** | Chapter 2.1-2.2 |
| **Estimated Reading Time** | 15 minutes |
| **Estimated Coding Time** | 30 minutes |

---

## Introduction

### The Problem

Hardcoded prompts are unmaintainable:

```python
# ❌ Can't reuse, can't test, can't A/B test
prompt = "Summarize the following article about machine learning in 3 bullet points: " + text
```

When you need different topics, counts, or formats across 10 scripts, you copy-paste everywhere.

### The Solution

Prompt templates separate **structure** from **data**:

```python
template = "Summarize the following {topic} article in {count} bullet points:\n\n{text}"
prompt = template.format(topic="AI", count=3, text=article)
```

One template → infinite variations. Reusable, testable, maintainable.

---

## Mental Model

### The Mad Libs Analogy

```
Template:  "The {adjective} {animal} jumped over the {object}."
Fill in:   adjective="lazy", animal="fox", object="moon"
Result:    "The lazy fox jumped over the moon."
```

Prompt templates work identically:
```
Template:  "Summarize {topic} in {count} points:\n\n{text}"
Fill in:   topic="AI", count=3, text="..."
Result:    "Summarize AI in 3 points:\n\n..."
```

---

## Theory

### The Evolution: Raw → Template

**Level 0 — Hardcoded (Bad):**
```python
prompt = "Translate 'hello' to Spanish"
```

**Level 1 — f-strings (Better but fragile):**
```python
prompt = f"Translate '{word}' to {language}"
# No validation. What if language is None?
```

**Level 2 — Template Function (Good):**
```python
def translation_prompt(word: str, language: str) -> str:
    if not word or not language:
        raise ValueError("word and language are required")
    return f"Translate '{word}' to {language}."
```

**Level 3 — Template Class (Production-grade):**
```python
class PromptTemplate:
    def __init__(self, template: str, input_variables: list[str]):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        return self.template.format(**kwargs)

# Define once, reuse everywhere
summarize = PromptTemplate(
    template="Summarize {topic} in {count} bullet points:\n\n{text}",
    input_variables=["topic", "count", "text"]
)

prompt = summarize.format(topic="AI", count=3, text="...")
```

This is exactly what LangChain's `PromptTemplate` does internally.

---

## Code Examples

### Prompt Library

```python
class PromptTemplate:
    def __init__(self, template: str, input_variables: list[str]):
        self.template = template
        self.input_variables = input_variables
    
    def format(self, **kwargs) -> str:
        missing = set(self.input_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")
        return self.template.format(**kwargs)

# === PROMPT LIBRARY ===

SUMMARIZE = PromptTemplate(
    template="Summarize the following {topic} text in {count} bullet points:\n\n{text}",
    input_variables=["topic", "count", "text"]
)

TRANSLATE = PromptTemplate(
    template="Translate from {source_lang} to {target_lang}:\n\n{text}",
    input_variables=["source_lang", "target_lang", "text"]
)

CODE_REVIEW = PromptTemplate(
    template="Review this {language} code for bugs, performance, and best practices:\n\n```{language}\n{code}\n```",
    input_variables=["language", "code"]
)

CLASSIFY = PromptTemplate(
    template="Classify into one of [{categories}]:\n\nText: {text}\n\nCategory:",
    input_variables=["categories", "text"]
)
```

### System Prompt Templates

```python
SYSTEM_PERSONA = PromptTemplate(
    template="""You are a {role} with expertise in {domain}.
Your communication style is {style}.
Always respond in {format} format.
If you don't know the answer, say "I don't have enough information.""",
    input_variables=["role", "domain", "style", "format"]
)

python_tutor = SYSTEM_PERSONA.format(
    role="Python programming tutor",
    domain="backend development",
    style="patient and encouraging",
    format="markdown"
)

code_reviewer = SYSTEM_PERSONA.format(
    role="senior code reviewer",
    domain="production Python applications",
    style="direct and concise",
    format="bullet points"
)
```

---

## Bridge to LangChain

```python
# YOUR VERSION
template = PromptTemplate(
    template="Translate '{word}' to {language}",
    input_variables=["word", "language"]
)
prompt = template.format(word="hello", language="Spanish")

# LANGCHAIN VERSION (coming in Phase 3!)
from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="Translate '{word}' to {language}",
    input_variables=["word", "language"]
)
prompt = template.format(word="hello", language="Spanish")
```

Nearly identical! LangChain adds: chain integration, `ChatPromptTemplate`, partial variables, and serialization.

---

## Common Mistakes

### Mistake 1: Literal braces in template text
```python
# ❌ Crashes — Python thinks {json} is a variable
template = "Return as {json} format: {data}"

# ✅ Escape literal braces with double braces
template = "Return as {{json}} format: {data}"
```

### Mistake 2: Not validating inputs
```python
# ❌ KeyError at runtime
"Translate {word} to {language}".format(word="hello")

# ✅ Validate first
missing = {"word", "language"} - {"word"}
if missing:
    raise ValueError(f"Missing: {missing}")
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Store templates as constants (UPPER_CASE) | Easy to find, immutable |
| Validate all variables before formatting | Catch errors early |
| Use descriptive variable names | `{user_question}` not `{q}` |
| Keep templates in a dedicated module | Single source of truth |
| Escape literal braces with `{{ }}` | Prevents crashes |

---

## Interview Preparation

### Easy
**Q: Why use prompt templates instead of hardcoded strings?**
**A:** Templates are reusable, testable, and maintainable. They separate prompt structure from data, enabling A/B testing and consistent formatting across the application.

### Medium
**Q: How does LangChain's PromptTemplate differ from a simple f-string?**
**A:** LangChain's PromptTemplate validates required variables, integrates with the Runnable chain interface, supports partial variables, provides ChatPromptTemplate for multi-message arrays, and can be serialized/deserialized.

---

## Summary

| Level | Approach | Use Case |
|-------|----------|----------|
| f-strings | Quick scripts | Prototyping |
| Template functions | Validated, reusable | Small projects |
| Template classes | Full validation, library | Production systems |
| LangChain PromptTemplate | Chain integration, serialization | LangChain applications |

---

## What's Next

In the next chapter, we'll learn **Output Parsing & Structured Output** — how to force LLMs to return JSON, Pydantic models, and structured data instead of free-form text. This is critical for building reliable applications where you need to programmatically process the LLM's response.

> [← Previous: Few-Shot & CoT](chapter-11-few-shot-cot.md) | [Next: Output Parsing & Structured Output →](chapter-13-output-parsing.md)

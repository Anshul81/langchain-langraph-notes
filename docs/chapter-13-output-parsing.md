# Chapter 2.4: Output Parsing & Structured Output

> **Phase 2 — Prompt Engineering** | [← Previous: Prompt Templates](chapter-12-prompt-templates.md) | [Next: Phase 3 — LangChain Installation & Hello World →](chapter-14-langchain-setup.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Force LLMs to return JSON instead of free-form text
- ✅ Parse LLM output into Python dictionaries and objects
- ✅ Use Pydantic to validate LLM responses at runtime
- ✅ Build a robust parser with retry logic
- ✅ Understand the bridge to LangChain's `with_structured_output()`

| | |
|---|---|
| **Prerequisites** | Chapter 0.4 (Pydantic), Chapter 2.1-2.3 |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 40 minutes |

---

## Introduction

### The Problem

LLMs return **free-form text**. But your code needs **structured data**:

```python
# ❌ LLM returns this:
"The movie Inception, directed by Christopher Nolan, was released in 2010 
and I'd rate it about 9.2 out of 10."

# ✅ Your code needs this:
{"title": "Inception", "director": "Christopher Nolan", "year": 2010, "rating": 9.2}
```

You can't do `response["title"]` on a paragraph of text. You need **parsing and validation**.

### The Solution

Combine three techniques:
1. **JSON mode** (`response_format`) — forces syntactically valid JSON
2. **Schema injection** — tell the LLM exactly what keys/types to use
3. **Pydantic validation** — validate the response matches your expected schema

---

## Mental Model

### The Customs Form Analogy

Without output parsing — tourist answers however they want:
> "Oh, my name is John, lovely weather, passport is GB-12345 somewhere..."

With output parsing — structured form:
```
┌──────────────────────────────┐
│ Name:     [John Smith      ] │
│ Passport: [GB-12345        ] │
│ Country:  [United Kingdom  ] │
└──────────────────────────────┘
```

The form **forces** structured output. That's what we're building.

---

## Theory

### Approach 1: Prompt-Based JSON (Naive)

Ask the LLM to return JSON in the prompt:

```python
import json

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Always respond in valid JSON. No markdown."},
        {"role": "user", "content": "Extract: title, director, year from: 'Inception by Nolan, 2010'"}
    ],
    temperature=0.0
)

data = json.loads(response.choices[0].message.content)
```

**Problems:** LLM might wrap in markdown, add commentary, misspell keys, or return invalid JSON.

### Approach 2: OpenAI JSON Mode (Better)

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    response_format={"type": "json_object"}  # ← Guarantees valid JSON!
)

data = json.loads(response.choices[0].message.content)  # Always works
```

**Guarantees valid JSON syntax** but NOT correct schema (keys might be wrong).

### Approach 3: Pydantic Validation (Production-grade)

```python
from pydantic import BaseModel, Field
from typing import Optional

class MovieReview(BaseModel):
    title: str
    director: str
    year: int = Field(ge=1888, le=2030)
    rating: float = Field(ge=0.0, le=10.0)
    genre: Optional[str] = None

def extract_movie(text: str) -> MovieReview:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""Extract movie info as JSON matching:
{json.dumps(MovieReview.model_json_schema(), indent=2)}"""},
            {"role": "user", "content": text}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    raw_json = json.loads(response.choices[0].message.content)
    return MovieReview(**raw_json)  # Pydantic validates!

movie = extract_movie("Inception by Nolan, 2010, 9.2/10, sci-fi")
print(movie.title)       # "Inception"
print(movie.model_dump())  # Full validated dict
```

### The Complete Flow

```
Text → LLM (json mode) → JSON string → json.loads() → dict → Pydantic → Typed object
```

### Robust Parser with Retry

```python
import re
from pydantic import ValidationError

def parse_llm_json(raw: str) -> dict:
    """Parse JSON from LLM output, handling markdown wrappers."""
    cleaned = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    cleaned = re.sub(r'\s*```$', '', cleaned)
    return json.loads(cleaned)

def extract_with_retry(text: str, model_class: type[BaseModel], max_retries: int = 2) -> BaseModel:
    schema = json.dumps(model_class.model_json_schema(), indent=2)
    
    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Extract data as JSON:\n{schema}"},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        try:
            raw_json = parse_llm_json(response.choices[0].message.content)
            return model_class(**raw_json)
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == max_retries:
                raise ValueError(f"Failed after {max_retries + 1} attempts: {e}")
```

---

## Bridge to LangChain

```python
# YOUR VERSION (manual)
schema = json.dumps(MovieReview.model_json_schema())
response = client.chat.completions.create(
    messages=[{"role": "system", "content": f"Return JSON: {schema}"}, ...],
    response_format={"type": "json_object"}
)
movie = MovieReview(**json.loads(response.choices[0].message.content))

# LANGCHAIN VERSION (one line!)
llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(MovieReview)
movie = structured_llm.invoke("Inception by Nolan, 2010, 9.2/10")
# Already a validated MovieReview object!
```

LangChain's `with_structured_output()` automates everything: schema generation, prompt injection, parsing, validation, and retry.

---

## Common Mistakes

### Mistake 1: Not using `response_format`
```python
# ❌ LLM might wrap JSON in markdown
response = client.chat.completions.create(messages=[...])

# ✅ Force JSON mode
response = client.chat.completions.create(messages=[...], response_format={"type": "json_object"})
```

### Mistake 2: Trusting LLM JSON without Pydantic
```python
# ❌ Keys might be wrong
data = json.loads(raw)
title = data["title"]  # KeyError if LLM used "movie_title"!

# ✅ Pydantic catches everything
movie = MovieReview(**data)
```

### Mistake 3: Not including schema in prompt
```python
# ❌ LLM guesses what keys to use
"Return movie info as JSON"

# ✅ Include exact schema
f"Return JSON matching:\n{json.dumps(MovieReview.model_json_schema())}"
```

---

## Interview Preparation

### Easy
**Q: Why can't you use LLM text output directly in code?**
**A:** LLM output is unstructured free-form text. Code needs structured data with guaranteed keys and types. Without parsing and validation, the code breaks on unexpected formats.

### Medium
**Q: What is the difference between `response_format=json_object` and Pydantic validation?**
**A:** `response_format` guarantees syntactically valid JSON (parseable by `json.loads()`), but doesn't validate the schema. Pydantic validates that data matches a specific schema with correct keys, types, and constraints.

### Hard
**Q: How does LangChain's `with_structured_output()` work internally?**
**A:** It generates a JSON schema from the Pydantic model, injects it into the prompt or uses function calling, sets response_format to JSON, parses with `json.loads()`, validates with Pydantic, and retries on failure.

---

## Summary

| Approach | JSON Valid? | Schema Valid? | Production? |
|----------|-------------|---------------|-------------|
| Prompt-based | ❌ Sometimes | ❌ No | ❌ No |
| `response_format` | ✅ Always | ❌ No | ⚠️ Partial |
| Pydantic + response_format | ✅ Always | ✅ Always | ✅ Yes |
| LangChain `with_structured_output` | ✅ Always | ✅ Always | ✅ Yes (easiest) |

---

## Phase 2 Complete! 🎉

You've mastered prompt engineering:

| Chapter | Topic | Status |
|---------|-------|--------|
| 2.1 | System / User / Assistant Messages | ✅ |
| 2.2 | Few-Shot Prompting & Chain-of-Thought | ✅ |
| 2.3 | Prompt Templates & Variables | ✅ |
| 2.4 | Output Parsing & Structured Output | ✅ |

## What's Next

**Phase 3: LangChain Core** — You finally start writing LangChain code! You'll install LangChain, make your first call through the LangChain interface, learn about the Runnable protocol and LCEL (LangChain Expression Language), and build your first chain. Everything from Phase 0-2 comes together here.

> [← Previous: Prompt Templates](chapter-12-prompt-templates.md) | [Next: Phase 3 — LangChain Installation & Hello World →](../phase-03-langchain-core/chapter-14-langchain-setup.md)

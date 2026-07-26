# Chapter 3.4: Structured Output with `with_structured_output()`

> **Phase 3 — LangChain Core** | [← Previous: ChatPromptTemplate](chapter-16-chat-prompt-template.md) | [Next: Phase 4 — Chains & Runnables →](../phase-04-chains-runnables/chapter-18-runnables-deep-dive.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Use `with_structured_output()` to get validated Pydantic objects from LLMs
- ✅ Understand the 3 methods under the hood (function calling, JSON mode, JSON schema)
- ✅ Handle nested models, enums, optionals, and lists
- ✅ Build extraction pipelines in LCEL chains
- ✅ Handle errors and implement retry logic
- ✅ Build real-world extractors (invoices, resumes, entities)

| | |
|---|---|
| **Prerequisites** | Chapter 0.4 (Pydantic), Chapter 2.4 (Output Parsing), Chapter 3.2-3.3 |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

### The Full Journey

This chapter is the payoff for everything you've learned:

```
Chapter 0.4 (Pydantic)     → BaseModel, Field, validation
Chapter 2.4 (Output Parse) → Manual JSON parsing + Pydantic validation
Chapter 3.2 (LCEL)         → Chaining components with |
Chapter 3.3 (Templates)    → ChatPromptTemplate

NOW: with_structured_output() does ALL of this in ONE LINE.
```

### The Problem

Without `with_structured_output()`, extracting structured data requires 5 manual steps: build schema → inject into prompt → call API with json mode → parse JSON → validate with Pydantic.

### The Solution

```python
structured_llm = llm.with_structured_output(Movie)
movie = structured_llm.invoke("Inception by Nolan, 2010, rated 9.2")
# Returns a validated Movie Pydantic object
```

---

## How It Works Internally

```
1. Reads Pydantic model's schema via model_json_schema()
2. Injects schema into the API call (function/tool calling or json_mode)
3. Parses the JSON response with json.loads()
4. Validates with Pydantic: Movie(**parsed_json)
5. Returns a typed Python object
```

---

## Code Examples

### Basic Usage

```python
from pydantic import BaseModel, Field
from typing import Optional

class Movie(BaseModel):
    """Information about a movie."""
    title: str = Field(description="The title of the movie")
    director: str = Field(description="The director's full name")
    year: int = Field(description="Release year", ge=1888, le=2030)
    rating: float = Field(description="Rating out of 10", ge=0, le=10)
    genre: Optional[str] = Field(default=None, description="Primary genre")

structured_llm = llm.with_structured_output(Movie)
movie = structured_llm.invoke("Inception, Christopher Nolan, 2010, 9.2/10, sci-fi")

print(movie.title)        # "Inception"
print(movie.year)         # 2010
print(movie.model_dump()) # Full dict
```

### Nested Models

```python
class Address(BaseModel):
    street: str = Field(description="Street name and number")
    city: str = Field(description="City name")
    country: str = Field(description="Country name")

class Person(BaseModel):
    name: str = Field(description="Full name")
    age: Optional[int] = Field(default=None, description="Age in years")
    address: Optional[Address] = Field(default=None, description="Physical address")

structured_llm = llm.with_structured_output(Person)
person = structured_llm.invoke("John Smith, 32, lives at 123 Main St, Springfield, USA")
print(person.address.city)  # "Springfield"
```

### Enums — Constraining Choices

```python
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class Category(str, Enum):
    HARDWARE = "hardware"
    SOFTWARE = "software"
    NETWORK = "network"
    ACCOUNT = "account"

class SupportTicket(BaseModel):
    summary: str = Field(description="One-line summary")
    category: Category = Field(description="Ticket category")
    priority: Priority = Field(description="Urgency level")

structured_llm = llm.with_structured_output(SupportTicket)
ticket = structured_llm.invoke("Email server down for 500 employees since 8 AM")
print(ticket.category.value)  # "software"
print(ticket.priority.value)  # "critical"
```

### Lists — Extracting Multiple Items

```python
class Entity(BaseModel):
    name: str = Field(description="Entity name")
    type: str = Field(description="Type: PERSON, ORGANIZATION, LOCATION, or DATE")

class EntityList(BaseModel):
    entities: list[Entity] = Field(description="All named entities found")

structured_llm = llm.with_structured_output(EntityList)
result = structured_llm.invoke("Tim Cook announced at Apple HQ in Cupertino on Jan 15, 2025...")
for e in result.entities:
    print(f"  {e.type}: {e.name}")
```

### In LCEL Chains

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a text analysis assistant. Analyze the given {content_type}."),
    ("human", "{text}")
])

chain = prompt | llm.with_structured_output(Summary)
result = chain.invoke({"content_type": "article", "text": "..."})
```

### Resume Parser (Complete Example)

```python
class Skill(BaseModel):
    name: str = Field(description="Skill name")
    level: str = Field(description="Level: beginner, intermediate, or advanced")

class Experience(BaseModel):
    company: str = Field(description="Company name")
    role: str = Field(description="Job title/role")
    years: float = Field(description="Duration in years")

class Resume(BaseModel):
    name: str = Field(description="Full name")
    email: Optional[str] = Field(default=None, description="Email address")
    skills: list[Skill] = Field(description="All skills")
    experience: list[Experience] = Field(description="Work history")
    total_years_experience: float = Field(description="Sum of all experience")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert resume parser."),
    ("human", "{text}")
])

chain = prompt | llm.with_structured_output(Resume)

result = chain.invoke({
    "text": "John Doe, john@email.com. 5 years at Google as Senior Engineer, "
            "2 years at startup as CTO. Expert in Python, intermediate in Rust."
})
```

---

## The `method` Parameter

| Method | How It Works | Reliability | Provider Support |
|--------|-------------|-------------|-----------------|
| `function_calling` | Uses tool/function calling API | ⭐⭐⭐ Highest | OpenAI, Anthropic |
| `json_mode` | Sets `response_format=json_object` | ⭐⭐ Good | OpenAI, some others |
| `json_schema` | Strict JSON schema enforcement | ⭐⭐⭐ Highest | OpenAI (newer models) |

Default is usually best. Only change if you have a specific reason.

---

## Advanced Features

### `include_raw=True`

```python
structured_llm = llm.with_structured_output(Movie, include_raw=True)
result = structured_llm.invoke("Inception, 2010")

print(result["raw"])            # Raw AIMessage
print(result["parsed"])         # Validated Movie object
print(result["parsing_error"])  # None if successful
```

### `.with_retry()`

```python
structured_llm = llm.with_structured_output(Movie).with_retry(
    stop_after_attempt=3
)
```

---

## Common Mistakes

### Mistake 1: Missing Field descriptions
```python
# ❌ LLM guesses what "num" means
class Bad(BaseModel):
    name: str
    num: int

# ✅ Descriptions guide the LLM
class Good(BaseModel):
    product_name: str = Field(description="Name of the product")
    quantity: int = Field(description="Number of items", ge=1)
```

### Mistake 2: Chaining StrOutputParser after structured output
```python
# ❌ StrOutputParser expects AIMessage, gets Pydantic object
chain = prompt | llm.with_structured_output(Movie) | StrOutputParser()

# ✅ Structured output IS the parser
chain = prompt | llm.with_structured_output(Movie)
```

### Mistake 3: High temperature for extraction
```python
# ❌ Inconsistent extractions
llm = ChatOpenAI(temperature=1.0)

# ✅ Deterministic
llm = ChatOpenAI(temperature=0.0)
```

### Mistake 4: Schema too complex
```python
# ❌ 30 fields, 5 levels deep — accuracy drops
# ✅ Break into multiple smaller extractions
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Add `Field(description=...)` to every field | Guides LLM extraction |
| Add docstring to the Pydantic class | Becomes function description |
| Use `temperature=0.0` | Deterministic results |
| Use `Enum` for constrained categories | Prevents hallucinated values |
| Keep schemas under ~15 fields | Accuracy drops with complexity |
| Use `Optional` for fields that might be missing | Prevents forced hallucination |
| Use `include_raw=True` for debugging | See raw LLM output |
| Add `ge`/`le` constraints on numerics | Validates ranges |

---

## Interview Preparation

### Easy
**Q: What does `with_structured_output()` do?**
**A:** Wraps an LLM to return validated Pydantic objects. It auto-generates the JSON schema, injects it into the API call, parses the response, and validates with Pydantic.

### Medium
**Q: Difference between `function_calling` and `json_mode`?**
**A:** `function_calling` provides the schema as a function definition via OpenAI's tool API — most reliable. `json_mode` sets `response_format=json_object` guaranteeing valid JSON but relies on the prompt for schema compliance.

### Hard
**Q: How to handle extraction of a complex schema?**
**A:** Split into smaller schemas. Extract high-level info first, then use results to inform subsequent extractions. Use `RunnableParallel` for independent parts, `RunnableSequence` for dependent ones.

### Senior
**Q: Production extraction pipeline with quality guarantees?**
**A:** (1) `include_raw=True` to capture all output. (2) `.with_retry()` for transient failures. (3) Cross-field validation beyond Pydantic. (4) Log all extractions. (5) `temperature=0`. (6) Confidence scoring. (7) Human review for low-confidence results.

---

## Summary

| Feature | What It Does |
|---------|-------------|
| `with_structured_output(Model)` | Returns validated Pydantic objects |
| `Field(description=...)` | Guides LLM on what to extract |
| Nested `BaseModel` | Handles complex hierarchical data |
| `Enum` fields | Constrains output to specific categories |
| `list[Model]` | Extracts multiple items |
| `include_raw=True` | Returns raw + parsed + error |
| `.with_retry()` | Auto-retry on failure |
| `method="function_calling"` | Uses tool calling API (default, most reliable) |

---

## Phase 3 (LangChain Core) Complete! 🎉

| Chapter | Topic | Status |
|---------|-------|--------|
| 3.1 | Installation & Hello World | ✅ |
| 3.2 | Runnable Protocol & LCEL | ✅ |
| 3.3 | ChatPromptTemplate Deep Dive | ✅ |
| 3.4 | Structured Output | ✅ |

## What's Next

In **Phase 4: Chains & Runnables**, we'll go deep on composing complex workflows — sequential chains, parallel branches, retry/fallback patterns, and building production-ready data pipelines.

> [← Previous: ChatPromptTemplate](chapter-16-chat-prompt-template.md) | [Next: Phase 4 — Chains & Runnables →](../phase-04-chains-runnables/chapter-18-runnables-deep-dive.md)

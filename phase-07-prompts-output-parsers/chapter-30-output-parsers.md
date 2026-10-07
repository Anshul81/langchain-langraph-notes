# Chapter 7.2: Output Parsers (StrOutputParser, JsonOutputParser, PydanticOutputParser)

> **Phase 7 — Prompts & Output Parsers** | [← Previous: ChatPromptTemplate](chapter-29-chat-prompt-template.md) | [Next: Structured Output →](chapter-31-structured-output.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Parse **`AIMessage`** objects into strings with **`StrOutputParser`**
- ✅ Extract JSON with **`JsonOutputParser`** and handle malformed output
- ✅ Enforce schemas with **`PydanticOutputParser`**
- ✅ Inject **format instructions** into prompts for reliable parsing
- ✅ Chain parsers in LCEL as the last step of a pipeline

| | |
|---|---|
| **Prerequisites** | Chapter 7.1 (ChatPromptTemplate), Phase 0.4 (Pydantic) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

### The Problem

LLMs return **text** (wrapped in `AIMessage`). Your application needs **types**:

```
LLM output:  "The capital is Paris."
App needs:   str  OR  {"capital": "Paris"}  OR  CapitalAnswer(city="Paris")
```

Without parsers, every route repeats `.content`, manual `json.loads()`, and brittle regex.

### The Solution

**Output parsers** are Runnables that transform model output into structured data — and can **tell the model** how to format its answer.

```
prompt (+ format instructions) ──► llm ──► OutputParser ──► Python types
                                              │
                         StrOutputParser ─────┼──► str
                         JsonOutputParser ───┼──► dict
                         PydanticOutputParser ┼──► BaseModel instance
```

---

## Part 1: `StrOutputParser` — The Default Finisher

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer briefly."),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()
print(chain.invoke({"question": "What is an output parser?"}))
print(type(chain.invoke({"question": "Hi"})))  # str
```

**Rule:** If downstream code expects a string, always end with `StrOutputParser()` — not raw `AIMessage`.

---

## Part 2: `JsonOutputParser` — Dict Outputs

```python
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Return JSON with keys: title (string), rating (float 0-10). "
        "No markdown fences.\n{format_instructions}",
    ),
    ("human", "Review the movie: {movie}"),
])

chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)

result = chain.invoke({"movie": "Inception"})
print(result)       # dict
print(result["title"])
```

### Handling failures

Models sometimes wrap JSON in markdown:

````text
```json
{"title": "Inception", "rating": 9.0}
```
````

Mitigations:

1. Prompt: *"Return raw JSON only, no code fences."*
2. Use **`with_structured_output()`** (Chapter 7.3) when the provider supports schema enforcement
3. Pre-process with `RunnableLambda` to strip fences before `json.loads`

---

## Part 3: `PydanticOutputParser` — Validated Models

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(ge=0, le=10, description="Score out of 10")
    summary: str = Field(description="One-sentence summary")

parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a film critic.\n{format_instructions}"),
    ("human", "Review: {movie}"),
])

chain = (
    prompt.partial(format_instructions=parser.get_format_instructions())
    | llm
    | parser
)

review = chain.invoke({"movie": "The Matrix"})
print(type(review))   # MovieReview
print(review.rating)
```

Pydantic validates types — invalid `rating: 99` raises a clear error at parse time.

---

## Part 4: Format Instructions in the Prompt

Parsers expose **`get_format_instructions()`** — text describing the expected schema. Always inject via `partial()` or a template variable:

```python
format_instructions = parser.get_format_instructions()
print(format_instructions[:200])  # schema hint for the model
```

```
┌─────────────────────────────────────────┐
│ System: ... {format_instructions}       │
│ Human: Review: Inception                │
└─────────────────────────────────────────┘
                    │
                    ▼
Model sees field names, types, and JSON shape → higher parse success rate
```

---

## Part 5: Chaining Parsers and Lambdas

```python
from langchain_core.runnables import RunnableLambda

def normalize_title(review: MovieReview) -> MovieReview:
    review.title = review.title.strip().title()
    return review

chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser | RunnableLambda(normalize_title)
```

Parsers output Python objects — lambdas can post-process before returning to your API.

---

## Part 6: Parser Comparison

| Parser | Output | Validation | Best when |
|--------|--------|------------|-----------|
| `StrOutputParser` | `str` | None | Chat, summaries |
| `JsonOutputParser` | `dict` | JSON syntax only | Flexible keys |
| `PydanticOutputParser` | `BaseModel` | Types + constraints | API contracts |
| `with_structured_output()` | `BaseModel` / schema | Provider-side | Production (Ch 7.3) |

---

## Common Mistakes

### Mistake 1: Forgetting the parser in LCEL

```python
# ❌ Returns AIMessage
chain = prompt | llm

# ✅ Returns str
chain = prompt | llm | StrOutputParser()
```

### Mistake 2: Omitting format instructions for JSON/Pydantic

Without `{format_instructions}`, parse failure rates spike.

### Mistake 3: Asking for JSON but parsing with StrOutputParser

Downstream services expect `dict` — wire the correct parser.

### Mistake 4: Overly complex schemas in one shot

Split extraction into two steps or use structured output API for nested models.

---

## Best Practices

| Practice | Why |
|----------|-----|
| End chains with an explicit parser | Stable types at boundaries |
| `temperature=0` for structured extraction | Fewer format drifts |
| Keep schemas small | Easier for the model to comply |
| Log raw `AIMessage` on parse failure | Debug prompts quickly |
| Prefer native structured output in prod | Less brittle than prompt-only JSON |

---

## Interview Preparation

### Easy
**Q: What does `StrOutputParser` do?**

> Extracts the string `.content` from an `AIMessage` (or chunk sequence) so LCEL chains return plain `str` to callers.

### Medium
**Q: Why use `PydanticOutputParser` over `JsonOutputParser`?**

> JSON parser only checks valid JSON. Pydantic parser validates **field types and constraints** (`ge`, `le`, required fields) and returns a typed model for IDE support and API serialization.

### Hard
**Q: Where should format instructions live in the prompt?**

> Typically in the **system** message via `prompt.partial(format_instructions=...)`, so user content stays clean and the schema stays stable across turns.

### Senior
**Q: Your JSON parser fails 15% of the time in production. What do you do?**

> Layer defenses: (1) switch to provider **`with_structured_output()`** or tool calling; (2) add automatic **retry** with repair prompt showing parse error; (3) reduce schema complexity; (4) monitor failure payloads; (5) fallback to human review queue for critical fields — never silently drop bad parses in finance/health domains.

---

## Summary

| Parser | Returns |
|--------|---------|
| `StrOutputParser` | `str` |
| `JsonOutputParser` | `dict` |
| `PydanticOutputParser` | Pydantic model |
| `get_format_instructions()` | Prompt text for schema |

---

## Hands-on Exercise

Build a chain that extracts `{"company": str, "ticker": str, "sentiment": "bullish"|"bearish"|"neutral"}` from a news blurb. Use `PydanticOutputParser` and print the model.

---

## Challenge Project

Create an **invoice extractor**: given OCR text, return Pydantic model with `vendor`, `date`, `line_items: list[{description, amount}]`, `total`. Handle one sample where the model omits a field — catch validation error and retry once with an error-aware prompt.

---

## What's Next

Chapter 7.3 covers **`with_structured_output()`** — provider-native schema enforcement that often replaces fragile JSON-in-prompt parsing.

> [← Previous: ChatPromptTemplate](chapter-29-chat-prompt-template.md) | [Next: Structured Output →](chapter-31-structured-output.md)

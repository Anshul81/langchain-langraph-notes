# Chapter 0.4: Type Hints & Pydantic — Data Validation for AI Applications

> **Phase 0 — Python Power-Up** | [← Previous: Generators](chapter-02-generators.md) | [Next: Async Python →](chapter-04-async-python.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand Python type hints and **WHY** they matter
- ✅ Use type hints on functions, variables, and collections
- ✅ Use Pydantic `BaseModel` for runtime data validation
- ✅ Use `Field()` for constraints, defaults, and descriptions
- ✅ Build nested models and convert to dictionaries
- ✅ Know why LangChain depends on Pydantic for **everything**

| | |
|---|---|
| **Prerequisites** | Python classes (basic), dictionaries |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

### The Problem

```python
# What does this function expect? What does it return? 🤷
def process(data, config, retries):
    pass

# Somebody calls it wrong — no error until RUNTIME (or worse, silent bug)
process("hello", 42, "three")  # Is this correct? Who knows!
```

Without type hints, you're **guessing** what every function expects. Bugs hide. Code is unreadable. Collaboration is painful.

And even with type hints, Python **doesn't enforce them at runtime**. You need something more — you need **Pydantic**.

### The Solution

- **Type hints** = labels that document what your code expects (checked by IDEs)
- **Pydantic** = runtime validation that enforces those types and catches bad data instantly

### History

Type hints were introduced in Python 3.5 via [PEP 484](https://peps.python.org/pep-0484/). The `typing` module provided `List`, `Dict`, `Optional`, etc. Python 3.10 simplified the syntax with `list[str]` and `str | None`. Pydantic was created by Samuel Colvin in 2017 and has become the de-facto data validation library in Python. Pydantic v2 (2023) was rewritten in Rust for massive performance gains.

### Industry Usage

- **LangChain** — Every tool, chain input, structured output uses Pydantic
- **FastAPI** — Request/response validation is 100% Pydantic
- **OpenAI SDK** — Uses Pydantic for structured outputs
- **Every major Python framework** — Django REST Framework, SQLModel, etc.

### Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| Type hints enforce types at runtime | No — Python ignores them. Only tools like mypy and Pydantic check them |
| Pydantic is only for web APIs | Pydantic is for ANY data validation — configs, LLM outputs, tool inputs |
| You need to validate data manually | Pydantic does it automatically when you create an instance |
| Type hints slow down Python | No — they have zero performance impact at runtime |

---

## Mental Model

### Type Hints = Labels on Boxes

Imagine you're packing boxes for a move:

**Without labels (no type hints):**
> 📦 📦 📦 — Which box has the kitchen stuff? Which has books? Open each one to find out.

**With labels (type hints):**
> 📦 Kitchen 📦 Books 📦 Clothes — Instantly know what's inside without opening.

Type hints **label** your code so everyone (including your future self and your IDE) knows what goes where.

### Pydantic = Security Guard at the Door

> Type hints are **suggestions** on the door sign: "Employees Only."  
> Pydantic is the **security guard** who actually checks your badge and turns you away if you don't belong.

```
                    Plain Python          Pydantic
                    ┌────────────┐     ┌────────────────┐
Input data    ──→   │ No checking │     │ Validates type  │
                    │ No defaults │     │ Applies defaults│
{"name": 123}       │ No errors   │     │ Coerces types   │
                    │ Silent bugs │     │ Clear errors    │
                    └────────────┘     └────────────────┘
                         │                    │
                         ▼                    ▼
                    name = 123           ValidationError!
                    (wrong type,         "name must be str"
                     no error 😱)        (caught instantly ✅)
```

---

## Theory

### Part 1: Type Hints Basics

#### Simple Type Hints

```python
# Variables
name: str = "LangChain"
version: float = 0.3
is_active: bool = True
token_count: int = 1500

# Functions
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def process(data: str, verbose: bool = False) -> None:
    if verbose:
        print(data)
```

#### What does `-> str` mean?

```python
def greet(name: str) -> str:
#         ──────┬───    ──┬──
#               │         │
#         Parameter       Return type
#         type hint       hint
```

#### Collection Type Hints

```python
from typing import List, Dict, Tuple, Optional, Set

# List of strings
names: List[str] = ["Alice", "Bob"]

# Dictionary with string keys and int values
scores: Dict[str, int] = {"Alice": 95, "Bob": 87}

# Tuple of specific types
coordinate: Tuple[float, float] = (3.14, 2.71)

# Optional — can be the type OR None
nickname: Optional[str] = None   # Same as str | None

# Set of integers
unique_ids: Set[int] = {1, 2, 3}
```

#### Python 3.10+ Simplified Syntax

```python
# Old way (requires imports)
from typing import List, Optional
def process(items: List[str], name: Optional[str] = None) -> None:
    pass

# New way (Python 3.10+, no imports needed!)
def process(items: list[str], name: str | None = None) -> None:
    pass
```

#### Type Hints DON'T Enforce Anything!

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"

greet(42)  # ⚠️ No error! Python IGNORES type hints at runtime!
```

Type hints are just **labels** — Python doesn't check them. They help:
- **Your IDE** (PyCharm!) catch bugs with red squiggly lines
- **Other developers** understand your code
- **Tools like mypy** do static analysis
- **Pydantic** enforce them at runtime

---

### Part 2: Pydantic — Type Hints That Actually Enforce

#### The Problem Type Hints Don't Solve

```python
# Type hints are ignored at runtime
def create_user(name: str, age: int) -> dict:
    return {"name": name, "age": age}

create_user(123, "twenty")  # No error! But data is garbage.
```

#### Enter Pydantic

Pydantic **validates data at runtime**. If the data is wrong, it raises a clear error immediately.

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# ✅ Valid data — works fine
user = User(name="Alice", age=30, email="alice@example.com")
print(user.name)   # "Alice"
print(user.age)    # 30

# ❌ Invalid data — IMMEDIATE error!
user = User(name="Alice", age="thirty", email="alice@example.com")
# ValidationError: 1 validation error for User
# age
#   Input should be a valid integer, unable to parse string as an integer
```

#### Why Does This Matter?

```
Without Pydantic:
    Bad data → Silently passes → Bug appears 500 lines later → 2 hours debugging

With Pydantic:
    Bad data → IMMEDIATE error with clear message → Fix in 30 seconds
```

#### Pydantic Auto-Converts When Possible

```python
class Config(BaseModel):
    temperature: float
    max_tokens: int

# Pydantic automatically converts "0.7" string → 0.7 float
config = Config(temperature="0.7", max_tokens="500")
print(config.temperature)  # 0.7 (float, not string!)
print(config.max_tokens)   # 500 (int, not string!)
```

This is called **coercion** — Pydantic tries to convert data to the right type when it safely can.

---

### Part 3: Pydantic Features You'll Use in LangChain

#### Default Values

```python
class LLMConfig(BaseModel):
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500
    streaming: bool = False

# Use all defaults
config = LLMConfig()
print(config.model)  # "gpt-3.5-turbo"

# Override some defaults
config = LLMConfig(model="gpt-4", temperature=0.2)
print(config.model)  # "gpt-4"
print(config.max_tokens)  # 500 (default)
```

#### Optional Fields

```python
from typing import Optional

class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None  # Can be a string or None

msg1 = Message(role="user", content="Hello")
print(msg1.name)  # None

msg2 = Message(role="assistant", content="Hi!", name="ChatBot")
print(msg2.name)  # "ChatBot"
```

**Important rule:** If a field can be `None`, you MUST declare it as `Optional[type]` (or `type | None`). Setting `field: int = None` without `Optional` will cause a Pydantic validation error.

#### Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str = "India"

class Company(BaseModel):
    name: str
    address: Address  # Nested Pydantic model!

# Pydantic validates the nested structure too
company = Company(
    name="AI Startup",
    address={"street": "123 Tech St", "city": "Bangalore"}
)
print(company.address.city)     # "Bangalore"
print(company.address.country)  # "India" (default)
```

Notice how you can pass a plain dictionary for the nested model — Pydantic automatically converts it to an `Address` object and validates it.

#### Converting to Dictionary and JSON

```python
class User(BaseModel):
    name: str
    age: int

user = User(name="Alice", age=30)

# Convert to dict — used EVERYWHERE in LangChain
user_dict = user.model_dump()
print(user_dict)  # {'name': 'Alice', 'age': 30}

# Convert to JSON string
user_json = user.model_dump_json()
print(user_json)  # '{"name":"Alice","age":30}'
```

#### Field Validation with `Field()`

```python
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    model: str = Field(description="The model name to use")
    temperature: float = Field(
        default=0.7, 
        ge=0.0,    # greater than or equal to 0
        le=2.0,    # less than or equal to 2
        description="Controls randomness"
    )
    max_tokens: int = Field(
        default=500,
        gt=0,      # greater than 0
        description="Maximum tokens to generate"
    )

# ✅ Valid
config = LLMConfig(model="gpt-4", temperature=0.5)

# ❌ Invalid — temperature > 2.0
config = LLMConfig(model="gpt-4", temperature=5.0)
# ValidationError: 1 validation error for LLMConfig
# temperature
#   Input should be less than or equal to 2.0 [type=less_than_equal]
```

**`Field()` constraint options:**

| Constraint | Meaning | Example |
|-----------|---------|---------|
| `gt` | Greater than | `Field(gt=0)` → must be > 0 |
| `ge` | Greater than or equal | `Field(ge=0)` → must be ≥ 0 |
| `lt` | Less than | `Field(lt=100)` → must be < 100 |
| `le` | Less than or equal | `Field(le=2.0)` → must be ≤ 2.0 |
| `min_length` | Minimum string length | `Field(min_length=1)` |
| `max_length` | Maximum string length | `Field(max_length=100)` |
| `pattern` | Regex pattern | `Field(pattern=r'^[a-z]+$')` |
| `description` | Documentation | `Field(description="...")` |

---

## Architecture

### How Pydantic Works Internally

```
┌───────────────────────────────────────────────────────┐
│                  Pydantic Validation Flow              │
├───────────────────────────────────────────────────────┤
│                                                       │
│  Input Data (dict, kwargs, JSON)                      │
│         │                                             │
│         ▼                                             │
│  ┌─────────────────┐                                  │
│  │ Type Checking    │ ← Does the type match?          │
│  └────────┬────────┘                                  │
│           ▼                                           │
│  ┌─────────────────┐                                  │
│  │ Type Coercion    │ ← Can it be safely converted?   │
│  └────────┬────────┘   ("0.7" → 0.7 ✅)              │
│           ▼                                           │
│  ┌─────────────────┐                                  │
│  │ Constraint Check │ ← Does it pass ge, le, etc.?    │
│  └────────┬────────┘                                  │
│           ▼                                           │
│  ┌─────────────────┐                                  │
│  │ Default Values   │ ← Apply defaults for missing    │
│  └────────┬────────┘                                  │
│           ▼                                           │
│  ┌─────────────────┐                                  │
│  │ Nested Validation│ ← Validate nested models        │
│  └────────┬────────┘                                  │
│           ▼                                           │
│  Valid Model Instance  OR  ValidationError             │
│                                                       │
└───────────────────────────────────────────────────────┘
```

---

## Real Industry Example

### How LangChain Uses Pydantic

#### 1. Every Tool Definition

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Max results to return")

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 5) -> str:
    """Search the web."""
    return f"Results for: {query}"
```

#### 2. Structured Output from LLMs

```python
class MovieReview(BaseModel):
    title: str
    rating: float = Field(ge=0, le=10)
    summary: str

# LangChain forces the LLM to output data matching this schema
llm_with_structure = llm.with_structured_output(MovieReview)
result = llm_with_structure.invoke("Review the movie Inception")
print(result.title)   # "Inception"
print(result.rating)  # 9.2
```

#### 3. Chain Inputs/Outputs

```python
class ChainInput(BaseModel):
    question: str
    context: str

class ChainOutput(BaseModel):
    answer: str
    confidence: float
```

**If you don't understand Pydantic, you cannot build LangChain applications.** It's not optional — it's the foundation.

---

## Common Mistakes

### Mistake 1: Using `dict` instead of Pydantic

```python
# ❌ No validation, no autocomplete, error-prone
config = {"model": "gpt-4", "temprature": 0.7}  # Typo! No error.

# ✅ Pydantic catches typos immediately
class Config(BaseModel):
    model: str
    temperature: float

config = Config(model="gpt-4", temprature=0.7)
# TypeError: unexpected keyword argument 'temprature'
```

### Mistake 2: Forgetting `BaseModel` inheritance

```python
# ❌ This is just a regular class — no validation
class User:
    name: str
    age: int

# ✅ Must inherit from BaseModel
class User(BaseModel):
    name: str
    age: int
```

### Mistake 3: `str` instead of `list[str]`

```python
# ❌ Single string, not a list
messages: str

# ✅ A list of strings
messages: list[str]
```

### Mistake 4: `int = None` without Optional

```python
# ❌ Pydantic error — int can't be None
max_tokens: int = None

# ✅ Optional means "int OR None"
from typing import Optional
max_tokens: Optional[int] = None

# ✅ Or Python 3.10+ syntax
max_tokens: int | None = None
```

### Mistake 5: Using Pydantic v1 syntax in v2

```python
# Pydantic v1 (old — you'll see this in older LangChain code)
user.dict()
user.json()
user.schema()

# Pydantic v2 (current — use this!)
user.model_dump()
user.model_dump_json()
user.model_json_schema()
```

---

## Debugging Guide

### Error: `ValidationError: X validation errors for Model`

**Cause:** One or more fields have invalid data.

**Fix:** Read the error message — it tells you exactly which field failed and why.

### Error: `TypeError: unexpected keyword argument`

**Cause:** You passed a field name that doesn't exist in the model (often a typo).

**Fix:** Check the model definition for the correct field names.

### Error: `Input should be a valid integer, unable to parse string as an integer`

**Cause:** You passed a string that can't be converted to an integer (e.g., `"hello"` for an `int` field).

**Fix:** Pass the correct type, or a string that can be parsed (e.g., `"42"`).

### Error: `Field required`

**Cause:** You didn't provide a value for a field that has no default.

**Fix:** Either provide the value or add a default: `field: str = "default"`.

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Always use type hints on function signatures | Self-documenting, IDE autocomplete and error detection |
| Use Pydantic for all external data (API inputs, configs, LLM outputs) | Runtime validation catches errors instantly |
| Use `Field()` with descriptions | Serves as documentation AND validation |
| Use `Optional[X]` with default `None` for optional fields | Explicit about what can be None |
| Use `model_dump()` not `dict()` | Pydantic v2 standard |
| Use nested models for complex structures | Validates all levels deeply |
| Use Python 3.10+ syntax when possible | `list[str]` and `str | None` — cleaner, no imports needed |

---

## Interview Preparation

### Easy

**Q: What are type hints in Python? Do they enforce types at runtime?**

**A:** Type hints annotate variables and functions with expected types using syntax like `name: str` and `-> int`. No — Python ignores them at runtime. They're used by IDEs for autocomplete and error detection, and by tools like mypy for static analysis.

### Medium

**Q: What is Pydantic and how is it different from regular type hints?**

**A:** Pydantic validates data at **runtime**. Unlike type hints (which are ignored by Python), Pydantic raises `ValidationError` immediately if data doesn't match the schema. It also supports default values, automatic type coercion, field constraints, and nested model validation.

### Hard

**Q: How does LangChain use Pydantic for structured output from LLMs?**

**A:** LangChain's `with_structured_output()` takes a Pydantic model and instructs the LLM to return data matching that schema. Internally, LangChain generates a JSON schema from the Pydantic model using `model_json_schema()`, sends it to the LLM's function-calling/tool API, parses the LLM's JSON response, and validates it against the Pydantic model.

### Senior

**Q: What is the difference between Pydantic v1 and v2? Why does it matter for LangChain?**

**A:** Pydantic v2 was rewritten with a Rust core for ~5-50x performance improvement. API changes include `model_dump()` replacing `.dict()`, `model_dump_json()` replacing `.json()`, and `model_json_schema()` replacing `.schema()`. LangChain migrated from v1 to v2, so older tutorials and code may use v1 syntax. Always use v2 syntax in new code.

---

## Summary

| Concept | What It Does |
|---------|--------------|
| `name: str` | Type hint — labels the expected type (not enforced) |
| `-> str` | Return type hint |
| `Optional[str]` or `str \| None` | Can be the type or None |
| `list[str]` | List containing strings |
| `BaseModel` | Pydantic base class — enables runtime validation |
| `Field()` | Adds constraints (`ge`, `le`, `gt`, `lt`) and descriptions |
| `model_dump()` | Converts Pydantic model to dictionary |
| `model_dump_json()` | Converts Pydantic model to JSON string |
| `ValidationError` | Raised when data doesn't match the schema |

---

## Cheat Sheet

```python
# TYPE HINTS
def func(name: str, age: int, active: bool = True) -> str:
    return f"{name} is {age}"

# COLLECTIONS
items: list[str] = ["a", "b"]
scores: dict[str, int] = {"a": 1}
point: tuple[float, float] = (1.0, 2.0)
maybe: str | None = None             # Python 3.10+
maybe: Optional[str] = None          # Older syntax

# PYDANTIC MODEL
from pydantic import BaseModel, Field

class MyModel(BaseModel):
    name: str
    count: int = 0
    score: float = Field(default=0.0, ge=0, le=100)
    tags: list[str] = []
    extra: str | None = None

# USAGE
obj = MyModel(name="test", count=5)
obj.model_dump()          # {'name': 'test', 'count': 5, ...}
obj.model_dump_json()     # '{"name":"test","count":5,...}'

# NESTED MODELS
class Inner(BaseModel):
    value: int

class Outer(BaseModel):
    inner: Inner

obj = Outer(inner={"value": 42})  # Auto-converts dict → Inner
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| Do type hints enforce types at runtime? | No — Python ignores them. Use Pydantic for enforcement |
| What does `Optional[str]` mean? | The value can be `str` or `None` |
| What does Pydantic do? | Validates data at runtime using type hints |
| What happens with invalid data in Pydantic? | Raises `ValidationError` with clear message |
| What is type coercion? | Pydantic auto-converts compatible types (e.g., `"42"` → `42`) |
| How to convert Pydantic model to dict? | `model.model_dump()` |
| How to add constraints in Pydantic? | Use `Field(ge=0, le=100)` |
| What must you inherit from? | `pydantic.BaseModel` |
| `list[str]` vs `str`? | `list[str]` = list of strings, `str` = single string |

---

## Hands-on Exercise

### Exercise 1: LLM Request Model

Create a Pydantic model `LLMRequest` with:
- `model`: str, default `"gpt-3.5-turbo"`
- `messages`: list of strings (required, no default)
- `temperature`: float, default 0.7, must be between 0.0 and 2.0
- `max_tokens`: optional int, default None

```python
from pydantic import BaseModel, Field
from typing import Optional

class LLMRequest(BaseModel):
    model: str = "gpt-3.5-turbo"
    messages: list[str]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = None

# Test
req = LLMRequest(messages=["Hello", "How are you?"], model="gpt-4")
print(req.model_dump())
```

### Exercise 2: Nested Chat Message Model

Build two models:
- `ChatMessage` with fields: `role` (str), `content` (str)
- `ChatRequest` with fields: `messages` (list of ChatMessage), `model` (str), `temperature` (float with bounds)

---

## Challenge Project

Build a configuration system:
1. Create a `DatabaseConfig` model (host, port, database name, credentials)
2. Create an `LLMConfig` model (model, temperature, max_tokens)
3. Create an `AppConfig` that nests both
4. Load config from a dictionary and validate it
5. Show what happens with invalid data

---

## Homework

1. **Reading:** Read [Pydantic v2 docs — Models](https://docs.pydantic.dev/latest/concepts/models/)
2. **Coding:** Complete exercises 1 and 2
3. **Experiment:** Try passing invalid data to your models and read the error messages
4. **Debugging:** Fix this code:
    ```python
    class Config(BaseModel):
        name: str
        count: int = None    # Bug 1
        items: str            # Bug 2 — should be list
    
    c = Config(name="test", items="hello")
    ```

---

## Additional Resources

- [Pydantic Official Docs](https://docs.pydantic.dev/latest/)
- [PEP 484 — Type Hints](https://peps.python.org/pep-0484/)
- [Python typing module docs](https://docs.python.org/3/library/typing.html)
- [Real Python — Python Type Checking](https://realpython.com/python-type-checking/)
- [mypy — Static Type Checker](https://mypy-lang.org/)

---

## What's Next

In the next chapter, we'll learn about **Async Python** — `async`/`await`, `asyncio`, and concurrent programming. Async is critical for AI applications because LLM API calls take seconds. Without async, your application blocks and waits for each API call to finish before starting the next one. With async, you can fire off multiple LLM calls simultaneously, drastically reducing total latency. LangChain provides async versions of nearly every method (`ainvoke`, `astream`, `abatch`).

> [← Previous: Generators](chapter-02-generators.md) | [Next: Async Python →](chapter-04-async-python.md)

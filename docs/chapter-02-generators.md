# Chapter 0.3: Generators & Iterators — Lazy Data Production

> **Phase 0 — Python Power-Up** | [← Previous: Context Managers](chapter-01-context-managers.md) | [Next: Type Hints & Pydantic →](chapter-03-type-hints-pydantic.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand what generators are and **WHY** they exist
- ✅ Know the difference between a list and a generator
- ✅ Use `yield` to create generator functions
- ✅ Understand lazy evaluation and memory efficiency
- ✅ Know why LangChain uses generators for **streaming LLM responses**

| | |
|---|---|
| **Prerequisites** | Python functions, loops, lists |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 40 minutes |

---

## Introduction

### The Problem

Imagine you need to process 10 million lines from a log file:

```python
# ❌ Loads ALL 10 million lines into memory at once
lines = open("huge_log.txt").readlines()  # 💥 Uses 5GB of RAM!
for line in lines:
    process(line)
```

Or imagine you're getting tokens from an LLM — you want to show them **as they arrive**, not wait for the entire response:

```python
# ❌ User stares at blank screen for 10 seconds
response = llm.generate("Write an essay")  # Waits for complete response
print(response)

# ✅ User sees words appearing one by one (streaming!)
for token in llm.stream("Write an essay"):  # Gets tokens as they arrive
    print(token, end="")
```

**The core problem:** Sometimes you can't (or shouldn't) load everything into memory at once. You need data **one piece at a time**.

### The Solution

Python **generators** produce values lazily — one at a time, only when asked. They use almost zero memory regardless of data size.

### History

Generators were introduced in Python 2.2 via [PEP 255](https://peps.python.org/pep-0255/). Generator expressions (one-liner syntax) were added in Python 2.4 via [PEP 289](https://peps.python.org/pep-0289/). The `yield from` syntax was added in Python 3.3 via [PEP 380](https://peps.python.org/pep-0380/).

### Industry Usage

- **LangChain** — Streaming LLM responses token by token
- **LangGraph** — Streaming state updates during graph execution
- **Web frameworks** — Streaming HTTP responses (FastAPI `StreamingResponse`)
- **Data processing** — Processing large files, databases, API pagination
- **ETL pipelines** — Extract-Transform-Load without memory overflow

### Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| Generators are just fancy lists | Generators don't store data — they compute on demand |
| You can iterate over a generator multiple times | Generators are **single-use** — once exhausted, they're empty |
| `yield` is the same as `return` | `return` ends the function; `yield` pauses it and resumes later |
| Generators are always better than lists | Use lists when you need random access, length, or re-iteration |

---

## Mental Model

### The Vending Machine vs The Warehouse

**List = Warehouse**
> You buy ALL 1,000 drinks at once, stack them in your house, then drink one at a time. Your house is full. Expensive. Wasteful.

**Generator = Vending Machine**
> You walk up, press the button, get ONE drink, drink it, then get the next one when you're ready. The machine only produces one at a time. Cheap. Efficient.

```
List (Warehouse):          Generator (Vending Machine):
┌────────────────────┐     ┌──────────┐
│ [1, 2, 3, 4, 5,   │     │ Press    │──→ 1
│  6, 7, 8, 9, 10,  │     │ button   │──→ 2
│  ... 999, 1000]    │     │ when     │──→ 3
│                    │     │ ready    │──→ ...
│ ALL in memory      │     │          │
│ at once!           │     │ ONE at   │
└────────────────────┘     │ a time!  │
     5GB RAM                └──────────┘
                               ~0 RAM
```

---

## Theory

### Part 1: Regular Function vs Generator Function

The **only difference** is `return` vs `yield`:

```python
# Regular function — returns EVERYTHING at once
def get_numbers_list(n):
    result = []
    for i in range(n):
        result.append(i)
    return result  # Returns a complete list

# Generator function — yields ONE item at a time
def get_numbers_gen(n):
    for i in range(n):
        yield i  # Produces one value, then PAUSES
```

```python
# Using them
numbers_list = get_numbers_list(5)  # [0, 1, 2, 3, 4] — all in memory NOW
numbers_gen = get_numbers_gen(5)    # <generator object> — nothing computed yet!
```

### Part 2: How `yield` Works

`yield` is like `return` but with a **pause button**:

```python
def count_to_3():
    print("About to yield 1")
    yield 1                        # Produces 1, PAUSES here
    print("About to yield 2")
    yield 2                        # Produces 2, PAUSES here
    print("About to yield 3")
    yield 3                        # Produces 3, PAUSES here
    print("Done!")
```

```python
gen = count_to_3()

print(next(gen))  # "About to yield 1" → 1
print(next(gen))  # "About to yield 2" → 2
print(next(gen))  # "About to yield 3" → 3
print(next(gen))  # "Done!" → 💥 StopIteration error!
```

**Key insight:** The function **remembers where it paused** and continues from there on the next call.

### Part 3: `yield` Execution Flow

```
gen = count_to_3()          # Nothing executes yet!

next(gen)                   # Runs until first yield
    │ print("About to yield 1")
    │ yield 1 ──────────────── PAUSE ──→ returns 1
    │
next(gen)                   # Resumes from pause
    │ print("About to yield 2")
    │ yield 2 ──────────────── PAUSE ──→ returns 2
    │
next(gen)                   # Resumes from pause
    │ print("About to yield 3")
    │ yield 3 ──────────────── PAUSE ──→ returns 3
    │
next(gen)                   # Resumes from pause
    │ print("Done!")
    │ Function ends ────────── StopIteration raised
```

### Part 4: Using Generators in `for` Loops

You almost never call `next()` directly. Use a `for` loop — it handles `StopIteration` automatically:

```python
def count_to(n):
    for i in range(1, n + 1):
        yield i

# for loop calls next() internally and stops at StopIteration
for number in count_to(5):
    print(number)
# 1, 2, 3, 4, 5
```

### Part 5: Memory Comparison

```python
import sys

# List — stores everything
numbers_list = [i for i in range(1_000_000)]
print(sys.getsizeof(numbers_list))  # ~8,448,728 bytes (8MB!)

# Generator — stores almost nothing
numbers_gen = (i for i in range(1_000_000))
print(sys.getsizeof(numbers_gen))   # ~200 bytes (always!)
```

**8 MB vs 200 bytes** — the generator uses **42,000x less memory!**

### Part 6: Generator Expressions (One-Liners)

Just like list comprehensions, but with `()` instead of `[]`:

```python
# List comprehension — creates entire list in memory
squares_list = [x**2 for x in range(1000)]   # list

# Generator expression — creates values lazily
squares_gen = (x**2 for x in range(1000))    # generator
```

```python
# Both work in for loops
for sq in squares_gen:
    print(sq)

# Generator expressions work inside functions too
total = sum(x**2 for x in range(1000))  # No extra brackets needed!
```

### Part 7: `yield from` — Delegating to Sub-Generators

When a generator needs to yield all values from another iterable:

```python
# Without yield from (manual)
def combined():
    for item in range(3):
        yield item
    for item in ['a', 'b', 'c']:
        yield item

# With yield from (cleaner)
def combined():
    yield from range(3)
    yield from ['a', 'b', 'c']

list(combined())  # [0, 1, 2, 'a', 'b', 'c']
```

---

## Architecture

### Iterator Protocol

Generators implement Python's **iterator protocol**:

```
┌─────────────────────────────────────────────┐
│           Iterator Protocol                  │
├─────────────────────────────────────────────┤
│                                             │
│  __iter__(self)                             │
│  └── Returns the iterator itself            │
│                                             │
│  __next__(self)                             │
│  ├── Returns the next value                 │
│  └── Raises StopIteration when done         │
│                                             │
│  Generators implement BOTH automatically!   │
│                                             │
└─────────────────────────────────────────────┘

   for item in generator:    ← calls __iter__() then __next__() repeatedly
       process(item)         ← until StopIteration
```

### How `for` Loops Work Internally

```python
# This:
for item in gen:
    process(item)

# Is equivalent to:
iterator = iter(gen)       # calls __iter__()
while True:
    try:
        item = next(iterator)  # calls __next__()
        process(item)
    except StopIteration:
        break
```

---

## Practical Example: Simulating LLM Streaming

```python
import time

def stream_words(sentence):
    """
    Simulate LLM streaming by yielding one word at a time.
    
    This is exactly what LangChain's .stream() does internally —
    it yields tokens/chunks as they arrive from the LLM API.
    
    Args:
        sentence: The complete text to stream.
    
    Yields:
        str: One word at a time with simulated network delay.
    """
    words = sentence.split()
    for word in words:
        time.sleep(0.3)  # Simulate network latency
        yield word


# Usage — words appear one at a time
for word in stream_words("LangChain is an amazing framework"):
    print(word, end=" ", flush=True)
# Output appears gradually: LangChain is an amazing framework
```

This is the **exact mental model** for how LangChain streams responses from LLMs. Instead of words, it yields tokens or message chunks.

---

## Real Industry Example

### How LangChain Uses Generators for Streaming

```python
# LangChain's streaming is built on generators
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", streaming=True)

# .stream() returns a GENERATOR — tokens arrive one at a time
for chunk in llm.stream("Explain quantum computing"):
    print(chunk.content, end="", flush=True)
# User sees: "Quantum" ... "computing" ... "is" ... "a" ... "field" ...
```

### Processing Large Document Sets for RAG

```python
def load_documents(file_paths):
    """Generator that lazily loads documents one at a time."""
    for path in file_paths:
        with open(path) as f:
            yield {"path": path, "content": f.read()}

# Process 10,000 documents without loading all into memory
for doc in load_documents(list_of_10000_paths):
    embeddings = embed(doc["content"])
    store_in_vector_db(embeddings)
```

---

## Common Mistakes

### Mistake 1: Generators are ONE-TIME use

```python
gen = (x for x in range(5))

list(gen)  # [0, 1, 2, 3, 4] ✅
list(gen)  # [] ❌ — EMPTY! Generator is exhausted!
```

**Fix:** Create a new generator if you need to iterate again.

### Mistake 2: Trying to index a generator

```python
gen = (x for x in range(5))
gen[0]  # ❌ TypeError: 'generator' object is not subscriptable
```

**Fix:** Convert to list first `list(gen)[0]`, or use `next(gen)`.

### Mistake 3: Confusing `return` and `yield`

```python
# ❌ This returns a list, not a generator
def not_a_generator(n):
    return [i for i in range(n)]

# ✅ This IS a generator
def is_a_generator(n):
    for i in range(n):
        yield i
```

If a function contains **even one** `yield`, it becomes a generator function.

### Mistake 4: Checking length of a generator

```python
gen = (x for x in range(1000))
len(gen)  # ❌ TypeError: object of type 'generator' has no len()
```

**Why:** Generators don't know their length — they produce values on demand. You'd have to consume the entire generator to count.

---

## Debugging Guide

### Error: `StopIteration`

**Cause:** You called `next()` on an exhausted generator.

**Fix:** Use `for` loops instead of manual `next()` calls. Or use `next(gen, default_value)` to provide a default.

### Error: `TypeError: 'generator' object is not subscriptable`

**Cause:** You tried to index a generator like `gen[0]`.

**Fix:** Convert to list first: `list(gen)[0]`, or use `next(gen)`.

### Error: `TypeError: object of type 'generator' has no len()`

**Cause:** You called `len()` on a generator.

**Fix:** Convert to list first: `len(list(gen))`. Warning: this loads everything into memory.

### Generator returns empty results on second iteration

**Cause:** Generators are single-use. After one complete iteration, they're exhausted.

**Fix:** Create a new generator instance, or use a list if you need to iterate multiple times.

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Use generators for large datasets | Saves memory — constant O(1) space |
| Use generators for streaming data | Process data as it arrives |
| Use `for` loops, not manual `next()` | Cleaner code, handles StopIteration automatically |
| Use generator expressions for simple cases | One-liner, readable, Pythonic |
| Remember generators are single-use | Create a new one for re-iteration |
| Use `yield from` to delegate to sub-generators | Cleaner than nested loops |
| Use lists when you need random access or length | Generators don't support indexing or `len()` |
| Use generators in function arguments | `sum(x**2 for x in range(n))` — no intermediate list |

---

## When to Use List vs Generator

| Use a List When... | Use a Generator When... |
|--------------------|------------------------|
| You need to access items by index | You only need to iterate sequentially |
| You need to know the length | Length doesn't matter |
| You need to iterate multiple times | Single pass is enough |
| The data is small | The data is large or infinite |
| You need to sort or reverse | You process items as they come |

---

## Interview Preparation

### Easy

**Q: What is the difference between a list and a generator?**

**A:** A list stores all elements in memory at once. A generator produces elements one at a time lazily — only computing the next value when asked. Lists support indexing and re-iteration; generators don't.

### Medium

**Q: What happens when you call a generator function? Does it execute the body?**

**A:** No! Calling a generator function returns a **generator object** without executing any code. The body only executes when you call `next()` or iterate over it with a `for` loop.

### Hard

**Q: Can you iterate over a generator twice? Why or why not?**

**A:** No. Generators are **single-use iterators**. Once exhausted (all values yielded), they cannot be reset. You must create a new generator to iterate again. This is because generators maintain internal state about where they paused, and once they reach the end, that state cannot be rewound.

### Scenario-Based

**Q: You need to process a 50GB log file line by line. How would you do it efficiently?**

**A:** Use a generator that reads and yields one line at a time:
```python
def read_lines(filepath):
    with open(filepath) as f:
        for line in f:  # file objects are iterators themselves
            yield line.strip()
```
This uses constant memory regardless of file size.

### Senior

**Q: How does LangChain use generators for streaming? What's the advantage?**

**A:** LangChain's `.stream()` method returns a generator that yields tokens/chunks as they arrive from the LLM API. This allows the UI to display partial responses immediately instead of waiting for the complete response — reducing perceived latency from seconds to milliseconds. It also enables processing of arbitrarily long responses without memory concerns.

---

## Summary

| Concept | What It Does |
|---------|--------------|
| `yield` | Produces a value and **pauses** the function |
| Generator function | A function containing `yield` — returns a generator object |
| Generator object | A lazy iterator that produces values on demand |
| `next(gen)` | Gets the next value from a generator |
| `StopIteration` | Raised when generator has no more values |
| Generator expression | `(x for x in iterable)` — one-liner generator |
| `yield from` | Delegates to another iterable/generator |
| Single-use | Generators can only be iterated once |

---

## Cheat Sheet

```python
# GENERATOR FUNCTION
def gen_func(n):
    for i in range(n):
        yield i

# GENERATOR EXPRESSION
gen_expr = (x**2 for x in range(10))

# USING GENERATORS
for item in gen_func(5):
    print(item)

# MANUAL ITERATION
gen = gen_func(3)
next(gen)  # 0
next(gen)  # 1
next(gen)  # 2
next(gen)  # StopIteration!

# SAFE MANUAL ITERATION
next(gen, "default")  # Returns "default" instead of StopIteration

# CONVERT TO LIST
values = list(gen_func(5))  # [0, 1, 2, 3, 4]

# YIELD FROM
def combined():
    yield from range(3)      # 0, 1, 2
    yield from "abc"         # 'a', 'b', 'c'

# MEMORY COMPARISON
import sys
sys.getsizeof([i for i in range(1000000)])   # ~8MB
sys.getsizeof((i for i in range(1000000)))   # ~200 bytes
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| What does `yield` do? | Produces a value and **pauses** the function until next iteration |
| Does calling a generator function execute its body? | No — it returns a generator object. Body executes on `next()` or iteration |
| What type does `*args` create? What about a generator? | `*args` = tuple, generator = generator object |
| Generator expression syntax? | `(expression for item in iterable)` — parentheses, not brackets |
| Can you index a generator? | No — generators don't support `gen[0]`. Convert to list first |
| Can you iterate a generator twice? | No — generators are single-use |
| How much memory does a generator use? | ~200 bytes regardless of data size |
| What error when generator is exhausted? | `StopIteration` |
| How does LangChain use generators? | For streaming LLM responses token by token |

---

## Hands-on Exercise

### Exercise 1: Streaming Word Simulator

Write a generator function `stream_words(sentence)` that yields one word at a time with a 0.3-second delay:

```python
import time

def stream_words(sentence):
    words = sentence.split()
    for word in words:
        time.sleep(0.3)
        yield word

for word in stream_words("LangChain is an amazing framework"):
    print(word, end=" ", flush=True)
```

### Exercise 2: Infinite Counter

Write a generator `infinite_counter(start=0)` that yields numbers forever:

```python
def infinite_counter(start=0):
    n = start
    while True:
        yield n
        n += 1

# Use it (with a break condition!)
for num in infinite_counter(100):
    if num > 105:
        break
    print(num)  # 100, 101, 102, 103, 104, 105
```

---

## Challenge Project

Build a `PaginatedAPI` generator that:
1. Simulates fetching data from a paginated API
2. Each "page" returns 10 items
3. Total items = 35 (so page 4 has only 5 items)
4. Yields individual items (not pages)

```python
def paginated_api(total_items=35, page_size=10):
    # Your code here
    pass

for item in paginated_api():
    print(item)
# Should print items 1 through 35, fetching in pages of 10
```

---

## Homework

1. **Reading:** Read [PEP 255 — Simple Generators](https://peps.python.org/pep-0255/)
2. **Coding:** Complete exercises 1 and 2 above
3. **Experiment:** Compare memory usage of `[i for i in range(10_000_000)]` vs `(i for i in range(10_000_000))` using `sys.getsizeof()`
4. **Debugging:** Why does this code print nothing?
    ```python
    gen = (x**2 for x in range(5))
    filtered = filter(lambda x: x > 5, gen)
    gen_list = list(gen)  # Bug is here
    print(gen_list)
    ```

---

## Additional Resources

- [Python Official Docs — Generators](https://docs.python.org/3/tutorial/classes.html#generators)
- [Python Official Docs — Generator Expressions](https://docs.python.org/3/tutorial/classes.html#generator-expressions)
- [PEP 255 — Simple Generators](https://peps.python.org/pep-0255/)
- [PEP 289 — Generator Expressions](https://peps.python.org/pep-0289/)
- [Real Python — Introduction to Generators](https://realpython.com/introduction-to-python-generators/)

---

## What's Next

In the next chapter, we'll learn about **Type Hints and Pydantic** — Python's system for declaring the types of your variables, function parameters, and return values. Type hints make your code self-documenting and catch bugs before runtime. **Pydantic** takes this further by validating data at runtime — ensuring your data matches expected shapes. LangChain uses Pydantic **everywhere**: every chain input, every tool definition, every structured output. Without Pydantic knowledge, you cannot build production LangChain applications.

> [← Previous: Context Managers](chapter-01-context-managers.md) | [Next: Type Hints & Pydantic →](chapter-03-type-hints-pydantic.md)

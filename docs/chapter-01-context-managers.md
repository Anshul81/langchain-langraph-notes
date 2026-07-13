# Chapter 0.2: Context Managers — The `with` Statement

> **Phase 0 — Python Power-Up** | [← Previous: *args & **kwargs](chapter-00-args-kwargs.md) | [Next: Generators →](chapter-02-generators.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand what context managers are and **WHY** they exist
- ✅ Use `with` statements for safe resource management
- ✅ Build your own context managers (both class-based and function-based)
- ✅ Understand how LangChain uses them internally

| | |
|---|---|
| **Prerequisites** | Basic Python functions, try/except basics |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 40 minutes |

---

## Introduction

### The Problem

Imagine you open a file to read data:

```python
# ❌ Dangerous code
file = open("data.txt", "r")
content = file.read()
# What if an ERROR happens here? The file stays open forever!
file.close()
```

**What goes wrong:**
- If an error occurs between `open()` and `close()`, the file **never gets closed**
- Open files consume system resources (memory, file handles)
- Your OS has a **limit** on open files — exceed it and your program crashes
- Same problem with: database connections, network sockets, locks, API sessions

**The core problem:** You need to guarantee cleanup happens **even if errors occur**.

### The Solution

Python's `with` statement — also called a **context manager** — guarantees that setup and cleanup code always runs, no matter what happens in between.

```python
with open("data.txt") as file:
    content = file.read()
# file.close() happens AUTOMATICALLY — even if an error occurred
```

### History

Context managers were introduced in Python 2.5 via [PEP 343](https://peps.python.org/pep-0343/). The `contextlib` module was added to provide utilities for creating context managers more easily. They have since become one of the most important Python patterns.

### Industry Usage

- **File I/O** — Every file operation should use `with`
- **Database connections** — SQLAlchemy, Django ORM, psycopg2
- **Network connections** — HTTP sessions, WebSocket connections
- **Threading** — Locks, semaphores, thread pools
- **LangChain** — Callback tracing, session management, observability
- **Testing** — pytest fixtures, mock patching

### Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| `with` is only for files | It works with ANY object that implements `__enter__` and `__exit__` |
| The variable is destroyed after `with` | The variable still exists, but the resource is released |
| `with` catches errors | It doesn't catch errors — it guarantees cleanup, then re-raises the error |
| You can only use one `with` at a time | You can nest them or use comma syntax for multiple |

---

## Mental Model

### The Hotel Room Analogy

Think of a **hotel stay:**

1. **Check in** → You get the room key (acquire the resource)
2. **Use the room** → Sleep, work, relax (use the resource)
3. **Check out** → Return the key, room gets cleaned (release the resource)

The hotel **guarantees** the room gets cleaned after you leave — even if you:
- Leave early due to an emergency *(exception)*
- Overstay your booking *(error in your code)*
- Trash the room *(unexpected behavior)*

**The `with` statement is the hotel's checkout system** — it guarantees cleanup no matter what.

```python
# The hotel analogy in code
with open("data.txt") as file:    # Check in (open file)
    content = file.read()          # Use the room (read data)
# Check out happens AUTOMATICALLY   # Room cleaned (file closed)
```

### Visual Diagram

```
                    with open("data.txt") as file:
                              │
                    ┌─────────▼──────────┐
                    │   __enter__()       │  ← Resource acquired
                    │   Returns: file     │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │   YOUR CODE RUNS   │
                    │   content = ...     │
                    │   process(...)      │
                    │                    │
                    │   Error? ─────┐    │
                    │   No error? ──┤    │
                    └──────────────┬┘────┘
                                  │
                    ┌─────────────▼──────┐
                    │   __exit__()        │  ← ALWAYS runs
                    │   file.close()      │     (cleanup guaranteed)
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  If error existed:  │
                    │  → Error raised now │
                    │  If no error:       │
                    │  → Continue normal  │
                    └────────────────────┘
```

---

## Theory

### The `with` Statement — Basic Syntax

```python
with open("data.txt", "r") as file:
    content = file.read()
    print(content)
# file.close() is called AUTOMATICALLY here — even if an error occurred
```

**What happens internally:**

```
with EXPRESSION as VARIABLE:
     ──────┬────   ───┬────
           │          │
     Creates the      Assigns the resource
     context manager  to this variable
           │
           ▼
     1. Calls __enter__() → returns the resource
     2. Executes the body (your code)
     3. Calls __exit__()  → cleanup (ALWAYS runs)
```

### Without `with` vs With `with`

```python
# ❌ WITHOUT context manager — manual cleanup (5 lines)
file = open("data.txt", "r")
try:
    content = file.read()
    process(content)
finally:
    file.close()  # Must remember this. Must use try/finally.

# ✅ WITH context manager — automatic cleanup (2 lines)
with open("data.txt", "r") as file:
    content = file.read()
    process(content)
# file.close() happens automatically!
```

### What Happens When Errors Occur?

```python
with open("data.txt", "r") as file:
    content = file.read()
    result = 1 / 0  # 💥 ZeroDivisionError!
    print("This never runs")

# file.close() STILL gets called before the error propagates!
```

**Execution flow:**
```
1. open("data.txt") → file object created
2. __enter__() called → file returned, assigned to `file`
3. file.read() → works fine
4. 1 / 0 → 💥 ZeroDivisionError!
5. __exit__() called → file.close() runs ✅ (cleanup guaranteed!)
6. Error propagates up → program crashes with ZeroDivisionError
```

### Multiple Context Managers

You can use multiple context managers in a single `with` statement:

```python
# Nested (works in all Python versions)
with open("input.txt") as infile:
    with open("output.txt", "w") as outfile:
        outfile.write(infile.read())

# Comma syntax (cleaner, Python 3.1+)
with open("input.txt") as infile, open("output.txt", "w") as outfile:
    outfile.write(infile.read())

# Parenthesized (Python 3.10+)
with (
    open("input.txt") as infile,
    open("output.txt", "w") as outfile,
):
    outfile.write(infile.read())
```

---

## Architecture

### The Context Manager Protocol

Any object that implements two special methods becomes a context manager:

```
┌─────────────────────────────────────────────┐
│           Context Manager Protocol          │
├─────────────────────────────────────────────┤
│                                             │
│  __enter__(self)                            │
│  ├── Called at the START of 'with' block    │
│  ├── Acquires the resource                  │
│  └── Returns the resource (for 'as' var)   │
│                                             │
│  __exit__(self, exc_type, exc_val, exc_tb)  │
│  ├── Called at the END of 'with' block      │
│  ├── ALWAYS runs (even on errors)           │
│  ├── Releases the resource                  │
│  ├── Returns False → let errors propagate   │
│  └── Returns True → suppress errors (rare)  │
│                                             │
└─────────────────────────────────────────────┘
```

---

## Step-by-Step Explanation

### Building a Class-Based Context Manager

```python
class DatabaseConnection:
    """A context manager for database connections."""
    
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = None
    
    def __enter__(self):
        """Called when entering the 'with' block. Acquire the resource."""
        print(f"Connecting to {self.host}:{self.port}...")
        self.connection = f"Connection({self.host}:{self.port})"  # Simulated
        return self.connection  # This is what 'as variable' receives
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Called when leaving the 'with' block. Release the resource."""
        print(f"Closing connection to {self.host}:{self.port}...")
        self.connection = None
        return False  # False = don't suppress exceptions
```

**Using it:**
```python
with DatabaseConnection("localhost", 5432) as conn:
    print(f"Using: {conn}")
    print("Running queries...")
```

**Output:**
```
Connecting to localhost:5432...
Using: Connection(localhost:5432)
Running queries...
Closing connection to localhost:5432...
```

### Understanding `__exit__` Parameters

```python
def __exit__(self, exc_type, exc_value, traceback):
    #              ────┬───  ────┬────  ────┬────
    #                  │        │          │
    #          Type of error  Error    Stack trace
    #          (e.g., ZeroDivisionError)  message   object
    #          None if no error  None if no error  None if no error
```

- If **no error** occurred: all three are `None`
- If **an error** occurred: they contain the error info
- Return `True` to **suppress** the error (swallow it — rarely desired)
- Return `False` to **let the error propagate** (almost always what you want)

### Building a Function-Based Context Manager

Python's `contextlib` module lets you create context managers with a simple generator function:

```python
from contextlib import contextmanager

@contextmanager
def timer(label):
    """A context manager that times code execution."""
    import time
    
    start = time.time()             # __enter__ equivalent (SETUP)
    print(f"[{label}] Starting...")
    
    yield  # <-- YOUR CODE RUNS HERE (the 'with' block body)
    
    elapsed = time.time() - start   # __exit__ equivalent (CLEANUP)
    print(f"[{label}] Finished in {elapsed:.2f}s")
```

**Using it:**
```python
with timer("Data Processing"):
    total = sum(range(1_000_000))
    print(f"Sum: {total}")
```

**Output:**
```
[Data Processing] Starting...
Sum: 499999500000
[Data Processing] Finished in 0.03s
```

### The `yield` Keyword — What Does It Do Here?

```
@contextmanager
def my_context():
    # === SETUP (runs before 'with' block) ===
    print("Before")
    
    yield value    # ← Pauses here. 'value' is assigned to 'as' variable.
                   #   Your 'with' block code runs now.
    
    # === CLEANUP (runs after 'with' block) ===
    print("After")
```

Think of `yield` as a **pause button**:
1. Code before `yield` → setup (`__enter__`)
2. `yield` → pause, run the user's code
3. Code after `yield` → cleanup (`__exit__`)

### Function-Based with Error Handling

```python
from contextlib import contextmanager

@contextmanager
def managed_resource(name):
    print(f"Acquiring {name}...")
    resource = {"name": name, "active": True}
    
    try:
        yield resource  # User's code runs here
    except Exception as e:
        print(f"Error occurred: {e}")
        raise  # Re-raise the error after cleanup
    finally:
        print(f"Releasing {name}...")
        resource["active"] = False
```

The `try/finally` inside the generator ensures cleanup even on errors — same guarantee as the class-based approach.

---

## Execution Walkthrough

Let's trace a full execution:

```python
with DatabaseConnection("localhost", 5432) as conn:
    print(f"Using: {conn}")
    result = 1 / 0  # Error!
```

```
Step 1: DatabaseConnection("localhost", 5432) 
        → Creates object with host="localhost", port=5432

Step 2: __enter__() called
        → Prints "Connecting to localhost:5432..."
        → Sets self.connection = "Connection(localhost:5432)"
        → Returns self.connection
        → conn = "Connection(localhost:5432)"

Step 3: print(f"Using: {conn}")
        → Prints "Using: Connection(localhost:5432)"

Step 4: 1 / 0
        → 💥 ZeroDivisionError!

Step 5: __exit__(ZeroDivisionError, "division by zero", <traceback>) called
        → Prints "Closing connection to localhost:5432..."
        → Sets self.connection = None
        → Returns False (don't suppress the error)

Step 6: ZeroDivisionError propagates
        → Program crashes with traceback
```

---

## Real Industry Example

### How LangChain Uses Context Managers

```python
# LangChain uses context managers for tracing/observability
from langchain_core.tracers import trace_as_chain_group

with trace_as_chain_group("my_group") as group_manager:
    # All LLM calls inside here are traced together
    result = chain.invoke({"input": "Hello"}, config={"callbacks": group_manager})
# Tracing data is flushed and sent automatically on exit
```

### Database Session Management (SQLAlchemy Pattern)

```python
# This is the pattern you'll use in your capstone project
@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()    # Commit on success
    except Exception:
        session.rollback()  # Rollback on error
        raise
    finally:
        session.close()     # Always close

# Usage
with get_db_session() as db:
    db.add(new_record)
    # commit happens automatically if no error
    # rollback happens automatically if error
    # close happens no matter what
```

---

## Common Mistakes

### Mistake 1: Using the resource OUTSIDE the `with` block

```python
with open("data.txt") as file:
    content = file.read()

# ❌ file is CLOSED here
print(file.read())  # ValueError: I/O operation on closed file
```

**Fix:** Do all work with the resource **inside** the `with` block.

### Mistake 2: Forgetting `yield` in `@contextmanager`

```python
@contextmanager
def bad_context():
    print("Setup")
    # ❌ Forgot yield — RuntimeError: generator didn't yield
    print("Cleanup")
```

**Fix:** Always include exactly **one** `yield` statement.

### Mistake 3: Multiple yields in `@contextmanager`

```python
@contextmanager
def bad_context():
    yield "first"   # ❌ Only ONE yield is allowed
    yield "second"  # RuntimeError
```

### Mistake 4: Not handling errors in `@contextmanager`

```python
@contextmanager
def leaky_resource():
    resource = acquire()
    yield resource
    release(resource)  # ❌ This line SKIPPED if error occurs inside 'with'!
```

**Fix:** Use `try/finally`:
```python
@contextmanager
def safe_resource():
    resource = acquire()
    try:
        yield resource
    finally:
        release(resource)  # ✅ Always runs
```

---

## Debugging Guide

### Error: `ValueError: I/O operation on closed file`

**Cause:** You're trying to read/write a file after the `with` block ended.

**Fix:** Move your file operations inside the `with` block.

### Error: `RuntimeError: generator didn't yield`

**Cause:** Your `@contextmanager` function doesn't contain a `yield` statement.

**Fix:** Add a `yield` statement (exactly one).

### Error: `AttributeError: __enter__`

**Cause:** You're using `with` on an object that isn't a context manager.

**Fix:** Ensure the object implements `__enter__` and `__exit__`, or use `@contextmanager`.

### Error: `TypeError: __exit__() takes 1 positional argument but 4 were given`

**Cause:** Your `__exit__` method doesn't accept the three exception parameters.

**Fix:** Define it as `def __exit__(self, exc_type, exc_value, traceback)`.

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Always use `with` for files, DB connections, locks | Guarantees cleanup |
| Use `@contextmanager` for simple context managers | Less boilerplate than class-based |
| Use class-based for complex state management | More control, reusable, testable |
| Always `return False` in `__exit__` | Explicit is better than implicit |
| Wrap `yield` in `try/finally` in `@contextmanager` | Ensures cleanup on errors |
| Never do heavy computation in `__enter__`/`__exit__` | They should be fast setup/cleanup |
| Use context managers for any acquire/release pattern | Not just files — connections, locks, sessions |

---

## Interview Preparation

### Easy

**Q: What does the `with` statement do in Python?**

**A:** It ensures a resource is properly acquired and released. It calls `__enter__` at the start and `__exit__` at the end, guaranteeing cleanup even if an error occurs.

### Medium

**Q: What are the two ways to create a context manager in Python?**

**A:** 1) Class-based: define `__enter__` and `__exit__` methods. 2) Function-based: use `@contextmanager` decorator from `contextlib` with a generator function containing one `yield`.

### Hard

**Q: What are the three parameters of `__exit__`? When are they not `None`?**

**A:** `exc_type`, `exc_value`, `traceback`. They are `None` when no exception occurred. When an exception occurs inside the `with` block, they contain the exception type, value, and traceback respectively.

### Senior

**Q: What happens if `__exit__` returns `True`?**

**A:** The exception is **suppressed** — it does NOT propagate. The program continues after the `with` block as if no error happened. This is rarely desired and should be used with extreme caution.

### System Design

**Q: How would you design a database connection pool using context managers?**

**A:** Create a context manager that checks out a connection from the pool in `__enter__` and returns it to the pool in `__exit__`. Use `try/finally` to ensure the connection is always returned, even on errors. The pool itself manages the lifecycle of connections (creation, health checks, max connections).

---

## Summary

| Concept | What It Does |
|---------|--------------|
| `with X as y:` | Acquires resource, runs your code, guarantees cleanup |
| `__enter__()` | Called at start — acquires and returns the resource |
| `__exit__(exc_type, exc_val, exc_tb)` | Called at end — releases the resource (ALWAYS runs) |
| `@contextmanager` | Decorator to create context managers from generator functions |
| `yield` in `@contextmanager` | Pause point — code before = setup, code after = cleanup |
| `return False` in `__exit__` | Let exceptions propagate (normal behavior) |
| `return True` in `__exit__` | Suppress exceptions (use with caution) |

---

## Cheat Sheet

```python
# USING a context manager
with open("file.txt") as f:
    data = f.read()

# CLASS-BASED context manager
class MyContext:
    def __enter__(self):
        # setup
        return resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # cleanup
        return False

# FUNCTION-BASED context manager
from contextlib import contextmanager

@contextmanager
def my_context():
    # setup
    try:
        yield resource
    finally:
        # cleanup

# MULTIPLE context managers
with open("a.txt") as a, open("b.txt") as b:
    pass
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| What problem do context managers solve? | Guaranteeing resource cleanup even when errors occur |
| What two methods make an object a context manager? | `__enter__` and `__exit__` |
| What does `__enter__` return? | The resource that gets assigned to the `as` variable |
| Does `__exit__` run if an error occurs? | Yes — it ALWAYS runs |
| What does `yield` do in `@contextmanager`? | Pauses the function — code before = setup, code after = cleanup |
| What happens if `__exit__` returns True? | The exception is suppressed (not raised) |
| Can you nest `with` statements? | Yes — you can nest them or use comma syntax |
| Is `with` only for files? | No — any object with `__enter__` and `__exit__` works |

---

## Hands-on Exercise

### Exercise 1: API Session Manager

Build a context manager called `APISession` that:
1. On enter: prints `"Opening API session..."` and returns a dictionary `{"session_id": "abc123", "active": True}`
2. On exit: prints `"Closing API session..."` and sets `active` to `False`
3. Build it **both ways** — class-based AND using `@contextmanager`

### Exercise 2: Timer with Logging

Create a `@contextmanager` that:
1. Records the start time
2. Yields a dictionary that the user can add data to
3. On exit, prints the elapsed time and the collected data

```python
with data_timer("Processing") as ctx:
    ctx["records"] = 1000
    time.sleep(1)
# [Processing] 1000 records in 1.00s
```

---

## Challenge Project

Build a `FileTransactionManager` context manager that:
1. Writes data to a **temporary file** first
2. If the `with` block completes without errors, **renames** the temp file to the final name
3. If an error occurs, **deletes** the temp file (no partial writes)

This is how production systems do safe file writes.

```python
with FileTransactionManager("output.json") as f:
    f.write('{"status": "complete"}')
# File renamed from output.json.tmp to output.json

with FileTransactionManager("output.json") as f:
    f.write('{"status": ')
    raise ValueError("oops")
# Temp file deleted — output.json unchanged
```

---

## Homework

1. **Reading:** Read [PEP 343 — The "with" Statement](https://peps.python.org/pep-0343/)
2. **Coding:** Complete exercises 1 and 2 above
3. **Debugging:** Fix this code:
    ```python
    @contextmanager
    def connection(host):
        conn = create_connection(host)
        yield conn
        conn.close()  # Bug: What happens if error occurs in with block?
    ```

---

## Additional Resources

- [Python Official Docs — The with Statement](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement)
- [Python Official Docs — contextlib](https://docs.python.org/3/library/contextlib.html)
- [PEP 343 — The "with" Statement](https://peps.python.org/pep-0343/)
- [Real Python — Context Managers](https://realpython.com/python-with-statement/)

---

## What's Next

In the next chapter, we'll learn about **Generators and Iterators** — Python's way of producing data lazily, one item at a time, without loading everything into memory. This is critical for LangChain because **streaming LLM responses** (getting tokens one at a time instead of waiting for the complete response) is built on generators. Without understanding generators, you won't understand how LangChain's streaming works.

> [← Previous: *args & **kwargs](chapter-00-args-kwargs.md) | [Next: Generators →](chapter-02-generators.md)

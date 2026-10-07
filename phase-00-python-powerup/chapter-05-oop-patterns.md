# Chapter 0.6: OOP Patterns for AI Applications

> **Phase 0 — Python Power-Up** | [← Previous: Async Python](chapter-04-async-python.md) | [Next: Phase 1 — What is an LLM? →](../phase-01-llm-fundamentals/chapter-06-what-is-an-llm.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand classes, objects, inheritance, and composition
- ✅ Use abstract classes (`ABC`) to define contracts
- ✅ Understand polymorphism — same interface, different implementations
- ✅ Know the OOP patterns LangChain uses internally
- ✅ Build a reusable `BaseLLM` class design

| | |
|---|---|
| **Prerequisites** | Basic Python classes, type hints |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 40 minutes |

---

## Introduction

### The Problem

You're building an AI app that supports multiple LLM providers (OpenAI, Gemini, Ollama). Each has:
- Different API calls
- Different authentication
- Same interface (send message → get response)

Without OOP:
```python
# ❌ Messy, repetitive, impossible to maintain
def call_openai(message, api_key, model, temperature):
    pass

def call_gemini(message, api_key, model, temperature):
    pass

def call_ollama(message, host, model, temperature):
    pass

# Every time you add a new provider, you change EVERYTHING
```

With OOP:
```python
# ✅ Clean, extensible, maintainable
class OpenAI(BaseLLM):
    def invoke(self, message): ...

class Gemini(BaseLLM):
    def invoke(self, message): ...

# Adding a new provider = just ONE new class. Nothing else changes.
```

### The Solution

Object-Oriented Programming (OOP) lets you create reusable, extensible, and maintainable code through:
- **Classes** — blueprints for objects
- **Inheritance** — shared behavior across related classes
- **Abstract classes** — enforced contracts (interfaces)
- **Polymorphism** — same interface, different implementations
- **Composition** — building complex objects from simpler ones

### History

OOP concepts originated in the 1960s with Simula and were popularized by Smalltalk. Python has supported OOP since its creation in 1991. The `abc` (Abstract Base Classes) module was added in Python 2.6/3.0 via [PEP 3119](https://peps.python.org/pep-3119/).

### Industry Usage

- **LangChain** — `BaseChatModel`, `BaseRetriever`, `Runnable` — all abstract classes
- **Django/Flask** — Class-based views, model inheritance
- **SQLAlchemy** — ORM model hierarchy
- **Design Patterns** — Strategy, Factory, Observer — all OOP-based
- **Every large Python project** uses OOP for organization and extensibility

---

## Mental Model

### The Power Outlet Analogy

Every electrical outlet has the **same interface** (two holes + ground), but behind the wall, the wiring differs by building. Your phone charger doesn't care about the wiring — it just plugs in.

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   OpenAI    │  │   Gemini    │  │   Ollama    │
│             │  │             │  │             │
│  .invoke()  │  │  .invoke()  │  │  .invoke()  │
│  .stream()  │  │  .stream()  │  │  .stream()  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
              Same interface (BaseLLM)
              Your code doesn't care
              which one is behind it
```

That's **polymorphism** — same interface, different implementations.

---

## Theory

### Part 1: Classes & Objects — Quick Refresher

```python
class Dog:
    """A simple class representing a dog."""
    
    def __init__(self, name: str, breed: str):
        self.name = name      # Instance attribute
        self.breed = breed    # Instance attribute
    
    def bark(self) -> str:
        return f"{self.name} says Woof!"

# Creating objects (instances)
dog1 = Dog("Rex", "German Shepherd")
dog2 = Dog("Buddy", "Golden Retriever")

print(dog1.bark())  # "Rex says Woof!"
print(dog2.bark())  # "Buddy says Woof!"
```

**Key concepts:**

| Term | What It Is |
|------|-----------|
| `class` | Blueprint/template for creating objects |
| `object` | Instance created from a class |
| `__init__` | Constructor — runs when you create an object |
| `self` | Reference to the current instance |
| Attribute | Data stored in an object (`self.name`) |
| Method | Function that belongs to a class (`def bark(self)`) |

### Part 2: Inheritance — Shared Behavior

```python
class Animal:
    def __init__(self, name: str):
        self.name = name
    
    def speak(self) -> str:
        raise NotImplementedError("Subclasses must implement speak()")

class Dog(Animal):
    def speak(self) -> str:
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return f"{self.name} says Meow!"

# Both are Animals, but speak differently
dog = Dog("Rex")
cat = Cat("Whiskers")
print(dog.speak())  # "Rex says Woof!"
print(cat.speak())  # "Whiskers says Meow!"
```

**Key insight:** `Dog` and `Cat` **inherit** from `Animal` — they get `__init__` for free but each provides its own `speak()`.

**Inheritance diagram:**
```
        Animal
       /      \
    Dog        Cat
    
    Dog inherits from Animal
    Cat inherits from Animal
    Both get __init__ for free
    Both override speak()
```

### Part 3: Abstract Classes — Enforcing a Contract

```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract base class — cannot be instantiated directly."""
    
    def __init__(self, model: str, temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
    
    @abstractmethod
    def invoke(self, message: str) -> str:
        """Every LLM MUST implement this method."""
        pass
    
    @abstractmethod
    def stream(self, message: str):
        """Every LLM MUST implement this method."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"

# ❌ Cannot create BaseLLM directly
# llm = BaseLLM("test")  # TypeError: Can't instantiate abstract class
```

`ABC` + `@abstractmethod` = **contract**. Any class inheriting from `BaseLLM` MUST implement `invoke()` and `stream()`, or Python raises an error.

**Why `ABC` instead of just `NotImplementedError`:**

| Approach | When Error Occurs |
|----------|------------------|
| `NotImplementedError` | At **call time** — bug hides until someone calls the method |
| `ABC` + `@abstractmethod` | At **object creation time** — instant error if method not implemented |

### Part 4: Implementing the Contract

```python
class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4", api_key: str = "", **kwargs):
        super().__init__(model, **kwargs)  # Call parent's __init__
        self.api_key = api_key
    
    def invoke(self, message: str) -> str:
        return f"[OpenAI/{self.model}] Response to: {message}"
    
    def stream(self, message: str):
        words = f"[OpenAI/{self.model}] Response to: {message}".split()
        for word in words:
            yield word

class OllamaLLM(BaseLLM):
    def __init__(self, model: str = "llama3", host: str = "localhost", **kwargs):
        super().__init__(model, **kwargs)
        self.host = host
    
    def invoke(self, message: str) -> str:
        return f"[Ollama/{self.model}@{self.host}] Response to: {message}"
    
    def stream(self, message: str):
        words = f"[Ollama/{self.model}] Response to: {message}".split()
        for word in words:
            yield word
```

Notice `super().__init__(model, **kwargs)` — this calls the parent's constructor. Without it, `self.model` and `self.temperature` wouldn't be set.

### Part 5: Polymorphism — Using Them Interchangeably

```python
def chat(llm: BaseLLM, message: str) -> str:
    """This function works with ANY LLM. It doesn't know or care which one."""
    return llm.invoke(message)

# Works with OpenAI
openai = OpenAILLM(model="gpt-4", api_key="sk-...")
print(chat(openai, "Hello"))
# [OpenAI/gpt-4] Response to: Hello

# Works with Ollama — SAME function!
ollama = OllamaLLM(model="llama3")
print(chat(ollama, "Hello"))
# [Ollama/llama3@localhost] Response to: Hello
```

**This is EXACTLY how LangChain works.** You write your chain once, then swap `ChatOpenAI` for `ChatAnthropic` or `ChatOllama` — everything still works.

### Part 6: Composition — Building with Pieces

Inheritance says "X **is a** Y." Composition says "X **has a** Y."

```python
class RAGPipeline:
    """Uses composition — HAS an LLM, HAS a retriever."""
    
    def __init__(self, llm: BaseLLM, retriever, prompt_template: str):
        self.llm = llm              # HAS an LLM
        self.retriever = retriever  # HAS a retriever
        self.prompt_template = prompt_template
    
    def query(self, question: str) -> str:
        # 1. Retrieve relevant documents
        docs = self.retriever.search(question)
        
        # 2. Format prompt with context
        prompt = self.prompt_template.format(
            context=docs,
            question=question
        )
        
        # 3. Generate answer
        return self.llm.invoke(prompt)

# Swap components without changing the pipeline
rag = RAGPipeline(
    llm=OpenAILLM(model="gpt-4"),
    retriever=ChromaRetriever(),
    prompt_template="Context: {context}\nQuestion: {question}"
)
```

**When to use which:**

| Relationship | Use | Example |
|-------------|-----|---------|
| "is a" | Inheritance | `ChatOpenAI` **is a** `BaseChatModel` |
| "has a" | Composition | `RAGPipeline` **has a** `BaseLLM` |

**Rule of thumb:** Prefer composition over inheritance. It's more flexible.

---

## Architecture

### LangChain's Class Hierarchy

LangChain's entire architecture is built on these patterns:

```
Runnable (Base interface for ALL components)
│
├── BaseChatModel (Abstract)
│   ├── ChatOpenAI
│   ├── ChatAnthropic
│   ├── ChatOllama
│   └── ChatGoogleGenerativeAI
│
├── BaseRetriever (Abstract)
│   ├── VectorStoreRetriever
│   ├── ParentDocumentRetriever
│   └── MultiQueryRetriever
│
├── RunnableLambda
├── RunnableParallel
├── RunnablePassthrough
└── RunnableSequence (chains)
```

Every component implements the `Runnable` interface with methods like `invoke()`, `stream()`, `batch()`. That's why you can chain any components together — they all speak the same language.

---

## Practical Example

```python
from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, message: str) -> str:
        pass


class MockOpenAI(BaseLLM):
    def invoke(self, message: str) -> str:
        return f"[OpenAI] {message}"


class MockGemini(BaseLLM):
    def invoke(self, message: str) -> str:
        return f"[Gemini] {message}"


def chat(llm: BaseLLM, message: str) -> str:
    return llm.invoke(message)


# Demo — same function, different LLMs
openai_llm = MockOpenAI()
gemini_llm = MockGemini()

print(chat(openai_llm, "Hello there"))  # [OpenAI] Hello there
print(chat(gemini_llm, "Hello there"))  # [Gemini] Hello there
```

This demonstrates polymorphism: the `chat()` function accepts **any** `BaseLLM` and works correctly regardless of which specific implementation is passed.

---

## Common Mistakes

### Mistake 1: Forgetting `super().__init__()`

```python
class Child(Parent):
    def __init__(self, extra_param):
        # ❌ Parent's __init__ never runs — parent attributes missing!
        self.extra = extra_param
    
    def __init__(self, extra_param):
        # ✅ Call parent's __init__ first
        super().__init__()
        self.extra = extra_param
```

### Mistake 2: Using inheritance when composition is better

```python
# ❌ A RAGPipeline is NOT a type of LLM
class RAGPipeline(BaseLLM):  # Wrong relationship!
    pass

# ✅ A RAGPipeline HAS an LLM
class RAGPipeline:
    def __init__(self, llm: BaseLLM):
        self.llm = llm  # Composition — correct!
```

### Mistake 3: Forgetting `self`

```python
class MyClass:
    def greet(name):  # ❌ Missing self — first arg IS the instance!
        print(f"Hello {name}")
    
    def greet(self, name):  # ✅ self is always the first parameter
        print(f"Hello {name}")
```

### Mistake 4: Not implementing all abstract methods

```python
class IncompleteModel(BaseLLM):
    def invoke(self, message: str) -> str:
        return message
    # ❌ Forgot to implement stream()!
    # TypeError at instantiation: Can't instantiate abstract class
```

---

## Debugging Guide

### Error: `TypeError: Can't instantiate abstract class X with abstract method Y`

**Cause:** You're trying to create an instance of a class that hasn't implemented all `@abstractmethod`s.

**Fix:** Implement all abstract methods in your subclass.

### Error: `AttributeError: 'X' object has no attribute 'Y'`

**Cause:** Often means you forgot `super().__init__()` in the child class, so parent attributes weren't set.

**Fix:** Add `super().__init__(...)` in the child's `__init__`.

### Error: `TypeError: greet() takes 1 positional argument but 2 were given`

**Cause:** Forgot `self` in the method definition.

**Fix:** Add `self` as the first parameter.

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Use abstract classes for shared interfaces | Enforces contract, prevents bugs at creation time |
| Prefer composition over inheritance | More flexible, easier to swap components |
| Use `super().__init__()` in child classes | Don't skip parent initialization |
| Keep classes focused (Single Responsibility) | Easier to test, maintain, and debug |
| Use type hints on all methods | Self-documenting, IDE support |
| Use `**kwargs` in `__init__` for flexibility | LangChain pattern — extensible without breaking changes |
| Use "is a" test for inheritance | If "X is a Y" doesn't make sense, use composition |

---

## Interview Preparation

### Easy

**Q: What is the difference between a class and an object?**

**A:** A class is a blueprint/template that defines attributes and methods. An object (instance) is a concrete thing created from that blueprint. `Dog` is a class; `rex = Dog("Rex")` is an object.

### Medium

**Q: What is the difference between inheritance and composition?**

**A:** Inheritance = "is a" relationship (Dog IS an Animal). Composition = "has a" relationship (Car HAS an Engine). Composition is more flexible because you can swap components at runtime without modifying the class hierarchy.

### Hard

**Q: What is polymorphism and how does LangChain use it?**

**A:** Polymorphism means "same interface, different implementations." LangChain defines `BaseChatModel` with methods like `invoke()` and `stream()`. Each provider (`ChatOpenAI`, `ChatAnthropic`, `ChatOllama`) implements these methods differently. Your application code uses the base interface and works with ANY provider without changes — you can swap providers by changing one line.

### Senior

**Q: What is the difference between `ABC` with `@abstractmethod` and just raising `NotImplementedError`?**

**A:** `ABC` + `@abstractmethod` prevents instantiation of the abstract class at **object creation time** — you get an immediate `TypeError` if a subclass doesn't implement all required methods. `NotImplementedError` only fails at **call time** — the bug hides until someone actually calls the unimplemented method, potentially in production.

### System Design

**Q: How would you design a pluggable LLM system that supports multiple providers?**

**A:** Define an abstract `BaseLLM` class with methods like `invoke()`, `stream()`, `batch()`. Each provider implements this interface. Use a factory function or configuration to select the provider at runtime. Application code only references `BaseLLM`, never specific providers. This is the Strategy Pattern — the algorithm (LLM provider) varies independently from the code that uses it.

---

## Summary

| Concept | What It Does |
|---------|--------------|
| `class` | Blueprint for creating objects |
| `__init__` | Constructor — initializes object attributes |
| `self` | Reference to the current instance |
| Inheritance (`class Child(Parent)`) | Child gets all parent's methods and attributes |
| `super().__init__()` | Calls parent's constructor from child |
| `ABC` + `@abstractmethod` | Creates an abstract class that enforces method implementation |
| Polymorphism | Same interface, different implementations — swap without code changes |
| Composition | "Has a" relationship — build objects from components |

---

## Cheat Sheet

```python
from abc import ABC, abstractmethod

# ABSTRACT CLASS (cannot instantiate)
class Base(ABC):
    @abstractmethod
    def required_method(self) -> str:
        pass

# CONCRETE CLASS (implements abstract methods)
class Concrete(Base):
    def required_method(self) -> str:
        return "implemented"

# INHERITANCE
class Child(Parent):
    def __init__(self, extra):
        super().__init__()  # Always call parent's init
        self.extra = extra

# COMPOSITION
class Pipeline:
    def __init__(self, llm: BaseLLM, retriever: BaseRetriever):
        self.llm = llm
        self.retriever = retriever

# POLYMORPHISM
def process(llm: BaseLLM):  # Accepts ANY BaseLLM subclass
    return llm.invoke("hello")
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| What is `self`? | Reference to the current instance of the class |
| What is `__init__`? | Constructor — runs when an object is created |
| What does `@abstractmethod` do? | Forces subclasses to implement the method |
| Can you instantiate an abstract class? | No — `TypeError` if you try |
| Inheritance vs Composition? | Inheritance = "is a", Composition = "has a" |
| What is polymorphism? | Same interface, different implementations |
| What does `super().__init__()` do? | Calls the parent class's constructor |
| When to use inheritance? | When "X is a Y" makes logical sense |
| When to use composition? | When "X has a Y" — prefer this by default |

---

## Hands-on Exercise

Build a mini LLM system:
1. Create `BaseLLM` abstract class with abstract `invoke(message: str) -> str` method
2. Create `MockOpenAI` that returns `"[OpenAI] {message}"`
3. Create `MockGemini` that returns `"[Gemini] {message}"`
4. Create a `chat()` function that accepts any `BaseLLM`
5. Demonstrate polymorphism

**Solution:**
```python
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, message: str) -> str:
        pass

class MockOpenAI(BaseLLM):
    def invoke(self, message: str) -> str:
        return f"[OpenAI] {message}"

class MockGemini(BaseLLM):
    def invoke(self, message: str) -> str:
        return f"[Gemini] {message}"

def chat(llm: BaseLLM, message: str) -> str:
    return llm.invoke(message)

openai_llm = MockOpenAI()
gemini_llm = MockGemini()
print(chat(openai_llm, "Hello there"))  # [OpenAI] Hello there
print(chat(gemini_llm, "Hello there"))  # [Gemini] Hello there
```

---

## Challenge Project

Build a `NotificationSystem`:
1. Abstract `BaseNotifier` with abstract `send(message: str, recipient: str) -> bool`
2. `EmailNotifier` — prints email notification
3. `SlackNotifier` — prints Slack notification
4. `SMSNotifier` — prints SMS notification
5. `NotificationService` that uses **composition** — accepts a list of notifiers and sends to all

```python
service = NotificationService([EmailNotifier(), SlackNotifier()])
service.notify_all("System alert!", "admin@company.com")
```

---

## Homework

1. **Reading:** Read [Python OOP docs](https://docs.python.org/3/tutorial/classes.html)
2. **Coding:** Build the `NotificationSystem` challenge project
3. **Research:** Look at LangChain's `BaseChatModel` source code — find `invoke()` and `stream()` methods
4. **Reflection:** Why does LangChain use `Runnable` as the base for everything?

---

## Additional Resources

- [Python Official Docs — Classes](https://docs.python.org/3/tutorial/classes.html)
- [Python Official Docs — abc module](https://docs.python.org/3/library/abc.html)
- [PEP 3119 — Abstract Base Classes](https://peps.python.org/pep-3119/)
- [Real Python — OOP in Python](https://realpython.com/python3-object-oriented-programming/)
- [Composition vs Inheritance](https://realpython.com/inheritance-composition-python/)

---

## Phase 0 Complete! 🎉

Congratulations! You've completed all Python foundations needed for LangChain:

| Chapter | Topic | Status |
|---------|-------|--------|
| 0.1 | `*args` & `**kwargs` | ✅ |
| 0.2 | Context Managers | ✅ |
| 0.3 | Generators & Iterators | ✅ |
| 0.4 | Type Hints & Pydantic | ✅ |
| 0.5 | Async Python | ✅ |
| 0.6 | OOP Patterns | ✅ |

## What's Next

**Phase 1: LLM Fundamentals** begins! In the next chapter, you'll learn **what a Large Language Model actually is** — how it was trained, what transformers are, how tokens work, and what happens when you send a message to ChatGPT. This foundational understanding will make everything in LangChain click — because LangChain is built on top of these models.

> [← Previous: Async Python](chapter-04-async-python.md) | [Next: Phase 1 — What is an LLM? →](../phase-01-llm-fundamentals/chapter-06-what-is-an-llm.md)

# Chapter 3.2: The Runnable Protocol & LCEL

> **Phase 3 — LangChain Core** | [← Previous: LangChain Setup](chapter-14-langchain-setup.md) | [Next: ChatPromptTemplate Deep Dive →](chapter-16-chat-prompt-template.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand the Runnable interface (backbone of LangChain)
- ✅ Use LCEL — the pipe `|` operator to chain components
- ✅ Build your first chain: Prompt → LLM → Parser
- ✅ Use `RunnableLambda`, `RunnablePassthrough`, `RunnableParallel`
- ✅ Stream through entire chains

| | |
|---|---|
| **Prerequisites** | Chapter 3.1 (LangChain Setup) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

### The Problem

Without LCEL, chaining steps requires verbose manual plumbing:

```python
# ❌ Manual plumbing
formatted = template.format(topic="AI")
messages = [HumanMessage(content=formatted)]
response = llm.invoke(messages)
parsed = parser.parse(response.content)
```

### The Solution: LCEL

```python
# ✅ One line — clean, readable, composable
chain = prompt | llm | parser
result = chain.invoke({"topic": "AI"})
```

---

## Mental Model

### Unix Pipes

```bash
cat file.txt | grep "error" | sort | head -10
```

Each command takes input, transforms it, passes output to the next. LCEL works identically:

```python
chain = prompt | llm | parser
#       ↑         ↑      ↑
#    Format     Generate  Parse
#    prompt     response  output
```

Data flows left → right through each component.

---

## Theory

### What is a Runnable?

Every LangChain component implements the Runnable interface with 6 methods:

```python
component.invoke(input)      # Single input
component.stream(input)      # Token-by-token
component.batch([inputs])    # Multiple inputs
component.ainvoke(input)     # Async single
component.astream(input)     # Async stream
component.abatch([inputs])   # Async batch
```

| Component | Is Runnable? | Input | Output |
|-----------|:---:|------|--------|
| `ChatOpenAI` | ✅ | Messages | AIMessage |
| `ChatPromptTemplate` | ✅ | Dict of variables | List of Messages |
| `StrOutputParser` | ✅ | AIMessage | String |
| `RunnableLambda` | ✅ | Anything | Anything |
| Any chain (`\|`) | ✅ | Chain's first input | Chain's last output |

### The Pipe Operator `|`

The `|` operator connects Runnables into a `RunnableSequence`:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

chain = prompt | llm | parser
result = chain.invoke({"question": "What is LangChain?"})
```

### Data Flow

```
{"question": "What is LangChain?"}
        │
        ▼
┌──────────────────────┐
│  ChatPromptTemplate  │  → Formats variables into messages
└────────┬─────────────┘
         │
    [SystemMessage, HumanMessage]
         │
         ▼
┌──────────────────────┐
│      ChatOpenAI      │  → Calls the LLM
└────────┬─────────────┘
         │
    AIMessage(content="LangChain is...")
         │
         ▼
┌──────────────────────┐
│   StrOutputParser    │  → Extracts .content
└────────┬─────────────┘
         │
    "LangChain is a framework..."
```

---

## Code Examples

### Basic Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert teacher. Explain topics for {audience}."),
    ("human", "Explain {topic} clearly.")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

chain = prompt | llm | parser

# invoke
result = chain.invoke({"topic": "async programming", "audience": "beginners"})
print(result)

# stream
for chunk in chain.stream({"topic": "generators", "audience": "beginners"}):
    print(chunk, end="", flush=True)
```

### RunnableLambda — Custom Functions

```python
from langchain_core.runnables import RunnableLambda

def word_count(text: str) -> dict:
    return {"text": text, "word_count": len(text.split())}

chain = prompt | llm | StrOutputParser() | RunnableLambda(word_count)
result = chain.invoke({"topic": "Python", "audience": "beginners"})
# {"text": "Python is...", "word_count": 45}
```

### RunnableParallel — Run Branches

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

chain = RunnableParallel(
    answer=prompt | llm | StrOutputParser(),
    question=RunnablePassthrough()
)

result = chain.invoke({"topic": "Python", "audience": "beginners"})
print(result["question"])  # Original input
print(result["answer"])    # LLM response
```

### Structured Output with Pydantic

```python
from pydantic import BaseModel, Field

class MovieReview(BaseModel):
    title: str
    rating: float = Field(ge=0, le=10)
    summary: str

structured_llm = llm.with_structured_output(MovieReview)
movie = structured_llm.invoke("Review the movie Inception")
print(movie.title)   # "Inception"
print(movie.rating)  # 9.2
```

---

## Common Mistakes

### Mistake 1: Wrong input type
```python
# ❌ Chain expects dict (prompt has variables)
chain.invoke("What is Python?")

# ✅ Pass dict matching template variables
chain.invoke({"question": "What is Python?"})
```

### Mistake 2: Forgetting StrOutputParser
```python
# ❌ Returns AIMessage object
chain = prompt | llm

# ✅ Parser extracts clean string
chain = prompt | llm | StrOutputParser()
```

### Mistake 3: Variable name mismatch
```python
prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

# ❌ KeyError
chain.invoke({"query": "Hello"})

# ✅ Match variable names
chain.invoke({"question": "Hello"})
```

---

## How the Pipe Operator Works Internally

```python
# a | b creates a RunnableSequence
# Internally:
class Runnable:
    def __or__(self, other):
        return RunnableSequence(first=self, last=other)

# RunnableSequence.invoke() does:
def invoke(self, input):
    result = input
    for step in self.steps:
        result = step.invoke(result)
    return result
```

---

## Interview Preparation

### Easy
**Q: What is LCEL?**
**A:** LangChain Expression Language — a declarative syntax using the pipe `|` operator to chain Runnable components. Data flows left-to-right through the chain.

### Medium
**Q: What is the Runnable interface?**
**A:** The base protocol every LangChain component implements. It guarantees 6 methods: `invoke`, `stream`, `batch` (sync) and `ainvoke`, `astream`, `abatch` (async). This uniform interface allows any components to be chained together.

### Hard
**Q: How does the pipe operator work internally?**
**A:** The `|` operator calls `__or__` on the Runnable class, creating a `RunnableSequence`. When invoked, it calls each step's `invoke()` sequentially, passing each output as input to the next step. The `RunnableSequence` itself is a Runnable, so it supports all 6 methods including streaming.

---

## Summary

| Concept | What It Does |
|---------|-------------|
| Runnable | Base interface with invoke/stream/batch |
| LCEL (`\|`) | Pipe operator to chain Runnables |
| `ChatPromptTemplate` | Template with variables → Messages |
| `StrOutputParser` | AIMessage → plain string |
| `RunnableLambda` | Wrap any function as a Runnable |
| `RunnableParallel` | Run multiple branches concurrently |
| `RunnablePassthrough` | Pass input through unchanged |
| `RunnableSequence` | Chain created by `\|` operator |

---

## What's Next

In the next chapter, we'll deep-dive into `ChatPromptTemplate` — advanced template features including `MessagesPlaceholder` for dynamic chat history, partial variables, and multi-turn conversation templates.

> [← Previous: LangChain Setup](chapter-14-langchain-setup.md) | [Next: ChatPromptTemplate Deep Dive →](chapter-16-chat-prompt-template.md)

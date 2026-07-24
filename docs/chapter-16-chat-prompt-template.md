# Chapter 3.3: ChatPromptTemplate Deep Dive

> **Phase 3 — LangChain Core** | [← Previous: Runnable & LCEL](chapter-15-runnable-lcel.md) | [Next: Structured Output →](chapter-17-structured-output.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Master `ChatPromptTemplate` — LangChain's prompt system
- ✅ Use `MessagesPlaceholder` for dynamic chat history
- ✅ Build multi-turn conversation templates
- ✅ Use partial variables and template composition
- ✅ Build a multi-persona chatbot factory

| | |
|---|---|
| **Prerequisites** | Chapter 3.2 (LCEL) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 40 minutes |

---

## Introduction

### The Problem

Real apps need more than simple prompt templates:
- **Chat history** injected dynamically (chatbots)
- **Few-shot examples** as message arrays
- **Partial templates** (pre-fill some variables, fill others later)
- **Composable templates** (combine smaller templates into bigger ones)

### The Solution

`ChatPromptTemplate` with `MessagesPlaceholder` provides a flexible system for building any prompt structure.

---

## Theory

### `from_messages` — The Core Method

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),   # → SystemMessage
    ("human", "{question}"),           # → HumanMessage
    ("ai", "..."),                     # → AIMessage
])
```

### `MessagesPlaceholder` — Dynamic Message Injection

Injects a variable-length list of messages at a specific position:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),  # ← Variable length!
    ("human", "{input}")
])
```

Template structure:
```
1. SystemMessage (fixed)
2. MessagesPlaceholder (0 to N messages — grows with conversation)
3. HumanMessage (current question)
```

### Partial Variables

Pre-fill some variables, fill others later:

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Respond in {language}."),
    ("human", "{question}")
])

spanish_tutor = prompt.partial(role="tutor", language="Spanish")
# Now only needs: {"question": "..."}
```

### Template Composition with Factory Functions

```python
def create_chat_prompt(system_msg: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="history", optional=True),
        ("human", "{input}")
    ])

python_chain = create_chat_prompt("You are a Python tutor.") | llm | parser
sql_chain = create_chat_prompt("You are a SQL expert.") | llm | parser
```

---

## Code Example: Multi-Persona Chatbot Factory

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
    temperature=0.7,
    max_tokens=200
)


def create_chatbot(persona: str):
    """Factory that returns a callable chatbot with its own memory."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", persona),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    history = []  # Each chatbot gets its OWN history (closure)
    
    def chat(user_message: str) -> str:
        response = chain.invoke({
            "history": history,
            "input": user_message
        })
        history.append(HumanMessage(content=user_message))
        history.append(AIMessage(content=response))
        return response
    
    return chat


# Two independent chatbots from one factory
pirate_chat = create_chatbot("You speak like a pirate. Arrr!")
scientist_chat = create_chatbot("You are a formal scientist.")

print(pirate_chat("What is gravity?"))
print(scientist_chat("What is gravity?"))
print(pirate_chat("Tell me more"))      # Remembers pirate context
print(scientist_chat("Tell me more"))   # Remembers scientist context
```

Key patterns:
- **Closure** — `history = []` lives inside `create_chatbot`; each chatbot gets isolated state
- **Factory** — one function creates unlimited specialized chatbots
- **`MessagesPlaceholder`** — injects growing history between system and current message

---

## Common Mistakes

### Mistake 1: Forgetting to pass history
```python
# ❌ KeyError — 'history' is required
chain.invoke({"input": "Hello"})

# ✅ Pass empty list or use optional=True
chain.invoke({"history": [], "input": "Hello"})
```

### Mistake 2: Strings instead of Message objects
```python
# ❌ History must be Message objects
chain.invoke({"history": ["Hello", "Hi!"], "input": "..."})

# ✅ Use typed messages
chain.invoke({"history": [HumanMessage(content="Hello"), AIMessage(content="Hi!")], "input": "..."})
```

### Mistake 3: Wrong message order
```python
# ❌ History after current question
[("system", "..."), ("human", "{input}"), MessagesPlaceholder("history")]

# ✅ History before current question
[("system", "..."), MessagesPlaceholder("history"), ("human", "{input}")]
```

---

## Interview Preparation

### Easy
**Q: What is `MessagesPlaceholder`?**
**A:** A template component that accepts a variable-length list of Message objects at runtime. Used for chat history and few-shot examples.

### Medium
**Q: How would you implement few-shot prompting with `ChatPromptTemplate`?**
**A:** Use `MessagesPlaceholder` with a list of alternating `HumanMessage`/`AIMessage` pairs as examples, placed between the system message and the current user question.

### Hard
**Q: How would you implement sliding window memory with `MessagesPlaceholder`?**
**A:** Maintain a message list. Before each call, count tokens with tiktoken. If over budget, keep system message at index 0 and trim oldest user/assistant pairs. Pass trimmed list to `MessagesPlaceholder`.

---

## Summary

| Feature | What It Does |
|---------|-------------|
| `from_messages()` | Create template from tuple list |
| `from_template()` | Shorthand for single human message |
| `MessagesPlaceholder` | Inject variable-length message lists |
| `optional=True` | Make placeholder non-required |
| `.partial()` | Pre-fill template variables |
| Factory functions | Create specialized templates/chains |

---

## What's Next

In the next chapter, we'll master **Structured Output with `with_structured_output()`** — the one-line method that combines everything from Chapter 0.4 (Pydantic) and Chapter 2.4 (Output Parsing) into LangChain's most powerful feature.

> [← Previous: Runnable & LCEL](chapter-15-runnable-lcel.md) | [Next: Structured Output →](chapter-17-structured-output.md)

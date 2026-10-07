# Chapter 1.3: Your First LLM API Call — Talking to GPT with Python

> **Phase 1 — LLM Fundamentals** | [← Previous: Tokens & Tokenization](chapter-07-tokens-tokenization.md) | [Next: Model Parameters →](chapter-09-model-parameters.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Set up an OpenAI API key securely
- ✅ Make your first API call to GPT-4
- ✅ Understand the Chat Completions API (messages, roles)
- ✅ Build a CLI chatbot with conversation memory
- ✅ Implement streaming responses
- ✅ Handle errors and track token usage

| | |
|---|---|
| **Prerequisites** | Chapter 1.1-1.2, Python basics, `pip` |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 60 minutes |

---

## Introduction

### The Problem

You've learned what LLMs are and how tokens work. Now it's time to actually **talk to one** using Python code. But there's a right way and a wrong way to do this — API key security, error handling, and conversation management all matter.

### Why Without LangChain First?

We call the OpenAI API **directly** before using LangChain for a critical reason:

> If you only know LangChain, you're a framework user. If you know what's underneath, you're an **engineer**.

When LangChain breaks or behaves unexpectedly, understanding the raw API lets you debug effectively.

### What You'll Build

1. A single API call to GPT
2. A CLI chatbot with conversation memory
3. A streaming response example

---

## Step 1: Get Your OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to **API Keys** (left sidebar)
4. Click **"Create new secret key"**
5. Copy the key — it starts with `sk-`
6. **⚠️ NEVER share this key or commit it to Git!**

> New accounts typically receive free credits ($5-18), enough for thousands of API calls during this course.

---

## Step 2: Project Setup

### Folder Structure

```
llm-basics/
├── .env              ← API key (NEVER commit!)
├── .gitignore         ← Tells Git to ignore .env
├── requirements.txt   ← Dependencies
└── main.py           ← Your code
```

### `.env` — Your Secrets

```
OPENAI_API_KEY=sk-your-actual-key-here
```

### `.gitignore` — Protect Your Secrets

```
.env
__pycache__/
*.pyc
.venv/
```

### `requirements.txt` — Dependencies

```
openai
python-dotenv
tiktoken
```

### Install

```bash
pip install openai python-dotenv tiktoken
```

---

## Step 3: Your First API Call

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env file
load_dotenv()

# Create the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Make your first API call!
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is LangChain in one sentence?"}
    ],
    temperature=0.7,
    max_tokens=100
)

# Extract the response
answer = response.choices[0].message.content
print(answer)
```

### The `messages` Array — The Most Important Concept

LLMs use 3 types of messages:

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    # System message: Sets the AI's behavior/personality
    # The user never sees this. It's your "instruction manual" for the AI.
    
    {"role": "user",      "content": "What is LangChain?"},
    # User message: The human's input
    
    {"role": "assistant", "content": "LangChain is a framework..."},
    # Assistant message: The AI's previous response
    # Used for conversation history
]
```

```
┌─────────────────────────────────────────────┐
│              MESSAGES ARRAY                  │
├─────────────────────────────────────────────┤
│                                             │
│  system:    "You are a helpful assistant"    │
│             ↳ Personality & rules           │
│                                             │
│  user:      "What is Python?"               │
│             ↳ Human's question              │
│                                             │
│  assistant: "Python is a language..."       │
│             ↳ AI's previous response        │
│                                             │
│  user:      "Tell me more"                  │
│             ↳ Follow-up question            │
│                                             │
│  ALL of this is sent every time!            │
│  The LLM has NO memory between calls.       │
│                                             │
└─────────────────────────────────────────────┘
```

### The Response Object

```python
response = client.chat.completions.create(...)

print(response.choices[0].message.content)  # The text response
print(response.choices[0].message.role)     # "assistant"
print(response.model)                       # "gpt-4o-mini"
print(response.usage.prompt_tokens)         # Input tokens used
print(response.usage.completion_tokens)     # Output tokens used
print(response.usage.total_tokens)          # Total tokens
```

---

## Step 4: Understanding Parameters

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",      # Which model to use
    messages=[...],            # The conversation
    temperature=0.7,           # Randomness (0.0 - 2.0)
    max_tokens=500,            # Max output length
    top_p=1.0,                 # Alternative to temperature
    frequency_penalty=0.0,     # Reduce repetition (-2.0 to 2.0)
    presence_penalty=0.0,      # Encourage new topics (-2.0 to 2.0)
    n=1,                       # Number of responses to generate
)
```

| Parameter | What It Does | Common Value |
|-----------|-------------|-------------|
| `model` | Which LLM to use | `"gpt-4o-mini"` (cheap) or `"gpt-4o"` (best) |
| `temperature` | Randomness | `0` for code, `0.7` for chat |
| `max_tokens` | Output length limit | `500` for short, `4096` for long |
| `top_p` | Nucleus sampling | Usually leave at `1.0` |
| `frequency_penalty` | Reduce word repetition | `0.0` to `0.5` |
| `presence_penalty` | Encourage new topics | `0.0` to `0.5` |

---

## Step 5: Build a CLI Chatbot

A real chatbot that maintains conversation history:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat():
    """A simple CLI chatbot with conversation memory."""
    print("🤖 ChatBot (type 'quit' to exit)")
    print("-" * 40)
    
    # Conversation history — this IS the memory
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Be concise."}
    ]
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break
        
        if not user_input:
            continue
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        try:
            # Call the API with FULL conversation history
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": assistant_message})
            
            tokens_used = response.usage.total_tokens
            print(f"\n🤖: {assistant_message}")
            print(f"   [tokens: {tokens_used}]")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    chat()
```

### How Conversation Memory Works

```
Turn 1:
    messages = [system, user("Hi")]
    → API returns "Hello!"
    messages = [system, user("Hi"), assistant("Hello!")]

Turn 2:
    messages = [system, user("Hi"), assistant("Hello!"), user("What's Python?")]
    → API returns "Python is..."
    messages = [system, user("Hi"), assistant("Hello!"),
                user("What's Python?"), assistant("Python is...")]

Turn 3:
    ENTIRE history sent every time!
    This is why long conversations cost more tokens.
```

**Key insight:** The LLM has **NO memory**. We simulate memory by sending the **entire conversation** with every request.

---

## Step 6: Streaming Responses

Using the generator pattern from Chapter 0.3:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Write a short poem about Python programming."}
    ],
    stream=True  # ← Enables streaming!
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)

print()
```

This uses the same pattern as your `stream_words()` generator — tokens arrive one at a time and are printed immediately.

---

## Common Mistakes

### Mistake 1: Hardcoding the API key

```python
# ❌ Key gets committed to Git!
client = OpenAI(api_key="sk-abc123...")

# ✅ Use environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### Mistake 2: Not sending conversation history

```python
# ❌ No memory
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me more"}]
)
# "Tell me more about what?" — No context!

# ✅ Send full history
messages.append({"role": "user", "content": "Tell me more"})
response = client.chat.completions.create(messages=messages)
```

### Mistake 3: No error handling

```python
# ❌ Crashes on API errors
response = client.chat.completions.create(...)

# ✅ Handle errors
try:
    response = client.chat.completions.create(...)
except Exception as e:
    print(f"Error: {e}")
```

### Mistake 4: Forgetting to add assistant response to history

```python
# ❌ Only adds user messages — AI loses context of its own responses
messages.append({"role": "user", "content": user_input})
response = client.chat.completions.create(messages=messages)
# Forgot to append assistant message!

# ✅ Add both sides
messages.append({"role": "user", "content": user_input})
response = client.chat.completions.create(messages=messages)
messages.append({"role": "assistant", "content": response.choices[0].message.content})
```

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Use `.env` + `python-dotenv` for API keys | Security — never hardcode secrets |
| Add `.env` to `.gitignore` | Prevents accidental key exposure |
| Use `gpt-4o-mini` for development | 10x cheaper than `gpt-4o` |
| Track `response.usage` | Monitor costs in production |
| Handle errors with try/except | APIs fail — rate limits, network issues |
| Start with low `max_tokens` while testing | Saves money during development |
| Always add both user AND assistant messages to history | Maintains proper conversation context |

---

## Interview Preparation

### Easy

**Q: What are the three message roles in the Chat Completions API?**

**A:** `system` (sets AI behavior/personality — invisible to user), `user` (human input), `assistant` (AI response). All three are sent in the `messages` array with every API call.

### Medium

**Q: How does conversation memory work with the OpenAI API?**

**A:** The API is **stateless** — it has no memory between calls. Memory is simulated by maintaining a `messages` list on the client side and sending the **entire conversation history** with every API request. This is why long conversations use more tokens and cost more.

### Hard

**Q: What is the difference between `temperature` and `top_p`?**

**A:** Both control output randomness but differently. `temperature` scales the logit scores before softmax — low temperature makes the distribution sharp (deterministic), high temperature makes it flat (random). `top_p` (nucleus sampling) only considers tokens whose cumulative probability reaches `p` — e.g., `top_p=0.1` means only the top 10% of probability mass is considered. OpenAI recommends adjusting one, not both.

### Senior

**Q: How would you handle the context window limit in a long-running chatbot?**

**A:** Strategies: (1) **Sliding window** — drop oldest messages when approaching the limit. (2) **Summarization** — periodically summarize old messages and replace them with the summary. (3) **Token counting** — use `tiktoken` to count before each call and truncate if needed. (4) **LangChain memory systems** automate these strategies (ConversationBufferWindowMemory, ConversationSummaryMemory).

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| `.env` + `python-dotenv` | Secure API key management |
| `client.chat.completions.create()` | The core API call |
| `messages` array | System + User + Assistant messages sent every call |
| `response.choices[0].message.content` | Extract the AI's response |
| `response.usage` | Token usage tracking |
| `stream=True` | Enable token-by-token streaming |
| Conversation memory | Client-side list — entire history sent each call |

---

## Cheat Sheet

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

# SETUP
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# BASIC CALL
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=500
)
answer = response.choices[0].message.content
tokens = response.usage.total_tokens

# STREAMING
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# CONVERSATION MEMORY
messages = [{"role": "system", "content": "..."}]
messages.append({"role": "user", "content": user_input})
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
messages.append({"role": "assistant", "content": response.choices[0].message.content})
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| How to securely store API keys? | `.env` file + `python-dotenv` + `.gitignore` |
| What are the 3 message roles? | `system`, `user`, `assistant` |
| Does the API remember previous conversations? | No — send full history each call |
| How to get the response text? | `response.choices[0].message.content` |
| How to enable streaming? | `stream=True` parameter |
| How to track token usage? | `response.usage.total_tokens` |
| Cheapest good model? | `gpt-4o-mini` |
| What does the system message do? | Sets AI behavior/personality, invisible to user |

---

## Homework

1. **Build:** Create the CLI chatbot and have a 5-turn conversation
2. **Experiment:** Try different system prompts (Python tutor, pirate, Shakespearean English)
3. **Observe:** Watch how `total_tokens` grows with each turn
4. **Stream:** Modify the chatbot to use streaming responses
5. **Cost track:** Print the estimated cost after each message

---

## Additional Resources

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)
- [OpenAI Cookbook](https://cookbook.openai.com/)
- [OpenAI Pricing](https://openai.com/pricing)
- [python-dotenv Documentation](https://pypi.org/project/python-dotenv/)

---

## What's Next

In the next chapter, we'll explore **Model Parameters in depth** — understanding temperature, top_p, frequency_penalty, and presence_penalty through hands-on experiments. You'll build a parameter playground that lets you see exactly how each parameter affects LLM output, preparing you for Phase 2: Prompt Engineering.

> [← Previous: Tokens & Tokenization](chapter-07-tokens-tokenization.md) | [Next: Model Parameters →](chapter-09-model-parameters.md)

# Chapter 1.2: Tokens & Tokenization — The Language LLMs Actually Speak

> **Phase 1 — LLM Fundamentals** | [← Previous: What is an LLM?](chapter-06-what-is-an-llm.md) | [Next: Your First LLM API Call →](chapter-08-first-api-call.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand what tokens are and why LLMs use them
- ✅ Tokenize text using `tiktoken` (OpenAI's tokenizer)
- ✅ Count tokens and estimate API costs
- ✅ Understand tokenization gotchas that affect real applications
- ✅ Build a token analyzer tool

| | |
|---|---|
| **Prerequisites** | Chapter 1.1 (What is an LLM) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 40 minutes |

---

## Introduction

### The Problem

Every LLM interaction involves tokens:
- **Billing** is per token (more tokens = more money)
- **Context window** limits are in tokens (not words or characters)
- **Document splitting** for RAG requires token-aware chunking
- **Debugging** "context length exceeded" errors requires counting tokens

If you can't count tokens, you can't control costs, manage context limits, or debug your LangChain applications.

### The Solution

Learn tokenization — how text is converted to numbers that LLMs process. Use `tiktoken` (OpenAI's tokenizer library) to count tokens programmatically.

### Industry Usage

- **Cost estimation** — Calculate API costs before sending requests
- **Context management** — Ensure prompts + history fit within limits
- **Document chunking** — Split documents at token boundaries for RAG
- **Rate limiting** — Track token usage across API calls
- **Monitoring** — Dashboard token consumption in production

---

## Mental Model

### The Translation Analogy

Think of tokenization as **translation**:

```
Human language:    "Hello, LangChain!"
Token language:    [9906, 11, 23988, 19281, 0]
```

The LLM is a foreigner that only speaks "token language." The tokenizer is the translator that converts back and forth. The model never sees your text — it only sees numbers.

---

## Theory

### What is Tokenization?

Tokenization converts text into **token IDs** (numbers the model understands):

```
Text:       "Hello, LangChain!"
Tokens:     ["Hello", ",", " Lang", "Chain", "!"]
Token IDs:  [9906, 11, 23988, 19281, 0]
```

### Why Not Characters?

```
Character-level:  "Hello" → ['H','e','l','l','o']  = 5 steps
Token-level:      "Hello" → ['Hello']              = 1 step
```

Characters are too granular — the model would need many more steps to process the same text, making it slow and expensive.

### Why Not Whole Words?

```
Word-level problems:
    "don't"       → How to split? Is it one word or two?
    "unhappiness" → One token for a rare word? Wasteful vocabulary
    "asdfghjkl"   → Unknown word! Can't process at all
    "🚀"          → Not a word!
    "Python3.12"  → One token? Multiple?
```

Whole words can't handle the infinite variety of human text. Tokens (subword units using BPE — Byte Pair Encoding) handle ALL cases.

### The Rule of Thumb

```
1 token ≈ 4 characters ≈ 0.75 words

"LangChain is amazing" = 4 tokens ≈ 16 characters ≈ 3 words
```

---

## Code Examples

### Setting Up tiktoken

```bash
pip install tiktoken
```

### Basic Tokenization

```python
import tiktoken

# Get the tokenizer for GPT-4o
encoder = tiktoken.encoding_for_model("gpt-4o")

text = "LangChain is an amazing framework for building AI applications."

# Encode: text → token IDs
tokens = encoder.encode(text)
print(f"Text: {text}")
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")

# Decode: token IDs → text
decoded = encoder.decode(tokens)
print(f"Decoded: {decoded}")
```

**Output:**
```
Text: LangChain is an amazing framework for building AI applications.
Token IDs: [28597, 26009, 374, 459, 8056, 12914, 369, 4857, 15592, 8522, 13]
Token count: 11
Decoded: LangChain is an amazing framework for building AI applications.
```

### Seeing Individual Tokens

```python
import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")
text = "Hello, LangChain! Let's build AI apps."

tokens = encoder.encode(text)

# Decode each token individually
for token_id in tokens:
    token_text = encoder.decode([token_id])
    print(f"  ID: {token_id:>6} → '{token_text}'")
```

### Cost Calculator

```python
import tiktoken

def estimate_cost(
    prompt: str,
    expected_response_words: int = 100,
    model: str = "gpt-4o",
    input_cost_per_million: float = 2.50,
    output_cost_per_million: float = 10.00
) -> dict:
    """Estimate the cost of an LLM API call."""
    encoder = tiktoken.encoding_for_model(model)
    
    input_tokens = len(encoder.encode(prompt))
    output_tokens = int(expected_response_words / 0.75)
    
    input_cost = (input_tokens / 1_000_000) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000) * output_cost_per_million
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": f"${input_cost:.6f}",
        "output_cost": f"${output_cost:.6f}",
        "total_cost": f"${input_cost + output_cost:.6f}",
    }

result = estimate_cost(
    prompt="Explain quantum computing in simple terms.",
    expected_response_words=200
)
for key, value in result.items():
    print(f"  {key}: {value}")
```

### Production-Quality Text Analyzer

```python
import tiktoken

def analyze_text(text: str) -> dict:
    """
    Analyze text for token count, character count, and estimated cost.
    
    Uses GPT-4o tokenizer with fallback to o200k_base encoding.
    """
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")

    tokens = encoding.encode(text)
    token_count = len(tokens)
    char_count = len(text)
    ratio = round(char_count / token_count, 2) if token_count > 0 else 0.0
    estimated_cost_input = round((token_count / 1_000_000) * 2.50, 8)

    return {
        "text": text,
        "token_count": token_count,
        "char_count": char_count,
        "ratio": ratio,
        "estimated_cost_input": estimated_cost_input,
    }

# Test
result = analyze_text("LangChain makes building AI applications easy and fun!")
for key, value in result.items():
    print(f"  {key}: {value}")
```

Note the production-quality touches:
- **Try/except with fallback** — handles cases where the model name isn't recognized
- **Division by zero guard** — `if token_count > 0`
- **Type hints** — `text: str` and `-> dict`

---

## Tokenization Gotchas

### Different Models = Different Tokenizers

```python
import tiktoken

text = "Hello world"

gpt4_enc = tiktoken.encoding_for_model("gpt-4o")
gpt3_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

print(f"GPT-4o tokens: {len(gpt4_enc.encode(text))}")
print(f"GPT-3.5 tokens: {len(gpt3_enc.encode(text))}")
# May differ! Always use the correct model's tokenizer
```

### Spaces Matter

```python
encoder = tiktoken.encoding_for_model("gpt-4o")

print(len(encoder.encode("hello")))    # Different count
print(len(encoder.encode(" hello")))   # Leading space = different token!
```

### Code Uses More Tokens Than English

```python
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
english = "Calculate the nth Fibonacci number using recursion."

encoder = tiktoken.encoding_for_model("gpt-4o")
print(f"Code tokens: {len(encoder.encode(code))}")
print(f"English tokens: {len(encoder.encode(english))}")
# Code typically uses MORE tokens per "concept"
```

### Non-English Text Uses More Tokens

```python
encoder = tiktoken.encoding_for_model("gpt-4o")

english = "Hello, how are you?"
hindi = "नमस्ते, आप कैसे हैं?"
chinese = "你好，你好吗？"

print(f"English: {len(encoder.encode(english))} tokens")
print(f"Hindi: {len(encoder.encode(hindi))} tokens")
print(f"Chinese: {len(encoder.encode(chinese))} tokens")
# Non-English typically uses more tokens = higher cost
```

---

## Common Mistakes

### Mistake 1: Counting words instead of tokens

```python
# ❌ Wrong — words ≠ tokens
text = "don't you think it's great?"
word_count = len(text.split())  # 6 words
# But "don't" → ["don", "'t"] = 2 tokens!

# ✅ Always use tiktoken
token_count = len(encoder.encode(text))  # Accurate
```

### Mistake 2: Using the wrong model's tokenizer

```python
# ❌ GPT-3.5 tokenizer for GPT-4 text
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
tokens = encoder.encode(text)  # Wrong count for GPT-4!

# ✅ Match the tokenizer to your model
encoder = tiktoken.encoding_for_model("gpt-4o")
```

### Mistake 3: Forgetting chat message overhead

```python
# ❌ Only counting the user message
tokens = len(encoder.encode("Hello"))  # 1 token

# ✅ Chat API adds overhead per message (~4 tokens)
# System prompt + message formatting adds tokens!
# Actual tokens used > just the text content
```

---

## Best Practices

| Practice | Reason |
|----------|--------|
| Always count tokens, never estimate from words | Accurate billing and context management |
| Use `tiktoken.encoding_for_model()` | Matches the exact tokenizer used by that model |
| Add error handling with fallback encoding | Production resilience |
| Account for message formatting overhead | Chat APIs add ~4 tokens per message |
| Monitor token usage in production | Cost control and debugging |
| Cache tokenizer instances | Creating encoders is expensive — reuse them |

---

## Interview Preparation

### Easy

**Q: What is a token?**

**A:** A token is a small unit of text (subword piece) that LLMs process — typically about 4 characters or 0.75 words. LLMs operate on token IDs (integers), not raw text. The tokenizer converts between text and token IDs.

### Medium

**Q: Why do LLM APIs charge separately for input and output tokens?**

**A:** Input tokens are processed in parallel through the model (one forward pass for all input). Output tokens are generated sequentially — each output token requires its own forward pass through the network, making output generation more compute-intensive and thus more expensive.

### Hard

**Q: Why might the same text have different token counts on different models?**

**A:** Different models use different tokenizers with different vocabularies. GPT-4o uses `o200k_base` (200K vocabulary), while GPT-3.5 uses `cl100k_base` (100K vocabulary). The vocabulary size, BPE merge rules, and training data all affect how text is split into tokens.

---

## Summary

| Concept | Key Takeaway |
|---------|-------------|
| Token | Subword unit, ~4 characters, ~0.75 words |
| Tokenizer | Converts text ↔ token IDs |
| `tiktoken` | OpenAI's tokenizer library |
| `encoding_for_model()` | Gets the correct tokenizer for a specific model |
| Token cost | Input: ~$2.50/1M, Output: ~$10/1M (GPT-4o) |
| Gotchas | Spaces matter, code costs more, non-English costs more |

---

## Cheat Sheet

```python
import tiktoken

# GET TOKENIZER
enc = tiktoken.encoding_for_model("gpt-4o")

# ENCODE (text → tokens)
tokens = enc.encode("Hello world")

# DECODE (tokens → text)
text = enc.decode(tokens)

# COUNT TOKENS
count = len(enc.encode("some text"))

# COST ESTIMATE
cost = (count / 1_000_000) * price_per_million

# FALLBACK PATTERN
try:
    enc = tiktoken.encoding_for_model("gpt-4o")
except KeyError:
    enc = tiktoken.get_encoding("o200k_base")

# RULE OF THUMB
# 1 token ≈ 4 characters ≈ 0.75 words
# 1,000 tokens ≈ 750 words ≈ 1.5 pages
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| What is a token? | Subword text unit, ~4 characters |
| How to count tokens in Python? | `len(tiktoken.encoding_for_model("gpt-4o").encode(text))` |
| 1 token ≈ ? characters | ~4 characters |
| 1 token ≈ ? words | ~0.75 words |
| Why not tokenize by characters? | Too many steps, slow and expensive |
| Why not tokenize by words? | Can't handle unknown words, contractions, emoji |
| Do different models use different tokenizers? | Yes — always match tokenizer to model |
| What library does OpenAI use for tokenization? | `tiktoken` |

---

## Homework

1. **Install:** `pip install tiktoken`
2. **Build:** Create the `analyze_text()` function and test with various inputs
3. **Experiment:** Compare token counts for English vs code vs non-English text
4. **Explore:** Visit [platform.openai.com/tokenizer](https://platform.openai.com/tokenizer) and try different inputs

---

## Additional Resources

- [OpenAI Tokenizer Tool (Web)](https://platform.openai.com/tokenizer)
- [tiktoken GitHub Repository](https://github.com/openai/tiktoken)
- [OpenAI Pricing Page](https://openai.com/pricing)
- [Hugging Face Tokenizers Library](https://huggingface.co/docs/tokenizers/)

---

## What's Next

In the next chapter, you'll make your **first real LLM API call** using Python and the OpenAI SDK. You'll set up your API key, send messages to GPT-4, and build a working CLI chatbot — all without LangChain, so you understand what LangChain abstracts away.

> [← Previous: What is an LLM?](chapter-06-what-is-an-llm.md) | [Next: Your First LLM API Call →](chapter-08-first-api-call.md)

# Chapter 2.2: Few-Shot Prompting & Chain-of-Thought (CoT)

> **Phase 2 — Prompt Engineering** | [← Previous: Messages](chapter-10-messages.md) | [Next: Prompt Templates & Variables →](chapter-12-prompt-templates.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand Zero-Shot vs Few-Shot prompting
- ✅ Learn Chain-of-Thought (CoT) prompting for complex reasoning
- ✅ Combine Few-Shot + CoT for maximum accuracy
- ✅ Apply formatting constraints to guarantee output formats
- ✅ Know when to use each technique

| | |
|---|---|
| **Prerequisites** | Chapter 2.1 (Messages) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 45 minutes |

---

## Introduction

### The Problem

LLMs are statistical next-token predictors. For simple tasks (conversation, summarization), they work well with zero guidance. But for **complex tasks** (multi-step logic, math, classification, structured outputs), they often fail:

**Failure 1 — Classification without examples:**
```
User: "Classify this ticket: 'My screen is blue'."
AI: "This sounds like a hardware issue related to your display..."
    ← Long essay instead of a label like [HARDWARE]
```

**Failure 2 — Math without reasoning:**
```
User: "A jug holds 5 liters. A cup holds 300 ml. Pour 3 cups in. Space left?"
AI: "4.1 liters"  ← WRONG! (correct: 4.1 liters... actually right, but often wrong on harder problems)
```

### The Solution

- **Few-Shot Prompting** — Show examples of correct input→output pairs
- **Chain-of-Thought (CoT)** — Force the model to reason step-by-step before answering

---

## Mental Model

### Zero-Shot = "Wing it"
> Ask a student to write a business proposal without showing them what one looks like. They'll guess.

### Few-Shot = "Here are examples"
> Show the student 3 perfect business proposals from last year. Now they know the exact format, tone, and structure expected.

### Chain-of-Thought = "Show your work"
> The math teacher says: "Don't just write the answer. Write out every step." This forces the brain (and the model) to process logic linearly, reducing mistakes.

---

## Theory

### 1. Zero-Shot Prompting

Ask the model directly without any examples:

```python
messages = [
    {"role": "user", "content": "Classify the sentiment of: 'This movie was okay.'"}
]
# Model might respond: "The sentiment is neutral" or a long explanation
```

**Works for:** Simple, well-understood tasks
**Fails for:** Ambiguous tasks, specific output formats, domain-specific logic

### 2. Few-Shot Prompting

Provide examples of inputs and expected outputs before the actual query:

```python
messages = [
    {"role": "system", "content": "You are a sentiment classifier."},
    {"role": "user", "content": 'Review: "I loved this film!" -> Sentiment: Positive'},
    {"role": "assistant", "content": "Understood."},
    {"role": "user", "content": 'Review: "Terrible acting." -> Sentiment: Negative'},
    {"role": "assistant", "content": "Understood."},
    {"role": "user", "content": 'Review: "This movie was okay." -> Sentiment:'},
]
# Model sees the pattern and outputs: "Neutral"
```

**Works for:** Classification, formatting, domain-specific patterns
**Key insight:** The model learns the OUTPUT FORMAT from examples, not just the task

### 3. Chain-of-Thought (CoT) Prompting

Instruct the model to reason step-by-step before producing the final answer:

```python
messages = [
    {"role": "user", "content": """
A jug holds 5 liters. A cup holds 300 ml. 
If I pour 3 cups into the jug, how much space is left?

Let's think step by step:
"""}
]
```

**Model output with CoT:**
```
Step 1: Convert cup volume to liters: 300 ml = 0.3 liters
Step 2: Calculate total poured: 3 × 0.3 = 0.9 liters
Step 3: Calculate remaining space: 5.0 - 0.9 = 4.1 liters
Answer: 4.1 liters
```

**Without CoT:** The model tries to jump directly to the answer and often fails.

**Why it works:**
```
Without CoT:  [Query] ──────────────────▶ [Final Answer] (often wrong)

With CoT:     [Query] ──▶ [Step 1] ──▶ [Step 2] ──▶ [Final Answer] (accurate)
```

Each generated token becomes context for the next token. The model uses its own reasoning as a scratchpad.

### 4. Combined Few-Shot + CoT

The most powerful technique — provide examples that INCLUDE step-by-step reasoning:

```python
SYSTEM_PROMPT = """You are a support ticket classification assistant.
Classify tickets into CATEGORY and PRIORITY.
Show your reasoning first, then output the final classification."""

FEW_SHOT_PROMPT = """Classify the following support tickets.

Ticket: "My laptop won't turn on, and the charger light is off."
Reasoning:
1. The user mentions the device won't power up.
2. The charger light is off, suggesting a physical power or adapter issue.
3. This prevents work, making it high priority.
Category: HARDWARE
Priority: HIGH
---
Ticket: "How do I change my password on the portal?"
Reasoning:
1. The user is asking a 'how-to' question about account settings.
2. This does not block their main work.
3. Category is account administration.
Category: ACCOUNT
Priority: LOW
---
Ticket: "The payroll system is down for all employees."
Reasoning:
1. The entire payroll system is unreachable.
2. It impacts all employees, blocking business operations.
3. Category is critical software infrastructure.
Category: SOFTWARE
Priority: CRITICAL
---
Ticket: "{ticket}"
Reasoning:"""
```

---

## Practical Example: Support Ticket Classifier

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a support ticket classification assistant.
Your task is to classify tickets into CATEGORY and PRIORITY.
You must show your reasoning first, and then output the final classification.
Follow the exact formatting of the examples provided by the user."""

FEW_SHOT_PROMPT = """Classify the following support tickets.

Ticket: "My laptop won't turn on, and the charger light is off."
Reasoning:
1. The user mentions the device won't power up.
2. The charger light is off, suggesting a physical power or adapter issue.
3. This prevents work, making it high priority.
Category: HARDWARE
Priority: HIGH
---
Ticket: "How do I change my password on the portal?"
Reasoning:
1. The user is asking a 'how-to' question about account settings.
2. This does not block their main work, nor is it system-wide.
3. Category is account administration.
Category: ACCOUNT
Priority: LOW
---
Ticket: "The payroll system is down for all employees. Nobody can submit timesheets."
Reasoning:
1. The entire payroll system is unreachable.
2. It impacts all employees, blocking business operations.
3. Category is critical software infrastructure.
Category: SOFTWARE
Priority: CRITICAL
---
Ticket: "{ticket}"
Reasoning:"""


def classify_ticket(ticket_text: str) -> str:
    """Classify a support ticket using few-shot + CoT prompting."""
    user_prompt = FEW_SHOT_PROMPT.format(ticket=ticket_text)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,  # Deterministic for classification
        max_tokens=300
    )
    
    return response.choices[0].message.content


# Test with an unseen ticket
new_ticket = "I am getting a 404 error when trying to log into the database dashboard."
result = classify_ticket(new_ticket)
print(result)
```

---

## When to Use Each Technique

| Technique | Best For | Example Tasks |
|-----------|----------|---------------|
| **Zero-Shot** | Simple, well-defined tasks | "Translate X to Spanish", "Summarize this" |
| **Few-Shot** | Format-specific, classification, domain tasks | Sentiment labels, ticket routing, entity extraction |
| **CoT** | Multi-step reasoning, math, logic | Word problems, code debugging, complex analysis |
| **Few-Shot + CoT** | Complex tasks needing both format AND reasoning | Structured classification with justification, multi-step data extraction |

---

## Common Mistakes

### Mistake 1: Bad or inconsistent examples
If your few-shot examples have typos, inconsistent formatting, or wrong classifications, the model will copy those mistakes. Examples are templates — make them perfect.

### Mistake 2: Too many examples
More isn't always better. 3-5 examples usually suffice. Too many waste tokens and can confuse the model if they're contradictory.

### Mistake 3: Missing delimiters between examples
Without clean separators (like `---` or numbered blocks), the model may confuse where examples end and the real query begins.

### Mistake 4: Using CoT for simple tasks
Chain-of-thought adds tokens (cost) and latency. Don't use it for trivial tasks like "What's the capital of France?"

---

## Interview Preparation

### Easy
**Q: What is the difference between zero-shot and few-shot prompting?**
**A:** Zero-shot asks the model to perform a task with no examples. Few-shot provides several input→output examples before the actual query, allowing the model to learn the expected pattern and format from context.

### Medium
**Q: How does Chain-of-Thought prompting improve accuracy?**
**A:** CoT forces the model to generate intermediate reasoning steps before the final answer. Each generated token becomes part of the context for subsequent tokens, effectively giving the model a "scratchpad" to work through logic sequentially instead of trying to jump directly to the answer.

### Hard
**Q: When would Few-Shot + CoT combined be better than either alone?**
**A:** When you need both a specific output format AND complex reasoning. For example, classifying support tickets into structured categories with justified reasoning — few-shot teaches the format, CoT ensures the logic is sound. Either technique alone would miss one aspect.

---

## Summary

| Technique | What It Does | Token Cost | When to Use |
|-----------|-------------|-----------|-------------|
| Zero-Shot | Direct question, no examples | Low | Simple, well-defined tasks |
| Few-Shot | Provides examples before query | Medium | Format-specific, classification |
| CoT | Forces step-by-step reasoning | Medium-High | Math, logic, complex analysis |
| Few-Shot + CoT | Examples with reasoning steps | High | Complex tasks needing format + logic |

---

## Flashcards

| Question | Answer |
|----------|--------|
| What is zero-shot? | Asking the model with no examples |
| What is few-shot? | Providing examples before the query |
| What phrase triggers CoT? | "Let's think step by step" |
| Why does CoT work? | Generated reasoning tokens become context for subsequent tokens |
| How many examples for few-shot? | 3-5 is usually optimal |
| When NOT to use CoT? | Simple factual lookups (wastes tokens) |
| What is the magic of combining both? | Few-shot teaches format, CoT ensures reasoning accuracy |

---

## Homework

1. **Build:** Write a few-shot prompt that converts informal dates to ISO format
2. **Experiment:** Compare zero-shot vs few-shot accuracy on 5 classification examples
3. **Reading:** [Google Research — Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)

---

## What's Next

In the next chapter, we'll learn about **Prompt Templates & Variables** — how to create reusable, parameterized prompts instead of hardcoding strings. This is the bridge between raw prompt engineering and LangChain's `PromptTemplate` system.

> [← Previous: Messages](chapter-10-messages.md) | [Next: Prompt Templates & Variables →](chapter-12-prompt-templates.md)

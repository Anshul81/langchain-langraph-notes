# Chapter 1.4: Model Parameters & the Parameter Playground

> **Phase 1 — LLM Fundamentals** | [← Previous: Your First LLM API Call](chapter-08-first-api-call.md) | [Next: Phase 2 — System / User / Assistant Messages →](chapter-10-messages.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand every critical LLM parameter deeply (temperature, top_p, frequency_penalty, presence_penalty, etc.)
- ✅ See visually and programmatically how each parameter changes the model's output
- ✅ Build a Parameter Playground tool to compare settings side by side
- ✅ Use a decision framework to choose the correct settings for any practical use case

| | |
|---|---|
| **Prerequisites** | Chapter 1.3 (Your First LLM API Call) |
| **Estimated Reading Time** | 20 minutes |
| **Estimated Coding Time** | 40 minutes |

---

## Introduction

### The Problem

Choosing the wrong parameters is like cooking at the wrong temperature — the ingredients are the same, but the result is ruined.
- A chatbot with a `temperature` of 1.5 will hallucinate wildly and make up facts.
- A creative writing tool with a `temperature` of 0.0 will be boring, repetitive, and sound like a robot.
- A code generator with too low a `max_tokens` limit will cut off mid-sentence, leaving syntax errors.

To build production-grade LLM applications, you cannot rely on default settings. You need to know exactly how to tune the model for your specific task.

### The Solution

We will dissect the core parameters of the OpenAI Chat Completions API. You will see how they affect the probability distribution of tokens and how to set them for tasks like coding, chatting, summarization, and data extraction.

---

## Mental Model

### The Probability Wheel

Think of the LLM's next-token selection as spinning a **Probability Wheel**.

Every time the model generates a word, it calculates a list of candidate words and assigns each a probability. The size of each slice on the wheel corresponds to its probability.

```
temperature = 0.0 (Deterministic)
┌─────────────────────────────────────────┐
│                 [PARIS]                 │  ← The wheel is 99% "Paris"
│                                         │  ← It will always land on Paris
│                                         │
└─────────────────────────────────────────┘

temperature = 0.7 (Balanced)
┌───────────────────────────┬─────────────┐
│          [PARIS]          │   [LYON]    │  ← Slices are distributed
│            60%            │     25%     │  ← Mostly Paris, but Lyon has a chance
├───────────────────────────┴─────────────┤
│         [ROME] 10% | other 5%           │
└─────────────────────────────────────────┘

temperature = 1.8 (Creative/Random)
┌───────┬───────┬───────┬───────┬─────────┐
│[PARIS]│[LYON] │[ROME] │ [THE] │ [DANCE] │  ← Almost equal slices
│  20%  │  20%  │  20%  │  20%  │   20%   │  ← Can choose highly unexpected tokens!
└───────┴───────┴───────┴───────┴─────────┘
```

---

## Theory & Parameters Deep Dive

### Parameter 1: `model` — The Engine

The choice of model is your primary decision. It determines the base capability, context size, and cost of the operations.

- **Fast & Cost-Effective:** `gpt-4o-mini` — Best for simple classification, summarization, extraction, and development/testing.
- **Premium Reasoning:** `gpt-4o` — Best for complex logic, multi-step math, writing high-quality code, or reasoning over large amounts of context.

### Parameter 2: `temperature` — Randomness Control

Temperature scales the raw prediction scores (called *logits*) before they are converted into probabilities using the Softmax function.

- **Low Temperature (0.0 - 0.3):** Sharpens the probability distribution. The highest-probability token gets almost all of the weight. The model becomes focused, factual, and deterministic.
- **Medium Temperature (0.4 - 0.7):** Balanced. It yields fluent, natural conversations with moderate variation.
- **High Temperature (0.8 - 1.2+):** Flattens the probability distribution. Unlikely tokens get a higher chance of being picked. This introduces creativity, brainstorming, and stylistic variation, but increases the risk of hallucination or grammatical collapse.

### Parameter 3: `max_tokens` — The Hard Ceiling

This parameter sets a hard limit on the number of tokens the model can generate in a single response.
- It acts as a safety guard to prevent expensive runaway loops.
- If the limit is reached before the model finishes its thought, the response will cut off mid-sentence (the `finish_reason` in the API response will be `"length"` instead of `"stop"`).
- Always budget enough tokens for your expected output.

### Parameter 4: `top_p` — Nucleus Sampling

`top_p` controls token selection using cumulative probability. Instead of looking at all tokens, the model only considers the top `p` percentage of the probability distribution.

- `top_p = 0.1` — Only the top 10% of likely tokens are considered. Any token outside this range is discarded entirely.
- `top_p = 1.0` — All tokens in the vocabulary are considered (default).

> **⚠️ Critical Rule:** Adjust `temperature` OR `top_p`, never both. Modifying both simultaneously makes the output distribution unpredictable.

### Parameter 5: `frequency_penalty` — Word Repetition Guard

This parameter discourages the model from repeating the same words or phrases. It applies a penalty to a token's probability based on how many times that token has *already appeared* in the output.

- Range: `-2.0` to `2.0` (Default: `0.0`)
- Positive values (e.g., `0.5` to `1.0`) make the model actively search for synonyms and vary its vocabulary.
- Extremely high values (e.g., `2.0`) can cause the model to write weird sentences to avoid common grammatical structure words.

### Parameter 6: `presence_penalty` — Topic Transition Encouragement

Similar to `frequency_penalty`, but the penalty is flat. It doesn't care if a token has appeared 1 time or 100 times; as long as it has appeared *at least once*, it gets penalized.

- Range: `-2.0` to `2.0` (Default: `0.0`)
- Pushes the model to talk about **new things** or explore new topics.
- Ideal for brainstorming tools where you want wide topic coverage.

---

## Code Example: Parameter Playground

Let's build a parameter playground that runs experiments comparing these settings side by side:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def compare_parameters(
    prompt: str,
    configs: list[dict],
    system_prompt: str = "You are a helpful assistant."
) -> None:
    """Run the same prompt with different parameter configs and compare."""
    
    print(f"Prompt: \"{prompt}\"")
    print("=" * 60)
    
    for i, config in enumerate(configs, 1):
        # Extract metadata from config dict
        label = config.pop("label", f"Config {i}")
        model = config.pop("model", "gpt-4o-mini")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                # Extract parameters with defaults
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 150),
                top_p=config.get("top_p", 1.0),
                frequency_penalty=config.get("frequency_penalty", 0.0),
                presence_penalty=config.get("presence_penalty", 0.0),
            )
            
            answer = response.choices[0].message.content
            tokens = response.usage.total_tokens
            
            print(f"\n[{label}]")
            print(f"  Settings: temp={config.get('temperature', 0.7)}, "
                  f"max_tokens={config.get('max_tokens', 150)}, "
                  f"freq_penalty={config.get('frequency_penalty', 0.0)}")
            print(f"  Tokens Used: {tokens}")
            print(f"  Response: {answer.strip()}")
            print("-" * 60)
            
        except Exception as e:
            print(f"\n[{label}] Error: {e}")


if __name__ == "__main__":
    # === EXPERIMENT 1: Temperature Comparison ===
    print("\n🔬 EXPERIMENT 1: Temperature Effect")
    compare_parameters(
        prompt="Write a one-sentence description of Python.",
        configs=[
            {"label": "Temp=0.0 (Deterministic)", "temperature": 0.0},
            {"label": "Temp=0.7 (Balanced)",      "temperature": 0.7},
            {"label": "Temp=1.5 (Creative/Wild)",  "temperature": 1.5},
        ]
    )

    # === EXPERIMENT 2: Frequency Penalty ===
    print("\n🔬 EXPERIMENT 2: Frequency Penalty")
    compare_parameters(
        prompt="List 5 benefits of Python. Start each with the word 'Python'.",
        configs=[
            {"label": "No penalty",   "frequency_penalty": 0.0, "max_tokens": 150},
            {"label": "Penalty=1.5",  "frequency_penalty": 1.5, "max_tokens": 150},
        ]
    )
```

---

## Parameter Selection Guide

Use this chart as your default configuration checklist for various applications:

| Application | Temperature | Max Tokens | Frequency Penalty | Presence Penalty | Rationale |
|-------------|-------------|------------|-------------------|------------------|-----------|
| **Code Generation** | `0.0` | `2000+` | `0.0` | `0.0` | Needs syntax precision and strict logic. |
| **Data Extraction** | `0.0` | `500` | `0.0` | `0.0` | Factual consistency, schema compliance. |
| **SQL Generation** | `0.0` | `300` | `0.0` | `0.0` | Syntax errors cannot be tolerated. |
| **Customer Support Chat** | `0.3` | `300` | `0.0` | `0.0` | Safe, coherent, and consistent branding. |
| **Document Summarization**| `0.3` | `400` | `0.2` | `0.0` | Stays close to facts, but avoids copying sentences. |
| **Creative Writing** | `1.0` | `1000+` | `0.3` | `0.3` | Encourages rich vocabulary and diverse narratives. |
| **Brainstorming / Ideation**| `1.0` | `500` | `0.5` | `0.5` | Forces the model to span new ideas and topics. |

---

## Common Mistakes

### Mistake 1: Setting high temperature for code or structural outputs
Using `temperature=1.0` when writing Python code or returning JSON schemas will result in invalid JSON syntax or code bugs that occur intermittently. Keep it at `0.0`.

### Mistake 2: Restricting output length too aggressively
Setting `max_tokens` too low (e.g., `20`) to save costs, which causes the response to cut off mid-thought. Always leave a buffer.

### Mistake 3: Adjusting both `temperature` and `top_p`
Adjusting both parameters shifts the token distribution vectors unpredictably. Stick to configuring `temperature` for general applications.

---

## Debugging Guide

### Issue: Output cuts off mid-sentence
- **Diagnosis:** Check `response.choices[0].finish_reason`. If it is `"length"`, your `max_tokens` ceiling was hit.
- **Fix:** Increase the value of `max_tokens`.

### Issue: Model keeps repeating the same phrase over and over
- **Diagnosis:** The model has entered a repetition loop (common in smaller models).
- **Fix:** Set `frequency_penalty` to `0.5` or `0.8`.

### Issue: Unpredictable output variations during testing
- **Diagnosis:** Your `temperature` is too high, leading to high variance across runs.
- **Fix:** Set `temperature=0.0` during development to make testing reproducible.

---

## Interview Preparation

### Easy
**Q: What is the effect of setting temperature to 0.0?**
**A:** Setting `temperature=0.0` makes the model deterministic. It will always pick the token with the absolute highest probability at each step, ensuring that identical inputs yield identical outputs.

### Medium
**Q: What is the difference between frequency penalty and presence penalty?**
**A:** `frequency_penalty` penalizes a token based on its *cumulative frequency* of appearance (penalizes more the more times it is repeated). `presence_penalty` penalizes a token with a flat rate as long as it has appeared *at least once* in the output, encouraging topic shifts.

### Hard
**Q: If you have a custom chatbot in production and users complain it is slowly going off-topic in long conversations, which parameters and architectural settings would you look at?**
**A:** First, look at the `presence_penalty` and check if it is accidentally set too high, pushing the model away from the conversational topic. Second, review the sliding context window architecture — if old context is dropped poorly, the model loses the system prompt instructions. Finally, check if `temperature` is too high, allowing drift to accumulate over multiple turns.

---

## Summary

| Parameter | Function | Value Range | Default |
|-----------|----------|-------------|---------|
| `model` | Core LLM engine | string | - |
| `temperature` | Output randomness/creativity | `0.0` to `2.0` | `1.0` |
| `max_tokens` | Limit on generated output | integer | Infinite |
| `top_p` | Cumulative probability filter | `0.0` to `1.0` | `1.0` |
| `frequency_penalty` | Word repetition deterrent | `-2.0` to `2.0` | `0.0` |
| `presence_penalty` | Topic variance promoter | `-2.0` to `2.0` | `0.0` |

---

## Cheat Sheet

```python
# DETERMINISTIC CONFIG (Code, JSON, Facts)
config = {
    "temperature": 0.0,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# BALANCED CONFIG (Chatbot, Q&A)
config = {
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# CREATIVE CONFIG (Writing, Ideation)
config = {
    "temperature": 1.0,
    "max_tokens": 1000,
    "top_p": 1.0,
    "frequency_penalty": 0.3,
    "presence_penalty": 0.3
}
```

---

## Flashcards

| Question | Answer |
|----------|--------|
| What is the maximum value for temperature? | `2.0` (though values > 1.2 are generally incoherent). |
| What parameter limits the exact response size? | `max_tokens`. |
| Should you adjust top_p and temperature together? | No, modify one and leave the other at 1.0. |
| Which penalty depends on cumulative count? | `frequency_penalty`. |
| Which penalty applies a flat rate for any usage? | `presence_penalty`. |
| What model is cheapest for basic tasks? | `gpt-4o-mini`. |

---

## Homework

1. **Build:** Write a script containing the `compare_parameters` function and run it.
2. **Observe:** Run Experiment 3 and compare the output vocabulary of "No penalty" vs "Penalty=1.5".
3. **Reading:** Read the OpenAI API documentation on Chat Completions parameter parameters.

---

## What's Next

**Phase 1 is complete!** You now understand LLM internals, tokenization, basic API usage, and parameter configurations. 

In **Phase 2: Prompt Engineering**, we will learn how to write elite system prompts, handle zero-shot/few-shot prompts, format templates dynamically, and output structured data schemas. This is the last step of engineering before we begin writing core LangChain code.

> [← Previous: Your First LLM API Call](chapter-08-first-api-call.md) | [Next: Phase 2 — System / User / Assistant Messages →](../phase-02-prompt-engineering/chapter-10-messages.md)

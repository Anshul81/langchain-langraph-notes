# Chapter 13.2: LangSmith — Tracing, Debugging & Evaluation at Scale

> **Phase 13 — Production** | [← Previous: Production LangChain](chapter-46-production-langchain.md) | [Next: Capstone Project →](chapter-48-capstone.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand what LangSmith is and why it's essential for production
- ✅ Set up LangSmith tracing for your LangChain application
- ✅ Use the LangSmith dashboard to debug chain/agent runs
- ✅ Create datasets and run evaluations in LangSmith
- ✅ Monitor latency, cost, and error rates in production
- ✅ Set up feedback collection and annotation

| | |
|---|---|
| **Prerequisites** | Chapter 13.1 (Production LangChain) |
| **Estimated Reading Time** | 25 minutes |
| **Estimated Coding Time** | 40 minutes |

---

## Introduction — Why LangSmith?

LLM applications are **non-deterministic** — the same input can produce different outputs. This makes debugging incredibly hard:

```
WITHOUT LANGSMITH:
  User: "My RAG answer was wrong."
  You:  "Hmm... let me check the logs... which part failed? 
         Was it retrieval? The prompt? The LLM? 
         I have no idea. Let me add some print statements..." 😫

WITH LANGSMITH:
  User: "My RAG answer was wrong."
  You:  *Opens LangSmith, finds the trace*
        "I can see exactly what happened:
         1. Retrieved 5 chunks — chunk #3 was irrelevant
         2. The prompt looked good
         3. The LLM hallucinated based on chunk #3
         Fix: Better retrieval filtering" ✅
```

**LangSmith** is a platform by LangChain for **tracing, debugging, evaluating, and monitoring** LLM applications.

---

## Part 1: Setup

### Getting Your API Key

```
1. Go to https://smith.langchain.com/
2. Sign up (free tier available)
3. Get your API key from Settings
```

### Environment Variables

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=my-rag-app    # Project name in LangSmith
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### That's It!

```python
import os
from dotenv import load_dotenv
load_dotenv()

# Once the env vars are set, LangChain AUTOMATICALLY sends traces!
# No code changes needed!

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0,
    openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
)

# This call is automatically traced in LangSmith
result = llm.invoke("What is LangSmith?")
print(result.content)

# Go to https://smith.langchain.com/ → your project → see the trace!
```

---

## Part 2: Understanding Traces

### What a Trace Looks Like

```
┌─────────────────────────────────────────────────────────────────┐
│ Trace: RAG Chain                                                 │
│ ID: run_abc123    Duration: 2.3s    Status: ✅ Success           │
│                                                                  │
│ ┌─ RunnableSequence ──────────────────────────── 2.3s ──────┐   │
│ │                                                            │   │
│ │ ┌─ RunnableParallel ───────────────────── 1.1s ─────┐     │   │
│ │ │                                                    │     │   │
│ │ │ ┌─ Retriever ─────────────────── 0.8s ──────┐     │     │   │
│ │ │ │  query: "What is our revenue?"             │     │     │   │
│ │ │ │  results: 5 documents                      │     │     │   │
│ │ │ │  Document 1: "Q3 revenue was ₹4.5 cr..."  │     │     │   │
│ │ │ │  Document 2: "Total ARR is ₹18 crore..."  │     │     │   │
│ │ │ │  Document 3: "SaaS subscriptions: 70%..."  │     │     │   │
│ │ │ └────────────────────────────────────────────┘     │     │   │
│ │ │                                                    │     │   │
│ │ │ ┌─ Passthrough ──────────── 0.001s ──────────┐    │     │   │
│ │ │ │  "What is our revenue?"                     │    │     │   │
│ │ │ └────────────────────────────────────────────┘    │     │   │
│ │ └────────────────────────────────────────────────────┘     │   │
│ │                                                            │   │
│ │ ┌─ ChatPromptTemplate ─────────── 0.001s ─────────────┐   │   │
│ │ │  Input: context + question                            │   │   │
│ │ │  Output: formatted prompt                             │   │   │
│ │ └──────────────────────────────────────────────────────┘   │   │
│ │                                                            │   │
│ │ ┌─ ChatOpenAI (gpt-4o-mini) ──── 1.2s ───────────────┐   │   │
│ │ │  Input: system + human message                       │   │   │
│ │ │  Output: "Q3 2024 revenue was ₹4.5 crore..."        │   │   │
│ │ │  Tokens: prompt=450, completion=85, total=535        │   │   │
│ │ │  Cost: $0.000119                                     │   │   │
│ │ └──────────────────────────────────────────────────────┘   │   │
│ │                                                            │   │
│ │ ┌─ StrOutputParser ──────── 0.001s ───────────────────┐   │   │
│ │ │  "Q3 2024 revenue was ₹4.5 crore ($540K USD)..."     │   │   │
│ │ └──────────────────────────────────────────────────────┘   │   │
│ └────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### What You Can See in Each Trace

| Information | What It Shows |
|-------------|--------------|
| **Run tree** | Visual hierarchy of all steps |
| **Inputs/Outputs** | Exact data at each step |
| **Latency** | Time taken by each component |
| **Token usage** | Prompt/completion tokens per LLM call |
| **Cost** | Estimated cost per LLM call |
| **Errors** | Full stack trace if any step fails |
| **Metadata** | Tags, user ID, session ID |

---

## Part 3: Custom Trace Metadata

### Adding Tags and Metadata

```python
from langchain_core.runnables import RunnableConfig

# Add custom metadata to traces
config = RunnableConfig(
    tags=["production", "user-query", "rag"],
    metadata={
        "user_id": "user-123",
        "session_id": "session-456",
        "environment": "production",
        "version": "1.2.0",
    },
    run_name="Customer Query: Revenue"  # Custom name in LangSmith
)

result = rag_chain.invoke("What is our revenue?", config=config)
```

### Custom Run Names for Debugging

```python
from langchain_core.runnables import RunnableLambda

# Name your chain steps for clearer traces
def retrieve_with_name(query):
    return retriever.invoke(query)

named_retriever = RunnableLambda(retrieve_with_name).with_config(
    run_name="Knowledge Base Retrieval"
)

# In the trace, this step will show as "Knowledge Base Retrieval" 
# instead of "RunnableLambda"
```

---

## Part 4: LangSmith Datasets & Evaluation

### Creating a Dataset

```python
from langsmith import Client

client = Client()  # Uses LANGCHAIN_API_KEY from env

# Create a dataset
dataset = client.create_dataset(
    "RAG Evaluation Set",
    description="Questions about our company knowledge base"
)

# Add examples
examples = [
    {"question": "What is CodeAssist?", "expected": "AI-powered code review tool"},
    {"question": "Q3 2024 revenue?", "expected": "₹4.5 crore"},
    {"question": "Who founded the company?", "expected": "Aarav Patel in 2020"},
    {"question": "Support hours?", "expected": "9 AM - 6 PM IST, Mon-Fri"},
    {"question": "Refund policy for annual?", "expected": "Full refund within 30 days"},
]

for ex in examples:
    client.create_example(
        inputs={"question": ex["question"]},
        outputs={"expected_answer": ex["expected"]},
        dataset_id=dataset.id
    )

print(f"Created dataset with {len(examples)} examples")
```

### Running Evaluations

```python
from langsmith.evaluation import evaluate


# Define your RAG function to evaluate
def predict(inputs: dict) -> dict:
    """Run RAG chain on an input question."""
    answer = rag_chain.invoke(inputs["question"])
    return {"answer": answer}


# Define evaluator (correctness)
def check_correctness(run, example) -> dict:
    """Check if the answer matches expected output."""
    predicted = run.outputs.get("answer", "")
    expected = example.outputs.get("expected_answer", "")
    
    # Simple substring check
    is_correct = expected.lower() in predicted.lower()
    
    return {"key": "correctness", "score": 1 if is_correct else 0}


# Run evaluation
results = evaluate(
    predict,
    data="RAG Evaluation Set",         # Dataset name
    evaluators=[check_correctness],
    experiment_prefix="rag-v1",         # Name this experiment
)

print(f"Results: {results}")
# View detailed results in LangSmith UI → Datasets → RAG Evaluation Set
```

### LLM-as-Judge Evaluator

```python
from langsmith.evaluation import evaluate, LangChainStringEvaluator

# Use LangChain's built-in LLM evaluator
qa_evaluator = LangChainStringEvaluator(
    "qa",                    # Question-answer evaluation
    config={"llm": ChatOpenAI(model="gpt-4o", temperature=0,
        openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
        openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
    )},
    prepare_data=lambda run, example: {
        "prediction": run.outputs["answer"],
        "reference": example.outputs["expected_answer"],
        "input": example.inputs["question"],
    }
)

results = evaluate(
    predict,
    data="RAG Evaluation Set",
    evaluators=[qa_evaluator],
    experiment_prefix="rag-v1-llm-judge",
)
```

---

## Part 5: Production Monitoring

### What to Monitor

```
REAL-TIME METRICS:
├── 📊 Request volume (requests/minute)
├── ⏱️ Latency (P50, P95, P99)
├── 💰 Cost (daily, per request)
├── ❌ Error rate (% of failed requests)
├── 🔄 Token usage (per request, daily total)
└── 📈 User satisfaction (thumbs up/down)

PERIODIC CHECKS:
├── 📋 Evaluation scores (weekly regression tests)
├── 🔍 Retrieval quality (are we finding the right docs?)
├── 🧪 A/B tests (new model vs old model)
└── 📉 Cost trends (are we spending more than expected?)
```

### Setting Up Monitoring with LangSmith

```python
# LangSmith automatically tracks:
# - All LLM calls with latency, tokens, cost
# - Chain/agent runs with full trace
# - Errors with stack traces
# - Custom metadata (user_id, session_id)

# You can filter in the dashboard by:
# - Time range
# - Tags
# - Error status
# - Latency range
# - Token count
# - Custom metadata
```

### Feedback Collection

```python
from langsmith import Client

client = Client()

# After a user interacts with your app:
def submit_feedback(run_id: str, score: float, comment: str = ""):
    """Submit user feedback for a specific run."""
    client.create_feedback(
        run_id=run_id,
        key="user-rating",
        score=score,           # 0 to 1
        comment=comment,
    )

# In your API:
# 1. Return the run_id with the response
# 2. User clicks 👍 or 👎
# 3. Submit feedback with the run_id
# 4. View feedback in LangSmith dashboard
```

---

## Part 6: Comparing Experiments

```python
# Run the same eval dataset with different configurations
# and compare results in LangSmith

# Experiment 1: Baseline
results_v1 = evaluate(
    predict_v1,  # chunk_size=500, k=3, gpt-4o-mini
    data="RAG Evaluation Set",
    experiment_prefix="rag-baseline",
)

# Experiment 2: Larger chunks
results_v2 = evaluate(
    predict_v2,  # chunk_size=1000, k=3, gpt-4o-mini
    data="RAG Evaluation Set",
    experiment_prefix="rag-larger-chunks",
)

# Experiment 3: More retrieval
results_v3 = evaluate(
    predict_v3,  # chunk_size=500, k=7, gpt-4o-mini
    data="RAG Evaluation Set",
    experiment_prefix="rag-more-docs",
)

# Compare all three experiments in the LangSmith UI!
# Dashboard → Datasets → RAG Evaluation Set → Experiments tab
```

---

## Common Mistakes

### Mistake 1: Not enabling tracing
```python
# ❌ No traces — flying blind in production
# (forgot to set environment variables)

# ✅ Set these in .env:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=lsv2_pt_...
# LANGCHAIN_PROJECT=my-app
```

### Mistake 2: Not naming runs and steps
```python
# ❌ All steps show as "RunnableLambda" — impossible to debug

# ✅ Name your steps and runs
config = {"run_name": "Revenue Query", "tags": ["finance"]}
result = chain.invoke(query, config=config)
```

### Mistake 3: Not creating evaluation datasets
```python
# ❌ "I'll test manually" — doesn't scale, not reproducible

# ✅ Build a dataset, run automated evaluations
dataset = client.create_dataset("eval-set")
# Add 30+ examples, run weekly
```

### Mistake 4: Ignoring latency patterns
```python
# ❌ Average latency is 2s — seems fine!
# But P99 is 15s — 1% of users wait 15 seconds!

# ✅ Monitor P50, P95, P99 — not just averages
# Set alerts: P95 > 5s → investigate
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Enable tracing in all environments | Debugging without traces is nearly impossible |
| Add user_id and session_id metadata | Trace user journeys, debug user-specific issues |
| Name your chain steps | Readable traces in the dashboard |
| Build evaluation datasets (30+ questions) | Reproducible, automated quality checks |
| Run evals on every major change | Catch regressions early |
| Collect user feedback (👍/👎) | Real-world quality signal |
| Monitor P95/P99 latency, not just averages | Tail latency affects user experience |
| Compare experiments before deploying | Data-driven decisions |
| Set up alerts on error rate and latency | Catch problems before users report them |

---

## Interview Preparation

### Easy
**Q: What is LangSmith and why is it needed?**

> LangSmith is a platform for tracing, debugging, evaluating, and monitoring LLM applications. It's needed because LLM apps are non-deterministic — the same input can produce different outputs, making traditional debugging insufficient. LangSmith captures every step of a chain/agent run (retrieval, prompting, LLM call, parsing) with full inputs, outputs, latency, token usage, and cost. This makes it possible to pinpoint exactly where a failure occurred.

### Medium
**Q: How would you set up evaluation for a RAG system using LangSmith?**

> (1) Create a dataset of 30-50+ question-answer pairs covering all topic areas. (2) Define evaluator functions — LLM-as-judge for correctness and faithfulness, or custom heuristic evaluators. (3) Run evaluations using `evaluate()` with your RAG chain as the predict function. (4) Compare experiments — run with different configurations (chunk sizes, retrieval k, models) and compare scores in the dashboard. (5) Automate — run evaluations on every deployment and set minimum score thresholds for release.

### Hard
**Q: Design a monitoring strategy for a production RAG system using LangSmith.**

> Multi-layer strategy: (1) **Real-time tracing** — every request traced with user_id, session_id, environment tags. (2) **Alerts** — error rate > 1%, P95 latency > 5s, daily cost > budget. (3) **Weekly regression testing** — automated evaluation against a golden dataset, compare with baseline scores. (4) **User feedback loop** — collect 👍/👎 on every response, review low-rated responses weekly to identify systematic failures. (5) **A/B testing** — route 10% of traffic to new model/config, compare evaluation scores. (6) **Cost tracking** — daily cost breakdown by model, alert on anomalies. (7) **Retrieval monitoring** — sample traces weekly, check if retrieved documents are relevant. (8) **Dashboard** — executive view showing pass rate, avg latency, daily cost, user satisfaction trend.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **LangSmith** | Tracing, debugging, evaluation, and monitoring platform |
| **Trace** | Complete record of a chain/agent run with all steps |
| **Tags/Metadata** | Custom labels for filtering and organization |
| **Dataset** | Collection of input-output pairs for evaluation |
| **Evaluator** | Function that scores a prediction against expected output |
| **Experiment** | A named evaluation run for comparison |
| **Feedback** | User ratings (👍/👎) linked to specific runs |
| **Monitoring** | Real-time tracking of latency, cost, errors |

---

> [← Previous: Production LangChain](chapter-46-production-langchain.md) | [Next: Capstone Project →](chapter-48-capstone.md)

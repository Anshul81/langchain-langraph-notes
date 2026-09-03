# Chapter 12.2: RAG Evaluation — Measuring What Matters

> **Phase 12 — Advanced RAG** | [← Previous: Advanced Retrieval](chapter-44-advanced-retrieval.md) | [Next: Phase 13 — Production →](../phase-12-production/chapter-46-production-langchain.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Understand the three pillars of RAG evaluation: Retrieval, Generation, End-to-End
- ✅ Build evaluation datasets with question-answer-context triples
- ✅ Measure retrieval quality (precision, recall, MRR)
- ✅ Measure generation quality (faithfulness, relevance, hallucination)
- ✅ Use RAGAS for automated RAG evaluation
- ✅ Use LLM-as-Judge for scalable evaluation
- ✅ Build a **RAG evaluation pipeline** for continuous monitoring

| | |
|---|---|
| **Prerequisites** | Chapter 12.1 (Advanced Retrieval), Phase 11 (Complete RAG) |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 50 minutes |

---

## Introduction — You Can't Improve What You Don't Measure

Most RAG systems are built, deployed, and... hoped for the best. That's a recipe for disaster:

```
WITHOUT EVALUATION:
"Is our RAG working well?"  →  "I think so?" 😬

WITH EVALUATION:
"Is our RAG working well?"  →  "Retrieval precision: 82%, faithfulness: 91%, 
                                 answer relevance: 87%. The main failure mode is 
                                 financial questions — we need better chunking for 
                                 the finance docs." ✅
```

---

## Part 1: The Three Pillars of RAG Evaluation

```
┌──────────────────────────────────────────────────────────┐
│                RAG Evaluation Framework                    │
│                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  RETRIEVAL   │  │  GENERATION  │  │  END-TO-END  │   │
│  │              │  │              │  │              │   │
│  │ Did we find  │  │ Did the LLM  │  │ Is the final │   │
│  │ the RIGHT    │  │ answer       │  │ answer       │   │
│  │ documents?   │  │ correctly    │  │ correct and  │   │
│  │              │  │ from the     │  │ useful?      │   │
│  │ Metrics:     │  │ context?     │  │              │   │
│  │ • Precision  │  │              │  │ Metrics:     │   │
│  │ • Recall     │  │ Metrics:     │  │ • Accuracy   │   │
│  │ • MRR        │  │ • Faithful-  │  │ • Usefulness │   │
│  │ • NDCG       │  │   ness       │  │ • Latency    │   │
│  │              │  │ • Relevance  │  │ • Cost       │   │
│  │              │  │ • Hallucin.  │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                           │
│  If retrieval     If generation is    Overall system      │
│  is bad, fix      bad, fix the        performance         │
│  chunking/        prompt/model                            │
│  embeddings                                               │
└──────────────────────────────────────────────────────────┘
```

### Why Separate Retrieval and Generation?

```
Scenario 1: Bad retrieval, good generation
  → Retrieved wrong docs, but LLM answered well from what it got
  → Fix: Better chunking, embeddings, or retrieval strategy

Scenario 2: Good retrieval, bad generation
  → Found the right docs, but LLM hallucinated or missed the answer
  → Fix: Better prompt, stronger model, or more context

Scenario 3: Both bad
  → Fix retrieval first (it's the foundation), then generation
```

---

## Part 2: Building an Evaluation Dataset

### The Golden Dataset

```python
# An evaluation dataset has three components:
# 1. Question: What the user asks
# 2. Ground truth answer: The correct answer (human-written)
# 3. Ground truth context: The document(s) containing the answer

eval_dataset = [
    {
        "question": "What is CodeAssist?",
        "ground_truth": "CodeAssist is an AI-powered code review tool that analyzes pull requests in real-time, providing feedback on code quality, security vulnerabilities, and best practices.",
        "ground_truth_context": "CodeAssist is an AI-powered code review tool built for modern development teams. It analyzes pull requests in real-time, providing instant feedback on code quality, security vulnerabilities, and best practices."
    },
    {
        "question": "How much does the Starter plan cost?",
        "ground_truth": "The Starter plan costs $29/month per developer, supporting up to 10 developers.",
        "ground_truth_context": "Starter: $29/month per developer (up to 10 developers)"
    },
    {
        "question": "When was the company founded?",
        "ground_truth": "TechAssist was founded in Q1 2020 by Aarav Patel in Bangalore, India.",
        "ground_truth_context": "TechAssist was founded in 2020 by Aarav Patel in Bangalore, India. Starting as a 3-person team..."
    },
    {
        "question": "What is the refund policy for annual plans?",
        "ground_truth": "Annual plans offer a full refund within 30 days of purchase. After 30 days, remaining months are credited toward future use, with no cash refunds.",
        "ground_truth_context": "Annual plans: Full refund within 30 days of purchase. After 30 days, remaining months are credited toward future use. No cash refunds after 30 days."
    },
    {
        "question": "What was Q3 2024 revenue?",
        "ground_truth": "Q3 2024 revenue was ₹4.5 crore ($540K USD), representing a 35% quarter-over-quarter increase and 120% year-over-year growth.",
        "ground_truth_context": "Q3 2024 Revenue: ₹4.5 crore ($540K USD). Year-over-year growth: 120%. Quarter-over-quarter growth: 35%."
    },
    {
        "question": "Is the company SOC 2 compliant?",
        "ground_truth": "Yes, TechAssist is SOC 2 Type II certified.",
        "ground_truth_context": "SOC 2 Type II certified"
    },
    {
        "question": "How many employees does the company have?",
        "ground_truth": "The company has 45 employees: 25 in engineering, 10 in sales & marketing, 5 in operations & HR, and 5 in leadership.",
        "ground_truth_context": "Team Composition (45 total): Engineering: 25, Sales & Marketing: 10, Operations & HR: 5, Leadership: 5"
    },
    {
        "question": "What are the support hours?",
        "ground_truth": "Standard support is available 9 AM to 6 PM IST, Monday through Friday. Enterprise customers get 24/7 support.",
        "ground_truth_context": "Standard: 9 AM - 6 PM IST, Monday through Friday. Enterprise: 24/7 support with guaranteed 1-hour response time."
    },
    {
        "question": "What cloud provider does the company use?",
        "ground_truth": "The company uses AWS with primary regions in Mumbai (ap-south-1) and US East (us-east-1).",
        "ground_truth_context": "Infrastructure: AWS (ap-south-1 and us-east-1)"
    },
    {
        "question": "What is the company's mission statement?",
        "ground_truth": "The company's mission is 'Make every developer's code better, automatically.'",
        "ground_truth_context": "Mission: 'Make every developer's code better, automatically.'"
    },
]

print(f"📋 Evaluation dataset: {len(eval_dataset)} questions")
```

### Generating Evaluation Data with LLM

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate_eval_questions(documents, llm, num_questions=5):
    """Auto-generate evaluation questions from documents."""
    
    prompt = ChatPromptTemplate.from_template(
        """Given this document, generate {n} question-answer pairs that can be answered 
from the document. Format each as:
Q: [question]
A: [answer]

Document:
{document}

Generate {n} diverse questions:"""
    )
    
    all_qa = []
    for doc in documents:
        result = (prompt | llm | StrOutputParser()).invoke({
            "document": doc.page_content,
            "n": num_questions
        })
        
        # Parse Q/A pairs
        lines = result.split("\n")
        current_q, current_a = None, None
        for line in lines:
            if line.strip().startswith("Q:"):
                current_q = line.strip()[2:].strip()
            elif line.strip().startswith("A:"):
                current_a = line.strip()[2:].strip()
                if current_q and current_a:
                    all_qa.append({
                        "question": current_q,
                        "ground_truth": current_a,
                        "ground_truth_context": doc.page_content,
                        "source": doc.metadata.get("source", "unknown")
                    })
                    current_q, current_a = None, None
    
    return all_qa
```

---

## Part 3: Retrieval Evaluation

### Measuring Retrieval Quality

```python
def evaluate_retrieval(retriever, eval_dataset, k=5):
    """Evaluate retrieval quality against ground truth."""
    
    results = {
        "hit_rate": [],      # Did the right doc appear in top-k?
        "mrr": [],           # Mean Reciprocal Rank — where did it appear?
        "precision": [],     # What fraction of retrieved docs are relevant?
    }
    
    for item in eval_dataset:
        question = item["question"]
        ground_truth_context = item["ground_truth_context"]
        
        # Retrieve
        retrieved_docs = retriever.invoke(question)
        retrieved_texts = [doc.page_content for doc in retrieved_docs[:k]]
        
        # Check if ground truth appears in retrieved docs
        hit = False
        rank = 0
        relevant_count = 0
        
        for i, text in enumerate(retrieved_texts):
            # Check if the ground truth context is contained in the retrieved text
            if ground_truth_context[:100].lower() in text.lower() or \
               any(sent.lower() in text.lower() 
                   for sent in ground_truth_context.split(". ")[:2]):
                if not hit:
                    rank = i + 1
                    hit = True
                relevant_count += 1
        
        results["hit_rate"].append(1 if hit else 0)
        results["mrr"].append(1 / rank if hit else 0)
        results["precision"].append(relevant_count / len(retrieved_texts) if retrieved_texts else 0)
    
    # Aggregate
    n = len(eval_dataset)
    metrics = {
        "hit_rate": sum(results["hit_rate"]) / n,
        "mrr": sum(results["mrr"]) / n,
        "avg_precision": sum(results["precision"]) / n,
        "total_questions": n,
    }
    
    return metrics


# Run evaluation
# retrieval_metrics = evaluate_retrieval(retriever, eval_dataset)
# print(f"\n📊 Retrieval Metrics:")
# print(f"   Hit Rate:      {retrieval_metrics['hit_rate']:.1%}")
# print(f"   MRR:           {retrieval_metrics['mrr']:.3f}")
# print(f"   Avg Precision: {retrieval_metrics['avg_precision']:.1%}")
```

### What the Metrics Mean

| Metric | What It Measures | Good Score |
|--------|-----------------|------------|
| **Hit Rate** | % of queries where the right doc is in top-k | > 85% |
| **MRR** | Average 1/rank of the first relevant doc | > 0.7 |
| **Precision@k** | % of retrieved docs that are relevant | > 50% |
| **Recall@k** | % of relevant docs that were retrieved | > 80% |

---

## Part 4: Generation Evaluation with LLM-as-Judge

### Faithfulness — Does the Answer Stick to the Context?

```python
def evaluate_faithfulness(question, answer, context, llm):
    """Check if the answer is faithful to the provided context (no hallucination)."""
    
    prompt = ChatPromptTemplate.from_template(
        """Evaluate if the answer is faithful to the provided context.
A faithful answer only contains information present in the context.

Context: {context}
Question: {question}
Answer: {answer}

Score the faithfulness from 1-5:
1 = Completely unfaithful (hallucinated information)
2 = Mostly unfaithful (significant fabrication)
3 = Partially faithful (some hallucination mixed with real info)
4 = Mostly faithful (minor unsupported claims)
5 = Completely faithful (everything in the answer comes from the context)

Score (just the number):"""
    )
    
    result = (prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": question,
        "answer": answer
    })
    
    try:
        return int(result.strip()[0])
    except:
        return 3  # Default


# Test
# score = evaluate_faithfulness(
#     "What is CodeAssist?",
#     "CodeAssist is an AI-powered code review tool.",
#     "CodeAssist is an AI-powered code review tool built for modern development teams.",
#     llm
# )
# print(f"Faithfulness: {score}/5")
```

### Answer Relevance — Does the Answer Address the Question?

```python
def evaluate_relevance(question, answer, llm):
    """Check if the answer actually addresses the question."""
    
    prompt = ChatPromptTemplate.from_template(
        """Evaluate if the answer is relevant to the question asked.

Question: {question}
Answer: {answer}

Score the relevance from 1-5:
1 = Completely irrelevant
2 = Slightly relevant but misses the point
3 = Partially relevant
4 = Mostly relevant with minor gaps
5 = Perfectly relevant — fully answers the question

Score (just the number):"""
    )
    
    result = (prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "answer": answer
    })
    
    try:
        return int(result.strip()[0])
    except:
        return 3


### Correctness — Is the Answer Factually Correct?

def evaluate_correctness(question, answer, ground_truth, llm):
    """Check if the answer matches the ground truth."""
    
    prompt = ChatPromptTemplate.from_template(
        """Compare the generated answer with the ground truth answer.

Question: {question}
Generated Answer: {answer}
Ground Truth: {ground_truth}

Score the correctness from 1-5:
1 = Completely wrong
2 = Mostly wrong with some correct elements
3 = Partially correct
4 = Mostly correct with minor inaccuracies
5 = Completely correct — matches the ground truth

Score (just the number):"""
    )
    
    result = (prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth
    })
    
    try:
        return int(result.strip()[0])
    except:
        return 3
```

---

## Part 5: Complete RAG Evaluation Pipeline

```python
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class RAGEvaluator:
    """Complete RAG evaluation pipeline."""
    
    def __init__(self, rag_chain, retriever):
        self.rag_chain = rag_chain
        self.retriever = retriever
        self.judge_llm = ChatOpenAI(
            model="gpt-4o-mini", temperature=0,
            openai_api_key=os.getenv("LITELLM_PROXY_API_KEY"),
            openai_api_base=os.getenv("LITELLM_PROXY_API_BASE")
        )
        self.results = []
    
    def evaluate(self, eval_dataset: list[dict]) -> dict:
        """Run full evaluation on a dataset."""
        print(f"\n🔍 Evaluating RAG on {len(eval_dataset)} questions...\n")
        
        all_scores = {
            "faithfulness": [],
            "relevance": [],
            "correctness": [],
            "retrieval_hit": [],
        }
        
        for i, item in enumerate(eval_dataset):
            question = item["question"]
            ground_truth = item["ground_truth"]
            gt_context = item.get("ground_truth_context", "")
            
            # Get RAG answer
            answer = self.rag_chain.invoke(question)
            
            # Get retrieved docs
            docs = self.retriever.invoke(question)
            retrieved_context = "\n".join(doc.page_content for doc in docs)
            
            # Check retrieval hit
            hit = any(
                gt_context[:80].lower() in doc.page_content.lower()
                for doc in docs
            ) if gt_context else True
            
            # LLM-as-Judge scores
            faithfulness = self._score_faithfulness(question, answer, retrieved_context)
            relevance = self._score_relevance(question, answer)
            correctness = self._score_correctness(question, answer, ground_truth)
            
            all_scores["faithfulness"].append(faithfulness)
            all_scores["relevance"].append(relevance)
            all_scores["correctness"].append(correctness)
            all_scores["retrieval_hit"].append(1 if hit else 0)
            
            status = "✅" if correctness >= 4 else "⚠️" if correctness >= 3 else "❌"
            print(f"  {status} Q{i+1}: {question[:50]}... "
                  f"[F:{faithfulness} R:{relevance} C:{correctness} H:{'Y' if hit else 'N'}]")
            
            self.results.append({
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "faithfulness": faithfulness,
                "relevance": relevance,
                "correctness": correctness,
                "retrieval_hit": hit,
            })
        
        # Aggregate metrics
        n = len(eval_dataset)
        metrics = {
            "avg_faithfulness": sum(all_scores["faithfulness"]) / n,
            "avg_relevance": sum(all_scores["relevance"]) / n,
            "avg_correctness": sum(all_scores["correctness"]) / n,
            "retrieval_hit_rate": sum(all_scores["retrieval_hit"]) / n,
            "total_questions": n,
            "passing_rate": sum(1 for s in all_scores["correctness"] if s >= 4) / n,
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 RAG Evaluation Summary")
        print(f"{'='*60}")
        print(f"  Faithfulness:       {metrics['avg_faithfulness']:.2f}/5")
        print(f"  Relevance:          {metrics['avg_relevance']:.2f}/5")
        print(f"  Correctness:        {metrics['avg_correctness']:.2f}/5")
        print(f"  Retrieval Hit Rate: {metrics['retrieval_hit_rate']:.1%}")
        print(f"  Passing Rate (≥4):  {metrics['passing_rate']:.1%}")
        
        # Identify failures
        failures = [r for r in self.results if r["correctness"] < 4]
        if failures:
            print(f"\n⚠️ Failed Questions ({len(failures)}):")
            for f in failures:
                print(f"   ❌ {f['question'][:60]}...")
                print(f"      Expected: {f['ground_truth'][:80]}...")
                print(f"      Got:      {f['answer'][:80]}...")
        
        return metrics
    
    def _score_faithfulness(self, question, answer, context):
        prompt = ChatPromptTemplate.from_template(
            "Is this answer faithful to the context? Context: {context}\n"
            "Question: {question}\nAnswer: {answer}\n"
            "Score 1-5 (5=completely faithful). Just the number:"
        )
        try:
            r = (prompt | self.judge_llm | StrOutputParser()).invoke({
                "context": context[:2000], "question": question, "answer": answer
            })
            return int(r.strip()[0])
        except:
            return 3
    
    def _score_relevance(self, question, answer):
        prompt = ChatPromptTemplate.from_template(
            "Does this answer address the question?\n"
            "Question: {question}\nAnswer: {answer}\n"
            "Score 1-5 (5=perfectly relevant). Just the number:"
        )
        try:
            r = (prompt | self.judge_llm | StrOutputParser()).invoke({
                "question": question, "answer": answer
            })
            return int(r.strip()[0])
        except:
            return 3
    
    def _score_correctness(self, question, answer, ground_truth):
        prompt = ChatPromptTemplate.from_template(
            "Compare answer vs ground truth.\n"
            "Question: {question}\nAnswer: {answer}\n"
            "Ground Truth: {ground_truth}\n"
            "Score 1-5 (5=completely correct). Just the number:"
        )
        try:
            r = (prompt | self.judge_llm | StrOutputParser()).invoke({
                "question": question, "answer": answer, "ground_truth": ground_truth
            })
            return int(r.strip()[0])
        except:
            return 3
    
    def export_results(self, filepath: str):
        """Export results to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"📁 Results exported to {filepath}")


# --- Usage ---
# evaluator = RAGEvaluator(rag_chain, retriever)
# metrics = evaluator.evaluate(eval_dataset)
# evaluator.export_results("rag_eval_results.json")
```

---

## Part 6: Using RAGAS (Popular Framework)

RAGAS is a popular open-source framework specifically for RAG evaluation:

```python
# pip install ragas

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare data in RAGAS format
ragas_data = {
    "question": [item["question"] for item in eval_dataset],
    "answer": [],         # Will be filled by RAG
    "contexts": [],       # Will be filled by retriever
    "ground_truth": [item["ground_truth"] for item in eval_dataset],
}

# Generate answers and retrieve contexts
for item in eval_dataset:
    answer = rag_chain.invoke(item["question"])
    docs = retriever.invoke(item["question"])
    
    ragas_data["answer"].append(answer)
    ragas_data["contexts"].append([doc.page_content for doc in docs])

# Create dataset
dataset = Dataset.from_dict(ragas_data)

# Evaluate
results = evaluate(
    dataset,
    metrics=[
        faithfulness,        # Is the answer grounded in the context?
        answer_relevancy,    # Does the answer address the question?
        context_precision,   # Are retrieved docs relevant?
        context_recall,      # Did we retrieve the right docs?
    ],
)

print(results)
# {'faithfulness': 0.92, 'answer_relevancy': 0.88, 
#  'context_precision': 0.85, 'context_recall': 0.78}
```

---

## Part 7: Iterative Improvement Workflow

```
Step 1: Build baseline RAG → Evaluate
        ↓
Step 2: Identify failure modes (e.g., "financial questions fail")
        ↓
Step 3: Fix the root cause:
        ├── Bad retrieval? → Fix chunking, add hybrid search
        ├── Bad generation? → Fix prompt, use stronger model
        └── Both? → Fix retrieval first
        ↓
Step 4: Re-evaluate → Compare with baseline
        ↓
Step 5: Repeat until metrics meet targets

TARGET METRICS:
├── Retrieval Hit Rate:  > 90%
├── Faithfulness:        > 4.0/5
├── Relevance:           > 4.0/5
├── Correctness:         > 4.0/5
└── Passing Rate:        > 85%
```

---

## Common Mistakes

### Mistake 1: Only evaluating end-to-end
```python
# ❌ "The final answer is wrong" — but WHY?
# Is it retrieval or generation?

# ✅ Evaluate retrieval and generation SEPARATELY
retrieval_metrics = evaluate_retrieval(retriever, eval_dataset)
generation_metrics = evaluate_generation(rag_chain, eval_dataset)
# Now you know WHERE the problem is
```

### Mistake 2: Using the same LLM for generation and judging
```python
# ❌ GPT-4o-mini generates AND judges → biased toward its own answers

# ✅ Use a different/stronger model for judging
generator = ChatOpenAI(model="gpt-4o-mini")  # Generates answers
judge = ChatOpenAI(model="gpt-4o")            # Evaluates them
```

### Mistake 3: Too few evaluation questions
```python
# ❌ 5 questions → not statistically meaningful

# ✅ 30-50 questions minimum for reliable metrics
# Cover all categories: product, finance, support, security, etc.
```

### Mistake 4: Not versioning evaluation results
```python
# ❌ "We improved! ...I think?"

# ✅ Save results with timestamps and configuration
results = {
    "timestamp": datetime.now().isoformat(),
    "config": {"chunk_size": 500, "k": 5, "model": "gpt-4o-mini"},
    "metrics": metrics,
    "version": "v2-hybrid-search"
}
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Build evaluation dataset BEFORE building RAG | Know what success looks like upfront |
| Evaluate retrieval and generation separately | Find the actual bottleneck |
| Use LLM-as-Judge for scalable evaluation | Human evaluation doesn't scale |
| Use a stronger LLM for judging than generating | Less biased evaluation |
| Version your evaluation results | Track improvement over time |
| Include edge cases in eval dataset | Test boundary conditions |
| Re-evaluate after every change | Ensure changes actually improve things |
| Target > 85% passing rate | Below this, users will notice failures |

---

## Interview Preparation

### Easy
**Q: How do you evaluate a RAG system?**

> Evaluate on three levels: (1) **Retrieval** — did we find the right documents? Measure with hit rate, precision, and MRR. (2) **Generation** — did the LLM answer correctly from the context? Measure faithfulness (no hallucination) and relevance. (3) **End-to-end** — is the final answer correct? Compare against ground truth answers. Use an evaluation dataset of 30-50+ question-answer pairs covering all topic areas.

### Medium
**Q: What is LLM-as-Judge and what are its limitations?**

> LLM-as-Judge uses a strong LLM (e.g., GPT-4) to evaluate the quality of another LLM's output. It scores answers on criteria like faithfulness, relevance, and correctness. **Advantages**: scalable (no human annotators needed), consistent, and fast. **Limitations**: (1) Self-bias — LLMs rate their own outputs higher. Mitigate by using a different model for judging. (2) Position bias — LLMs may favor the first option in comparisons. (3) Can't evaluate factual accuracy against external sources — only against provided context/ground truth. (4) Costs money per evaluation. Best practice: use a stronger model as judge and validate with periodic human evaluation.

### Hard
**Q: Your RAG system has 70% correctness. Walk through your debugging process.**

> Systematic approach: (1) **Separate retrieval from generation** — evaluate retrieval alone first. If hit rate is below 85%, the problem is retrieval: check chunk quality (too large? split badly?), try different chunk sizes, add hybrid search, or try a better embedding model. (2) If retrieval is good but generation is poor, check **faithfulness** — if low, the LLM is hallucinating: strengthen the "only use context" prompt, add "I don't know" instruction, or use a stronger model. (3) Check **failure distribution** — are failures concentrated in one topic area? That category may need better chunking or more data. (4) Examine specific failures manually — read the retrieved chunks and ask "could a human answer from these?" If not, retrieval is the issue. If yes, generation needs fixing. (5) Iterate: fix one thing at a time, re-evaluate, compare with baseline.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **Retrieval evaluation** | Did we find the right documents? (hit rate, MRR, precision) |
| **Generation evaluation** | Did the LLM answer correctly? (faithfulness, relevance) |
| **End-to-end evaluation** | Is the final answer correct? (correctness vs ground truth) |
| **LLM-as-Judge** | Using an LLM to score another LLM's output |
| **RAGAS** | Open-source framework for RAG evaluation |
| **Evaluation dataset** | Question + ground truth answer + ground truth context |
| **Faithfulness** | Does the answer only contain info from the context? |
| **Hit rate** | % of queries where correct doc appears in top-k |

---

> [← Previous: Advanced Retrieval](chapter-44-advanced-retrieval.md) | [Next: Phase 13 — Production →](../phase-12-production/chapter-46-production-langchain.md)

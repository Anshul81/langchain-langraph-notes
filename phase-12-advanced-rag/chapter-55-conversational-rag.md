# Chapter 12.5: Conversational RAG — Chat + Retrieval for Enterprise Document Q&A

> **Phase 12 — Advanced RAG** | [← Previous: Multi-Query & RAG Fusion](chapter-54-multi-query-rag-fusion.md) | [Next: Why LangGraph →](../phase-13-langgraph-fundamentals/chapter-56-why-langgraph.md)

---

## Learning Objectives

By the end of this chapter, you will:

- ✅ Build **multi-turn RAG** that understands follow-ups ("What about Pro?")
- ✅ Use a **history-aware retriever** to rewrite queries with chat context
- ✅ Structure **ConversationalRetrievalChain** patterns in modern LCEL
- ✅ Manage session memory without leaking documents across tenants
- ✅ Add citations and "I don't know" guardrails for enterprise trust
- ✅ Ship a **document chat bot** over an internal policy corpus

| | |
|---|---|
| **Prerequisites** | Phase 11 RAG, Chapters 12.1–12.4, Phase 8 Memory |
| **Estimated Reading Time** | 30 minutes |
| **Estimated Coding Time** | 55 minutes |

---

## Introduction — Follow-Ups Break Naive RAG

Naive RAG embeds **only the latest user message**. Chat users speak in **shorthand**:

```
Turn 1: "What's included in the Pro plan?"
Turn 2: "Does it include SSO?"
Turn 3: "And the SOC 2 report?"
```

Turn 2 and 3 are **not standalone search queries** — "it" and "And" refer to Pro.

### The Problem

```
Retrieve("Does it include SSO?")
  → embedding of vague pronoun query
  → wrong chunks (generic SSO blog, wrong product)
```

### The Solution — Conversational RAG Pipeline

```
 Chat history + latest question
            │
            ▼
   History-aware query rewriter (LLM)
            │
            ▼
 Standalone search query: "Does CodeAssist Pro include SSO?"
            │
            ▼
 Retriever → context → Answer LLM (with history)
```

**Rewrite for retrieval; answer with full conversation.**

---

## Part 1: Setup

```bash
pip install langchain langchain-openai langchain-chroma python-dotenv
```

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(
    model=os.getenv("LITE_LLM_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("LITELLM_PROXY_API_KEY"),
    base_url=os.getenv("LITELLM_PROXY_API_BASE"),
)
```

---

## Part 2: Index Enterprise Policies

```python
docs = [
    Document(
        page_content="Pro plan ($79/dev/mo) includes SSO (SAML/OIDC), audit logs, and SOC 2 Type II report on request.",
        metadata={"source": "plans/pro.md"},
    ),
    Document(
        page_content="Starter plan ($29/dev/mo) includes email support only; SSO is not available.",
        metadata={"source": "plans/starter.md"},
    ),
    Document(
        page_content="Enterprise plan uses custom MSAs, dedicated CSM, and annual SOC 2 + pen test summary sharing.",
        metadata={"source": "plans/enterprise.md"},
    ),
]

vectorstore = Chroma.from_documents(
    docs, embeddings, collection_name="conv_rag", persist_directory="./chroma_conv_rag"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
```

---

## Part 3: History-Aware Retriever

```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system = (
    "Given chat history and the latest user question, formulate a standalone question "
    "that can be understood without history. Do NOT answer — only reformulate if needed."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

When the user says *"Does it include SSO?"*, the rewriter produces *"Does CodeAssist Pro include SSO?"* using prior turns.

---

## Part 4: Conversational QA Chain

```python
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system = (
    "You are an internal policy assistant. Answer from context only. "
    "Cite sources in brackets like [plans/pro.md]. "
    "If the answer is not in context, say you don't know."
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

conv_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

### Multi-Turn Session

```python
chat_history = []

def chat(user_input: str) -> str:
    global chat_history
    result = conv_rag_chain.invoke({"input": user_input, "chat_history": chat_history})
    chat_history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=result["answer"]),
    ])
    return result["answer"]

print(chat("What's included in the Pro plan?"))
print(chat("Does it include SSO?"))
print(chat("And the SOC 2 report?"))
```

ASCII flow:

```
Turn 2 internal rewrite:
  History: [Human: Pro plan?, AI: ...Pro features...]
  Input: "Does it include SSO?"
  Rewritten query → retrieve → answer with history in QA prompt
```

---

## Part 5: LCEL-Style Equivalent (Explicit)

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

rewrite_chain = contextualize_q_prompt | llm | StrOutputParser()

def retrieve_context(inputs: dict) -> str:
    question = inputs["input"]
    if inputs.get("chat_history"):
        question = rewrite_chain.invoke({
            "chat_history": inputs["chat_history"],
            "input": inputs["input"],
        })
    docs = retriever.invoke(question)
    return "\n\n".join(
        f"[{d.metadata.get('source')}]\n{d.page_content}" for d in docs
    )

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system + "\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

lcel_conv_chain = (
    RunnablePassthrough.assign(context=retrieve_context)
    | answer_prompt
    | llm
    | StrOutputParser()
)
```

Same architecture — useful when you customize retrieval (hybrid, multi-query from Chapter 12.4).

---

## Part 6: Session Isolation for Enterprise

```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store: dict[str, ChatMessageHistory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Wrap chain that accepts chat_history key
def run_turn(session_id: str, text: str) -> str:
    history = get_history(session_id)
    out = lcel_conv_chain.invoke({
        "input": text,
        "chat_history": history.messages,
    })
    history.add_user_message(text)
    history.add_ai_message(out)
    return out

print(run_turn("employee-100", "Compare Starter vs Pro for SSO."))
print(run_turn("employee-200", "What's Enterprise SOC 2 coverage?"))
print(run_turn("employee-100", "Which is cheaper for 5 devs?"))
```

**Tenant rule:** never share `session_id` across customers; filter vector collections by `tenant_id` in metadata.

---

## Part 7: Production Hardening

| Concern | Pattern |
|---------|---------|
| Hallucination | "Context only" + evals |
| Stale index | Version tag in system prompt |
| PII in logs | Redact `chat_history` in traces |
| Long chats | Trim history; summarize older turns |
| Access control | Filter retriever by user ACL metadata |

```python
def retriever_with_acl(user_roles: list[str]):
    def _get_relevant(query: str):
        docs = retriever.invoke(query)
        return [d for d in docs if d.metadata.get("acl", "all") in user_roles + ["all"]]
    return _get_relevant
```

---

## Part 8: Combining Advanced Retrieval

For high-stakes internal chat, plug **hybrid** or **multi-query** retriever into `create_history_aware_retriever`:

```python
# pseudo: history_aware_retriever = create_history_aware_retriever(llm, hybrid_retriever, prompt)
```

Order: **rewrite question → advanced retrieve → stuff → answer**.

---

## Part 9: Streaming Conversational RAG

```python
from langchain_core.output_parsers import StrOutputParser

stream_answer = answer_prompt | llm | StrOutputParser()

def chat_stream(user_input: str, chat_history: list):
    context = retrieve_context({"input": user_input, "chat_history": chat_history})
    for chunk in stream_answer.stream({
        "input": user_input,
        "chat_history": chat_history,
        "context": context,
    }):
        print(chunk, end="", flush=True)
    print()
```

Retrieve **once** per turn (after rewrite), then stream tokens — users perceive latency improvement even when retrieval time is unchanged.

---

## Part 10: Clarifying Questions (Optional Guardrail)

```python
clarify_prompt = ChatPromptTemplate.from_template(
    """If the user question is ambiguous AND chat history lacks the entity,
ask ONE clarifying question instead of guessing. Otherwise reply GO.

Question: {input}
History: {history}
"""
)

def maybe_clarify(user_input: str, history: list) -> str | None:
    text = (clarify_prompt | llm | StrOutputParser()).invoke({
        "input": user_input,
        "history": history,
    })
    if text.strip() == "GO":
        return None
    return text.strip()
```

Use sparingly — enterprise users prefer smart rewriting over frequent clarifications.

---

## Part 11: End-to-End Architecture Diagram

```
┌──────────────┐     ┌─────────────────────┐     ┌─────────────┐
│ Web / Slack  │────→│ Session store       │────→│ Vector DB   │
│   client     │     │ (history per thread)│     │ + optional  │
└──────────────┘     └──────────┬──────────┘     │ BM25 index  │
                                │                └──────▲──────┘
                                ▼                       │
                     ┌──────────────────────┐            │
                     │ Rewrite w/ history │            │
                     └──────────┬─────────┘            │
                                ▼                       │
                     ┌──────────────────────┐   retrieve │
                     │ Hybrid / multi-query │──────────┘
                     └──────────┬─────────┘
                                ▼
                     ┌──────────────────────┐
                     │ Answer + citations   │
                     └──────────────────────┘
```

This is the reference architecture for internal **enterprise document chat** before you graduate to LangGraph agents that can retrieve repeatedly.

---

## Common Mistakes

### Mistake 1: Skipping query contextualization
```python
# ❌ retriever.invoke(latest_message_only)
# ✅ history-aware retriever or rewrite step
```

### Mistake 2: Dumping entire chat into retrieval embedding
```python
# ❌ embed concatenation of 30 turns — noisy vector
# ✅ rewrite to standalone question first
```

### Mistake 3: No citation requirement
```python
# ❌ "Pro has SSO" without source — untrusted in enterprise
# ✅ Prompt requires [source] tags from metadata
```

### Mistake 4: Global chat history variable in web apps
```python
# ❌ one list for all HTTP users
# ✅ session-scoped history store
```

---

## Best Practices

| Practice | Why |
|----------|-----|
| Rewrite for retrieval, keep history for answer | pronouns fixed at search time |
| Cap stored turns | control tokens and cost |
| Use separate models for rewrite vs answer (optional) | cheap rewrite, strong answer |
| Log rewritten query, not only user text | debug retrieval misses |
| Run conversational eval sets | follow-ups are where RAG fails |
| Align with Chapter 11.5 metrics | measure retrieval after rewrite |

---

## Interview Preparation

### Easy
**Q: Why does conversational RAG need query rewriting?**

> Follow-up questions often contain pronouns and omitted entities. Embedding the raw follow-up yields poor retrieval. A history-aware step reformulates the user message into a standalone question that includes entities from prior turns, so the retriever finds the right documents.

### Medium
**Q: What is the difference between `create_history_aware_retriever` and stuffing history into the QA prompt only?**

> The history-aware retriever uses the LLM to reformulate the question **before** search. Stuffing history only into the answer prompt helps the model interpret context but does not fix bad retrieval if the retriever still sees an ambiguous query. You need both: good retrieval query + conversational answer generation.

### Hard
**Q: How would you build multi-tenant conversational RAG?**

> Partition vector indexes or enforce metadata filters per tenant on every retrieval. Session IDs scoped to tenant+user. Never reuse chat history stores across tenants. Apply document ACLs at retrieval time. Audit logs with tenant ID. Separate encryption keys per tenant for stored transcripts where required.

### Senior
**Q: When does conversational RAG become agentic RAG?**

> When an LLM **decides** whether to retrieve, reformulate, or call tools (calculator, SQL) across multiple steps — typically LangGraph loops. Conversational RAG is a structured chain (rewrite → retrieve → answer). Move to agents when you need self-correction ("retrieve again if confidence low") or multi-source orchestration beyond one vector store.

---

## Summary

| Concept | What It Means |
|---------|--------------|
| **History-aware retriever** | Rewrites question using chat history |
| **`create_retrieval_chain`** | Wires retriever + document QA chain |
| **Standalone query** | Retrieval input with resolved entities |
| **Session history** | Isolated per user/thread |
| **Citations** | Enterprise trust and auditability |
| **Advanced retriever plug-in** | Hybrid/multi-query after rewrite |

---

## Hands-on Exercise

Implement `run_turn` with **max 6 messages** in history; when exceeded, summarize older messages into one `SystemMessage` note before rewrite. Test a 4-turn conversation about plan upgrades and verify retrieval quality on turn 3.

---

## What's Next

You have advanced **retrieval and chat over documents**. **Phase 13** introduces **LangGraph** for explicit workflows — starting with **Chapter 13.1: Why LangGraph**.

---

> [← Previous: Multi-Query & RAG Fusion](chapter-54-multi-query-rag-fusion.md) | [Next: Why LangGraph →](../phase-13-langgraph-fundamentals/chapter-56-why-langgraph.md)

# Apollo Hospitals Chennai — Intelligent Policy & Procedure Assistant

A production-style Retrieval-Augmented Generation (RAG) system that lets clinical and administrative staff query Apollo Hospitals Chennai's internal policies in plain English and receive accurate, source-cited answers in seconds.

---

## Business Problem

Hospital staff routinely need to look up discharge procedures, infection control protocols, billing rules, and compliance policies — often under time pressure. Traditional keyword search across static documents is slow, imprecise, and returns raw text rather than direct answers. This system eliminates that friction: a nurse can ask *"What is the ICU discharge checklist?"* and receive a concise, cited answer drawn directly from the relevant policy document, without reading through pages of text.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                                                                  │
│  documents/*.txt  ──►  RecursiveCharacterTextSplitter  ──►  Chunks│
│  (policies,            (500 chars, 50 overlap)                   │
│   protocols)                   │                                 │
│                                ▼                                 │
│                   SentenceTransformer (all-MiniLM-L6-v2)        │
│                   Dense vector embeddings (384-dim)              │
│                                │                                 │
│                                ▼                                 │
│                   ChromaDB  (persistent local store)            │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     QUERY PIPELINE      │
                    │                         │
                    │  User Question          │
                    │       │                 │
                    │       ▼                 │
                    │  all-MiniLM-L6-v2       │
                    │  (encode query)         │
                    │       │                 │
                    │       ▼                 │
                    │  ChromaDB similarity    │
                    │  search  (top-k = 3)    │
                    │       │                 │
                    │       ▼                 │
                    │  Retrieved chunks +     │
                    │  source metadata        │
                    │       │                 │
                    │       ▼                 │
                    │  Claude Haiku           │
                    │  (claude-haiku-4-5)     │
                    │  Grounded generation    │
                    │  with source citations  │
                    │       │                 │
                    │       ▼                 │
                    │  Answer + Sources       │
                    └─────────────────────────┘
                                 │
                    ┌────────────▼────────────────────────────────┐
                    │          EVALUATION PIPELINE (RAGAS)        │
                    │                                             │
                    │  Faithfulness      ──►  Claude Haiku LLM   │
                    │  Answer Relevancy  ──►  Claude Haiku LLM   │
                    │                        + Ollama embeddings  │
                    │  Context Precision ──►  Ollama qwen2.5:0.5b│
                    │  Context Recall    ──►  Ollama qwen2.5:0.5b│
                    └─────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Document chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embedding model | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Vector store | ChromaDB (persistent, local) |
| Generation LLM | Anthropic Claude Haiku (`claude-haiku-4-5-20251001`) |
| Evaluation framework | RAGAS |
| Evaluation LLM | Claude Haiku (faithfulness, answer relevancy) |
| Evaluation embeddings | Ollama `qwen2.5:0.5b` (context precision, context recall) |
| LLM orchestration | LangChain |
| Language | Python 3.11+ |

---

## Project Structure

```
apollo-rag/
├── documents/
│   ├── apollo_discharge_policy.txt
│   ├── apollo_infection_control.txt
│   └── apollo_insurance_billing.txt
├── src/
│   ├── document_loader.py    # Chunk and split source documents
│   ├── embed_and_store.py    # Embed chunks and persist to ChromaDB
│   ├── retriever.py          # Standalone retrieval with distance scores
│   ├── rag_pipeline.py       # End-to-end query: retrieve → generate → cite
│   └── evaluate_rag.py       # RAGAS evaluation with mixed LLM strategy
├── data/
│   └── chromadb/             # Persisted vector index (git-ignored)
├── .env                      # ANTHROPIC_API_KEY (never commit)
└── README.md
```

---

## Installation

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) running locally with `qwen2.5:0.5b` pulled.

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd apollo-rag

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install anthropic chromadb sentence-transformers langchain \
            langchain-text-splitters langchain-anthropic langchain-ollama \
            ragas datasets python-dotenv

# 4. Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# 5. Pull the local evaluation model
ollama pull qwen2.5:0.5b
```

---

## Running the Pipeline

### Step 1 — Ingest documents into ChromaDB

```bash
cd apollo-rag
python src/embed_and_store.py
```

Expected output:
```
apollo_discharge_policy.txt: 12 chunks
apollo_infection_control.txt: 9 chunks
apollo_insurance_billing.txt: 11 chunks
Stored 32 vectors in ChromaDB at 'data/chromadb'
```

### Step 2 — Ask a question

```bash
python src/rag_pipeline.py
```

Type any question at the prompt, or press Enter to use the default.

### Step 3 — Run RAGAS evaluation

```bash
python src/evaluate_rag.py
```

This runs all 5 test questions through the full pipeline, then scores the results using the mixed LLM strategy.

---

## Sample Questions

```
What is the ICU discharge procedure at Apollo Hospitals?
What documents are required for patient admission?
What are the visiting hours for ICU and critical care units?
How does Apollo Hospitals protect patient data confidentiality?
What payment methods are accepted and when is the deposit collected?
```

**Sample answer (abbreviated):**

> According to `apollo_discharge_policy.txt`, the ICU discharge process begins with a clinical assessment by the treating physician confirming patient stability. The patient is then transferred to a general ward or discharged home with written care instructions and a scheduled follow-up appointment.

---

## Evaluation Results

Evaluated on 5 clinically relevant question/ground-truth pairs using the RAGAS framework.

| Metric | Score | Interpretation |
|---|---|---|
| **Context Precision** | **1.00** | Every retrieved chunk was relevant — no noise in retrieval |
| **Context Recall** | **0.90** | 90% of ground-truth information was present in retrieved context |
| **Faithfulness** | **0.67** | Answers stay grounded in source text; some paraphrasing drift noted |
| **Answer Relevancy** | **0.55** | Answers address the question but occasionally include extraneous detail |

**Key insight:** The retrieval layer is performing close to perfectly (precision 1.0, recall 0.90). The gap in faithfulness and answer relevancy points to generation-side behaviour — the LLM occasionally elaborates beyond the retrieved context. This is a prompt engineering and model temperature problem, not a retrieval problem.

---

## Future Improvements

- **Prompt hardening** — tighten the system prompt with explicit "answer only from context" constraints and lower temperature to improve faithfulness toward 0.90+
- **Hybrid retrieval** — combine dense (semantic) and sparse (BM25) retrieval to handle precise terminology like drug names and procedure codes that semantic search can miss
- **Metadata filtering** — tag chunks by document type (policy, protocol, billing) and filter at query time so a billing question never retrieves an infection control chunk
- **Streaming API** — expose the pipeline as a FastAPI service with streaming responses for integration into hospital staff portals or WhatsApp bots
- **Multi-document expansion** — ingest structured data (formulary lists, department directories, appointment schedules) alongside unstructured policy text
- **Re-ranking** — add a cross-encoder re-ranker between retrieval and generation to further improve context relevancy
- **Continuous evaluation** — integrate RAGAS into a CI workflow to catch retrieval or generation regressions when documents are updated

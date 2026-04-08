---
title: RAG Document Dashboard
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---

# RAG Document Dashboard

A **production-ready Retrieval-Augmented Generation (RAG) application** — fully free to run and deploy, with zero API costs. Built with Streamlit, Hugging Face models, and FAISS. All AI runs locally; no OpenAI, no paid APIs, no billing.

---

## Live Demo

**Try it here:** [huggingface.co/spaces/AdamRuane99/rag-project-documents](https://huggingface.co/spaces/AdamRuane99/rag-project-documents)

> **Note on load time:** The app is hosted on Hugging Face Spaces' free tier. If it hasn't been visited in a while, the Space goes to sleep to save resources. When you first open it, it may show a loading screen for **1-2 minutes** while it wakes up and downloads the AI models. This is a one-time warm-up — once it's running, it responds normally. If you see a grey screen, just wait and refresh.

---

## What It Does

Load a set of documents, ask a natural-language question, and the app:

1. **Embeds** your documents into a vector space using `all-MiniLM-L6-v2`
2. **Retrieves** the most relevant chunks using FAISS similarity search
3. **Generates** a grounded answer using `flan-t5-base`, citing only what's in your documents

No hallucination from the open web — answers are always grounded in the documents you loaded.

---

## Why It's Free (and Production-Ready)

| | Detail |
|---|---|
| **Hosting** | Hugging Face Spaces — free CPU tier, public URL, no card required |
| **Models** | Open-source, downloaded once and cached inside the container |
| **No API keys** | Everything runs inside the container — no OpenAI, no Anthropic, no cloud LLM calls |
| **Dockerised** | Runs identically locally or in any cloud environment |
| **Persistent index** | FAISS index can be saved and reloaded between sessions |

The only trade-off on the free tier is speed — CPU inference takes ~10-20 seconds per query. Swap to a GPU-backed host and it drops to under 2 seconds, with no code changes needed.

---

## Architecture

```
User query
    │
    ▼
[Streamlit UI]  ──── app.py
    │
    ▼
[Retrieval]  ─────── rag_pipeline.py
  ├── Embedding:  sentence-transformers/all-MiniLM-L6-v2
  └── Vector DB:  FAISS (IndexFlatIP / cosine similarity)
    │
    ▼
[Generation] ─────── rag_pipeline.py
  └── LLM:  google/flan-t5-base  (seq2seq, CPU-friendly)
    │
    ▼
Answer + source references  →  Streamlit UI
```

**Key files**

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — data loading, query form, results display |
| `rag_pipeline.py` | Ingestion, FAISS indexing, retrieval, generation |
| `utils.py` | Text chunking, sample data, file parsing helpers |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container definition for local and cloud deployment |

---

## Quick Start (Run Locally)

### 1. Clone the repo

```bash
git clone https://github.com/AdamRuane99/rag-system-project
cd rag-system-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run will download `all-MiniLM-L6-v2` (~90 MB) and `flan-t5-base` (~250 MB). These are cached locally afterwards.

### 4. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Using the Dashboard

1. **Load data** — choose *"Use built-in sample data"* in the sidebar and click **Load sample documents** (10 pre-built articles covering tech, finance, health, space, and more). Or upload your own `.txt` or `.csv` file.

2. **Ask a question** — type any natural-language question and press **Ask**.

3. **Review the answer** — the generated answer appears at the top. Expand **Source references** to see exactly which document chunks were retrieved and their similarity scores.

### Example questions (with sample data)

- *"What drove Apex Technologies' revenue growth in Q3 2024?"*
- *"How are GLP-1 drugs being used beyond obesity treatment?"*
- *"What progress has been made in quantum error correction?"*
- *"Which countries led EV sales growth?"*

---

## Tech Stack

| Component | Library / Model |
|-----------|----------------|
| UI | [Streamlit](https://streamlit.io) |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) |
| Language model | [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) |
| ML framework | [PyTorch](https://pytorch.org) + [Hugging Face Transformers](https://huggingface.co/docs/transformers) |
| Deployment | [Docker](https://www.docker.com) + [Hugging Face Spaces](https://huggingface.co/spaces) |

---

## License

MIT — free to use, modify, and distribute.

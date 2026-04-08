# RAG Dashboard

A **portfolio-ready Retrieval-Augmented Generation (RAG) application** built with
Streamlit, Hugging Face models, and FAISS — no paid APIs, fully local.

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
  └── Vector DB:  FAISS (in-memory, IndexFlatIP / cosine)
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
| `.env.example` | Optional environment variable template |

---

## Quick Start

### 1. Clone / download the repo

```bash
git clone <your-repo-url>
cd rag-system-project
```

### 2. Create a virtual environment (recommended)

```bash
# Python 3.11
python -m venv .venv

# Activate — macOS/Linux
source .venv/bin/activate

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **First run note:** Hugging Face will download `all-MiniLM-L6-v2` (~90 MB) and
> `flan-t5-base` (~990 MB) on first use. These are cached locally afterwards.

### 4. Run the app

```bash
streamlit run app.py
```

OR Run in Powershell terminal
```bash
cd "c:\Users\user\Desktop\source\repos\rag-system-project"; rag-env\Scripts\streamlit.exe run app.py
```


Open the URL shown in your terminal (usually `http://localhost:8501`).

---

## Using the Dashboard

1. **Load data** — choose *"Use built-in sample data"* in the sidebar and click
   **Load sample documents** (10 pre-generated articles on tech, finance, science, etc.).
   Alternatively, upload your own `.txt` or `.csv` file.

2. **Ask a question** — type a natural-language question in the main input and
   press **Ask**.

3. **Review the answer** — the generated answer appears with a card at the top.
   Expand **Source references** to see which chunks were retrieved and their
   cosine-similarity scores.

### Example questions (sample data)

- *"What drove Apex Technologies' revenue growth in Q3 2024?"*
- *"How are GLP-1 drugs being used beyond obesity treatment?"*
- *"What progress has been made in quantum error correction?"*
- *"Which countries drove EV sales growth?"*

---

## Configuration

### Retrieval settings

Use the **Top-K** slider in the sidebar (1–6) to control how many chunks are
passed to the language model as context.

### Chunking

In `rag_pipeline.py`, adjust `RAGPipeline` defaults:

```python
RAGPipeline(
    chunk_size=300,    # words per chunk
    chunk_overlap=50,  # shared words between adjacent chunks
    top_k=3,           # chunks retrieved per query
)
```

### Switching the LLM

To use a larger / GPU-accelerated model, edit `rag_pipeline.py`:

```python
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"   # requires GPU + ~14 GB VRAM
```

> For decoder-only models (Mistral, LLaMA, Phi) also switch from
> `AutoModelForSeq2SeqLM` → `AutoModelForCausalLM` and update `generate()` to use
> a causal prompt format.

### Environment variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

---

## GPU acceleration

The app runs on CPU by default. To use a CUDA GPU:

```bash
# Replace faiss-cpu with faiss-gpu
pip install faiss-gpu

# PyTorch with CUDA (example — match your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

The models will automatically use the GPU when `torch.cuda.is_available()`.

---

## Project structure

```
rag-system-project/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # Ingestion, retrieval, generation
├── utils.py            # Chunking, sample data, file loaders
├── requirements.txt    # Dependencies
├── .env.example        # Optional env-var template
└── README.md           # This file
```

---

## Tech stack

| Component | Library / Model |
|-----------|----------------|
| UI | [Streamlit](https://streamlit.io) |
| Embeddings | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss) |
| Language model | [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) |
| ML framework | [PyTorch](https://pytorch.org) + [Hugging Face Transformers](https://huggingface.co/docs/transformers) |

**No OpenAI. No paid APIs. Fully offline after initial model download.**

---

## License

MIT — free to use, modify, and distribute.

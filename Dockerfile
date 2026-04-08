# ── Stage: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# libgomp1  — required by faiss-cpu (OpenMP threading)
# curl      — used by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker layer cache is efficient.
# The pip layer only rebuilds when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application source (excludes rag-env/, saved_index/, etc.
# via .dockerignore)
COPY app.py rag_pipeline.py utils.py ./

# Pre-create the index directory so the volume mount works on first run
RUN mkdir -p /app/saved_index

# ── Runtime config ───────────────────────────────────────────────────────────
ENV TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4

EXPOSE 7860

# Streamlit has a built-in health endpoint at /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

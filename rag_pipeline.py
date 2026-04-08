"""
rag_pipeline.py — Core RAG logic.

Pipeline stages:
  1. Ingest  : chunk documents and embed with sentence-transformers
  2. Index   : store vectors in an in-memory FAISS index
  3. Retrieve: find the top-k most relevant chunks for a query
  4. Generate: produce an answer with google/flan-t5-base (CPU-friendly)
"""

from __future__ import annotations

import os
import pickle
import pathlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from utils import chunk_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single text chunk with metadata."""
    text: str
    source_title: str
    chunk_index: int          # position within the source document


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever, augmented with a similarity score."""
    chunk: Chunk
    score: float              # cosine similarity [0, 1]


@dataclass
class RAGResult:
    """Final output returned to the UI."""
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    query: str


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest(documents)           # list of {"title":..., "content":...}
        result = pipeline.query("What is …")
    """

    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL_NAME   = "google/flan-t5-base"

    def __init__(
        self,
        chunk_size: int = 80,
        chunk_overlap: int = 15,
        top_k: int = 3,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k         = top_k

        self._chunks:       List[Chunk]         = []
        self._index                             = None   # faiss.IndexFlatIP
        self._embed_model                       = None
        self._tokenizer                         = None
        self._llm                               = None
        self._embeddings: Optional[np.ndarray]  = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True once at least one document has been ingested."""
        return self._index is not None and len(self._chunks) > 0

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------
    # Stage 1 — Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: List[dict]) -> None:
        """
        Chunk all documents, embed them, and build the FAISS index.

        Args:
            documents: List of {"title": str, "content": str} dicts.
        """
        if not documents:
            raise ValueError("No documents provided for ingestion.")

        # 1a. Chunk documents
        self._chunks = []
        for doc in documents:
            title   = doc.get("title", "Untitled")
            content = doc.get("content", "")
            if not content.strip():
                continue
            parts = chunk_text(content, self.chunk_size, self.chunk_overlap)
            for idx, part in enumerate(parts):
                self._chunks.append(Chunk(text=part, source_title=title, chunk_index=idx))

        if not self._chunks:
            raise ValueError("All documents were empty after chunking.")

        logger.info("Ingested %d chunks from %d documents.", len(self._chunks), len(documents))

        # 1b. Embed chunks
        self._load_embed_model()
        texts = [c.text for c in self._chunks]
        self._embeddings = self._embed_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # needed for cosine via inner-product
        )

        # 1c. Build FAISS index (inner product == cosine on normalised vectors)
        import faiss

        dim = self._embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings.astype(np.float32))
        logger.info("FAISS index built — %d vectors, dim=%d.", self._index.ntotal, dim)

    # ------------------------------------------------------------------
    # Stage 2 — Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[RetrievedChunk]:
        """
        Embed the query and return the top-k most similar chunks.

        Args:
            query: Natural-language question.

        Returns:
            List of RetrievedChunk (sorted by score, descending).
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline not ready — call ingest() first.")

        q_emb = self._embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self._index.search(q_emb, self.top_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:            # FAISS returns -1 when fewer results exist
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=float(score)))

        return results

    # ------------------------------------------------------------------
    # Stage 3 — Generation
    # ------------------------------------------------------------------

    def generate(self, query: str, context_chunks: List[RetrievedChunk]) -> str:
        """
        Produce an answer given the query and retrieved context.

        Args:
            query:          User's question.
            context_chunks: Top-k retrieved chunks.

        Returns:
            Generated answer string.
        """
        self._load_llm()

        # Build context block from retrieved chunks
        context = "\n\n".join(
            f"[Source: {rc.chunk.source_title}]\n{rc.chunk.text}"
            for rc in context_chunks
        )

        # Prompt template for flan-t5 instruction-following
        prompt = (
            f"Answer the question based only on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )

        output_ids = self._llm.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            num_beams=4,
        )

        answer = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return answer.strip() if answer.strip() else "I could not find a specific answer in the provided documents."

    # ------------------------------------------------------------------
    # Convenience — single call for the full RAG loop
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResult:
        """
        Retrieve relevant chunks and generate an answer in one call.

        Args:
            question: Natural-language question from the user.

        Returns:
            RAGResult with answer, source chunks, and echoed query.
        """
        if not self.is_ready:
            raise RuntimeError("Pipeline not ready — call ingest() first.")

        retrieved = self.retrieve(question)
        answer    = self.generate(question, retrieved)

        return RAGResult(answer=answer, retrieved_chunks=retrieved, query=question)

    # ------------------------------------------------------------------
    # Lazy model loading (avoids slow startup)
    # ------------------------------------------------------------------

    def _load_embed_model(self) -> None:
        if self._embed_model is not None:
            return
        logger.info("Loading embedding model: %s", self.EMBED_MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        self._embed_model = SentenceTransformer(self.EMBED_MODEL_NAME)

    def _load_llm(self) -> None:
        if self._llm is not None:
            return
        logger.info("Loading LLM: %s", self.LLM_MODEL_NAME)
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._tokenizer = AutoTokenizer.from_pretrained(self.LLM_MODEL_NAME)
        self._llm = AutoModelForSeq2SeqLM.from_pretrained(self.LLM_MODEL_NAME)
        self._llm.eval()

    # ------------------------------------------------------------------
    # Index persistence — save / load the FAISS index to/from disk
    # ------------------------------------------------------------------

    def save_index(self, directory: str = "saved_index") -> None:
        """
        Write the FAISS index and chunk metadata to disk.

        Creates two files inside *directory*:
            index.faiss  — the FAISS binary index
            chunks.pkl   — pickled list of Chunk objects

        Args:
            directory: Path to the directory where files are written.
                       Created automatically if it does not exist.
        """
        if not self.is_ready:
            raise RuntimeError("Nothing to save — ingest documents first.")

        import faiss

        path = pathlib.Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

        logger.info("Index saved to '%s' (%d chunks).", directory, len(self._chunks))

    def load_index(self, directory: str = "saved_index") -> None:
        """
        Restore a previously saved FAISS index from disk.

        Also loads the embedding model so the pipeline is immediately
        ready for retrieval without needing to call ingest() again.

        Args:
            directory: Path that contains index.faiss and chunks.pkl.

        Raises:
            FileNotFoundError: If either required file is missing.
        """
        import faiss

        path = pathlib.Path(directory)
        index_path  = path / "index.faiss"
        chunks_path = path / "chunks.pkl"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No saved index found in '{directory}'. "
                "Ingest documents and click Save Index first."
            )

        self._index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self._chunks = pickle.load(f)

        # Embedding model must be ready for future retrieve() calls
        self._load_embed_model()

        logger.info(
            "Index loaded from '%s' — %d chunks, %d vectors.",
            directory, len(self._chunks), self._index.ntotal,
        )

    # ------------------------------------------------------------------
    # Reset — allows re-ingesting with new documents
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the index and all loaded data (keeps models in memory)."""
        self._chunks     = []
        self._index      = None
        self._embeddings = None
        logger.info("Pipeline reset.")

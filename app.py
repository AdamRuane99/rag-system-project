"""
app.py — Streamlit UI for the RAG Dashboard.

Run with:
    streamlit run app.py
"""

import logging
import streamlit as st

from rag_pipeline import RAGPipeline, RAGResult
from utils import get_sample_documents, load_text_from_upload

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* ── Main background ── */
    .stApp {
        background-color: #f5f7fa;
    }

    /* ── Reduce top padding ── */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* ── Hero header ── */
    .hero-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 12px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.75rem;
        color: white;
    }
    .hero-header h1 {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
        color: white !important;
    }
    .hero-header p {
        font-size: 0.95rem;
        opacity: 0.82;
        margin: 0;
        color: white !important;
    }

    /* ── Query box container ── */
    .query-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
    }

    /* ── Answer card ── */
    .answer-card {
        background: white;
        border-left: 4px solid #2563eb;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin: 0.75rem 0 1.25rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(37,99,235,0.06);
        font-size: 1rem;
        line-height: 1.65;
        color: #1e293b;
    }

    /* ── Result wrapper ── */
    .result-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    }

    /* ── Query echo label ── */
    .query-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.2rem;
    }
    .query-text {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.1rem;
    }

    /* ── Section label ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 1rem 0 0.4rem 0;
    }

    /* ── Source card ── */
    .source-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem 1.1rem;
        border-radius: 8px;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #334155;
    }
    .source-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.6rem;
    }
    .source-title {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.9rem;
    }

    /* ── Score badge ── */
    .score-badge {
        display: inline-flex;
        align-items: center;
        background: #eff6ff;
        color: #2563eb;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
        border: 1px solid #bfdbfe;
        white-space: nowrap;
    }

    /* ── Rank pill ── */
    .rank-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 22px;
        height: 22px;
        background: #2563eb;
        color: white;
        font-size: 0.7rem;
        font-weight: 700;
        border-radius: 50%;
        margin-right: 0.5rem;
        flex-shrink: 0;
    }

    /* ── History divider label ── */
    .history-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin: 1.5rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .history-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #e2e8f0;
    }

    /* ── Empty state ── */
    .empty-state {
        background: white;
        border-radius: 10px;
        padding: 3rem 2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        color: #64748b;
    }
    .empty-state .icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .empty-state h3 {
        font-size: 1.1rem;
        font-weight: 600;
        color: #334155;
        margin: 0 0 0.5rem 0;
    }
    .empty-state p {
        font-size: 0.9rem;
        margin: 0;
        max-width: 360px;
        margin: 0 auto;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0f1f3d;
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #cbd5e1 !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: background 0.15s ease;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #1d4ed8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 8px !important;
        padding: 0.6rem 0.75rem !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
    }

    /* ── Sidebar title ── */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.25rem;
    }
    .sidebar-brand-icon {
        font-size: 1.4rem;
    }
    .sidebar-brand-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: white !important;
        letter-spacing: -0.01em;
    }
    .sidebar-brand-sub {
        font-size: 0.78rem;
        color: #94a3b8 !important;
        margin-bottom: 0;
        padding-left: 0.1rem;
    }

    /* ── Model info box ── */
    .model-info {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.78rem;
        color: #94a3b8 !important;
        line-height: 1.6;
    }
    .model-info b {
        color: #cbd5e1 !important;
    }

    /* ── Status badge ── */
    .status-ready {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(34,197,94,0.15);
        color: #4ade80 !important;
        border: 1px solid rgba(74,222,128,0.3);
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        margin-bottom: 0.75rem;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #4ade80;
        display: inline-block;
    }

    /* ── Stale previous results ── */
    .result-card.previous {
        opacity: 0.85;
    }

    /* ── Clear button ── */
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid #e2e8f0 !important;
        color: #64748b !important;
        border-radius: 7px !important;
        font-size: 0.85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(chunk_size=80, chunk_overlap=15, top_k=3)

if "history" not in st.session_state:
    st.session_state.history: list[RAGResult] = []

if "ingested" not in st.session_state:
    st.session_state.ingested = False

if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

pipeline: RAGPipeline = st.session_state.pipeline

# ---------------------------------------------------------------------------
# Sidebar — data loading
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="sidebar-brand-icon">🔍</span>
            <span class="sidebar-brand-name">RAG Dashboard</span>
        </div>
        <p class="sidebar-brand-sub">Local retrieval-augmented generation</p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("**Data source**")

    data_source = st.radio(
        "Choose data source",
        options=["Use built-in sample data", "Upload your own file"],
        index=0,
        label_visibility="collapsed",
    )

    # ---- Option A: sample data ----
    if data_source == "Use built-in sample data":
        if st.button("Load sample documents", type="primary", use_container_width=True):
            with st.spinner("Embedding documents and building index…"):
                try:
                    docs = get_sample_documents()
                    pipeline.reset()
                    pipeline.ingest(docs)
                    st.session_state.ingested  = True
                    st.session_state.doc_count = len(docs)
                    st.session_state.history   = []
                    st.success(f"Loaded {len(docs)} sample documents ({pipeline.num_chunks} chunks).")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    # ---- Option B: file upload ----
    elif data_source == "Upload your own file":
        uploaded = st.file_uploader(
            "Upload a .txt or .csv file",
            type=["txt", "csv"],
            help="CSV files should have a text/content column.",
        )

        if uploaded is not None:
            if st.button("Ingest uploaded file", type="primary", use_container_width=True):
                with st.spinner("Embedding documents and building index…"):
                    try:
                        docs = load_text_from_upload(uploaded.read(), uploaded.name)
                        pipeline.reset()
                        pipeline.ingest(docs)
                        st.session_state.ingested  = True
                        st.session_state.doc_count = len(docs)
                        st.session_state.history   = []
                        st.success(
                            f"Loaded {len(docs)} document(s) from '{uploaded.name}' "
                            f"({pipeline.num_chunks} chunks)."
                        )
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")

    # ---- Status summary ----
    if st.session_state.ingested:
        st.divider()
        st.markdown(
            '<span class="status-ready"><span class="status-dot"></span>Index ready</span>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        col1.metric("Documents", st.session_state.doc_count)
        col2.metric("Chunks", pipeline.num_chunks)

    # ---- Settings ----
    st.divider()
    st.markdown("**Retrieval settings**")
    top_k = st.slider("Top-K chunks to retrieve", min_value=1, max_value=6, value=3)
    pipeline.top_k = top_k

    # ---- Index persistence ----
    st.divider()
    st.markdown("**Index persistence**")
    index_dir = st.text_input(
        "Index directory",
        value="./saved_index",
        label_visibility="visible",
        help="Folder where the FAISS index is saved and loaded from.",
    )
    col_save, col_load = st.columns(2)
    with col_save:
        if st.button("Save", use_container_width=True, disabled=not st.session_state.ingested):
            try:
                pipeline.save_index(index_dir)
                st.success("Saved.")
            except Exception as e:
                st.error(str(e))
    with col_load:
        if st.button("Load", use_container_width=True):
            try:
                pipeline.load_index(index_dir)
                st.session_state.ingested  = True
                st.session_state.doc_count = len(set(c.source_title for c in pipeline._chunks))
                st.session_state.history   = []
                st.success("Loaded.")
                st.rerun()
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Load failed: {e}")

    st.divider()
    st.markdown(
        """
        <div class="model-info">
            <b>Embeddings</b><br>all-MiniLM-L6-v2<br>
            <b>Generation</b><br>flan-t5-base<br><br>
            All models run locally — no API keys required.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main area — hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-header">
        <h1>Ask your documents anything</h1>
        <p>Load a knowledge base, then ask questions — the system retrieves relevant context and generates a grounded answer.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Not yet ingested — empty state
# ---------------------------------------------------------------------------

if not st.session_state.ingested:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">📂</div>
            <h3>No documents loaded yet</h3>
            <p>Choose a data source in the sidebar and click <strong>Load</strong> to index your documents before asking questions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # ---- Query form ----
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    with st.form("query_form", clear_on_submit=False):
        query = st.text_input(
            "Your question",
            placeholder="e.g. What drove Apex Technologies' revenue growth in Q3?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →", type="primary", use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    result: RAGResult = pipeline.query(query.strip())
                    st.session_state.history.insert(0, result)
                except Exception as e:
                    st.error(f"Query failed: {e}")

    # ---- Display results ----
    if st.session_state.history:
        for idx, result in enumerate(st.session_state.history):
            is_latest = idx == 0

            if not is_latest and idx == 1:
                st.markdown(
                    '<div class="history-label">Previous queries</div>',
                    unsafe_allow_html=True,
                )

            card_class = "result-card" if is_latest else "result-card previous"
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

            # Query echo
            st.markdown(
                f'<p class="query-label">{"Latest query" if is_latest else f"Query {idx + 1}"}</p>'
                f'<p class="query-text">{result.query}</p>',
                unsafe_allow_html=True,
            )

            # Answer
            st.markdown('<p class="section-label">Answer</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="answer-card">{result.answer}</div>',
                unsafe_allow_html=True,
            )

            # Source references
            with st.expander(f"View source references  ({len(result.retrieved_chunks)} chunks)"):
                for rank, rc in enumerate(result.retrieved_chunks, start=1):
                    score_pct = f"{rc.score * 100:.1f}%"
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-card-header">'
                        f'<span><span class="rank-pill">{rank}</span>'
                        f'<span class="source-title">{rc.chunk.source_title}</span></span>'
                        f'<span class="score-badge">&#9675; {score_pct} match</span>'
                        f'</div>'
                        f'{rc.chunk.text}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Clear history ----
        col_spacer, col_btn = st.columns([6, 1])
        with col_btn:
            if st.button("Clear history", type="secondary"):
                st.session_state.history = []
                st.rerun()

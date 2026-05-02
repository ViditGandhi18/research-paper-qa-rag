"""
app.py — Streamlit UI for Research Paper Q&A Assistant.

Run:
    streamlit run app.py
"""

import json
import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from loguru import logger

from src.utils import cfg, setup_logger
from src.ingest import ingest

setup_logger()

# Page Config
st.set_page_config(
    page_title="Research Paper Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(90deg, #1a73e8, #0d47a1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1rem; border: 1px solid #e0e0e0;
        text-align: center;
    }
    .source-badge {
        background: #e3f2fd; color: #1565c0;
        border-radius: 4px; padding: 2px 8px;
        font-size: 0.8rem; margin-right: 4px;
    }
    .answer-box {
        background: #f0f7ff; border-left: 4px solid #1a73e8;
        border-radius: 0 8px 8px 0; padding: 1rem 1.25rem;
        margin: 0.5rem 0;
    }
    .eval-good  { color: #2e7d32; font-weight: 600; }
    .eval-warn  { color: #f57c00; font-weight: 600; }
    .eval-bad   { color: #c62828; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Session State
if "assistant" not in st.session_state:
    st.session_state.assistant  = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "use_lora" not in st.session_state:
    st.session_state.use_lora = False

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["OpenAI (GPT)", "HuggingFace (Local)"],
        index=0,
    )
    os.environ["LLM_PROVIDER"] = "openai" if "OpenAI" in llm_provider else "huggingface"

    if "OpenAI" in llm_provider:
        api_key = st.text_input("OpenAI API Key", type="password", value=cfg.OPENAI_API_KEY)
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

    use_lora = st.toggle("Use LoRA Fine-tuned Model", value=False)
    st.session_state.use_lora = use_lora

    st.divider()
    st.markdown("## 📄 Upload Papers")

    uploaded_files = st.file_uploader(
        "Upload PDF research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files to index.",
    )

    if uploaded_files and st.button("📥 Index Papers", use_container_width=True, type="primary"):
        with st.spinner("Ingesting and indexing PDFs..."):
            tmp_dir = Path("data/papers")
            tmp_dir.mkdir(parents=True, exist_ok=True)

            for ufile in uploaded_files:
                save_path = tmp_dir / ufile.name
                with open(save_path, "wb") as f:
                    f.write(ufile.read())

            try:
                ingest(input_dir="data/papers/", reset=False)
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.session_state.assistant = None
                st.success(f"Indexed {len(uploaded_files)} paper(s)!")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    if st.session_state.indexed_files:
        st.markdown("**Indexed Papers:**")
        for fname in st.session_state.indexed_files:
            st.markdown(f"- 📄 `{fname}`")

    st.divider()
    st.markdown("## 🔬 Run Evaluation")
    if st.button("Run RAGAS Evaluation", use_container_width=True):
        st.switch_page("pages/evaluation.py") if Path("pages").exists() else st.info(
            "Run: `python src/evaluate.py` in your terminal."
        )

# Main Panel 
st.markdown('<p class="main-header">📚 Research Paper Q&A Assistant</p>', unsafe_allow_html=True)
st.caption("Ask questions about your uploaded research papers. Powered by RAG + FAISS + ChromaDB.")

# Quick stats row
if st.session_state.chat_history:
    col1, col2, col3, col4 = st.columns(4)
    latencies = [h["timing"]["total_ms"] for h in st.session_state.chat_history if "timing" in h]
    avg_lat   = sum(latencies) / len(latencies) if latencies else 0
    with col1:
        st.metric("Questions Asked",   len(st.session_state.chat_history))
    with col2:
        st.metric("Avg Latency",       f"{avg_lat:.0f}ms")
    with col3:
        papers_n = len(st.session_state.indexed_files)
        st.metric("Papers Indexed",    papers_n)
    with col4:
        lora_status = "Active" if st.session_state.use_lora else "Off"
        st.metric("LoRA Model",        lora_status)
    st.divider()

# Chat History 
for entry in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(entry["question"])

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            f'<div class="answer-box">{entry["answer"]}</div>',
            unsafe_allow_html=True,
        )

        if entry.get("source_docs"):
            with st.expander("📎 Source Documents", expanded=False):
                for doc in entry["source_docs"]:
                    src  = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "?")
                    st.markdown(
                        f'<span class="source-badge">📄 {src} | Page {page}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(doc.page_content[:200] + "...")
                    st.divider()

        if entry.get("timing"):
            t = entry["timing"]
            c1, c2, c3 = st.columns(3)
            c1.caption(f"🔍 Retrieval: {t['retrieval_ms']}ms")
            c2.caption(f"✍️ Generation: {t['generation_ms']}ms")
            c3.caption(f"⏱️ Total: {t['total_ms']}ms")

# Chat Input
question = st.chat_input("Ask a question about your research papers...")

if question:
    if not st.session_state.indexed_files and not list(Path("data/vectorstore/chroma").glob("*") if Path("data/vectorstore/chroma").exists() else []):
        st.warning("⚠️ Please upload and index at least one PDF first (use the sidebar).")
    else:
        if st.session_state.assistant is None:
            with st.spinner("Loading RAG pipeline..."):
                try:
                    from src.rag_chain import RAGAssistant
                    st.session_state.assistant = RAGAssistant(
                        use_lora=st.session_state.use_lora
                    )
                except Exception as e:
                    st.error(f"Failed to load assistant: {e}")
                    st.stop()

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching papers and generating answer..."):
                try:
                    result = st.session_state.assistant.ask(question)

                    st.markdown(
                        f'<div class="answer-box">{result["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    if result.get("source_docs"):
                        with st.expander("📎 Source Documents", expanded=True):
                            for doc in result["source_docs"]:
                                src  = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", "?")
                                st.markdown(
                                    f'<span class="source-badge">📄 {src} | Page {page}</span>',
                                    unsafe_allow_html=True,
                                )
                                st.caption(doc.page_content[:200] + "...")
                                st.divider()

                    t = result["timing"]
                    c1, c2, c3 = st.columns(3)
                    c1.caption(f"🔍 Retrieval: {t['retrieval_ms']}ms")
                    c2.caption(f"✍️ Generation: {t['generation_ms']}ms")
                    c3.caption(f"⏱️ Total: {t['total_ms']}ms")

                    st.session_state.chat_history.append({
                        "question":    question,
                        "answer":      result["answer"],
                        "source_docs": result.get("source_docs", []),
                        "timing":      result["timing"],
                    })

                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    logger.exception(e)

st.markdown("---")
st.caption(
    "Research Paper Q&A Assistant · RAG + FAISS + ChromaDB + LangChain + LoRA · "
    "[GitHub](https://github.com/ViditGandhi18/research-paper-qa-rag)"
)
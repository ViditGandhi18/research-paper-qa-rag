"""
retriever.py — Hybrid retrieval: FAISS (dense ANN) + ChromaDB (metadata-aware).
Results merged with Reciprocal Rank Fusion (RRF).

Usage:
    from src.retriever import HybridRetriever
    retriever = HybridRetriever()
    docs = retriever.retrieve("What is attention mechanism?", top_k=5)
"""

import pickle
import time
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np
import chromadb
from chromadb.config import Settings
from loguru import logger

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.utils import cfg


# Data class 
@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float          
    retriever: str        


# Helpers 
def _reciprocal_rank_fusion(
    faiss_ids: list[str],
    chroma_ids: list[str],
    k: int = 60,
) -> dict[str, float]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = Σ 1 / (k + rank_i)
    Higher score = better.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(faiss_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(chroma_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# Main Retriever
class HybridRetriever:
    """
    Hybrid retriever combining FAISS dense search + ChromaDB semantic search.
    Uses RRF to merge ranked results from both sources.
    """

    def __init__(self):
        self._embedding_model: HuggingFaceEmbeddings | None = None
        self._chroma_store: Chroma | None = None
        self._faiss_index: faiss.Index | None = None
        self._faiss_chunks: list[dict] | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        logger.info("Loading embedding model...")
        self._embedding_model = HuggingFaceEmbeddings(
            model_name=cfg.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Load ChromaDB
        logger.info("Loading ChromaDB store...")
        client = chromadb.PersistentClient(
            path=cfg.CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self._chroma_store = Chroma(
            client=client,
            collection_name="research_papers",
            embedding_function=self._embedding_model,
        )

        # Load FAISS
        index_file = Path(cfg.FAISS_INDEX_PATH) / "index.faiss"
        meta_file  = Path(cfg.FAISS_INDEX_PATH) / "metadata.pkl"
        if not index_file.exists():
            raise FileNotFoundError(
                "FAISS index not found. Run `python src/ingest.py` first."
            )
        logger.info("Loading FAISS index...")
        self._faiss_index = faiss.read_index(str(index_file))
        with open(meta_file, "rb") as f:
            self._faiss_chunks = pickle.load(f)

        self._loaded = True
        logger.success("HybridRetriever loaded.")

    # FAISS search
    def _faiss_search(self, query: str, top_k: int = 20) -> list[tuple[str, dict]]:
        """Returns list of (chunk_id, chunk_dict) sorted by cosine similarity."""
        query_vec = np.array(
            [self._embedding_model.embed_query(query)], dtype=np.float32
        )
        scores, indices = self._faiss_index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self._faiss_chunks):
                chunk = self._faiss_chunks[idx]
                results.append((chunk["chunk_id"], chunk))
        return results

    # ChromaDB search
    def _chroma_search(
        self,
        query: str,
        top_k: int = 20,
        filter_source: str | None = None,
    ) -> list[tuple[str, Document]]:
        """Returns list of (chunk_id, Document) sorted by chroma similarity."""
        where = {"source": filter_source} if filter_source else None
        docs = self._chroma_store.similarity_search_with_score(
            query, k=top_k, filter=where
        )
        # docs = [(Document, score), ...]
        results = []
        for doc, _score in docs:
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:40])
            results.append((chunk_id, doc))
        return results

    # Public API 
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_source: str | None = None,
    ) -> tuple[list[RetrievedChunk], dict]:
        """
        Main retrieval function.

        Args:
            query         : natural language question
            top_k         : number of final chunks to return (default: cfg.TOP_K_RETRIEVAL)
            filter_source : restrict ChromaDB to a specific PDF filename

        Returns:
            (chunks, timing_info)
        """
        self._load()
        top_k = top_k or cfg.TOP_K_RETRIEVAL

        t0 = time.time()

        # 1. FAISS search
        t_faiss_start = time.time()
        faiss_results = self._faiss_search(query, top_k=20)
        t_faiss = time.time() - t_faiss_start

        # 2. ChromaDB search
        t_chroma_start = time.time()
        chroma_results = self._chroma_search(query, top_k=20, filter_source=filter_source)
        t_chroma = time.time() - t_chroma_start

        # 3. Build ID → data maps
        faiss_ids  = [cid for cid, _ in faiss_results]
        chroma_ids = [cid for cid, _ in chroma_results]

        faiss_map  = {cid: chunk for cid, chunk in faiss_results}
        chroma_map = {cid: doc   for cid, doc   in chroma_results}

        # 4. Merge with RRF
        merged_scores = _reciprocal_rank_fusion(faiss_ids, chroma_ids)

        # 5. Build final result list
        final_chunks: list[RetrievedChunk] = []
        seen = set()

        for chunk_id, rrf_score in merged_scores.items():
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            in_faiss  = chunk_id in faiss_map
            in_chroma = chunk_id in chroma_map
            source_tag = "both" if (in_faiss and in_chroma) else ("faiss" if in_faiss else "chroma")

            if in_faiss:
                chunk = faiss_map[chunk_id]
                text, source, page = chunk["text"], chunk["source"], chunk["page"]
            else:
                doc = chroma_map[chunk_id]
                text   = doc.page_content
                source = doc.metadata.get("source", "unknown")
                page   = doc.metadata.get("page", 0)

            final_chunks.append(RetrievedChunk(
                text=text,
                source=source,
                page=page,
                score=round(rrf_score, 6),
                retriever=source_tag,
            ))

            if len(final_chunks) >= top_k:
                break

        timing = {
            "faiss_ms":    round(t_faiss  * 1000, 1),
            "chroma_ms":   round(t_chroma * 1000, 1),
            "total_ms":    round((time.time() - t0) * 1000, 1),
            "n_retrieved": len(final_chunks),
        }

        logger.debug(
            f"Retrieved {len(final_chunks)} chunks | "
            f"FAISS: {timing['faiss_ms']}ms | Chroma: {timing['chroma_ms']}ms"
        )
        return final_chunks, timing

    def as_langchain_retriever(self, top_k: int | None = None):
        """
        Wraps this retriever as a LangChain BaseRetriever for use in LCEL chains.
        """
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun

        outer = self
        _k    = top_k or cfg.TOP_K_RETRIEVAL

        class _LCRetriever(BaseRetriever):
            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun,
            ) -> list[Document]:
                chunks, _ = outer.retrieve(query, top_k=_k)
                return [
                    Document(
                        page_content=c.text,
                        metadata={"source": c.source, "page": c.page, "score": c.score},
                    )
                    for c in chunks
                ]
        return _LCRetriever()
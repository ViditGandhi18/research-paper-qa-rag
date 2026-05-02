"""
ingest.py — Load PDFs → chunk → embed → store in ChromaDB + FAISS.

Usage:
    python src/ingest.py --input_dir data/papers/
    python src/ingest.py --input_dir data/papers/ --reset   # wipe & re-index
"""

import argparse
import pickle
import time
from pathlib import Path

import fitz                          # PyMuPDF
import numpy as np
import faiss
from tqdm import tqdm
from loguru import logger

import chromadb
from chromadb.config import Settings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.utils import cfg, save_json


# PDF Loader
def load_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text page-by-page from a PDF.
    Returns list of dicts: {text, page, source}
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if len(text) > 50:                  
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": Path(pdf_path).name,
            })
    doc.close()
    logger.info(f"Loaded {len(pages)} pages from '{Path(pdf_path).name}'")
    return pages

def load_all_pdfs(input_dir: str) -> list[dict]:
    """Load all PDFs in a directory."""
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in '{input_dir}'")
    logger.info(f"Found {len(pdf_files)} PDF(s) in '{input_dir}'")

    all_pages = []
    for pdf_path in pdf_files:
        all_pages.extend(load_pdf(str(pdf_path)))
    return all_pages

# Chunking 
def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split pages into overlapping chunks.
    Preserves metadata (source, page).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "text": split,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": f"{page['source']}_p{page['page']}_c{i}",
            })

    logger.info(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

# Embeddings
def get_embedding_model() -> HuggingFaceEmbeddings:
    logger.info(f"Loading embedding model: {cfg.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=cfg.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ChromaDB Store
def build_chroma_store(
    chunks: list[dict],
    embedding_model: HuggingFaceEmbeddings,
    reset: bool = False,
) -> Chroma:
    """
    Upsert chunks into ChromaDB with metadata.
    Supports incremental updates (skip existing chunk_ids).
    """
    persist_dir = cfg.CHROMA_PERSIST_DIR
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    if reset:
        try:
            client.delete_collection("research_papers")
            logger.warning("Existing ChromaDB collection deleted.")
        except Exception:
            pass

    vectorstore = Chroma(
        client=client,
        collection_name="research_papers",
        embedding_function=embedding_model,
    )

    texts      = [c["text"]     for c in chunks]
    metadatas  = [{"source": c["source"], "page": c["page"]} for c in chunks]
    ids        = [c["chunk_id"] for c in chunks]

    existing_ids = set(vectorstore._collection.get(ids=ids)["ids"])
    new_indices  = [i for i, cid in enumerate(ids) if cid not in existing_ids]

    if new_indices:
        logger.info(f"Adding {len(new_indices)} new chunks to ChromaDB...")
        vectorstore.add_texts(
            texts=[texts[i]     for i in new_indices],
            metadatas=[metadatas[i] for i in new_indices],
            ids=[ids[i]         for i in new_indices],
        )
    else:
        logger.info("ChromaDB already up-to-date. No new chunks added.")

    logger.success(f"ChromaDB store ready at '{persist_dir}'")
    return vectorstore


# FAISS Store 
def build_faiss_index(
    chunks: list[dict],
    embedding_model: HuggingFaceEmbeddings,
    reset: bool = False,
) -> None:
    """
    Build a FAISS IVF Flat index from chunk embeddings.
    Saves index + chunk metadata to disk.
    """
    index_dir = Path(cfg.FAISS_INDEX_PATH)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_file = index_dir / "index.faiss"
    meta_file  = index_dir / "metadata.pkl"

    if index_file.exists() and not reset:
        logger.info("FAISS index already exists. Skipping rebuild (use --reset to rebuild).")
        return

    logger.info(f"Embedding {len(chunks)} chunks for FAISS...")
    texts = [c["text"] for c in chunks]

    embeddings = []
    batch_size = 64
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        embeddings.extend(embedding_model.embed_documents(batch))

    vectors = np.array(embeddings, dtype=np.float32)
    dim     = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(index_file))
    with open(meta_file, "wb") as f:
        pickle.dump(chunks, f)

    logger.success(f"FAISS index saved → {index_file} ({len(chunks)} vectors, dim={dim})")

# Main 
def ingest(input_dir: str = "data/papers/", reset: bool = False) -> None:
    start = time.time()

    pages  = load_all_pdfs(input_dir)
    chunks = chunk_pages(pages)
    model  = get_embedding_model()

    build_chroma_store(chunks, model, reset=reset)
    build_faiss_index(chunks, model, reset=reset)

    save_json(
        [{"chunk_id": c["chunk_id"], "source": c["source"], "page": c["page"]} for c in chunks],
        "data/vectorstore/chunk_manifest.json",
    )

    logger.success(f"Ingestion complete in {time.time() - start:.1f}s — {len(chunks)} chunks indexed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into vector stores")
    parser.add_argument("--input_dir", default="data/papers/", help="Folder containing PDF files")
    parser.add_argument("--reset", action="store_true", help="Wipe existing indexes and rebuild")
    args = parser.parse_args()
    ingest(args.input_dir, args.reset)
"""
utils.py — Shared helpers: config loading, logging, path management.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Config
class Config:
    """Central config loaded from .env"""

    OPENAI_API_KEY: str        = os.getenv("OPENAI_API_KEY", "")
    HF_TOKEN: str              = os.getenv("HUGGINGFACE_TOKEN", "")

    CHROMA_PERSIST_DIR: str    = os.getenv("CHROMA_PERSIST_DIR", "data/vectorstore/chroma")
    FAISS_INDEX_PATH: str      = os.getenv("FAISS_INDEX_PATH", "data/vectorstore/faiss_index")

    EMBEDDING_MODEL: str       = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    LLM_PROVIDER: str          = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_MODEL: str          = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    HF_MODEL_ID: str           = os.getenv("HF_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    LORA_ADAPTER_PATH: str     = os.getenv("LORA_ADAPTER_PATH", "models/lora_adapter")

    CHUNK_SIZE: int            = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP: int         = int(os.getenv("CHUNK_OVERLAP", 64))
    TOP_K_RETRIEVAL: int       = int(os.getenv("TOP_K_RETRIEVAL", 5))

    EVAL_RESULTS_PATH: str     = os.getenv("EVAL_RESULTS_PATH", "evaluation/eval_results.json")

    @classmethod
    def ensure_dirs(cls):
        """Create all required directories if they don't exist."""
        dirs = [
            cls.CHROMA_PERSIST_DIR,
            Path(cls.FAISS_INDEX_PATH).parent,
            "models/lora_adapter",
            "evaluation",
            "data/papers",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        logger.info("All directories verified.")

cfg = Config()

# JSON helpers
def save_json(data: dict | list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved → {path}")

def load_json(path: str) -> dict | list:
    with open(path, "r") as f:
        return json.load(f)

# Logging setup
def setup_logger(level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level=level,
        colorize=True,
    )
"""
rag_chain.py — LangChain LCEL RAG chain.
Supports OpenAI GPT and local HuggingFace (+ LoRA fine-tuned) models.

Usage:
    from src.rag_chain import build_rag_chain
    chain = build_rag_chain()
    result = chain.invoke("What is self-attention in transformers?")
"""

import time
from pathlib import Path

from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from src.utils import cfg
from src.retriever import HybridRetriever


# Prompt Template
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert research assistant. Answer the user's question \
using ONLY the provided context from research papers. 
If the answer is not in the context, say: "I couldn't find relevant information \
in the uploaded papers."
Always cite the source paper name and page number at the end of your answer.

Context:
{context}
""",
    ),
    ("human", "{question}"),
])

# LLM Loaders

def _load_openai_llm():
    from langchain_openai import ChatOpenAI
    logger.info(f"Loading OpenAI model: {cfg.OPENAI_MODEL}")
    return ChatOpenAI(
        model=cfg.OPENAI_MODEL,
        temperature=0.1,
        openai_api_key=cfg.OPENAI_API_KEY,
    )

def _load_hf_llm(use_lora: bool = False):
    """
    Load a local HuggingFace model.
    If use_lora=True and adapter path exists, load LoRA fine-tuned model.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel
    from langchain_huggingface import HuggingFacePipeline

    model_id = cfg.HF_MODEL_ID
    adapter_path = cfg.LORA_ADAPTER_PATH

    logger.info(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        load_in_4bit=torch.cuda.is_available(),   # 4-bit only on GPU
    )

    if use_lora and Path(adapter_path).exists():
        logger.info(f"Loading LoRA adapter from '{adapter_path}'...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        logger.success("LoRA adapter merged into base model.")
    else:
        if use_lora:
            logger.warning(f"LoRA adapter not found at '{adapter_path}'. Using base model.")

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        repetition_penalty=1.1,
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)

def get_llm(use_lora: bool = False):
    """Return the appropriate LLM based on LLM_PROVIDER env var."""
    if cfg.LLM_PROVIDER == "openai":
        return _load_openai_llm()
    return _load_hf_llm(use_lora=use_lora)

# Context Formatter 

def _format_context(docs) -> str:
    """Format retrieved documents into a clean context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[{i}] Source: {source} | Page: {page}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)

# RAG Chain Builder 
def build_rag_chain(use_lora: bool = False):
    """
    Build and return a LangChain LCEL RAG chain.

    The chain:
      question → retriever → format context → prompt → LLM → string output

    Returns a callable: chain.invoke({"question": "...", "filter_source": None})
    """
    retriever_wrapper = HybridRetriever()
    lc_retriever      = retriever_wrapper.as_langchain_retriever()
    llm               = get_llm(use_lora=use_lora)

    def retrieve_and_format(inputs: dict) -> dict:
        question = inputs["question"]
        docs     = lc_retriever.invoke(question)
        return {
            "context":  _format_context(docs),
            "question": question,
            "source_docs": docs,
        }

    chain = (
        RunnableLambda(retrieve_and_format)
        | RunnablePassthrough.assign(
            answer=RAG_PROMPT | llm | StrOutputParser()
        )
    )

    logger.success("RAG chain built successfully.")
    return chain

# Convenience wrapper 
class RAGAssistant:
    """
    High-level wrapper around the RAG chain.
    Tracks per-query timing and source attribution.
    """

    def __init__(self, use_lora: bool = False):
        self._chain = None
        self._retriever = HybridRetriever()
        self.use_lora = use_lora

    def _ensure_chain(self):
        if self._chain is None:
            self._chain = build_rag_chain(use_lora=self.use_lora)

    def ask(self, question: str) -> dict:
        """
        Ask a question. Returns:
            {
                "answer"       : str,
                "source_docs"  : list of LangChain Documents,
                "timing"       : {"retrieval_ms", "generation_ms", "total_ms"},
            }
        """
        self._ensure_chain()

        t0 = time.time()

        # Retrieval timing
        t_ret_start = time.time()
        chunks, ret_timing = self._retriever.retrieve(question)
        t_ret = time.time() - t_ret_start

        # Full chain (includes retrieval + generation)
        result = self._chain.invoke({"question": question})

        t_total = time.time() - t0
        t_gen   = t_total - t_ret

        return {
            "answer":      result["answer"],
            "source_docs": result.get("source_docs", []),
            "context":     result.get("context", ""),
            "timing": {
                "retrieval_ms":  round(t_ret * 1000, 1),
                "generation_ms": round(t_gen * 1000, 1),
                "total_ms":      round(t_total * 1000, 1),
            },
        }
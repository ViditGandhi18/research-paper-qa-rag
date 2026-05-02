# Hybrid RAG System for Research Paper Q&A with LoRA Fine-Tuning and Automated LLM Evaluation

> A production-ready RAG (Retrieval-Augmented Generation) system for answering questions from research papers, with hybrid vector search, LoRA fine-tuning, automated LLM evaluation, and a Streamlit UI.

# Project Structure

research_rag/
├── src/
│   ├── ingest.py            # PDF ingestion, chunking, embedding
│   ├── retriever.py         # FAISS + ChromaDB hybrid retriever
│   ├── rag_chain.py         # LangChain RAG pipeline
│   ├── finetune.py          # LoRA fine-tuning with QLoRA
│   ├── evaluate.py          # LLM evaluation (RAGAS + latency)
│   └── utils.py             # Shared helpers
├── app.py                   # Streamlit frontend
├── data/
│   ├── papers/              # Upload your PDFs here
│   └── vectorstore/         # Persisted FAISS & ChromaDB indexes
├── models/
│   └── lora_adapter/        # Saved LoRA weights after training
├── evaluation/
│   └── eval_results.json    # Evaluation outputs
├── notebooks/
│   ├── 01_ingestion_demo.ipynb
│   ├── 02_retrieval_demo.ipynb
│   └── 03_finetune_demo.ipynb
├── requirements.txt
├── .env.example
└── README.md

# Quick Start

# 1. Clone the repository
```bash
git clone https://github.com/ViditGandhi18/research-rag-assistant.git
cd research-rag-assistant
```

# 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
venv\Scripts\activate             # Windows
```
# 3. Install dependencies
```bash
pip install -r requirements.txt
```
# 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys
```

# 5. Ingest your research papers
```bash
python src/ingest.py --input_dir data/papers/
```

# 6. Run the Streamlit app
```bash
streamlit run app.py
```

#  Step-by-Step Methodology

# Step 1 — PDF Ingestion & Chunking
- Load PDF research papers using `PyMuPDF`
- Split into semantic chunks using LangChain's `RecursiveCharacterTextSplitter` (chunk size: 512, overlap: 64)
- Extract metadata: title, authors, year, page number

# Step 2 — Embedding & Vector Storage
- Embed chunks using `sentence-transformers/all-MiniLM-L6-v2`
- Store in **ChromaDB** (persistent, metadata-filtered queries)
- Also index in **FAISS** (fast ANN search for large corpora)
- Both stores are kept in sync via `src/ingest.py`

# Step 3 — Hybrid Retrieval
- Query hits both FAISS (top-20) and ChromaDB (top-20)
- Results merged using **Reciprocal Rank Fusion (RRF)**
- Final top-5 chunks passed to the generator

# Step 4 — LangChain RAG Chain
- LangChain LCEL chain: `retriever | prompt | llm | output_parser`
- Custom prompt template injects retrieved context + question
- Supports both OpenAI GPT and local HuggingFace models

# Step 5 — LoRA Fine-Tuning (Optional Enhancement)
- Fine-tune `TinyLlama-1.1B` or `Mistral-7B` on custom QA pairs
- Uses **QLoRA** (4-bit quantization + LoRA rank-8 adapters via PEFT)
- Swap the base LLM in the RAG chain with your fine-tuned model

# Step 6 — LLM Evaluation
- **RAGAS metrics**: faithfulness, answer relevancy, context recall
- **Latency tracking**: retrieval time, generation time, total time
- **Hallucination detection**: NLI-based entailment check
- All results logged to `evaluation/eval_results.json`

# Step 7 — Streamlit UI
- Upload PDFs directly from the browser
- Ask questions and see answers with source citations
- View evaluation scores per query in real time

# Tech Stack

| Component           | Technology               |
| Vector Databasese   | ChromaDB, FAISS          |
| Orchestration       | LangChain LCEL           |
| Embeddings          | sentence-transformers    |
| LLM (API)           | OpenAI GPT-3.5 / GPT-4   |
| Fine-tuned LLM      | TinyLlama + QLoRA / PEFT |
| Fine-tuning         | PEFT, TRL, bitsandbytes  |
| Evaluation          | RAGAS, DeepEval          |
| UI                  | Streamlit                |

# Sample Evaluation Results

Faithfulness Score    : 0.874
Answer Relevancy      : 0.912
Context Recall        : 0.856
Avg Retrieval Latency : 0.31s
Avg Generation Latency: 1.09s
Hallucination Rate    : ~4.8%

# Environment Variables

```env
OPENAI_API_KEY=your_openai_key_here
HUGGINGFACE_TOKEN=your_hf_token_here
CHROMA_PERSIST_DIR=data/vectorstore/chroma
FAISS_INDEX_PATH=data/vectorstore/faiss_index
```

# License
MIT License — free to use, modify, and distribute.

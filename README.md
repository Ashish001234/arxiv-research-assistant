# ArXiv RAG — Academic Research Search Engine

A Retrieval-Augmented Generation (RAG) research assistant built for CSCE 670 (Information Storage and Retrieval) at Texas A&M University. Search 893K+ arXiv CS papers using hybrid retrieval and get LLM-generated cited answers.

**Team:** Ashish Molakalapalli, Gagan Kumar Chowdary, Nandhini Valiveti
**Course:** CSCE 670 — Spring 2026, Texas A&M University

---

## Features

- **4 retrieval methods** you can compare side-by-side:
  - BM25 (sparse keyword search)
  - Dense (semantic vector search via `all-MiniLM-L6-v2`)
  - Hybrid (BM25 + Dense fused with Reciprocal Rank Fusion)
  - Hybrid + Cross-Encoder Re-ranking
- **RAG answer generation** — ask a question, get a cited answer from top papers (Groq/Gemini)
- **893K+ CS papers** from the arXiv dataset (filtered from ~2.97M total)
- Dark sci-fi UI with filters, pagination, and autocomplete

---

## Prerequisites

- Python 3.10+
- NVIDIA GPU recommended for encoding (CPU works but is slower)
- A free [Groq API key](https://console.groq.com/keys) or [Gemini API key](https://aistudio.google.com/apikey) for the RAG answer feature

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd academic-rag
pip install -r requirements.txt
```

> **GPU users (Linux/Mac):** Replace `faiss-cpu` in `requirements.txt` with `faiss-gpu` for faster vector search.
> **PyTorch with CUDA:** `pip install torch --index-url https://download.pytorch.org/whl/cu124`

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

```env
LLM_PROVIDER=groq          # "groq" or "gemini"
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=            # optional fallback
```

Search mode works without any API key. Only "Ask a Question" mode requires one.

### 3. Download the dataset

```bash
python -m backend.data.download_data
```

This downloads the arXiv dataset (~4 GB) via kagglehub and saves it to `data/`.

### 4. Preprocess (filter CS papers)

```bash
python -m backend.data.preprocess
```

Filters ~2.97M papers down to ~893K CS papers and saves `data/arxiv_cs_papers.jsonl`.

### 5. Build the BM25 index

```bash
python -m backend.indexing.bm25_index
```

Builds and pickles the BM25 index (~1.2 GB). Takes ~10–15 minutes.

### 6. Build the FAISS dense index

```bash
python -m backend.indexing.faiss_index
```

Encodes all abstracts with `all-MiniLM-L6-v2` and builds the FAISS index (~1.3 GB).
Takes ~5–10 min on GPU, ~30–60 min on CPU.

---

## Running the Server

```bash
uvicorn backend.api:app --reload
```

Open **http://localhost:8000** in your browser.

**Startup time:** ~20–30 seconds (loads BM25 + FAISS + models into memory once).

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/search` | GET | Search papers (`q`, `method`, `top_k`, `category`, `year_min`, `year_max`) |
| `/api/ask` | POST | RAG answer generation (`{"query": "..."}`) |
| `/api/stats` | GET | Dataset stats (total papers, categories) |
| `/api/suggest` | GET | Autocomplete suggestions (`q`) |

Interactive docs: **http://localhost:8000/docs**

**Search methods:** `bm25` · `dense` · `hybrid` · `hybrid_rerank`

```bash
# Example search
curl "http://localhost:8000/api/search?q=transformer+attention&method=hybrid_rerank&top_k=5"

# Example ask
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What are recent techniques for reducing hallucination in LLMs?"}'
```

---

## Project Structure

```
academic-rag/
├── backend/
│   ├── config.py                 # Paths, model names, hyperparameters
│   ├── api.py                    # FastAPI app (search + ask + stats endpoints)
│   ├── data/
│   │   ├── download_data.py      # Download arXiv dataset via kagglehub
│   │   └── preprocess.py         # Filter to CS papers, save as JSONL
│   ├── indexing/
│   │   ├── bm25_index.py         # Build and pickle BM25 index
│   │   └── faiss_index.py        # Encode abstracts, build FAISS index
│   ├── retrieval/
│   │   ├── bm25_retriever.py     # BM25 search
│   │   ├── dense_retriever.py    # Dense vector search
│   │   ├── hybrid_fusion.py      # Reciprocal Rank Fusion (RRF, k=60)
│   │   └── reranker.py           # Cross-encoder re-ranking
│   ├── generation/
│   │   └── answer_generator.py   # LLM answer generation (Groq / Gemini)
│   └── evaluation/
│       ├── auto_label.py         # LLM-as-judge labeling of test_queries.json
│       ├── evaluate.py           # P@5, nDCG@10, MRR across 4 methods
│       ├── benchmark_http.py     # End-to-end HTTP latency benchmark
│       ├── test_queries.json     # 10 evaluation queries with relevance pools
│       ├── results.md            # Retrieval evaluation results
│       ├── latency_results.json  # Aggregated latency stats from benchmark_http
│       └── latency_raw.csv       # Raw per-query latency timings
├── frontend/
│   └── index.html                # Single-file dark UI
├── data/                         # Auto-generated (gitignored)
│   ├── arxiv_cs_papers.jsonl     # 893K filtered CS papers
│   ├── bm25_index.pkl            # BM25 index (~1.2 GB)
│   ├── faiss_index.bin           # FAISS dense index (~1.3 GB)
│   └── embeddings.npy            # Dense embeddings
├── .env                          # API keys (gitignored)
├── .env.example                  # Template
├── requirements.txt
└── README.md
```

---

## Evaluation

The repo ships with a 10-query test set in `backend/evaluation/test_queries.json` and pre-computed silver labels (LLM-as-judge via Groq).

**Retrieval ablation** — compares P@5, nDCG@10, and MRR across all four methods:

```bash
python -m backend.evaluation.evaluate
```

**Re-label test queries** (regenerate `relevant_ids` via Llama 3.3 70B):

```bash
python -m backend.evaluation.auto_label
```

**Latency benchmark** — measures end-to-end HTTP latency for every retrieval method, including cache hit vs miss. The server must be running (in another terminal):

```bash
python -m backend.evaluation.benchmark_http
```

Aggregated results are written to `backend/evaluation/latency_results.json` and raw per-query timings to `latency_raw.csv`.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Sparse retrieval | `rank-bm25` (BM25Okapi) |
| Dense retrieval | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector index | `faiss-cpu` / `faiss-gpu` (384-dim cosine similarity) |
| Hybrid fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| LLM generation | Groq (`llama-3.3-70b-versatile`) / Gemini (`gemini-2.0-flash`) |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla HTML/CSS/JS |

---

## References

- Gao et al., 2023 — RAG survey (arXiv:2312.10997)
- Reimers and Gurevych, 2019 — Sentence-BERT (EMNLP 2019)
- Johnson et al., 2019 — FAISS (IEEE Transactions on Big Data)
- Cormack et al., 2009 — RRF (SIGIR 2009)
- Nogueira and Cho, 2019 — Passage re-ranking with BERT (arXiv:1901.04085)

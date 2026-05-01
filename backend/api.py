"""
FastAPI application — loads all models/indexes at startup, serves search + RAG endpoints.
"""
import os
import json
from contextlib import asynccontextmanager
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.config import (
    PAPERS_JSONL_PATH,
    BM25_INDEX_PATH,
    FAISS_INDEX_PATH,
    DENSE_MODEL_NAME,
    RERANKER_MODEL_NAME,
    BM25_TOP_K,
    DENSE_TOP_K,
    RERANK_TOP_N,
    RERANK_CANDIDATES,
    BASE_DIR,
)
from backend.retrieval.bm25_retriever import bm25_search
from backend.retrieval.dense_retriever import dense_search
from backend.retrieval.hybrid_fusion import reciprocal_rank_fusion
from backend.retrieval.reranker import load_reranker, rerank
from backend.generation.answer_generator import generate_answer


# ---------------------------------------------------------------------------
# Global state — populated at startup
# ---------------------------------------------------------------------------
state: dict = {}

# Simple LRU-style caches (search results + LLM answers)
_search_cache: dict = {}   # key: (query, method, top_k, category, year_min, year_max)
_ask_cache: dict = {}      # key: query (normalised)
_SEARCH_CACHE_MAX = 500
_ASK_CACHE_MAX = 200


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading papers...")
    papers = []
    with open(PAPERS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    state["papers"] = papers
    print(f"  {len(papers):,} papers loaded.")

    print("Loading BM25 index...")
    import pickle
    with open(BM25_INDEX_PATH, "rb") as f:
        state["bm25"] = pickle.load(f)
    print("  BM25 index loaded.")

    print("Loading sentence-transformer model...")
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["dense_model"] = SentenceTransformer(DENSE_MODEL_NAME, device=device)
    print(f"  Model loaded on {device}.")

    print("Loading FAISS index...")
    import faiss
    state["faiss_index"] = faiss.read_index(FAISS_INDEX_PATH)
    print(f"  FAISS index loaded ({state['faiss_index'].ntotal:,} vectors, CPU).")

    print("Loading cross-encoder re-ranker...")
    state["reranker"] = load_reranker(RERANKER_MODEL_NAME)
    print("  Re-ranker loaded.")

    print("All models ready. Server starting...")
    yield

    state.clear()


app = FastAPI(title="Academic RAG Search Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_dir = os.path.join(BASE_DIR, "frontend")
app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dir, "assets")), name="assets")


@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _paper_to_dict(paper: dict, rank: int, score: float, method: str) -> dict:
    year = paper.get("update_date", "")[:4] if paper.get("update_date") else ""
    cats = paper.get("categories", "").split()
    # arXiv link
    arxiv_id = paper.get("id", "")
    return {
        "rank": rank,
        "id": arxiv_id,
        "title": paper["title"],
        "authors": paper.get("authors", ""),
        "year": year,
        "categories": cats,
        "abstract": paper["abstract"],
        "score": round(score, 4),
        "method": method,
        "url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def _run_retrieval(query: str, method: str, top_k: int) -> list[tuple[int, float]]:
    papers = state["papers"]
    bm25 = state["bm25"]
    dense_model = state["dense_model"]
    faiss_index = state["faiss_index"]
    reranker = state["reranker"]

    if method == "bm25":
        return bm25_search(query, bm25, top_k=top_k)

    elif method == "dense":
        return dense_search(query, dense_model, faiss_index, top_k=top_k)

    elif method == "hybrid":
        bm25_results = bm25_search(query, bm25, top_k=BM25_TOP_K)
        dense_results = dense_search(query, dense_model, faiss_index, top_k=DENSE_TOP_K)
        fused = reciprocal_rank_fusion([bm25_results, dense_results])
        return fused[:top_k]

    elif method == "hybrid_rerank":
        bm25_results = bm25_search(query, bm25, top_k=BM25_TOP_K)
        dense_results = dense_search(query, dense_model, faiss_index, top_k=DENSE_TOP_K)
        fused = reciprocal_rank_fusion([bm25_results, dense_results])
        reranked = rerank(
            query, fused, papers, reranker,
            top_n=RERANK_TOP_N,
            n_candidates=RERANK_CANDIDATES,
        )
        return reranked

    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    method: str = Query("hybrid_rerank", pattern="^(bm25|dense|hybrid|hybrid_rerank)$"),
    top_k: int = Query(10, ge=1, le=50),
    category: str = Query(None),
    year_min: int = Query(None),
    year_max: int = Query(None),
):
    cache_key = (q.lower().strip(), method, top_k, category, year_min, year_max)
    if cache_key in _search_cache:
        return _search_cache[cache_key]

    papers = state["papers"]
    results = _run_retrieval(q, method, top_k=top_k if method != "hybrid_rerank" else BM25_TOP_K)

    # For non-rerank methods, trim to top_k
    if method != "hybrid_rerank":
        results = results[:top_k]

    # Apply filters and build response
    output = []
    rank = 1
    for doc_id, score in results:
        paper = papers[doc_id]

        # Year filter
        year = paper.get("update_date", "")[:4]
        if year_min and year and int(year) < year_min:
            continue
        if year_max and year and int(year) > year_max:
            continue

        # Category filter
        if category and category not in paper.get("categories", ""):
            continue

        output.append(_paper_to_dict(paper, rank, score, method))
        rank += 1
        if len(output) >= top_k:
            break

    response = {"query": q, "method": method, "total": len(output), "results": output}

    if len(_search_cache) >= _SEARCH_CACHE_MAX:
        _search_cache.pop(next(iter(_search_cache)))
    _search_cache[cache_key] = response
    return response


class AskRequest(BaseModel):
    query: str


@app.post("/api/ask")
async def ask(body: AskRequest):
    q = body.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    cache_key = q.lower()
    if cache_key in _ask_cache:
        return _ask_cache[cache_key]

    papers = state["papers"]
    results = _run_retrieval(q, "hybrid_rerank", top_k=BM25_TOP_K)
    top_results = results[:RERANK_TOP_N]

    top_papers = [papers[doc_id] for doc_id, _ in top_results]
    answer = await generate_answer(q, top_papers)

    sources = [
        _paper_to_dict(papers[doc_id], rank + 1, score, "hybrid_rerank")
        for rank, (doc_id, score) in enumerate(top_results)
    ]

    response = {"query": q, "answer": answer, "sources": sources}

    if len(_ask_cache) >= _ASK_CACHE_MAX:
        _ask_cache.pop(next(iter(_ask_cache)))
    _ask_cache[cache_key] = response
    return response


@app.get("/api/stats")
def stats():
    papers = state.get("papers", [])
    categories = set()
    for p in papers:
        for cat in p.get("categories", "").split():
            if cat.startswith("cs."):
                categories.add(cat)
    return {
        "total_papers": len(papers),
        "cs_categories": len(categories),
        "index_loaded": bool(state),
    }


@app.get("/api/suggest")
def suggest(q: str = Query(..., min_length=2)):
    """Simple autocomplete — returns paper titles containing the query prefix."""
    q_lower = q.lower()
    papers = state["papers"]
    suggestions = []
    for p in papers:
        if q_lower in p["title"].lower():
            suggestions.append(p["title"])
            if len(suggestions) >= 8:
                break
    return {"suggestions": suggestions}

"""
Evaluation script — computes P@5, nDCG@10, MRR across retrieval configurations.
Loads indexes directly (no server required).

Run:
    py -3.11 -m backend.evaluation.evaluate
"""
import json
import math
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

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
)
from backend.retrieval.bm25_retriever import bm25_search
from backend.retrieval.dense_retriever import dense_search
from backend.retrieval.hybrid_fusion import reciprocal_rank_fusion
from backend.retrieval.reranker import rerank

EVAL_PATH = Path(__file__).parent / "test_queries.json"
METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    hits = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hits / k


def dcg_at_k(relevant: set, retrieved: list, k: int) -> float:
    return sum(
        1.0 / math.log2(i + 1)
        for i, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant
    )


def ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
    ideal = sorted([1] * min(len(relevant), k) + [0] * max(k - len(relevant), 0), reverse=True)
    ideal_dcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return 0.0 if ideal_dcg == 0 else dcg_at_k(relevant, retrieved, k) / ideal_dcg


def mean_reciprocal_rank(relevant: set, retrieved: list) -> float:
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

def load_resources():
    print("Loading papers...")
    papers = []
    with open(PAPERS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            papers.append(json.loads(line))
    print(f"  {len(papers):,} papers loaded.")

    print("Loading BM25 index...")
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)

    print("Loading dense model + FAISS index...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(DENSE_MODEL_NAME, device=device)
    index = faiss.read_index(FAISS_INDEX_PATH)

    print("Loading cross-encoder...")
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    print("All resources loaded.\n")
    return papers, bm25, model, index, reranker


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def run_retrieval(query: str, method: str, papers, bm25, model, index, reranker, top_k=10) -> list[str]:
    if method == "bm25":
        results = bm25_search(query, bm25, top_k=top_k)

    elif method == "dense":
        results = dense_search(query, model, index, top_k=top_k)

    elif method == "hybrid":
        bm25_r = bm25_search(query, bm25, top_k=BM25_TOP_K)
        dense_r = dense_search(query, model, index, top_k=DENSE_TOP_K)
        results = reciprocal_rank_fusion([bm25_r, dense_r])[:top_k]

    elif method == "hybrid_rerank":
        bm25_r = bm25_search(query, bm25, top_k=BM25_TOP_K)
        dense_r = dense_search(query, model, index, top_k=DENSE_TOP_K)
        fused = reciprocal_rank_fusion([bm25_r, dense_r])
        results = rerank(query, fused, papers, reranker,
                         top_n=RERANK_TOP_N, n_candidates=RERANK_CANDIDATES)

    return [papers[doc_id]["id"] for doc_id, _ in results[:top_k]]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate():
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"Test queries not found: {EVAL_PATH}")

    with open(EVAL_PATH) as f:
        test_queries = json.load(f)

    unannotated = [q for q in test_queries if not q["relevant_ids"]]
    if unannotated:
        print(f"WARNING: {len(unannotated)} queries have no relevant_ids — run auto_label.py first.\n")

    papers, bm25, model, index, reranker = load_resources()

    print(f"Evaluating {len(test_queries)} queries across {len(METHODS)} methods...\n")
    aggregate = {m: {"p5": [], "ndcg10": [], "mrr": []} for m in METHODS}

    for item in test_queries:
        query = item["query"]
        relevant = set(item["relevant_ids"])
        print(f"Query: {query!r}  ({len(relevant)} relevant)")

        for method in METHODS:
            retrieved = run_retrieval(query, method, papers, bm25, model, index, reranker, top_k=10)
            p5     = precision_at_k(relevant, retrieved, 5)
            ndcg10 = ndcg_at_k(relevant, retrieved, 10)
            mrr    = mean_reciprocal_rank(relevant, retrieved)
            aggregate[method]["p5"].append(p5)
            aggregate[method]["ndcg10"].append(ndcg10)
            aggregate[method]["mrr"].append(mrr)
            print(f"  {method:20s}  P@5={p5:.3f}  nDCG@10={ndcg10:.3f}  MRR={mrr:.3f}")
        print()

    print("=" * 60)
    print(f"{'Method':<22} {'P@5':>8} {'nDCG@10':>10} {'MRR':>8}")
    print("-" * 60)
    for method in METHODS:
        scores = aggregate[method]
        n = len(scores["p5"])
        print(f"{method:<22} {sum(scores['p5'])/n:>8.4f} {sum(scores['ndcg10'])/n:>10.4f} {sum(scores['mrr'])/n:>8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
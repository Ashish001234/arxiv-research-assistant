"""
Auto-label test_queries.json using LLM-as-judge (Groq).

For each query, fetches top-10 results via hybrid_rerank, then asks
the LLM to judge each paper's relevance. Saves relevant_ids back to
test_queries.json.

Run:
    py -3.11 -m backend.evaluation.auto_label
"""
import json
import os
import pickle
import time

import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

from backend.config import (
    PAPERS_JSONL_PATH,
    BM25_INDEX_PATH,
    FAISS_INDEX_PATH,
    DENSE_MODEL_NAME,
    RERANKER_MODEL_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
    BM25_TOP_K,
    DENSE_TOP_K,
    RERANK_TOP_N,
    RERANK_CANDIDATES,
)
from backend.retrieval.bm25_retriever import bm25_search
from backend.retrieval.dense_retriever import dense_search
from backend.retrieval.hybrid_fusion import reciprocal_rank_fusion
from backend.retrieval.reranker import rerank

TEST_QUERIES_PATH = os.path.join(os.path.dirname(__file__), "test_queries.json")


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

    return papers, bm25, model, index, reranker


def retrieve_top(query, papers, bm25, model, index, reranker, top_n=10):
    bm25_results = bm25_search(query, bm25, top_k=BM25_TOP_K)
    dense_results = dense_search(query, model, index, top_k=DENSE_TOP_K)
    fused = reciprocal_rank_fusion([bm25_results, dense_results])
    reranked = rerank(query, fused, papers, reranker,
                      top_n=top_n, n_candidates=RERANK_CANDIDATES)
    return [(papers[doc_id], score) for doc_id, score in reranked]


def judge_relevance(client, query: str, paper: dict) -> bool:
    prompt = (
        f"Query: {query}\n\n"
        f"Paper title: {paper['title']}\n"
        f"Abstract: {paper['abstract'][:600]}\n\n"
        "Is this paper directly relevant to the query? "
        "Answer with a single word: YES or NO."
    )
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print(f"    LLM error: {e}")
        return False


def main():
    with open(TEST_QUERIES_PATH, "r") as f:
        test_queries = json.load(f)

    papers, bm25, model, index, reranker = load_resources()
    client = Groq(api_key=GROQ_API_KEY)

    for item in tqdm(test_queries, desc="Labelling queries"):
        query = item["query"]
        # Skip if already labelled
        if item["relevant_ids"]:
            print(f'  Skipping "{query}" (already labelled)')
            continue

        print(f'\n  Query: "{query}"')
        top_papers = retrieve_top(query, papers, bm25, model, index, reranker, top_n=10)

        relevant_ids = []
        for paper, score in top_papers:
            is_relevant = judge_relevance(client, query, paper)
            status = "YES" if is_relevant else "NO "
            print(f"    [{status}] {paper['title'][:70]}")
            if is_relevant:
                relevant_ids.append(paper["id"])
            time.sleep(0.3)  # avoid hitting Groq rate limit

        item["relevant_ids"] = relevant_ids
        print(f"  -> {len(relevant_ids)} relevant papers found")

    with open(TEST_QUERIES_PATH, "w") as f:
        json.dump(test_queries, f, indent=2)

    print(f"\nDone. Updated {TEST_QUERIES_PATH}")


if __name__ == "__main__":
    main()
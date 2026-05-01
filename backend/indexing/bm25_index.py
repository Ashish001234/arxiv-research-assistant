"""
Build and save the BM25 index from the filtered CS papers.
Run after preprocess.py.

Memory-efficient: streams JSONL line-by-line so only tokenized text
(not full paper dicts) is held in RAM during index construction.
"""
import json
import pickle
import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from backend.config import PAPERS_JSONL_PATH, BM25_INDEX_PATH


def build_bm25_index_from_jsonl(jsonl_path: str) -> BM25Okapi:
    """Stream JSONL and build BM25 index without loading full paper dicts."""
    print("Streaming and tokenizing documents...")
    tokenized = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Tokenizing"):
            p = json.loads(line)
            text = (p["title"] + " " + p["abstract"]).lower()
            tokenized.append(text.split())

    print(f"Building BM25 index over {len(tokenized):,} documents...")
    bm25 = BM25Okapi(tokenized)
    return bm25


def save_bm25_index(bm25: BM25Okapi, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bm25, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"BM25 index saved to {path} ({size_mb:.1f} MB)")


def load_bm25_index(path: str) -> BM25Okapi:
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    if not os.path.exists(PAPERS_JSONL_PATH):
        raise FileNotFoundError(
            f"Papers JSONL not found at {PAPERS_JSONL_PATH}. "
            "Run backend/data/preprocess.py first."
        )

    bm25 = build_bm25_index_from_jsonl(PAPERS_JSONL_PATH)
    save_bm25_index(bm25, BM25_INDEX_PATH)
    print("Done.")

"""
Encode all CS paper abstracts with sentence-transformers and build a FAISS index.
Run after preprocess.py.

Note: Uses GPU for encoding (fast), CPU FAISS index (faiss-gpu has no Windows wheels).
CPU FAISS search over 893K flat vectors is still sub-100ms.
"""
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from backend.config import (
    PAPERS_JSONL_PATH,
    FAISS_INDEX_PATH,
    EMBEDDINGS_PATH,
    DENSE_MODEL_NAME,
    EMBEDDING_DIM,
)


def encode_papers_streaming(jsonl_path: str, model_name: str, batch_size: int = 512) -> np.ndarray:
    """Stream JSONL and encode in batches — avoids loading all paper dicts into RAM."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Encoding papers on {device} with {model_name}...")

    model = SentenceTransformer(model_name, device=device)

    # Count lines first for tqdm
    with open(jsonl_path, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    all_embeddings = []
    batch = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="Encoding"):
            p = json.loads(line)
            batch.append(p["title"] + " " + p["abstract"])
            if len(batch) == batch_size:
                embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
                all_embeddings.append(embs.astype("float32"))
                batch = []

    if batch:
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.append(embs.astype("float32"))

    embeddings = np.vstack(all_embeddings)
    print(f"Encoded {len(embeddings):,} papers, shape {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (dim={dim}, {len(embeddings):,} vectors) on CPU...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index: faiss.Index, embeddings: np.ndarray):
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDINGS_PATH, embeddings)
    idx_mb = os.path.getsize(FAISS_INDEX_PATH) / 1e6
    emb_mb = os.path.getsize(EMBEDDINGS_PATH) / 1e6
    print(f"FAISS index saved to {FAISS_INDEX_PATH} ({idx_mb:.1f} MB)")
    print(f"Embeddings saved to {EMBEDDINGS_PATH} ({emb_mb:.1f} MB)")


def load_faiss_index(path: str) -> faiss.Index:
    index = faiss.read_index(path)
    print(f"FAISS index loaded from {path} ({index.ntotal:,} vectors).")
    return index


if __name__ == "__main__":
    if not os.path.exists(PAPERS_JSONL_PATH):
        raise FileNotFoundError(
            f"Papers JSONL not found at {PAPERS_JSONL_PATH}. "
            "Run backend/data/preprocess.py first."
        )

    embeddings = encode_papers_streaming(PAPERS_JSONL_PATH, DENSE_MODEL_NAME)
    index = build_faiss_index(embeddings)
    save_faiss_index(index, embeddings)
    print("Done.")

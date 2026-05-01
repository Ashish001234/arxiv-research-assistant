import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def dense_search(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    top_k: int = 100,
) -> list[tuple[int, float]]:
    """
    Returns list of (doc_index, score) sorted by score descending.
    Score is cosine similarity (inner product on normalized vectors).
    """
    query_vec = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(query_vec.astype("float32"), top_k)
    return [(int(indices[0][i]), float(scores[0][i])) for i in range(top_k)]

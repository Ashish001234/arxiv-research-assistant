from rank_bm25 import BM25Okapi


def bm25_search(query: str, bm25: BM25Okapi, top_k: int = 100) -> list[tuple[int, float]]:
    """
    Returns list of (doc_index, score) sorted by score descending.
    doc_index is the position in the papers list used to build the index.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(int(idx), float(scores[idx])) for idx in top_indices]

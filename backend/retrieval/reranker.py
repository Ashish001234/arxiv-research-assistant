from sentence_transformers import CrossEncoder
from backend.config import RERANKER_MODEL_NAME, RERANK_TOP_N, RERANK_CANDIDATES


def load_reranker(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    return CrossEncoder(model_name)


def rerank(
    query: str,
    candidates: list[tuple[int, float]],
    papers: list[dict],
    reranker: CrossEncoder,
    top_n: int = RERANK_TOP_N,
    n_candidates: int = RERANK_CANDIDATES,
) -> list[tuple[int, float]]:
    """
    Re-rank top candidates using a cross-encoder.

    Args:
        query: search query string
        candidates: [(doc_id, score), ...] from hybrid fusion, sorted by score
        papers: full list of paper dicts (indexed by doc_id)
        reranker: loaded CrossEncoder model
        top_n: number of results to return after re-ranking
        n_candidates: how many candidates to pass to the cross-encoder

    Returns:
        [(doc_id, cross_encoder_score), ...] sorted descending, length top_n
    """
    pool = candidates[:n_candidates]
    pairs = [
        (query, papers[doc_id]["title"] + " " + papers[doc_id]["abstract"])
        for doc_id, _ in pool
    ]
    scores = reranker.predict(pairs)
    reranked = sorted(
        zip(pool, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return [(doc_id, float(score)) for (doc_id, _), score in reranked[:top_n]]

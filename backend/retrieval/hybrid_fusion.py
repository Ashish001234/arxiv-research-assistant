from backend.config import RRF_K


def reciprocal_rank_fusion(
    results_list: list[list[tuple[int, float]]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        results_list: list of ranked result lists, each being [(doc_id, score), ...]
                      sorted by score descending.
        k: RRF smoothing constant (default 60, per Cormack et al. 2009).

    Returns:
        Fused list of (doc_id, rrf_score) sorted by rrf_score descending.
    """
    fused_scores: dict[int, float] = {}
    for results in results_list:
        for rank, (doc_id, _) in enumerate(results, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

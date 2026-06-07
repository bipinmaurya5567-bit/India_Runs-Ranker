"""
fusion.py — Step 6
Reciprocal Rank Fusion (RRF) to combine semantic + behavioral rankings.
RRF formula: score_i = Σ 1/(k + rank_i) across all score lists.
k=60 is the standard constant from the original RRF paper.
"""
from typing import Dict, Any, List, Tuple

import numpy as np

RRF_K = 60   # Standard RRF constant


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to 1-indexed ranks (highest score → rank 1).
    Returns array of same shape as scores.
    """
    n = len(scores)
    order = np.argsort(scores)[::-1]   # indices that sort descending
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = np.arange(1, n + 1)
    return ranks


def reciprocal_rank_fusion(
    scores_list: List[np.ndarray],
    k: int = RRF_K,
) -> np.ndarray:
    """
    Combine multiple ranked lists via Reciprocal Rank Fusion.

    Args:
        scores_list: List of score arrays, each of length N.
                     Higher score = better candidate in each list.
        k:           RRF constant (default 60)

    Returns:
        rrf_scores: np.ndarray of shape (N,); higher = better fused rank.
    """
    n = len(scores_list[0])
    rrf = np.zeros(n, dtype=np.float64)

    for scores in scores_list:
        ranks = _ranks_from_scores(scores)
        rrf += 1.0 / (k + ranks)

    return rrf


def get_top_k_candidates(
    candidates: List[Dict[str, Any]],
    rrf_scores: np.ndarray,
    semantic_scores: np.ndarray,
    behavioral_scores: np.ndarray,
    penalties: List[List[str]],
    k: int = 100,
) -> List[Dict[str, Any]]:
    """
    Sort candidates by RRF score (desc) and return the top-k with metadata.

    Tie-breaking: equal RRF scores → sort by candidate_id ascending
    (required by validate_submission.py).

    Returns:
        List of dicts, each containing:
          candidate, rrf_score, semantic_score, behavioral_score,
          local_idx, penalties
    """
    n = len(rrf_scores)
    entries = []
    for i in range(n):
        cid = candidates[i].get("candidate_id", f"UNKNOWN_{i}")
        entries.append({
            "candidate":        candidates[i],
            "rrf_score":        float(rrf_scores[i]),
            "semantic_score":   float(semantic_scores[i]),
            "behavioral_score": float(behavioral_scores[i]),
            "local_idx":        i,
            "penalties":        penalties[i],
            "candidate_id":     cid,
        })

    # Primary sort: RRF descending (formatted to 6 decimals to match CSV output); tie-break: candidate_id ascending
    entries.sort(key=lambda e: (-float(f"{e['rrf_score']:.6f}"), e["candidate_id"]))

    return entries[:k]

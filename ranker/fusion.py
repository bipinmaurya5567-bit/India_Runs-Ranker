"""
fusion.py — Stage 4 (upgraded: 3-way weighted RRF)
Combines cross-encoder, behavioral, and bi-encoder semantic rankings.

Formula (per candidate i):
  score_i = 0.6 × 1/(60 + cross_rank_i)
           + 0.3 × 1/(60 + behavioral_rank_i)
           + 0.1 × 1/(60 + semantic_rank_i)

Weights reflect: cross-encoder quality ≫ behavioral ≫ semantic (tiebreaker).
k=60 is the standard constant from the original RRF paper (Cormack et al. 2009).
"""
from typing import Dict, Any, List

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


def weighted_reciprocal_rank_fusion(
    cross_encoder_scores: np.ndarray,
    behavioral_scores: np.ndarray,
    semantic_scores: np.ndarray,
    k: int = RRF_K,
    w_cross: float = 0.6,
    w_behavioral: float = 0.3,
    w_semantic: float = 0.1,
) -> np.ndarray:
    """
    3-way weighted Reciprocal Rank Fusion.

    Args:
        cross_encoder_scores : Raw logits from cross-encoder (Stage 2.5), shape (N,).
        behavioral_scores    : Behavioral signal scores (Stage 3), shape (N,).
        semantic_scores      : Bi-encoder cosine similarities (Stage 2), shape (N,).
        k                    : RRF smoothing constant (default 60).
        w_cross              : Weight for cross-encoder rank  (default 0.6).
        w_behavioral         : Weight for behavioral rank      (default 0.3).
        w_semantic           : Weight for semantic rank        (default 0.1).

    Returns:
        rrf_scores : np.ndarray of shape (N,); higher = better fused rank.
    """
    cross_ranks    = _ranks_from_scores(cross_encoder_scores)
    behav_ranks    = _ranks_from_scores(behavioral_scores)
    semantic_ranks = _ranks_from_scores(semantic_scores)

    rrf = (
        w_cross      * (1.0 / (k + cross_ranks))    +
        w_behavioral * (1.0 / (k + behav_ranks))     +
        w_semantic   * (1.0 / (k + semantic_ranks))
    )
    return rrf


# Keep the old name as an alias for backwards compatibility with any
# scripts that still call reciprocal_rank_fusion directly.
def reciprocal_rank_fusion(
    scores_list: List[np.ndarray],
    k: int = RRF_K,
) -> np.ndarray:
    """
    Legacy equal-weight RRF (kept for compatibility).
    Prefer weighted_reciprocal_rank_fusion for new code.
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
    cross_encoder_scores: np.ndarray,
    penalties: List[List[str]],
    k: int = 100,
) -> List[Dict[str, Any]]:
    """
    Sort candidates by weighted RRF score (desc) and return the top-k with metadata.

    Tie-breaking: equal RRF scores → sort by candidate_id ascending
    (required by validate_submission.py).

    Returns:
        List of dicts, each containing:
          candidate, rrf_score, semantic_score, behavioral_score,
          cross_encoder_score, local_idx, penalties
    """
    n = len(rrf_scores)
    entries = []
    for i in range(n):
        cid = candidates[i].get("candidate_id", f"UNKNOWN_{i}")
        entries.append({
            "candidate":           candidates[i],
            "rrf_score":           float(rrf_scores[i]),
            "semantic_score":      float(semantic_scores[i]),
            "behavioral_score":    float(behavioral_scores[i]),
            "cross_encoder_score": float(cross_encoder_scores[i]),
            "local_idx":           i,
            "penalties":           penalties[i],
            "candidate_id":        cid,
        })

    # Primary sort: RRF descending; tie-break: candidate_id ascending
    entries.sort(key=lambda e: (-float(f"{e['rrf_score']:.6f}"), e["candidate_id"]))

    return entries[:k]

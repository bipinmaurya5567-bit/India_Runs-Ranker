"""
embed_stage.py — Stage 2 + Stage 2.5

Stage 2 : Bi-encoder semantic filtering (all-MiniLM-L6-v2)
          Encodes JD + top-2000 BM25 candidates, keeps top-500 by cosine sim.
          ~15 s on CPU.

Stage 2.5: Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
           Jointly encodes [JD, candidate] pairs → single relevance score.
           Processes top-500 in batches of 32.  ~120 s on CPU.
           This is where the biggest quality jump comes from.
"""
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

MODEL_NAME         = "all-MiniLM-L6-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """
    Load the sentence-transformers model.
    First call downloads ~90 MB; subsequent calls load from local cache.
    """
    print(f"    Loading model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print(f"    Model loaded — max seq len: {model.max_seq_length}")
    return model


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 128,
    desc: str = "Encoding",
) -> np.ndarray:
    """
    Encode a list of texts into L2-normalised embeddings.

    normalize_embeddings=True means cosine similarity == dot product,
    which is faster to compute at inference time.
    """
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → dot = cosine sim
        convert_to_numpy=True,
    )
    return embeddings


def compute_cosine_similarities(
    jd_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarities between the JD embedding and all candidate embeddings.

    Since both are L2-normalised, cosine similarity = dot product.

    Args:
        jd_embedding:          shape (dim,) or (1, dim)
        candidate_embeddings:  shape (N, dim)

    Returns:
        similarities: shape (N,) in range [-1, 1]
    """
    jd = jd_embedding.reshape(1, -1)          # (1, dim)
    sims = candidate_embeddings @ jd.T         # (N, 1)
    return sims.flatten()                      # (N,)


# ---------------------------------------------------------------------------
# Stage 2.5 — Cross-Encoder Reranking
# ---------------------------------------------------------------------------

class CrossEncoderStage:
    """
    Wraps cross-encoder/ms-marco-MiniLM-L-6-v2 for relevance reranking.

    A cross-encoder jointly encodes the query and each document together,
    attending across both, which gives far better relevance scores than a
    bi-encoder at the cost of O(N) full-model forward passes.

    Typical gain vs bi-encoder: +3–5 NDCG@10 on standard benchmarks.

    Usage
    -----
    stage = CrossEncoderStage()          # loads ~90 MB model
    scores = stage.rerank(jd_text, candidate_texts)  # shape (N,)
    """

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_NAME,
        max_length: int = 512,
    ) -> None:
        """
        Load the cross-encoder model.

        Parameters
        ----------
        model_name : HuggingFace model id or local path.
        max_length : Token budget per (query, document) pair.
                     512 covers most JD + candidate summary combos.
        """
        print(f"  [CrossEncoder] Loading {model_name} ...")
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )
        print(f"  [CrossEncoder] Model ready (max_length={max_length})")
        # Warm-up: run one dummy pair to trigger any JIT / lazy init
        # so the timing of the first real batch is not inflated.
        self.model.predict([("warm up query", "warm up document")])
        print(f"  [CrossEncoder] Warm-up complete")

    # ------------------------------------------------------------------
    def rerank(
        self,
        jd_text: str,
        candidate_texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Score every (jd_text, candidate_text) pair.

        Parameters
        ----------
        jd_text          : The job description string.
        candidate_texts  : List of N candidate profile strings.
        batch_size       : Number of pairs per forward pass.
                           32 keeps CPU memory comfortable and throughput
                           at ~4 pairs/s → ~120 s for 500 candidates.

        Returns
        -------
        scores : np.ndarray of shape (N,), higher = more relevant.
                 Raw logits from the MS-MARCO model (not probabilities).
                 Safe to rank-sort directly.
        """
        pairs: List[Tuple[str, str]] = [
            (jd_text, cand) for cand in candidate_texts
        ]

        print(
            f"  [CrossEncoder] Scoring {len(pairs)} pairs "
            f"(batch_size={batch_size}) ..."
        )
        scores: np.ndarray = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print(
            f"  [CrossEncoder] Done — "
            f"score range [{scores.min():.3f}, {scores.max():.3f}]"
        )
        return scores  # shape (N,)

    # ------------------------------------------------------------------
    def top_k_indices(
        self,
        jd_text: str,
        candidate_texts: List[str],
        k: int,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Convenience wrapper: returns indices of the top-k candidates
        sorted by cross-encoder score (descending).

        Parameters
        ----------
        jd_text          : Job description string.
        candidate_texts  : List of candidate profile strings.
        k                : How many top indices to return.
        batch_size       : Batch size for predict().

        Returns
        -------
        indices : np.ndarray of shape (min(k, N),), dtype int64.
        """
        scores = self.rerank(jd_text, candidate_texts, batch_size=batch_size)
        # argsort ascending → reverse for descending
        sorted_indices = np.argsort(scores)[::-1]
        return sorted_indices[:k]

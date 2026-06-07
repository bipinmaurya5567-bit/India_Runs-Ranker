"""
embed_stage.py — Step 4
Stage 2: Dense semantic reranking using sentence-transformers all-MiniLM-L6-v2.
Encodes only the top-2000 BM25 candidates (not all 100K).
~10-30s on CPU for 2000 candidates.
"""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


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

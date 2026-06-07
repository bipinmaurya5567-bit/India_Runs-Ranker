"""
bm25_stage.py — Step 3
Stage 1: BM25 sparse retrieval over all 100K candidates.
Retrieves top-2000 candidates for downstream semantic reranking.
"""
import re
import gc
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Stop-words that add noise without signal
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "it", "they", "their", "this", "that", "these", "those",
    "also", "as", "not", "no", "nor", "so", "yet", "both", "either",
    "neither", "whether", "while", "although", "because", "since",
    "though", "unless", "until", "when", "where", "who", "which",
}


def tokenize(text: str) -> List[str]:
    """
    Tokenize text for BM25:
    - lowercase
    - keep alphanumeric + hyphens (preserves "fine-tuning", "bi-encoder")
    - split on whitespace and punctuation
    - filter stop-words and single-char tokens
    """
    text = text.lower()
    # Replace punctuation except hyphens with spaces
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    # Split and filter
    tokens = [
        t for t in text.split()
        if len(t) > 1 and t not in _STOP_WORDS
    ]
    return tokens


def build_bm25_index(texts: List[str]) -> BM25Okapi:
    """
    Tokenize all candidate texts and build a BM25Okapi index.

    Memory note: BM25Okapi stores per-doc frequency dicts, not raw tokens.
    We explicitly delete the tokenized list after building to free RAM.
    """
    print(f"    Tokenizing {len(texts):,} documents...")
    tokenized: List[List[str]] = []
    for text in tqdm(texts, desc="    Tokenizing", unit=" docs", ncols=80):
        tokenized.append(tokenize(text))

    print("    Building BM25 index (this may take 30-60s)...")
    index = BM25Okapi(tokenized)

    # Free the tokenized corpus — BM25Okapi already extracted what it needs
    del tokenized
    gc.collect()

    return index


def retrieve_top_k(
    index: BM25Okapi,
    query_text: str,
    k: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score all documents against the query and return the top-k indices and scores.

    Args:
        index:      Trained BM25Okapi index
        query_text: Raw query string (job description text)
        k:          Number of candidates to retrieve

    Returns:
        top_indices: int array of shape (k,) — indices into the original corpus
        top_scores:  float array of shape (k,) — BM25 scores (descending)
    """
    query_tokens = tokenize(query_text)
    print(f"    Query tokens ({len(query_tokens)}): {query_tokens[:15]} ...")

    scores: np.ndarray = index.get_scores(query_tokens)

    # Partial-sort: get top-k without sorting all 100K
    if k >= len(scores):
        top_indices = np.argsort(scores)[::-1]
    else:
        # np.argpartition is O(n); then sort only the top-k
        part = np.argpartition(scores, -k)[-k:]
        top_indices = part[np.argsort(scores[part])[::-1]]

    top_scores = scores[top_indices]
    return top_indices, top_scores

"""
main.py — Step 8
Orchestrates the 3-stage ranking pipeline:
1. Load candidates.jsonl and build text representations.
2. Build BM25 index and retrieve top 2000 candidates using job description query.
3. Rerank top 2000 candidates using sentence-transformers (all-MiniLM-L6-v2) semantic embedding.
4. Score top 2000 candidates on behavioral signals from redrob_signals, applying penalties.
5. Fuse rankings using Reciprocal Rank Fusion (RRF) and perform tie-breaking.
6. Generate non-templated reasoning for the top 100 candidates.
7. Save output to submission.csv and run the validator.
"""

import csv
import time
from pathlib import Path
from typing import List, Dict, Any

import docx
import numpy as np

# Import our package modules
from ranker.loader import load_candidates, build_candidate_text
from ranker.bm25_stage import build_bm25_index, retrieve_top_k
from ranker.embed_stage import load_model, encode_texts, compute_cosine_similarities
from ranker.signals import compute_all_behavioral_scores
from ranker.fusion import reciprocal_rank_fusion, get_top_k_candidates
from ranker.reasoning import generate_reasoning

# ── Paths ─────────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR = Path(
    r"C:\Users\Bipin Maurya\OneDrive\Documents\[PUB] India_runs_data_and_ai_challenge"
    r"\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
)
CANDIDATES_JSONL = DEFAULT_DATA_DIR / "candidates.jsonl"
JD_DOCX = DEFAULT_DATA_DIR / "job_description.docx"
OUTPUT_CSV = Path("submission.csv")


def load_jd_text(docx_path: Path) -> str:
    """Load text paragraphs from the Job Description docx file."""
    print(f"Reading Job Description from: {docx_path.name}")
    doc = docx.Document(docx_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def get_cleaned_query(jd_text: str) -> str:
    """
    Construct a clean positive search query from the job description.
    Excludes negative matching traps (such as names of consulting firms, which 
    would boost consulting candidates in BM25 if included).
    """
    exclude_keywords = {
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "consulting", "do not want", "framework enthusiasts", "title-chasers",
        "google", "meta"  # exclude Google/Meta to avoid matching just any candidate who mentions them
    }
    
    lines = jd_text.split("\n")
    positive_lines = []
    
    for line in lines:
        line_lower = line.lower()
        # Skip lines detailing exclusions or negative constraints
        if any(ex in line_lower for ex in exclude_keywords):
            continue
        positive_lines.append(line)
        
    merged = " ".join(positive_lines)
    
    # Prepend highly weighted curated search terms
    curated_keywords = (
        "Senior AI Engineer Founding Team "
        "embeddings retrieval ranking search recommendation NLP sentence-transformers "
        "vector databases Pinecone Weaviate Milvus FAISS machine learning deep learning "
        "Python offline evaluation NDCG MRR MAP RAG learning-to-rank LLMs fine-tuning "
        "Pune Noida India"
    )
    
    return f"{curated_keywords} {merged}"


def main():
    start_time = time.time()
    
    print("======================================================================")
    print("REDROB Hackathon Candidate Discovery & Ranking Pipeline starting...")
    print("======================================================================")
    
    # ── 1. Read Job Description ──────────────────────────────────────────────
    t_jd = time.time()
    if not JD_DOCX.exists():
        raise FileNotFoundError(f"Job Description not found at {JD_DOCX}")
    
    jd_raw = load_jd_text(JD_DOCX)
    jd_query = get_cleaned_query(jd_raw)
    print(f"  Processed JD query. Cleaned length: {len(jd_query)} chars.")
    print(f"  JD Processing completed in {time.time() - t_jd:.2f}s\n")
    
    # ── 2. Load Candidate Data ────────────────────────────────────────────────
    t_load = time.time()
    if not CANDIDATES_JSONL.exists():
        raise FileNotFoundError(f"Candidates file not found at {CANDIDATES_JSONL}")
        
    print(f"Loading candidates from: {CANDIDATES_JSONL.name}...")
    candidates = load_candidates(str(CANDIDATES_JSONL))
    num_candidates = len(candidates)
    print(f"  Successfully loaded {num_candidates:,} candidates.")
    print(f"  Data loading completed in {time.time() - t_load:.2f}s\n")
    
    # ── 3. Build candidate texts ──────────────────────────────────────────────
    t_text = time.time()
    print("Building text representations for BM25 and embeddings...")
    candidate_texts = []
    for c in candidates:
        candidate_texts.append(build_candidate_text(c))
    print(f"  Text representations built in {time.time() - t_text:.2f}s\n")
    
    # ── 4. STAGE 1: BM25 Retrieval (Top-2000) ──────────────────────────────────
    t_bm25 = time.time()
    print("STAGE 1: Building BM25 index & retrieving top-2000 candidates...")
    bm25_index = build_bm25_index(candidate_texts)
    
    top_indices, bm25_scores = retrieve_top_k(bm25_index, jd_query, k=2000)
    print(f"  Retrieved {len(top_indices)} candidates by BM25.")
    print(f"  Stage 1 completed in {time.time() - t_bm25:.2f}s\n")
    
    # Subset candidates and texts to the top-2000 to save memory and CPU
    sub_candidates = [candidates[idx] for idx in top_indices]
    sub_texts = [candidate_texts[idx] for idx in top_indices]
    
    # ── 5. STAGE 2: Dense Semantic Embedding Reranking ────────────────────────
    t_embed = time.time()
    print("STAGE 2: Encoding JD and top-2000 candidates with all-MiniLM-L6-v2...")
    model = load_model()
    
    print("    Encoding job description query...")
    jd_embedding = encode_texts(model, [jd_query], desc="Query")[0]
    
    print("    Encoding top-2000 candidates...")
    candidate_embeddings = encode_texts(model, sub_texts, desc="Candidates")
    
    print("    Computing cosine similarities...")
    semantic_scores = compute_cosine_similarities(jd_embedding, candidate_embeddings)
    print(f"  Stage 2 completed in {time.time() - t_embed:.2f}s\n")
    
    # ── 6. STAGE 3: Behavioral Signal Scoring ─────────────────────────────────
    t_signals = time.time()
    print("STAGE 3: Scoring behavioral signals & applying honeypot checks...")
    behavioral_scores, penalties_list = compute_all_behavioral_scores(sub_candidates)
    print(f"  Stage 3 completed in {time.time() - t_signals:.2f}s\n")
    
    # ── 7. RRF Fusion & Tie Breaking ──────────────────────────────────────────
    t_fusion = time.time()
    print("Fusing ranks via Reciprocal Rank Fusion (RRF)...")
    rrf_scores = reciprocal_rank_fusion([semantic_scores, behavioral_scores])
    
    top_100_entries = get_top_k_candidates(
        candidates=sub_candidates,
        rrf_scores=rrf_scores,
        semantic_scores=semantic_scores,
        behavioral_scores=behavioral_scores,
        penalties=penalties_list,
        k=100
    )
    print(f"  Selected top-100 candidates with tie-breaking rules applied.")
    print(f"  RRF Fusion completed in {time.time() - t_fusion:.2f}s\n")
    
    # ── 8. Generate Reasoning and Write Output ───────────────────────────────
    t_write = time.time()
    print("Generating personalized reasoning and writing to CSV...")
    
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for i, entry in enumerate(top_100_entries):
            rank = i + 1
            reasoning = generate_reasoning(
                candidate=entry["candidate"],
                rank=rank,
                semantic_score=entry["semantic_score"],
                behavioral_score=entry["behavioral_score"],
                rrf_score=entry["rrf_score"],
                penalties=entry["penalties"]
            )
            
            # Format score to 4 decimal places for clean reading, matching sample submissions
            score_formatted = f"{entry['rrf_score']:.6f}"
            
            writer.writerow([
                entry["candidate_id"],
                rank,
                score_formatted,
                reasoning
            ])
            
    print(f"  Output saved successfully to: {OUTPUT_CSV.absolute()}")
    print(f"  Reasoning generation and output completed in {time.time() - t_write:.2f}s\n")
    
    # ── 9. Final Stats ────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print("======================================================================")
    print(f"Ranking Pipeline Finished in: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Peak RAM expected to be well below 16 GB.")
    print("======================================================================")


if __name__ == "__main__":
    main()

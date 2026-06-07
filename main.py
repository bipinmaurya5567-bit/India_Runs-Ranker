"""
main.py — Upgraded 5-stage ranking pipeline

1. Load candidates.jsonl and build text representations.
2. BM25: build index, retrieve top-2000.
3. Bi-encoder (all-MiniLM-L6-v2): filter top-2000 → top-500 by cosine sim.
4. Cross-encoder (ms-marco-MiniLM-L-6-v2): rerank top-500 → relevance scores.  [NEW]
5. Behavioral signals (JD-tuned): score top-500, apply honeypot penalties.
6. 3-way weighted RRF: fuse cross-encoder (0.6), behavioral (0.3), semantic (0.1).
7. Generate improved reasoning for top-100 (varied structure, cross-encoder confidence).
8. Save output to submission.csv and run the validator.

Timing budget (CPU, no GPU):
  Stage 1 BM25        : ~60 s
  Stage 2 Bi-encoder  : ~15 s   (2000 → 500)
  Stage 2.5 Cross-enc : ~120 s  (500, batch=32)  ← biggest quality gain
  Stage 3 Signals     : ~10 s
  Stage 4 RRF         : ~5 s
  Stage 5 Reasoning   : ~30 s
  TOTAL               : ~240 s = 4 min  (within 5-min limit)
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
from ranker.embed_stage import (
    load_model, encode_texts, compute_cosine_similarities,
    CrossEncoderStage,
)
from ranker.signals import compute_all_behavioral_scores
from ranker.fusion import weighted_reciprocal_rank_fusion, get_top_k_candidates
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
    
    # ── 5. STAGE 2: Bi-encoder filter 2000 → 500 ──────────────────────────────
    t_embed = time.time()
    print("STAGE 2: Encoding JD + top-2000 with all-MiniLM-L6-v2 (filter to top-500)...")
    bi_model = load_model()

    print("    Encoding job description query...")
    jd_embedding = encode_texts(bi_model, [jd_query], desc="Query")[0]

    print("    Encoding top-2000 candidates...")
    candidate_embeddings = encode_texts(bi_model, sub_texts, desc="Candidates")

    print("    Computing cosine similarities...")
    all_semantic_scores = compute_cosine_similarities(jd_embedding, candidate_embeddings)

    # Keep top-500 by cosine similarity for cross-encoder
    top500_local = np.argsort(all_semantic_scores)[::-1][:500]
    sub500_candidates = [sub_candidates[i] for i in top500_local]
    sub500_texts      = [sub_texts[i]      for i in top500_local]
    semantic_scores   = all_semantic_scores[top500_local]   # shape (500,)
    print(f"  Filtered to top-500 candidates.")
    print(f"  Stage 2 completed in {time.time() - t_embed:.2f}s\n")

    # ── 6. STAGE 2.5: Cross-encoder reranking 500 candidates ─────────────────
    t_cross = time.time()
    print("STAGE 2.5: Cross-encoder reranking top-500 (batch_size=32)...")
    cross_stage = CrossEncoderStage()   # loads ~90 MB model, runs warm-up
    cross_encoder_scores = cross_stage.rerank(
        jd_text=jd_query,
        candidate_texts=sub500_texts,
        batch_size=32,
    )
    print(f"  Stage 2.5 completed in {time.time() - t_cross:.2f}s\n")

    # ── 7. STAGE 3: Behavioral Signal Scoring ─────────────────────────────────
    t_signals = time.time()
    print("STAGE 3: Scoring behavioral signals & applying honeypot checks (top-500)...")
    behavioral_scores, penalties_list = compute_all_behavioral_scores(sub500_candidates)
    print(f"  Stage 3 completed in {time.time() - t_signals:.2f}s\n")
    
    # ── 8. STAGE 4: 3-way Weighted RRF Fusion ─────────────────────────────────
    t_fusion = time.time()
    print("STAGE 4: 3-way weighted RRF (cross=0.6, behavioral=0.3, semantic=0.1)...")
    rrf_scores = weighted_reciprocal_rank_fusion(
        cross_encoder_scores=cross_encoder_scores,
        behavioral_scores=behavioral_scores,
        semantic_scores=semantic_scores,
    )

    top_100_entries = get_top_k_candidates(
        candidates=sub500_candidates,
        rrf_scores=rrf_scores,
        semantic_scores=semantic_scores,
        behavioral_scores=behavioral_scores,
        cross_encoder_scores=cross_encoder_scores,
        penalties=penalties_list,
        k=100
    )
    print(f"  Selected top-100 candidates with tie-breaking rules applied.")
    print(f"  Stage 4 completed in {time.time() - t_fusion:.2f}s\n")
    
    # ── 9. STAGE 5: Generate Reasoning and Write Output ──────────────────────
    t_write = time.time()
    print("STAGE 5: Generating personalized reasoning and writing to CSV...")

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
                cross_encoder_score=entry["cross_encoder_score"],
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
    print(f"  Stage 5 completed in {time.time() - t_write:.2f}s\n")

    # ── 10. Final Stats ───────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print("======================================================================")
    print(f"Upgraded Pipeline Finished in: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Stages: BM25 → Bi-encoder(2000→500) → Cross-encoder(500) → Signals → 3-way RRF")
    print(f"Peak RAM expected to be well below 16 GB.")
    print("======================================================================")


if __name__ == "__main__":
    main()

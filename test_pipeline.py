"""
test_pipeline.py — STEP 7: Smoke-test the upgraded pipeline on synthetic data.

Creates 5 hand-crafted candidates that span the quality spectrum:
  C001 — ideal: Senior ML Engineer, IIT, active, GitHub, Pune, 0d notice
  C002 — good:  AI Engineer, NIT, active, moderate GitHub, 30d notice
  C003 — mid:   Data Scientist, active, no GitHub, 60d notice
  C004 — low:   Software Engineer (non-AI title), inactive 90d, high notice
  C005 — ghost: HR Manager (non-tech), inactive 200d, no response

Expected order: C001 > C002 > C003 > C004 >> C005 (C005 should be last due to honeypot).

Run with:
    python test_pipeline.py
"""

import sys
import time
import json
from pathlib import Path

import numpy as np

# ── Synthetic candidates ──────────────────────────────────────────────────────
CANDIDATES = [
    {
        "candidate_id": "C001",
        "profile": {
            "current_title": "Senior Machine Learning Engineer",
            "years_of_experience": 7,
            "location": "Pune, Maharashtra",
        },
        "education": [
            {"institution": "IIT Bombay", "tier": "tier_1", "degree": "B.Tech"}
        ],
        "skills": [
            {"name": "PyTorch",        "proficiency": "expert",        "endorsements": 25},
            {"name": "sentence-transformers", "proficiency": "expert", "endorsements": 18},
            {"name": "FAISS",          "proficiency": "advanced",      "endorsements": 12},
            {"name": "Python",         "proficiency": "expert",        "endorsements": 30},
        ],
        "career_history": [
            {
                "company": "Sarvam AI",
                "description": "Built retrieval pipeline using FAISS and sentence-transformers for 10M+ document corpus."
            }
        ],
        "redrob_signals": {
            "open_to_work_flag":         True,
            "last_active_date":          "2026-06-01",   # 6 days ago — active
            "recruiter_response_rate":   0.92,
            "github_activity_score":     85,
            "profile_completeness_score": 95,
            "saved_by_recruiters_30d":   22,
            "interview_completion_rate": 0.90,
            "skill_assessment_scores":   {"PyTorch": 91, "NLP": 87},
            "notice_period_days":        0,
            "verified_email":            True,
            "verified_phone":            True,
            "willing_to_relocate":       True,
            "offer_acceptance_rate":     0.80,
            "avg_response_time_hours":   4,
        },
    },
    {
        "candidate_id": "C002",
        "profile": {
            "current_title": "AI Engineer",
            "years_of_experience": 5,
            "location": "Noida, UP",
        },
        "education": [
            {"institution": "NIT Trichy", "tier": "tier_2", "degree": "B.Tech"}
        ],
        "skills": [
            {"name": "LLM fine-tuning",  "proficiency": "advanced", "endorsements": 10},
            {"name": "RAG",              "proficiency": "advanced",  "endorsements": 8},
            {"name": "Python",           "proficiency": "expert",    "endorsements": 20},
        ],
        "career_history": [
            {
                "company": "Razorpay",
                "description": "Deployed RAG pipeline using LLMs for internal search, improving retrieval NDCG by 8 points."
            }
        ],
        "redrob_signals": {
            "open_to_work_flag":         True,
            "last_active_date":          "2026-05-25",   # ~13 days ago
            "recruiter_response_rate":   0.75,
            "github_activity_score":     55,
            "profile_completeness_score": 82,
            "saved_by_recruiters_30d":   8,
            "interview_completion_rate": 0.80,
            "skill_assessment_scores":   {"NLP": 82, "RAG": 78},
            "notice_period_days":        30,
            "verified_email":            True,
            "verified_phone":            False,
            "willing_to_relocate":       False,
            "offer_acceptance_rate":     0.70,
            "avg_response_time_hours":   12,
        },
    },
    {
        "candidate_id": "C003",
        "profile": {
            "current_title": "Data Scientist",
            "years_of_experience": 4,
            "location": "Bengaluru, Karnataka",
        },
        "education": [
            {"institution": "BITS Pilani", "tier": "tier_2", "degree": "M.Tech"}
        ],
        "skills": [
            {"name": "scikit-learn",   "proficiency": "advanced",     "endorsements": 9},
            {"name": "TensorFlow",     "proficiency": "intermediate",  "endorsements": 6},
            {"name": "SQL",            "proficiency": "advanced",      "endorsements": 14},
        ],
        "career_history": [
            {
                "company": "Flipkart",
                "description": "Built recommendation model for product search using TensorFlow."
            }
        ],
        "redrob_signals": {
            "open_to_work_flag":         False,
            "last_active_date":          "2026-05-10",   # ~28 days ago
            "recruiter_response_rate":   0.45,
            "github_activity_score":     -1,             # no GitHub
            "profile_completeness_score": 65,
            "saved_by_recruiters_30d":   3,
            "interview_completion_rate": 0.60,
            "skill_assessment_scores":   {"ML": 70},
            "notice_period_days":        60,
            "verified_email":            True,
            "verified_phone":            False,
            "willing_to_relocate":       True,
            "offer_acceptance_rate":     -1,             # no offer history
            "avg_response_time_hours":   36,
        },
    },
    {
        "candidate_id": "C004",
        "profile": {
            "current_title": "Software Engineer",
            "years_of_experience": 3,
            "location": "Hyderabad, Telangana",
        },
        "education": [
            {"institution": "VIT Vellore", "tier": "tier_3", "degree": "B.Tech"}
        ],
        "skills": [
            {"name": "Java",       "proficiency": "advanced",      "endorsements": 15},
            {"name": "Spring Boot", "proficiency": "advanced",     "endorsements": 10},
            {"name": "Python",     "proficiency": "intermediate",  "endorsements": 5},
        ],
        "career_history": [
            {
                "company": "TCS",
                "description": "Developed REST APIs using Java Spring Boot for banking clients."
            }
        ],
        "redrob_signals": {
            "open_to_work_flag":         False,
            "last_active_date":          "2026-03-10",   # ~89 days ago
            "recruiter_response_rate":   0.20,
            "github_activity_score":     15,
            "profile_completeness_score": 50,
            "saved_by_recruiters_30d":   1,
            "interview_completion_rate": 0.40,
            "skill_assessment_scores":   {},
            "notice_period_days":        90,
            "verified_email":            False,
            "verified_phone":            False,
            "willing_to_relocate":       False,
            "offer_acceptance_rate":     0.20,           # low → penalty
            "avg_response_time_hours":   80,             # slow → penalty
        },
    },
    {
        "candidate_id": "C005",
        "profile": {
            "current_title": "HR Manager",              # non-tech → honeypot
            "years_of_experience": 8,
            "location": "Delhi",
        },
        "education": [],
        "skills": [
            {"name": "PyTorch",    "proficiency": "beginner",  "endorsements": 0},
            {"name": "TensorFlow", "proficiency": "beginner",  "endorsements": 0},
        ],
        "career_history": [],
        "redrob_signals": {
            "open_to_work_flag":         True,
            "last_active_date":          "2025-11-15",   # >200 days → ghost
            "recruiter_response_rate":   0.05,           # ghost threshold
            "github_activity_score":     -1,
            "profile_completeness_score": 30,
            "saved_by_recruiters_30d":   0,
            "interview_completion_rate": 0.10,
            "skill_assessment_scores":   {},
            "notice_period_days":        120,
            "verified_email":            False,
            "verified_phone":            False,
            "willing_to_relocate":       False,
            "offer_acceptance_rate":     -1,
            "avg_response_time_hours":   200,
        },
    },
]

JD_TEXT = (
    "Senior AI Engineer Founding Team embeddings retrieval ranking search recommendation NLP "
    "sentence-transformers vector databases Pinecone Weaviate Milvus FAISS machine learning "
    "deep learning Python offline evaluation NDCG MRR MAP RAG learning-to-rank LLMs fine-tuning "
    "Pune Noida India Series A startup product-minded engineer who can build and ship retrieval "
    "systems fast. You will design embedding pipelines, reranking models, and evaluation frameworks."
)


def run_smoke_test():
    print("=" * 70)
    print("SMOKE TEST — Upgraded Pipeline (5 synthetic candidates)")
    print("=" * 70)
    t0 = time.time()

    # ── Stage 1: BM25 ────────────────────────────────────────────────────────
    from ranker.loader import build_candidate_text
    from ranker.bm25_stage import build_bm25_index, retrieve_top_k

    print("\n[Stage 1] Building BM25 index...")
    texts = [build_candidate_text(c) for c in CANDIDATES]
    bm25 = build_bm25_index(texts)
    bm25_indices, bm25_scores = retrieve_top_k(bm25, JD_TEXT, k=len(CANDIDATES))
    sub_candidates = [CANDIDATES[i] for i in bm25_indices]
    sub_texts      = [texts[i]      for i in bm25_indices]
    print(f"  BM25 order: {[CANDIDATES[i]['candidate_id'] for i in bm25_indices]}")

    # ── Stage 2: Bi-encoder (keep all 5 for smoke test) ──────────────────────
    from ranker.embed_stage import (
        load_model, encode_texts, compute_cosine_similarities,
        CrossEncoderStage,
    )

    print("\n[Stage 2] Bi-encoder semantic scoring...")
    bi_model        = load_model()
    jd_emb          = encode_texts(bi_model, [JD_TEXT])[0]
    cand_embs       = encode_texts(bi_model, sub_texts)
    all_sem_scores  = compute_cosine_similarities(jd_emb, cand_embs)

    # Keep all 5 (small test; in production this becomes top-500)
    top_k_local     = np.argsort(all_sem_scores)[::-1]
    sub500          = [sub_candidates[i] for i in top_k_local]
    sub500_texts    = [sub_texts[i]      for i in top_k_local]
    sem_scores      = all_sem_scores[top_k_local]
    print(f"  Bi-encoder order: {[c['candidate_id'] for c in sub500]}")
    print(f"  Scores: {[f'{s:.4f}' for s in sem_scores]}")

    # ── Stage 2.5: Cross-encoder ─────────────────────────────────────────────
    print("\n[Stage 2.5] Cross-encoder reranking...")
    cross = CrossEncoderStage()
    ce_scores = cross.rerank(JD_TEXT, sub500_texts, batch_size=5)
    print(f"  Cross-encoder scores: {dict(zip([c['candidate_id'] for c in sub500], [f'{s:.3f}' for s in ce_scores]))}")

    # ── Stage 3: Behavioral signals ──────────────────────────────────────────
    from ranker.signals import compute_all_behavioral_scores

    print("\n[Stage 3] Behavioral signals...")
    behav_scores, penalties = compute_all_behavioral_scores(sub500)
    for c, s, p in zip(sub500, behav_scores, penalties):
        pen_str = f" [PENALTY] {p}" if p else ""
        print(f"  {c['candidate_id']}: {s:.4f}{pen_str}")

    # ── Stage 4: 3-way weighted RRF ──────────────────────────────────────────
    from ranker.fusion import weighted_reciprocal_rank_fusion, get_top_k_candidates

    print("\n[Stage 4] 3-way weighted RRF (cross=0.6, behav=0.3, sem=0.1)...")
    rrf = weighted_reciprocal_rank_fusion(ce_scores, behav_scores, sem_scores)
    top_entries = get_top_k_candidates(
        candidates=sub500,
        rrf_scores=rrf,
        semantic_scores=sem_scores,
        behavioral_scores=behav_scores,
        cross_encoder_scores=ce_scores,
        penalties=penalties,
        k=len(sub500),
    )

    # ── Stage 5: Reasoning ───────────────────────────────────────────────────
    from ranker.reasoning import generate_reasoning

    print("\n[Stage 5] Generating reasoning...\n")
    print("=" * 70)
    print(f"{'Rank':<5} {'CandID':<6} {'RRF':>8} {'CE':>7} {'Behav':>7}  Reasoning")
    print("-" * 70)

    for i, entry in enumerate(top_entries):
        rank = i + 1
        r = generate_reasoning(
            candidate=entry["candidate"],
            rank=rank,
            semantic_score=entry["semantic_score"],
            behavioral_score=entry["behavioral_score"],
            rrf_score=entry["rrf_score"],
            cross_encoder_score=entry["cross_encoder_score"],
            penalties=entry["penalties"],
        )
        print(f"#{rank:<4} {entry['candidate_id']:<6} "
              f"{entry['rrf_score']:>8.5f} "
              f"{entry['cross_encoder_score']:>7.2f} "
              f"{entry['behavioral_score']:>7.4f}")
        print(f"       {r}\n")

    print("=" * 70)
    print(f"\n✅ Smoke test completed in {time.time() - t0:.1f}s")

    # ── Assertions ───────────────────────────────────────────────────────────
    ranked_ids = [e["candidate_id"] for e in top_entries]
    print(f"\nFinal order: {ranked_ids}")

    assert ranked_ids[0] == "C001", f"Expected C001 at rank 1, got {ranked_ids[0]}"
    assert ranked_ids[-1] == "C005", f"Expected C005 at rank 5 (ghost/honeypot), got {ranked_ids[-1]}"
    print("✅ Assertions passed: C001 is #1, C005 (ghost honeypot) is last.")


if __name__ == "__main__":
    run_smoke_test()

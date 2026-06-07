"""
loader.py — Step 2
Load candidates from plain JSONL and build rich text representations.
"""
import json
import gc
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from tqdm import tqdm


# ── AI/ML skill keywords (lowercase) ────────────────────────────────────────
AI_ML_KEYWORDS = {
    "machine learning", "deep learning", "neural network", "nlp",
    "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
    "sklearn", "transformers", "bert", "gpt", "llm", "embedding", "embeddings",
    "vector", "retrieval", "ranking", "recommendation", "rag", "langchain",
    "pinecone", "faiss", "milvus", "chromadb", "weaviate", "elasticsearch",
    "mlops", "model deployment", "inference", "fine-tuning", "fine tuning",
    "lora", "rlhf", "hugging face", "huggingface", "data science",
    "feature engineering", "xgboost", "lightgbm", "catboost", "opencv",
    "image classification", "object detection", "semantic segmentation",
    "generative ai", "diffusion", "gans", "stable diffusion",
    "speech recognition", "tts", "asr", "statistical modeling", "bayesian",
    "reinforcement learning", "kubeflow", "mlflow", "weights & biases",
    "wandb", "spark", "databricks", "sagemaker", "vertex ai", "azure ml",
    "vector database", "semantic search", "information retrieval",
    "knowledge graph", "graph neural", "attention mechanism", "multi-modal",
    "multimodal", "vision transformer", "vit", "roberta", "llama", "mistral",
    "mixtral", "bentoml", "triton", "onnx", "anomaly detection", "clustering",
    "natural language processing", "text classification", "sentiment analysis",
    "question answering", "summarization", "named entity recognition",
    "time series", "causal inference", "a/b testing", "recommender",
    "search ranking", "bm25", "tfidf", "tf-idf", "word2vec", "doc2vec",
    "sentence transformer", "cross-encoder", "bi-encoder",
}

# ── Engineering/tech title keywords (for honeypot detection) ────────────────
TECH_TITLE_KEYWORDS = {
    "engineer", "developer", "scientist", "researcher", "architect",
    "programmer", "ml", "ai", "data", "software", "backend", "frontend",
    "fullstack", "full-stack", "full stack", "devops", "sre", "platform",
    "infrastructure", "cloud", "nlp", "computer vision", "machine learning",
    "deep learning", "analytics", "technical", "lead", "principal", "staff",
    "algorithms", "modeling", "tech lead", "cto", "vp engineering",
    "head of engineering", "head of ai", "director of engineering",
    "director of ai", "quantitative", "research scientist", "applied scientist",
}


def load_candidates(
    jsonl_path: str,
    max_candidates: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load candidates from a plain JSONL file.
    Handles malformed lines gracefully.
    """
    candidates: List[Dict[str, Any]] = []
    path = Path(jsonl_path)

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc="  Loading candidates", unit=" docs")):
            if max_candidates is not None and i >= max_candidates:
                break
            line = line.strip()
            if not line:
                continue
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip bad lines silently

    return candidates


def build_candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Build a rich text blob for BM25 indexing and dense embedding.

    Strategy:
    - Repeat title and skills (they're the strongest signal)
    - Include headline, summary (truncated), career descriptions
    - Include education field-of-study for relevance
    """
    parts: List[str] = []
    profile = candidate.get("profile", {}) or {}

    # ── Core identity (repeat for BM25 term-frequency boost) ────────
    title = (profile.get("current_title") or "").strip()
    headline = (profile.get("headline") or "").strip()
    summary = (profile.get("summary") or "").strip()

    if title:
        parts.extend([title, title])          # 2x weight
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary[:700])

    # ── Skills (repeat for BM25 boost) ──────────────────────────────
    skills = candidate.get("skills") or []
    skill_names = [s.get("name", "") for s in skills if s.get("name")]
    if skill_names:
        blob = " ".join(skill_names)
        parts.extend([blob, blob])            # 2x weight

    # ── Certifications ───────────────────────────────────────────────
    certs = candidate.get("certifications") or []
    cert_names = [c.get("name", "") for c in certs if c.get("name")]
    if cert_names:
        parts.append(" ".join(cert_names))

    # ── Career history: top-3 roles ──────────────────────────────────
    career = candidate.get("career_history") or []
    for role in career[:3]:
        role_title = (role.get("title") or "").strip()
        role_desc = (role.get("description") or "").strip()
        if role_title:
            parts.append(role_title)
        if role_desc:
            parts.append(role_desc[:450])

    # ── Education ────────────────────────────────────────────────────
    education = candidate.get("education") or []
    for edu in education:
        field = (edu.get("field_of_study") or "").strip()
        degree = (edu.get("degree") or "").strip()
        if field:
            parts.append(field)
        if degree:
            parts.append(degree)

    return " ".join(p for p in parts if p)


def get_top_skills(candidate: Dict[str, Any], n: int = 3) -> List[str]:
    """
    Return the top-N skills ranked by: proficiency × 10 + endorsements.
    Prefers AI/ML skills over generic ones.
    """
    proficiency_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    skills = candidate.get("skills") or []

    ai_skills: List[Tuple[str, float]] = []
    other_skills: List[Tuple[str, float]] = []

    for s in skills:
        name = s.get("name", "")
        if not name:
            continue
        prof_val = proficiency_map.get(s.get("proficiency", "beginner"), 1)
        endorsements = s.get("endorsements", 0) or 0
        score = prof_val * 10 + endorsements

        if is_ai_skill(name):
            ai_skills.append((name, score))
        else:
            other_skills.append((name, score))

    ai_skills.sort(key=lambda x: -x[1])
    other_skills.sort(key=lambda x: -x[1])

    selected: List[str] = []
    for name, _ in ai_skills[:n]:
        selected.append(name)
    for name, _ in other_skills:
        if len(selected) >= n:
            break
        if name not in selected:
            selected.append(name)

    return selected[:n]


def is_ai_skill(skill_name: str) -> bool:
    """Return True if the skill name matches any AI/ML keyword."""
    name_lower = skill_name.lower()
    return any(kw in name_lower for kw in AI_ML_KEYWORDS)


def has_tech_title(title: str) -> bool:
    """Return True if the job title contains an engineering/tech keyword."""
    if not title:
        return False
    title_lower = title.lower()
    return any(kw in title_lower for kw in TECH_TITLE_KEYWORDS)

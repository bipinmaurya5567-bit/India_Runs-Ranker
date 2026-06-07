"""
signals.py — Step 5
Stage 3: Behavioral signal scoring from redrob_signals.
Computes a 0-1 normalized score per candidate with anti-honeypot penalties.
"""
import math
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ranker.loader import has_tech_title, is_ai_skill

# Reference date for time-decay calculations (set once at import time)
REF_DATE: date = date.today()

# ── Weights sum to ~1.0 for positive signals ─────────────────────────────────
_WEIGHT_OPEN_TO_WORK        = 0.15   # Boolean bonus
_WEIGHT_ACTIVITY_DECAY      = 0.12   # Recency of platform use
_WEIGHT_RESPONSE_RATE       = 0.10   # Recruiter response rate
_WEIGHT_GITHUB              = 0.08   # GitHub activity (0-100 → 0-1)
_WEIGHT_COMPLETENESS        = 0.06   # Profile completeness (0-100 → 0-1)
_WEIGHT_SAVED_RECRUITERS    = 0.05   # Log-normalised saves in 30d
_WEIGHT_INTERVIEW_COMPLETE  = 0.05   # Interview completion rate
_WEIGHT_SKILL_ASSESSMENT    = 0.07   # Avg AI-skill assessment score
_BONUS_BOTH_VERIFIED        = 0.03   # verified_email AND verified_phone
_BONUS_ONE_VERIFIED         = 0.01   # either verified

_PENALTY_LONG_NOTICE        = -0.05  # notice_period_days > 90
_PENALTY_SLOW_RESPONSE      = -0.03  # avg_response_time_hours > 72
_PENALTY_LOW_OFFER_RATE     = -0.02  # offer_acceptance_rate < 0.3 (not -1)

# Log-normalisation ceiling for saved_by_recruiters_30d
_SAVED_MAX_CAP = 50


def _safe(d: Dict, key: str, default=None):
    """Get dict value, treating None as missing."""
    v = d.get(key, default)
    return default if v is None else v


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# ── Activity Decay ────────────────────────────────────────────────────────────

def _activity_decay(last_active_str: str, ref: date = REF_DATE) -> float:
    """
    Piecewise linear decay:
      ≤ 30d  → 1.00
      60d    → 0.85
      90d    → 0.65
      180d   → 0.30
      365d   → 0.05
      >365d  → 0.05
    """
    try:
        last_dt = date.fromisoformat(last_active_str)
        days = max(0, (ref - last_dt).days)
    except (ValueError, TypeError):
        return 0.10   # unknown → low

    if days <= 30:
        return 1.00
    if days <= 60:
        return 0.85 + (1.00 - 0.85) * (60 - days) / 30
    if days <= 90:
        return 0.65 + (0.85 - 0.65) * (90 - days) / 30
    if days <= 180:
        return 0.30 + (0.65 - 0.30) * (180 - days) / 90
    if days <= 365:
        return 0.05 + (0.30 - 0.05) * (365 - days) / 185
    return 0.05


# ── Skill Assessment ──────────────────────────────────────────────────────────

def _avg_ai_assessment(skill_scores: Dict[str, float]) -> float:
    """
    Average the assessment scores for AI-related skills.
    Falls back to all available scores if none are AI-labelled.
    Returns value in [0, 1].
    """
    if not isinstance(skill_scores, dict) or not skill_scores:
        return 0.0

    ai_vals = [float(v) for k, v in skill_scores.items() if is_ai_skill(k)]
    if not ai_vals:
        ai_vals = [float(v) for v in skill_scores.values()]

    return sum(ai_vals) / len(ai_vals) / 100.0 if ai_vals else 0.0


# ── Core behavioral scorer ────────────────────────────────────────────────────

def compute_behavioral_score(
    candidate: Dict[str, Any],
    ref: date = REF_DATE,
) -> float:
    """
    Compute normalised behavioral score in [0, 1] for a single candidate.
    Does NOT apply honeypot multipliers (see apply_honeypot_penalties).
    """
    sig = candidate.get("redrob_signals") or {}
    score = 0.0

    # 1. open_to_work_flag
    if _safe(sig, "open_to_work_flag", False):
        score += _WEIGHT_OPEN_TO_WORK

    # 2. Activity recency
    score += _WEIGHT_ACTIVITY_DECAY * _activity_decay(
        _safe(sig, "last_active_date", ""), ref
    )

    # 3. Recruiter response rate [0, 1]
    rr = _clamp(float(_safe(sig, "recruiter_response_rate", 0)))
    score += _WEIGHT_RESPONSE_RATE * rr

    # 4. GitHub activity score (-1 = not linked → 0; else normalise 0-100 → 0-1)
    gh = float(_safe(sig, "github_activity_score", -1))
    gh_norm = 0.0 if gh < 0 else _clamp(gh / 100.0)
    score += _WEIGHT_GITHUB * gh_norm

    # 5. Profile completeness [0, 100] → [0, 1]
    pc = _clamp(float(_safe(sig, "profile_completeness_score", 0)) / 100.0)
    score += _WEIGHT_COMPLETENESS * pc

    # 6. Saved by recruiters (log-normalised)
    saved = max(0, int(_safe(sig, "saved_by_recruiters_30d", 0)))
    saved_norm = _clamp(math.log1p(saved) / math.log1p(_SAVED_MAX_CAP))
    score += _WEIGHT_SAVED_RECRUITERS * saved_norm

    # 7. Interview completion rate [0, 1]
    icr = _clamp(float(_safe(sig, "interview_completion_rate", 0)))
    score += _WEIGHT_INTERVIEW_COMPLETE * icr

    # 8. Skill assessment scores (avg AI-related, normalised)
    sa = _safe(sig, "skill_assessment_scores", {})
    score += _WEIGHT_SKILL_ASSESSMENT * _avg_ai_assessment(sa)

    # 9. Verified email / phone bonus
    ve = bool(_safe(sig, "verified_email", False))
    vp = bool(_safe(sig, "verified_phone", False))
    if ve and vp:
        score += _BONUS_BOTH_VERIFIED
    elif ve or vp:
        score += _BONUS_ONE_VERIFIED

    # ── Penalties ────────────────────────────────────────────────────
    notice = int(_safe(sig, "notice_period_days", 0))
    if notice > 90:
        score += _PENALTY_LONG_NOTICE

    oar = float(_safe(sig, "offer_acceptance_rate", -1))
    # -1 means no offer history → treat as neutral, NOT penalised
    if oar != -1 and oar < 0.3:
        score += _PENALTY_LOW_OFFER_RATE

    art = float(_safe(sig, "avg_response_time_hours", 0))
    if art > 72:
        score += _PENALTY_SLOW_RESPONSE

    return _clamp(score)


# ── Anti-honeypot penalties ───────────────────────────────────────────────────

# Consulting firm keywords to identify consulting-only profiles
CONSULTING_KEYWORDS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "deloitte", "ernst & young", " ey ",
    "pwc", "kpmg", "ltimindtree", "l&t infotech", "lnt infotech", "genpact",
}

def is_consulting_only(candidate: Dict[str, Any]) -> bool:
    """Return True if the candidate has ONLY worked at service/consulting firms."""
    companies = []
    
    # Current company
    curr = (candidate.get("profile") or {}).get("current_company")
    if curr:
        companies.append(curr)
        
    # Career history
    for role in candidate.get("career_history") or []:
        comp = role.get("company")
        if comp:
            companies.append(comp)
            
    if not companies:
        return False
        
    for comp in companies:
        comp_lower = comp.lower()
        is_consulting = False
        for kw in CONSULTING_KEYWORDS:
            if kw in comp_lower:
                is_consulting = True
                break
        if not is_consulting:
            return False  # found a non-consulting company
            
    return True


def apply_honeypot_penalties(
    candidate: Dict[str, Any],
    base_score: float,
    ref: date = REF_DATE,
) -> Tuple[float, List[str]]:
    """
    Apply dataset-specific anti-honeypot multipliers.

    Rules:
      1. Non-tech title + AI skills listed → 0.40× (title-mismatch trap)
      2. Inactive >180d AND response_rate <0.1 → 0.50× (ghost candidate)
      3. Service/consulting-only career history → 0.20× (consulting-only trap)

    Returns: (adjusted_score, list_of_penalty_labels_applied)
    """
    profile = candidate.get("profile") or {}
    sig     = candidate.get("redrob_signals") or {}
    skills  = candidate.get("skills") or []

    multiplier = 1.0
    labels: List[str] = []

    # Rule 1: Title mismatch
    title = profile.get("current_title") or ""
    if not has_tech_title(title):
        ai_skills = [s.get("name") for s in skills if is_ai_skill(s.get("name", ""))]
        if ai_skills:
            multiplier *= 0.40
            labels.append(f"title_mismatch:0.40x ({title!r})")

    # Rule 2: Ghost candidate
    last_active_str = _safe(sig, "last_active_date", "")
    rr = float(_safe(sig, "recruiter_response_rate", 0))
    try:
        days_inactive = max(0, (ref - date.fromisoformat(last_active_str)).days)
        if days_inactive > 180 and rr < 0.1:
            multiplier *= 0.50
            labels.append(
                f"ghost:0.50x ({days_inactive}d inactive, rr={rr:.2f})"
            )
    except (ValueError, TypeError):
        pass

    # Rule 3: Consulting only
    if is_consulting_only(candidate):
        multiplier *= 0.20
        labels.append("consulting_only:0.20x (service-only career)")

    return _clamp(base_score * multiplier), labels


# ── Batch scorer ──────────────────────────────────────────────────────────────

def compute_all_behavioral_scores(
    candidates: List[Dict[str, Any]],
    ref: date = REF_DATE,
) -> Tuple[np.ndarray, List[List[str]]]:
    """
    Compute behavioral scores (with honeypot penalties) for a list of candidates.

    Returns:
        scores:      np.ndarray of shape (N,), values in [0, 1]
        penalties:   list of N penalty-label lists (empty if no penalties)
    """
    scores: List[float] = []
    all_penalties: List[List[str]] = []

    for c in tqdm(candidates, desc="    Scoring signals", unit=" cands", ncols=80):
        base = compute_behavioral_score(c, ref)
        adj, labels = apply_honeypot_penalties(c, base, ref)
        scores.append(adj)
        all_penalties.append(labels)

    return np.array(scores, dtype=np.float32), all_penalties

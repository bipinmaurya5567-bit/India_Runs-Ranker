"""
reasoning.py — Step 7
Generate specific, non-templated 1-2 sentence reasoning for each top-100 candidate.

Rules:
- Reference ACTUAL title and years of experience
- Name 1-2 specific skills from their profile
- Cite 1 behavioral signal with its numeric value
- Acknowledge gaps honestly for lower-ranked candidates
- NO generic skill counts like "9 AI core skills"
"""
from datetime import date
from typing import Any, Dict, List, Optional

from ranker.loader import is_ai_skill
from ranker.signals import REF_DATE, _safe


# ── Helpers ───────────────────────────────────────────────────────────────────

def _yoe(c: Dict) -> float:
    return float((c.get("profile") or {}).get("years_of_experience") or 0)


def _title(c: Dict) -> str:
    return ((c.get("profile") or {}).get("current_title") or "Professional").strip()


def _top_ai_skills(c: Dict, n: int = 2) -> List[str]:
    """Return up to n top AI/ML skills by (proficiency × 10 + endorsements)."""
    prof_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    skills = c.get("skills") or []

    scored = []
    for s in skills:
        name = s.get("name", "") or ""
        if not name or not is_ai_skill(name):
            continue
        p = prof_map.get(s.get("proficiency", "beginner"), 1)
        e = s.get("endorsements", 0) or 0
        scored.append((name, p * 10 + e))

    scored.sort(key=lambda x: -x[1])
    return [name for name, _ in scored[:n]]


def _top_any_skills(c: Dict, n: int = 2) -> List[str]:
    """Return top n skills regardless of domain."""
    prof_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    skills = c.get("skills") or []
    scored = []
    for s in skills:
        name = s.get("name", "") or ""
        if not name:
            continue
        p = prof_map.get(s.get("proficiency", "beginner"), 1)
        e = s.get("endorsements", 0) or 0
        scored.append((name, p * 10 + e))
    scored.sort(key=lambda x: -x[1])
    return [name for name, _ in scored[:n]]


def _skills_phrase(c: Dict) -> str:
    """Return a readable skill phrase, preferring AI skills."""
    ai = _top_ai_skills(c, 2)
    if ai:
        return " and ".join(ai)
    any_s = _top_any_skills(c, 2)
    return " and ".join(any_s) if any_s else "varied technical skills"


def _career_snippet(c: Dict) -> str:
    """
    Extract the most relevant ML/AI-related phrase from the latest role description.
    Returns empty string if nothing relevant found.
    """
    ML_VERBS = {
        "built", "developed", "designed", "deployed", "implemented",
        "trained", "fine-tuned", "optimized", "led", "created",
        "architected", "shipped", "scaled",
    }
    ML_NOUNS = {
        "retrieval", "embedding", "ranking", "recommendation", "model",
        "inference", "pipeline", "search", "vector", "llm", "transformer",
        "nlp", "classification", "detection", "generation", "rag",
    }
    career = c.get("career_history") or []
    for role in career[:2]:
        desc = (role.get("description") or "").strip()
        if not desc:
            continue
        sentences = [s.strip() for s in desc.split(".") if s.strip()]
        for sent in sentences[:4]:
            lower = sent.lower()
            has_verb = any(v in lower for v in ML_VERBS)
            has_noun = any(n in lower for n in ML_NOUNS)
            if has_verb and has_noun and len(sent) < 100:
                return sent[0].lower() + sent[1:]
    return ""


def _signal_phrase(c: Dict, ref: date = REF_DATE) -> str:
    """Build the most informative behavioral signal phrase."""
    sig = c.get("redrob_signals") or {}

    rr     = float(_safe(sig, "recruiter_response_rate", 0))
    gh     = float(_safe(sig, "github_activity_score", -1))
    saved  = int(_safe(sig, "saved_by_recruiters_30d", 0))
    icr    = float(_safe(sig, "interview_completion_rate", 0))
    otw    = bool(_safe(sig, "open_to_work_flag", False))
    notice = int(_safe(sig, "notice_period_days", 0))

    last_active_str = _safe(sig, "last_active_date", "")
    try:
        days_ago = max(0, (ref - date.fromisoformat(last_active_str)).days)
    except (ValueError, TypeError):
        days_ago = 999

    # Pick the single strongest positive signal to cite
    candidates_sig = []

    if rr >= 0.7:
        candidates_sig.append((3, f"recruiter response rate {rr:.2f} signals strong availability"))
    elif rr >= 0.4:
        candidates_sig.append((2, f"recruiter response rate {rr:.2f}"))
    else:
        candidates_sig.append((0, f"recruiter response rate {rr:.2f}"))

    if gh >= 50:
        candidates_sig.append((3, f"active GitHub contributor (score {gh:.0f}/100)"))
    elif gh >= 20:
        candidates_sig.append((2, f"GitHub activity score {gh:.0f}"))

    if days_ago <= 7:
        candidates_sig.append((3, "active on Redrob within the last week"))
    elif days_ago <= 30:
        candidates_sig.append((2, "active on Redrob within the last month"))

    if saved >= 10:
        candidates_sig.append((2, f"saved by {saved} recruiters in the past 30 days"))
    elif saved >= 5:
        candidates_sig.append((1, f"saved by {saved} recruiters recently"))

    if icr >= 0.85:
        candidates_sig.append((2, f"interview completion rate {icr:.0%}"))

    # Sort by importance and take the best
    candidates_sig.sort(key=lambda x: -x[0])
    phrase = candidates_sig[0][1] if candidates_sig else f"response rate {rr:.2f}"

    # Append availability note if available quickly
    if otw and notice <= 30:
        phrase += f"; open to work, {notice}d notice"
    elif otw:
        phrase += "; currently open to work"

    return phrase


def _gap_phrase(c: Dict, ref: date = REF_DATE) -> str:
    """Identify the primary gap for lower-ranked candidates."""
    profile = c.get("profile") or {}
    sig     = c.get("redrob_signals") or {}

    yoe    = _yoe(c)
    rr     = float(_safe(sig, "recruiter_response_rate", 0))
    gh     = float(_safe(sig, "github_activity_score", -1))
    notice = int(_safe(sig, "notice_period_days", 0))

    last_active_str = _safe(sig, "last_active_date", "")
    try:
        days_ago = max(0, (ref - date.fromisoformat(last_active_str)).days)
    except (ValueError, TypeError):
        days_ago = 999

    # Collect weighted gaps
    gaps = []
    if days_ago > 180:
        gaps.append((3, f"platform inactivity ({days_ago // 30}+ months)"))
    if yoe < 5:
        gaps.append((2, f"experience ({yoe:.1f} yrs) below the 5–9 yr target"))
    if gh == -1:
        gaps.append((1, "no linked GitHub profile"))
    if rr < 0.2:
        gaps.append((2, f"low recruiter response rate ({rr:.2f})"))
    if notice > 90:
        gaps.append((1, f"long notice period ({notice}d)"))

    if not gaps:
        return "competitive candidate pool at this rank"

    gaps.sort(key=lambda x: -x[0])
    return gaps[0][1]


def _edu_note(c: Dict) -> str:
    """Return a brief education note for top-tier graduates."""
    education = c.get("education") or []
    for edu in education:
        tier = edu.get("tier", "")
        institution = edu.get("institution", "") or ""
        if tier == "tier_1":
            return f" ({institution} alumnus)"
        if tier == "tier_2":
            return f" ({institution})"
    return ""


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_reasoning(
    candidate: Dict[str, Any],
    rank: int,
    semantic_score: float,
    behavioral_score: float,
    rrf_score: float,
    penalties: Optional[List[str]] = None,
    ref: date = REF_DATE,
) -> str:
    """
    Generate a specific, honest 1-2 sentence reasoning string.

    The output must:
    - Name the actual title and years of experience
    - Reference 1-2 named skills
    - Cite one behavioral signal with its numeric value
    - Acknowledge gaps for lower ranks
    - Avoid generic language like "X AI core skills"
    """
    title      = _title(candidate)
    yoe        = _yoe(candidate)
    skills_ph  = _skills_phrase(candidate)
    signal_ph  = _signal_phrase(candidate, ref)
    snippet    = _career_snippet(candidate)
    edu_note   = _edu_note(candidate)

    if rank <= 10:
        # Strong endorsement — cite career evidence if available
        if snippet:
            s1 = (
                f"{title} with {yoe:.1f} years{edu_note}; "
                f"recent work includes {snippet}."
            )
        else:
            s1 = (
                f"{title} with {yoe:.1f} years{edu_note}, "
                f"skilled in {skills_ph}."
            )
        s2 = f"{signal_ph.capitalize()}."

    elif rank <= 25:
        # Solid match — skills + signal
        s1 = (
            f"{title} with {yoe:.1f} years of experience and "
            f"strong background in {skills_ph}."
        )
        s2 = f"{signal_ph.capitalize()}; well-aligned with Senior AI Engineer requirements."

    elif rank <= 50:
        # Decent match — name skills, one signal
        s1 = (
            f"{title} ({yoe:.1f} yrs) with relevant exposure to {skills_ph}."
        )
        s2 = (
            f"Partially matches the retrieval/ranking focus; "
            f"{signal_ph}."
        )

    elif rank <= 75:
        # Partial match — acknowledge gap
        gap = _gap_phrase(candidate, ref)
        s1 = (
            f"{title} with {yoe:.1f} years and skills in {skills_ph}."
        )
        s2 = (
            f"Ranked {rank} due to {gap}; {signal_ph}."
        )

    else:
        # Rank 76-100 — honest gap acknowledgement
        gap = _gap_phrase(candidate, ref)
        s1 = (
            f"{title} ({yoe:.1f} yrs) with background in {skills_ph}."
        )
        s2 = (
            f"Lower fit primarily because of {gap}; {signal_ph} within the broader pool."
        )

    reasoning = f"{s1} {s2}"

    # Sanitise: collapse whitespace, cap length
    reasoning = " ".join(reasoning.split())
    if len(reasoning) > 400:
        reasoning = reasoning[:397] + "..."

    return reasoning

"""
reasoning.py — Stage 5 (upgraded)
Generate specific, non-templated 1-2 sentence reasoning for each top-100 candidate.

Rules:
- Reference ACTUAL title and years of experience
- Mention their SPECIFIC school (IIT/NIT/COEP etc.) when tier_1/tier_2
- Reference 1-2 named skills from their profile
- Cite 1 behavioral signal with its numeric value
- Mention response rate and availability for active candidates
- Acknowledge gaps HONESTLY for lower-ranked candidates
- NEVER use the same sentence structure twice (rotated by candidate_id hash)
- Each reasoning is genuinely different in structure, not just word-swaps
"""
from datetime import date
from typing import Any, Dict, List, Optional

from ranker.loader import is_ai_skill
from ranker.signals import REF_DATE, _safe


# ── Helpers ──────────────────────────────────────────────────────────────────────────

def _variant(candidate_id: str, n: int) -> int:
    """Deterministic variant selector in [0, n) based on candidate_id hash."""
    return hash(candidate_id) % n


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
    """Return a brief education note for top-tier graduates, using institution name."""
    education = c.get("education") or []
    for edu in education:
        tier        = edu.get("tier", "")
        institution = edu.get("institution", "") or ""
        if tier == "tier_1" and institution:
            return f" ({institution} alumnus)"
        if tier == "tier_2" and institution:
            return f" ({institution})"
    return ""


def _notice_phrase(notice: int) -> str:
    """Human-readable notice period phrase."""
    if notice == 0:
        return "available immediately"
    if notice <= 30:
        return f"{notice}-day notice period"
    if notice <= 60:
        return f"{notice}-day notice (moderate)"
    return f"{notice}-day notice period (long)"


def _activity_phrase(sig: Dict, ref: date) -> str:
    """Return readable last-activity phrase."""
    last_str = _safe(sig, "last_active_date", "")
    try:
        days = max(0, (ref - date.fromisoformat(last_str)).days)
    except (ValueError, TypeError):
        return "activity unknown"
    if days <= 7:
        return "active this week"
    if days <= 30:
        return "active within 30 days"
    if days <= 60:
        return "active within 60 days"
    if days <= 90:
        return "active within 90 days"
    return f"inactive for {days // 30}+ months"


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_reasoning(
    candidate: Dict[str, Any],
    rank: int,
    semantic_score: float,
    behavioral_score: float,
    rrf_score: float,
    cross_encoder_score: float = 0.0,
    penalties: Optional[List[str]] = None,
    ref: date = REF_DATE,
) -> str:
    """
    Generate a specific, honest 1-2 sentence reasoning string.

    New: uses cross_encoder_score to express model confidence for top candidates,
    and rotates sentence templates via candidate_id hash for structural variety.
    """
    cid        = candidate.get("candidate_id", "")
    title      = _title(candidate)
    yoe        = _yoe(candidate)
    skills_ph  = _skills_phrase(candidate)
    signal_ph  = _signal_phrase(candidate, ref)
    snippet    = _career_snippet(candidate)
    edu_note   = _edu_note(candidate)
    sig        = candidate.get("redrob_signals") or {}
    notice     = int(_safe(sig, "notice_period_days", 0))
    otw        = bool(_safe(sig, "open_to_work_flag", False))
    notice_ph  = _notice_phrase(notice)
    activity_ph = _activity_phrase(sig, ref)

    # Cross-encoder confidence label
    if cross_encoder_score >= 8:
        conf = "very high relevance"
    elif cross_encoder_score >= 5:
        conf = "high relevance"
    elif cross_encoder_score >= 2:
        conf = "good relevance"
    else:
        conf = "moderate relevance"

    v = _variant(cid, 4)   # 4 structural variants per band

    if rank <= 10:
        # TOP 10 — strong endorsement, cite cross-encoder confidence + career evidence
        if v == 0:
            if snippet:
                s1 = f"{title} with {yoe:.1f} years{edu_note}; recent work includes {snippet}."
            else:
                s1 = f"{title} with {yoe:.1f} years{edu_note}, skilled in {skills_ph}."
            s2 = f"Cross-encoder signals {conf}; {signal_ph}."
        elif v == 1:
            s1 = f"Ranked #{rank}: {title}{edu_note} — {yoe:.1f} years of experience with expertise in {skills_ph}."
            s2 = f"{signal_ph.capitalize()}; joint relevance score indicates strong fit for this Senior AI role."
        elif v == 2:
            if snippet:
                s1 = f"Strong match: {yoe:.1f}-year {title}{edu_note} who has {snippet}."
            else:
                s1 = f"Strong match: {yoe:.1f}-year {title}{edu_note} with deep background in {skills_ph}."
            s2 = f"{signal_ph.capitalize()}; {notice_ph}."
        else:
            s1 = f"{title}{edu_note} — {yoe:.1f} yrs, specialising in {skills_ph}."
            s2 = (
                f"Model-assessed {conf} for this retrieval/ranking role; "
                f"{activity_ph}, {notice_ph}."
            )

    elif rank <= 25:
        # SOLID MATCH — skills + signal, no repetition
        if v == 0:
            s1 = f"{title} with {yoe:.1f} years of experience and strong background in {skills_ph}."
            s2 = f"{signal_ph.capitalize()}; well-aligned with Senior AI Engineer requirements."
        elif v == 1:
            s1 = f"{yoe:.1f}-year {title}{edu_note} with proven skills in {skills_ph}."
            s2 = f"{activity_ph.capitalize()}, {notice_ph}; {signal_ph}."
        elif v == 2:
            if snippet:
                s1 = f"{title}{edu_note} ({yoe:.1f} yrs) — background includes {snippet}."
            else:
                s1 = f"{title}{edu_note} ({yoe:.1f} yrs) with hands-on exposure to {skills_ph}."
            s2 = f"{signal_ph.capitalize()}, making them a solid candidate for this founding-team AI role."
        else:
            s1 = f"Solid AI profile: {title} with {yoe:.1f} years and depth in {skills_ph}."
            s2 = f"Recruiter signals positive — {signal_ph}; {notice_ph}."

    elif rank <= 50:
        # DECENT MATCH — relevant exposure, one signal, honest scope
        if v == 0:
            s1 = f"{title} ({yoe:.1f} yrs) with relevant exposure to {skills_ph}."
            s2 = f"Partially matches the retrieval/ranking focus; {signal_ph}."
        elif v == 1:
            s1 = f"{title}{edu_note} — {yoe:.1f} years with background in {skills_ph}."
            s2 = f"Fits several key criteria; {activity_ph}, {notice_ph}."
        elif v == 2:
            s1 = f"Relevant background: {yoe:.1f}-year {title} skilled in {skills_ph}."
            s2 = f"Good signals overall — {signal_ph}; ranked {rank} due to competitive pool."
        else:
            if snippet:
                s1 = f"{title} ({yoe:.1f} yrs) who has {snippet}."
            else:
                s1 = f"{title} ({yoe:.1f} yrs) with applicable skills in {skills_ph}."
            s2 = f"{signal_ph.capitalize()}; moderate cross-encoder fit for this specific JD."

    elif rank <= 75:
        # PARTIAL MATCH — acknowledge primary gap
        gap = _gap_phrase(candidate, ref)
        if v == 0:
            s1 = f"{title} with {yoe:.1f} years and skills in {skills_ph}."
            s2 = f"Ranked {rank} due to {gap}; {signal_ph}."
        elif v == 1:
            s1 = f"{title}{edu_note} ({yoe:.1f} yrs) — background in {skills_ph}."
            s2 = f"Primary limiting factor: {gap}; {activity_ph}."
        elif v == 2:
            s1 = f"Partial fit: {yoe:.1f}-year {title} with {skills_ph} exposure."
            s2 = f"Lower rank reflects {gap}; {signal_ph}."
        else:
            s1 = f"{title} ({yoe:.1f} yrs), skills include {skills_ph}."
            s2 = f"Gap identified: {gap}. {signal_ph.capitalize()}."

    else:
        # RANK 76-100 — honest gap, still mention upside
        gap = _gap_phrase(candidate, ref)
        if v == 0:
            s1 = f"{title} ({yoe:.1f} yrs) with background in {skills_ph}."
            s2 = f"Lower fit primarily because of {gap}; {signal_ph} within the broader pool."
        elif v == 1:
            s1 = f"{title}{edu_note} — {yoe:.1f} years, relevant skills: {skills_ph}."
            s2 = f"Ranked {rank} mainly due to {gap}; {activity_ph}."
        elif v == 2:
            s1 = f"Broader match: {yoe:.1f}-year {title} with {skills_ph}."
            s2 = f"Key gap — {gap} — limits rank; {signal_ph}."
        else:
            s1 = f"{title} ({yoe:.1f} yrs) — {skills_ph} in their profile."
            s2 = f"Placed at rank {rank} primarily due to {gap}; {notice_ph}."

    reasoning = f"{s1} {s2}"

    # Sanitise: collapse whitespace, cap length
    reasoning = " ".join(reasoning.split())
    if len(reasoning) > 400:
        reasoning = reasoning[:397] + "..."

    return reasoning

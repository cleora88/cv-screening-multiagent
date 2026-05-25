from __future__ import annotations

import re

from src.agents.schemas import AgentOutput, ScreeningInput


_LEADERSHIP_TERMS = ("lead", "senior", "principal", "manager", "head", "director")
_PROJECT_TERMS = ("project", "delivery", "built", "deployed", "ownership", "launched")
_COMMUNICATION_TERMS = ("team", "communication", "stakeholder", "collaboration", "client")
_DEGREE_TERMS = ("bachelor", "master", "phd", "doctorate", "engineer", "licence", "ingénieur")


def _extract_years(text: str) -> int:
    """Find the largest explicit experience number, e.g. '5 years'."""
    matches = [int(value) for value in re.findall(r"(\d+)\+?\s*(?:years?|ans?)", text.lower())]
    return max(matches, default=0)


def profile_match(screening_input: ScreeningInput) -> AgentOutput:
    """Score non-technical profile fit with transparent HR-style signals.

    This agent checks experience, seniority, project ownership, communication,
    and education. These signals complement the technical agent so the final
    decision is not based only on keyword matching.
    """
    cv = screening_input.cv_text.lower()
    job = screening_input.job_text.lower()
    evidence: list[str] = []
    score = 0.25

    # Start from a small baseline so candidates are not given a zero score just
    # because one signal is absent; each detected profile signal adds evidence.
    cv_years = _extract_years(cv)
    job_years = _extract_years(job)
    years_alignment = min(cv_years / max(job_years, 1), 1.0) if job_years else float(cv_years > 0)
    score += 0.25 * years_alignment
    if cv_years:
        evidence.append(f"Experience detected: {cv_years} years")

    leadership = any(term in cv for term in _LEADERSHIP_TERMS)
    if leadership:
        score += 0.2
        evidence.append("Leadership or seniority signals present")

    project_ownership = any(term in cv for term in _PROJECT_TERMS)
    if project_ownership:
        score += 0.15
        evidence.append("Project ownership or delivery evidence present")

    communication = any(term in cv for term in _COMMUNICATION_TERMS)
    if communication:
        score += 0.1
        evidence.append("Communication or collaboration signals present")

    education = any(term in cv for term in _DEGREE_TERMS)
    if education:
        score += 0.1
        evidence.append("Formal degree signal detected")

    score = min(score, 0.95)

    label = "High" if score >= 0.67 else "Medium" if score >= 0.45 else "Low"
    recommendation = "shortlist" if label == "High" else "review" if label == "Medium" else "reject"
    rationale = (
        f"Profile score based on years alignment ({years_alignment:.2f}), leadership={leadership}, "
        f"project_ownership={project_ownership}, communication={communication}, education={education}."
    )
    return AgentOutput(
        agent_name="profile_analyzer",
        score=score,
        label=label,
        rationale=rationale,
        success=True,
        error=None,
        recommendation=recommendation,
        evidence=evidence,
        metadata={
            "cv_years": cv_years,
            "job_years": job_years,
            "leadership": leadership,
            "project_ownership": project_ownership,
            "communication": communication,
            "education": education,
        },
    )

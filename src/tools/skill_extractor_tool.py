from __future__ import annotations

from dataclasses import dataclass

from src.data.dataset import extract_skills as extract_skill_set


@dataclass
class SkillToolResult:
    success: bool
    matched_skills: list[str]
    coverage: float
    error: str | None
    job_skills: list[str] | None = None
    missing_skills: list[str] | None = None


def extract_skills(cv_text: str, job_text: str) -> SkillToolResult:
    """Compare required job skills with skills detected in the CV.

    The returned coverage ratio is easy to explain: matched job skills divided
    by total detected job skills. Missing skills are kept for evidence in the UI.
    """
    try:
        cv_skills = extract_skill_set(cv_text)
        job_skills = extract_skill_set(job_text)
        matched = sorted(list(cv_skills & job_skills))
        missing = sorted(list(job_skills - cv_skills))
        coverage = len(matched) / max(len(job_skills), 1)
        return SkillToolResult(True, matched, coverage, None, sorted(job_skills), missing)
    except Exception as exc:
        return SkillToolResult(False, [], 0.0, str(exc), [], [])

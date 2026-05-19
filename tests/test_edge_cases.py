"""Edge-case tests for tools and orchestrator robustness."""
from pathlib import Path

import pytest

from src.pipeline.hitl import human_checkpoint
from src.tools.model_tool import run_model_tool
from src.tools.skill_extractor_tool import extract_skills


MODEL_PATH = Path("models/cv_fit_model.pt")


class TestModelToolEdgeCases:
    """CVFitNet model tool robustness."""

    def test_empty_cv_returns_result(self):
        result = run_model_tool("", "Python developer needed", MODEL_PATH)
        assert result.fit_label in {"High", "Medium", "Low", "Error"}

    def test_empty_job_returns_result(self):
        result = run_model_tool("Python expert with 5 years", "", MODEL_PATH)
        assert result.fit_label in {"High", "Medium", "Low", "Error"}

    def test_very_long_cv_does_not_crash(self):
        long_cv = ("Python machine learning deep learning pytorch " * 500).strip()
        result = run_model_tool(long_cv, "Python ML engineer", MODEL_PATH)
        assert result.fit_label in {"High", "Medium", "Low", "Error"}

    def test_numeric_only_input(self):
        result = run_model_tool("12345 67890", "99999 00000", MODEL_PATH)
        assert result.fit_label in {"High", "Medium", "Low", "Error"}

    def test_score_is_bounded(self):
        result = run_model_tool("Python developer", "Python engineer", MODEL_PATH)
        assert 0.0 <= result.score <= 1.0


class TestSkillExtractorEdgeCases:
    """Skill extractor robustness."""

    def test_empty_cv(self):
        result = extract_skills("", "Python machine learning")
        assert result.coverage == 0.0
        assert result.matched_skills == []

    def test_empty_job(self):
        result = extract_skills("Python machine learning", "")
        assert result.coverage == 0.0

    def test_both_empty(self):
        result = extract_skills("", "")
        assert result.coverage == 0.0

    def test_all_skills_match(self):
        skill = "python"
        result = extract_skills(skill, skill)
        assert result.coverage == pytest.approx(1.0)
        assert skill in result.matched_skills

    def test_unicode_input(self):
        result = extract_skills("Développeur Python senior", "Python développeur expérimenté")
        assert isinstance(result.coverage, float)
        assert 0.0 <= result.coverage <= 1.0

    def test_numeric_only_input(self):
        result = extract_skills("12345", "67890")
        assert result.coverage == 0.0


class TestHitlStrictMode:
    def test_noninteractive_defaults_to_flagged_review(self, monkeypatch):
        class _FakeStdin:
            @staticmethod
            def isatty() -> bool:
                return False

        monkeypatch.setattr("sys.stdin", _FakeStdin())
        result = human_checkpoint(
            candidate_id="cand_noninteractive",
            score=0.5,
            rationale="borderline",
            reasons=["borderline final score"],
            require_human_approval=False,
        )

        assert result["status"] == "auto-flagged"
        assert result["reasons"] == ["borderline final score"]

    def test_pending_when_human_required_and_noninteractive(self, monkeypatch):
        class _FakeStdin:
            @staticmethod
            def isatty() -> bool:
                return False

        monkeypatch.setattr("sys.stdin", _FakeStdin())
        result = human_checkpoint(
            candidate_id="cand_noninteractive",
            score=0.5,
            rationale="borderline",
            reasons=["borderline final score"],
            require_human_approval=True,
        )

        assert result["status"] == "pending-human-approval"

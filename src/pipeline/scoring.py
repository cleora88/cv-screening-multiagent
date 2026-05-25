from __future__ import annotations

DISAGREEMENT_REVIEW_THRESHOLD = 0.25


def label_from_score(score: float) -> str:
    """Convert a normalized 0..1 score into the project labels."""
    if score >= 0.67:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def recommendation_from_label(label: str) -> str:
    """Map model labels to HR-facing actions used by the UI and reports."""
    if label == "High":
        return "shortlist"
    if label == "Medium":
        return "review"
    return "reject"

from __future__ import annotations

DISAGREEMENT_REVIEW_THRESHOLD = 0.25


def label_from_score(score: float) -> str:
    if score >= 0.67:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


def recommendation_from_label(label: str) -> str:
    if label == "High":
        return "shortlist"
    if label == "Medium":
        return "review"
    return "reject"

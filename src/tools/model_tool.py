from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.dataset import extract_skills, featurize
from src.pipeline.scoring import label_from_score


MODEL_FEATURE_COUNT = 8
HEURISTIC_WEIGHTS = torch.tensor([0.35, 0.15, 0.15, 0.10, 0.05, 0.08, 0.07, 0.05], dtype=torch.float32)


class CVFitNet(torch.nn.Module):
    """Small neural network that classifies CV/job fit into Low/Medium/High."""

    def __init__(self, in_features: int = MODEL_FEATURE_COUNT, hidden: int = 16, out_features: int = 3) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ModelToolResult:
    success: bool
    fit_label: str
    score: float
    model_used: str
    error: str | None
    matched_skills: list[str] | None = None
    missing_skills: list[str] | None = None
    feature_vector: list[float] | None = None
    confidence: float | None = None


_SENIORITY = {"lead", "senior", "principal", "manager", "head", "director"}
_DEGREES = {"bachelor", "master", "phd", "doctorate", "engineer", "licence", "ingénieur"}


def _simple_featureize(cv_text: str, job_text: str) -> torch.Tensor:
    """Convert raw text into the 8 numeric features expected by CVFitNet."""
    return torch.tensor([featurize(cv_text, job_text)], dtype=torch.float32)


def _heuristic_predict(features: torch.Tensor) -> tuple[str, float]:
    """Fallback score used when the trained .pt file is missing or unreadable."""
    weighted_score = float(torch.dot(features[0], HEURISTIC_WEIGHTS).item())
    return label_from_score(weighted_score), round(weighted_score, 4)


def run_model_tool(cv_text: str, job_text: str, model_path: Path) -> ModelToolResult:
    """Run the trained PyTorch classifier, with a deterministic fallback.

    In the presentation, this is the "DL model as a callable tool": agents call
    this function, not the model directly. If the model artifact is unavailable,
    the tool still returns a reasonable heuristic result so the pipeline remains
    usable.
    """
    try:
        features = _simple_featureize(cv_text, job_text)
        matched_skills = sorted(list(extract_skills(cv_text) & extract_skills(job_text)))
        missing_skills = sorted(list(extract_skills(job_text) - extract_skills(cv_text)))
        model = CVFitNet()
        model_name = "feature_heuristic"
        load_error = None

        if model_path.exists():
            try:
                # The trained model outputs class probabilities for
                # Low/Medium/High; the weighted average converts those
                # probabilities into a normalized fit score.
                state = torch.load(model_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.eval()
                with torch.no_grad():
                    logits = model(features)
                    probs = torch.softmax(logits, dim=1)[0]
                    idx = int(torch.argmax(probs).item())
                    confidence = float(torch.max(probs).item())
                    score = float(sum(probs[i].item() * (i / 2.0) for i in range(len(probs))))
                label_map = {0: "Low", 1: "Medium", 2: "High"}
                return ModelToolResult(
                    success=True,
                    fit_label=label_map[idx],
                    score=round(score, 4),
                    model_used="trained_pytorch_model",
                    error=None,
                    matched_skills=matched_skills,
                    missing_skills=missing_skills,
                    feature_vector=features[0].tolist(),
                    confidence=round(confidence, 4),
                )
            except Exception as exc:
                load_error = str(exc)

        label, score = _heuristic_predict(features)
        return ModelToolResult(
            success=True,
            fit_label=label,
            score=score,
            model_used=model_name,
            error=load_error,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            feature_vector=features[0].tolist(),
            confidence=None,
        )
    except Exception as exc:
        # Hard fallback prevents pipeline crash on model/tool errors.
        return ModelToolResult(
            success=False,
            fit_label="Medium",
            score=0.5,
            model_used="fallback_on_error",
            error=str(exc),
            matched_skills=[],
            missing_skills=[],
            feature_vector=[],
            confidence=None,
        )

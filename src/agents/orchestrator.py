from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from src.agents.profile_analyzer import profile_match
from src.agents.schemas import ScreeningInput
from src.agents.technical_matcher import technical_match
from src.pipeline.hitl import human_checkpoint_cli, needs_human_review

if TYPE_CHECKING:
    from src.llm_client import OllamaClient


def run_screening(
    screening_input: ScreeningInput,
    model_path,
    low: float,
    high: float,
    logger,
    ollama_client: "OllamaClient | None" = None,
    interactive_human_review: bool = True,
) -> dict[str, Any]:
    logger.log("screening_started", {"candidate_id": screening_input.candidate_id, "job_id": screening_input.job_id})

    tech = technical_match(screening_input, model_path)
    logger.log("agent_result", asdict(tech))

    profile = profile_match(screening_input)
    logger.log("agent_result", asdict(profile))

    score = (tech.score * 0.65) + (profile.score * 0.35)
    label = "High" if score >= 0.67 else "Medium" if score >= 0.45 else "Low"
    recommendation = "shortlist" if label == "High" else "review" if label == "Medium" else "reject"
    disagreement = abs(tech.score - profile.score)
    review_reasons: list[str] = []
    if low <= score <= high:
        review_reasons.append("borderline final score")
    if disagreement >= 0.25:
        review_reasons.append("specialist disagreement")
    if not tech.success or not profile.success:
        review_reasons.append("agent failure fallback")

    review = None
    if needs_human_review(score, low, high, disagreement=disagreement, agent_failure=(not tech.success or not profile.success)):
        rationale = f"Tech: {tech.rationale} | Profile: {profile.rationale}"
        if interactive_human_review:
            review = human_checkpoint_cli(screening_input.candidate_id, score, rationale, review_reasons)
        else:
            review = {
                "candidate_id": screening_input.candidate_id,
                "score": round(score, 4),
                "status": "auto-flagged",
                "reviewer": "system",
                "reason": "interactive review disabled",
                "reasons": review_reasons,
                "timestamp": None,
            }
        logger.log("human_checkpoint", review)
        if review["status"] == "approved":
            recommendation = "shortlist"
        elif review["status"] == "rejected":
            recommendation = "reject"
        else:
            recommendation = "review"

    # Optional LLM enrichment via Ollama
    llm_rationale: str | None = None
    if ollama_client is not None:
        try:
            llm_rationale = ollama_client.screening_rationale(
                candidate_id=screening_input.candidate_id,
                final_score=score,
                final_label=label,
                tech_rationale=tech.rationale,
                profile_rationale=profile.rationale,
            )
            logger.log("llm_rationale", {
                "candidate_id": screening_input.candidate_id,
                "model": ollama_client.model,
                "rationale": llm_rationale,
            })
        except RuntimeError as exc:
            llm_rationale = f"[Ollama unavailable: {exc}]"
            logger.log("llm_rationale_error", {"error": str(exc)})

    final = {
        "candidate_id": screening_input.candidate_id,
        "job_id": screening_input.job_id,
        "final_score": round(score, 4),
        "final_label": label,
        "recommendation": recommendation,
        "orchestrator_summary": (
            f"Technical agent={tech.label} ({tech.score:.2f}); "
            f"Profile agent={profile.label} ({profile.score:.2f}); disagreement={disagreement:.2f}."
        ),
        "review_reasons": review_reasons,
        "llm_rationale": llm_rationale,
        "technical": asdict(tech),
        "profile": asdict(profile),
        "human_review": review,
    }
    logger.log("orchestrator_decision", {
        "candidate_id": screening_input.candidate_id,
        "final_score": round(score, 4),
        "final_label": label,
        "recommendation": recommendation,
        "review_reasons": review_reasons,
    })
    logger.log("screening_completed", final)
    return final

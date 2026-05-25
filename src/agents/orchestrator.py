from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from src.agents.profile_analyzer import profile_match
from src.agents.schemas import ScreeningInput
from src.pipeline.scoring import (
    DISAGREEMENT_REVIEW_THRESHOLD,
    label_from_score,
    recommendation_from_label,
)
from src.agents.technical_matcher import technical_match
from src.pipeline.hitl import human_checkpoint, needs_human_review

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
    require_human_approval: bool = False,
) -> dict[str, Any]:
    """Run the full screening workflow for one candidate/job pair.

    The orchestrator is the coordinator: it calls the two specialist agents,
    blends their scores, checks whether a human review is needed, optionally
    asks Ollama for a readable rationale, and writes audit logs.
    """
    logger.log(
        "screening_started",
        {"candidate_id": screening_input.candidate_id, "job_id": screening_input.job_id},
        agent_name="orchestrator",
        action="start_screening",
        input_summary=f"candidate_id={screening_input.candidate_id}, job_id={screening_input.job_id}",
        tool_used="technical_matcher,profile_analyzer",
        status="started",
    )

    # Specialist agents work independently so their scores can be compared.
    # A large difference means the automated decision should be reviewed.
    tech = technical_match(screening_input, model_path)
    logger.log(
        "agent_result",
        asdict(tech),
        agent_name=tech.agent_name,
        action="produce_assessment",
        input_summary="cv_text + job_text",
        output_summary=f"label={tech.label}, score={tech.score:.4f}, recommendation={tech.recommendation}",
        tool_used="model_tool,skill_extractor_tool",
        status="success" if tech.success else "failure",
        error=tech.error,
    )

    profile = profile_match(screening_input)
    logger.log(
        "agent_result",
        asdict(profile),
        agent_name=profile.agent_name,
        action="produce_assessment",
        input_summary="cv_text + job_text",
        output_summary=f"label={profile.label}, score={profile.score:.4f}, recommendation={profile.recommendation}",
        tool_used="heuristic_profile_rules",
        status="success" if profile.success else "failure",
        error=profile.error,
    )

    # Technical fit is weighted more because this project screens for role
    # requirements, while profile fit adds HR-style context.
    score = (tech.score * 0.65) + (profile.score * 0.35)
    label = label_from_score(score)
    recommendation = recommendation_from_label(label)
    disagreement = abs(tech.score - profile.score)
    review_reasons: list[str] = []
    if low <= score <= high:
        review_reasons.append("borderline final score")
    if disagreement >= DISAGREEMENT_REVIEW_THRESHOLD:
        review_reasons.append("specialist disagreement")
    if not tech.success or not profile.success:
        review_reasons.append("agent failure fallback")

    review = None
    if needs_human_review(score, low, high, disagreement=disagreement, agent_failure=(not tech.success or not profile.success)):
        # HITL means "human in the loop": borderline, conflicted, or fallback
        # cases are routed to a reviewer instead of being blindly accepted.
        rationale = f"Tech: {tech.rationale} | Profile: {profile.rationale}"
        if interactive_human_review:
            review = human_checkpoint(
                screening_input.candidate_id,
                score,
                rationale,
                review_reasons,
                require_human_approval=require_human_approval,
            )
        else:
            if require_human_approval:
                review = {
                    "candidate_id": screening_input.candidate_id,
                    "score": round(score, 4),
                    "status": "pending-human-approval",
                    "reviewer": "unassigned",
                    "reason": "interactive review disabled while human approval is required",
                    "reasons": review_reasons,
                    "timestamp": None,
                }
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
        logger.log(
            "human_checkpoint",
            review,
            agent_name="orchestrator",
            action="request_human_decision",
            input_summary=f"score={score:.4f}, disagreement={disagreement:.4f}, reasons={review_reasons}",
            output_summary=f"status={review.get('status')}, reviewer={review.get('reviewer')}",
            tool_used="human_checkpoint",
            status="success",
        )
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
            logger.log(
                "llm_rationale",
                {
                    "candidate_id": screening_input.candidate_id,
                    "model": ollama_client.model,
                    "rationale": llm_rationale,
                },
                agent_name="orchestrator",
                action="generate_llm_rationale",
                input_summary=f"candidate_id={screening_input.candidate_id}, model={ollama_client.model}",
                output_summary="short hiring rationale generated",
                tool_used="ollama_generate",
                status="success",
            )
        except RuntimeError as exc:
            llm_rationale = f"[Ollama unavailable: {exc}]"
            logger.log(
                "llm_rationale_error",
                {"error": str(exc)},
                agent_name="orchestrator",
                action="generate_llm_rationale",
                tool_used="ollama_generate",
                status="failure",
                error=str(exc),
            )

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
    logger.log(
        "orchestrator_decision",
        {
            "candidate_id": screening_input.candidate_id,
            "final_score": round(score, 4),
            "final_label": label,
            "recommendation": recommendation,
            "review_reasons": review_reasons,
        },
        agent_name="orchestrator",
        action="final_decision",
        input_summary=f"tech_score={tech.score:.4f}, profile_score={profile.score:.4f}",
        output_summary=f"final_label={label}, recommendation={recommendation}",
        tool_used="decision_policy",
        status="success",
    )
    logger.log(
        "screening_completed",
        final,
        agent_name="orchestrator",
        action="complete_screening",
        output_summary=f"candidate_id={screening_input.candidate_id}, final_label={label}",
        status="success",
    )
    return final

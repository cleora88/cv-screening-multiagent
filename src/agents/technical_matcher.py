from __future__ import annotations

from src.agents.schemas import AgentOutput, ScreeningInput
from src.pipeline.scoring import label_from_score, recommendation_from_label
from src.tools.model_tool import run_model_tool
from src.tools.skill_extractor_tool import extract_skills


def technical_match(screening_input: ScreeningInput, model_path) -> AgentOutput:
    model_result = run_model_tool(screening_input.cv_text, screening_input.job_text, model_path)
    skill_result = extract_skills(screening_input.cv_text, screening_input.job_text)

    if not model_result.success and not skill_result.success:
        return AgentOutput(
            agent_name="technical_matcher",
            score=0.5,
            label="Medium",
            rationale="Both model and skills tools failed; safe fallback used.",
            success=False,
            error="model_and_skill_tool_failure",
            recommendation="review",
            evidence=["DL model unavailable", "Skill extractor unavailable"],
        )

    blended = (model_result.score * 0.7) + (skill_result.coverage * 0.3)
    label = label_from_score(blended)
    recommendation = recommendation_from_label(label)
    missing_skills = skill_result.missing_skills or []
    evidence = [
        f"Model label: {model_result.fit_label} via {model_result.model_used}",
        f"Model confidence: {model_result.confidence:.2f}" if model_result.confidence is not None else "Model confidence: n/a",
        f"Matched skills: {', '.join(skill_result.matched_skills) if skill_result.matched_skills else 'none'}",
        f"Missing skills: {', '.join(missing_skills) if missing_skills else 'none'}",
    ]
    rationale = (
        f"Model={model_result.fit_label}({model_result.score:.2f}), "
        f"skills_coverage={skill_result.coverage:.2f}, matched={skill_result.matched_skills}, "
        f"missing={missing_skills}"
    )
    return AgentOutput(
        agent_name="technical_matcher",
        score=blended,
        label=label,
        rationale=rationale,
        success=True,
        error=None,
        recommendation=recommendation,
        evidence=evidence,
        metadata={
            "model_used": model_result.model_used,
            "model_confidence": model_result.confidence,
            "feature_vector": model_result.feature_vector or [],
            "matched_skills": skill_result.matched_skills,
            "missing_skills": missing_skills,
        },
    )

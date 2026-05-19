from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agents.orchestrator import run_screening
from src.agents.schemas import ScreeningInput

if TYPE_CHECKING:
    from pathlib import Path

    from src.llm_client import OllamaClient
    from src.utils.json_logger import JsonLogger


def run_batch_screening(
    *,
    cv_records: list[dict[str, Any]],
    job_record: dict[str, Any],
    model_path: "Path",
    low: float,
    high: float,
    logger: "JsonLogger",
    ollama_client: "OllamaClient | None" = None,
    require_human_approval: bool = False,
) -> list[dict[str, Any]]:
    """Screen one or many CVs against one job using the normal agent workflow."""
    job_id = str(job_record.get("job_id", "job_demo")).strip() or "job_demo"
    job_text = str(job_record.get("job_text", "")).strip()
    total = len(cv_records)

    logger.log(
        "batch_screening_started",
        {"job_id": job_id, "candidate_count": total},
        agent_name="orchestrator",
        action="start_batch_screening",
        input_summary=f"job_id={job_id}, candidate_count={total}",
        tool_used="technical_matcher,profile_analyzer,decision_policy",
        status="started",
    )

    results: list[dict[str, Any]] = []
    for index, cv_record in enumerate(cv_records, start=1):
        candidate_id = str(cv_record.get("candidate_id", f"candidate_{index:03d}")).strip()
        cv_text = str(cv_record.get("cv_text", "")).strip()
        result = run_screening(
            ScreeningInput(
                candidate_id=candidate_id or f"candidate_{index:03d}",
                cv_text=cv_text,
                job_id=job_id,
                job_text=job_text,
            ),
            model_path=model_path,
            low=low,
            high=high,
            logger=logger,
            ollama_client=ollama_client,
            interactive_human_review=False,
            require_human_approval=require_human_approval,
        )
        result["batch_rank_input_order"] = index
        results.append(result)

    results.sort(key=lambda item: item.get("final_score", 0.0), reverse=True)
    for rank, result in enumerate(results, start=1):
        result["batch_rank"] = rank

    logger.log(
        "batch_screening_completed",
        {
            "job_id": job_id,
            "candidate_count": total,
            "ranked_candidates": [
                {
                    "rank": result["batch_rank"],
                    "candidate_id": result["candidate_id"],
                    "final_score": result["final_score"],
                    "recommendation": result["recommendation"],
                }
                for result in results
            ],
        },
        agent_name="orchestrator",
        action="complete_batch_screening",
        output_summary=f"screened={total}, top_candidate={results[0]['candidate_id'] if results else 'none'}",
        tool_used="decision_policy",
        status="success",
    )
    return results

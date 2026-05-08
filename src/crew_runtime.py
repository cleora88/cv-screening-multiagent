from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agents.schemas import ScreeningInput
from src.config import settings
from src.utils.json_logger import JsonLogger

_CREWAI_AVAILABLE = importlib.util.find_spec("crewai") is not None


def is_crewai_available() -> bool:
    return _CREWAI_AVAILABLE


def _build_tools(model_path: Path):
    """Return CrewAI tool functions wrapping the two project tools."""
    from crewai.tools import tool as crewai_tool  # type: ignore[reportMissingImports]
    from src.tools.model_tool import run_model_tool
    from src.tools.skill_extractor_tool import extract_skills

    def _normalize_inputs(cv_and_job: str = "", cv_text: str = "", job_text: str = "") -> tuple[str, str]:
        if cv_text or job_text:
            return cv_text or cv_and_job, job_text
        try:
            parts = cv_and_job.split("|||")
            norm_cv = parts[0].replace("CV:", "").strip() if len(parts) > 0 else cv_and_job
            norm_job = parts[1].replace("JOB:", "").strip() if len(parts) > 1 else ""
            return norm_cv, norm_job
        except Exception:
            return cv_and_job, ""

    @crewai_tool("CV Fit Classifier")
    def dl_classifier_tool(cv_and_job: str = "", cv_text: str = "", job_text: str = "") -> str:
        """
        Classify CV fit using the trained PyTorch model.
        Input format: 'CV: <cv_text> ||| JOB: <job_text>'
        Returns fit label (High/Medium/Low) and confidence score.
        """
        cv_text, job_text = _normalize_inputs(cv_and_job=cv_and_job, cv_text=cv_text, job_text=job_text)

        result = run_model_tool(cv_text, job_text, model_path)
        return (
            f"Fit={result.fit_label}, confidence={result.score:.3f}, "
            f"model={result.model_used}, error={result.error}"
        )

    @crewai_tool("Skill Extractor")
    def skill_extractor_tool(cv_and_job: str = "", cv_text: str = "", job_text: str = "") -> str:
        """
        Extract and compare skills between a CV and job description.
        Input format: 'CV: <cv_text> ||| JOB: <job_text>'
        Returns matched skills and coverage ratio.
        """
        cv_text, job_text = _normalize_inputs(cv_and_job=cv_and_job, cv_text=cv_text, job_text=job_text)

        result = extract_skills(cv_text, job_text)
        return (
            f"Matched skills: {result.matched_skills}, "
            f"coverage={result.coverage:.3f}, error={result.error}"
        )

    return dl_classifier_tool, skill_extractor_tool


def run_with_crewai(
    screening_input: ScreeningInput,
    logger: JsonLogger | None = None,
    ollama_host: str = "http://localhost:11434",
    ollama_model: str | None = None,
) -> dict[str, Any]:
    if not _CREWAI_AVAILABLE:
        return {
            "success": False,
            "error": "CrewAI is not installed. Run: pip install -r requirements.txt",
            "candidate_id": screening_input.candidate_id,
        }

    from crewai import Agent, Crew, Process, Task  # type: ignore[reportMissingImports]

    if logger is None:
        logger = JsonLogger(settings.log_dir_abs)

    # Keep CLI runs non-interactive and avoid trace prompts in defense demos.
    import os
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    # If an Ollama model is specified, configure LiteLLM to route through Ollama.
    # CrewAI uses LiteLLM under the hood; "ollama/<model>" is the canonical format.
    llm_spec: str | None = None
    if ollama_model:
        os.environ.setdefault("OPENAI_API_BASE", f"{ollama_host}/v1")
        os.environ.setdefault("OPENAI_API_KEY", "ollama")
        llm_spec = f"ollama/{ollama_model}"

    payload_str = f"CV: {screening_input.cv_text} ||| JOB: {screening_input.job_text}"
    logger.log(
        "crewai_run_started",
        {
            "candidate_id": screening_input.candidate_id,
            "job_id": screening_input.job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        agent_name="crewai_runtime",
        action="start_crewai_workflow",
        input_summary=f"candidate_id={screening_input.candidate_id}, job_id={screening_input.job_id}",
        tool_used="CrewAI",
        status="started",
    )

    dl_tool, skill_tool = _build_tools(settings.model_path_abs)

    _agent_kwargs: dict = {"verbose": True}
    if llm_spec:
        _agent_kwargs["llm"] = llm_spec

    technical_agent = Agent(
        role="Technical Matcher",
        goal="Assess technical skill alignment between the candidate CV and job requirements.",
        backstory=(
            "You are an expert technical recruiter specialised in hard skills, "
            "tools, and technology stacks. You use provided tools to extract and score skills."
        ),
        tools=[dl_tool, skill_tool],
        **_agent_kwargs,
    )
    profile_agent = Agent(
        role="Profile Analyzer",
        goal="Assess overall candidate profile: seniority, experience level, and role alignment.",
        backstory=(
            "You are an expert HR analyst who evaluates career trajectory, "
            "leadership signals, and role fit beyond pure technical skills."
        ),
        tools=[skill_tool],
        **_agent_kwargs,
    )

    t1 = Task(
        description=(
            f"Use your tools to analyse the technical fit for candidate {screening_input.candidate_id}.\n"
            f"Input for tools: {payload_str}\n"
            "Provide: (1) a fit label High/Medium/Low, (2) key evidence, (3) confidence score."
        ),
        expected_output="Technical fit assessment: label, evidence, and confidence score.",
        agent=technical_agent,
    )
    t2 = Task(
        description=(
            f"Analyse the overall profile fit for candidate {screening_input.candidate_id}.\n"
            f"Input for tools: {payload_str}\n"
            "Provide: (1) seniority level assessment, (2) role alignment, (3) recommendation."
        ),
        expected_output="Profile fit assessment: seniority level, alignment rationale, recommendation.",
        agent=profile_agent,
    )

    crew = Crew(
        agents=[technical_agent, profile_agent],
        tasks=[t1, t2],
        process=Process.sequential,
        verbose=False,
    )

    try:
        crew_output = crew.kickoff()
        raw = str(crew_output)
    except Exception as exc:
        logger.log(
            "crewai_run_error",
            {"error": str(exc), "candidate_id": screening_input.candidate_id},
            agent_name="crewai_runtime",
            action="execute_crewai_workflow",
            tool_used="CrewAI",
            status="failure",
            error=str(exc),
        )
        return {
            "success": False,
            "error": str(exc),
            "candidate_id": screening_input.candidate_id,
        }

    result: dict[str, Any] = {
        "success": True,
        "candidate_id": screening_input.candidate_id,
        "job_id": screening_input.job_id,
        "llm_backend": llm_spec or "crewai_default",
        "crew_output": raw,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.log(
        "crewai_run_completed",
        result,
        agent_name="crewai_runtime",
        action="complete_crewai_workflow",
        output_summary=f"candidate_id={screening_input.candidate_id}, llm_backend={result['llm_backend']}",
        tool_used="CrewAI",
        status="success",
    )
    return result

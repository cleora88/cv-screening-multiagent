from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScreeningInput:
    """One screening request: compare a single CV against a single job."""

    candidate_id: str
    cv_text: str
    job_id: str
    job_text: str


@dataclass
class AgentOutput:
    """Standard response shape returned by every specialist agent.

    Keeping all agents on this structure makes orchestration, logging, tests,
    and the Streamlit UI simpler because they can read the same fields from
    both the technical and profile agents.
    """

    agent_name: str
    score: float
    label: str
    rationale: str
    success: bool
    error: str | None
    recommendation: str = "review"
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScreeningInput:
    candidate_id: str
    cv_text: str
    job_id: str
    job_text: str


@dataclass
class AgentOutput:
    agent_name: str
    score: float
    label: str
    rationale: str
    success: bool
    error: str | None
    recommendation: str = "review"
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

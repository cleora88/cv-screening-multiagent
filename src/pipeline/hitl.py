from __future__ import annotations

import sys
from datetime import datetime, timezone


def needs_human_review(
    score: float,
    low: float,
    high: float,
    disagreement: float = 0.0,
    agent_failure: bool = False,
) -> bool:
    return low <= score <= high or disagreement >= 0.25 or agent_failure


def human_checkpoint_cli(candidate_id: str, score: float, rationale: str = "", reasons: list[str] | None = None) -> dict:
    """
    Prompt the human reviewer for a borderline candidate.
    Falls back to auto-approve when stdin is not a TTY (e.g. CI/batch).
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    if not sys.stdin.isatty():
        return {
            "candidate_id": candidate_id,
            "score": round(score, 4),
            "status": "auto-approved",
            "reviewer": "system",
            "reason": "non-interactive environment",
            "timestamp": timestamp,
        }

    separator = "-" * 60
    print(f"\n{separator}")
    print("  HUMAN REVIEW REQUIRED")
    print(separator)
    print(f"  Candidate : {candidate_id}")
    print(f"  Score     : {score:.4f}  (borderline — needs your decision)")
    if rationale:
        print(f"  Rationale : {rationale}")
    if reasons:
        print(f"  Why now   : {', '.join(reasons)}")
    print(separator)
    print("  Options:")
    print("    a  — Approve (advance to next stage)")
    print("    r  — Reject  (exclude from shortlist)")
    print("    f  — Flag    (needs further manual review)")
    print(separator)

    status_map = {"a": "approved", "r": "rejected", "f": "flagged"}
    choice = ""
    for attempt in range(3):
        try:
            raw = input("  Your decision [a/r/f]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raw = ""
        if raw in status_map:
            choice = raw
            break
        print(f"  Invalid input '{raw}'. Please enter a, r, or f.")

    if not choice:
        print("  No valid input received — defaulting to 'flagged'.")
        choice = "f"

    reviewer_name = ""
    try:
        reviewer_name = input("  Reviewer name (optional, press Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        pass

    return {
        "candidate_id": candidate_id,
        "score": round(score, 4),
        "status": status_map[choice],
        "reviewer": reviewer_name or "anonymous",
        "reasons": reasons or [],
        "timestamp": timestamp,
    }

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JsonLogger:
    """Append one JSON object per event for traceability and defense evidence."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.log_file = self.log_dir / f"run_{stamp}.jsonl"

    def log(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        agent_name: str | None = None,
        action: str | None = None,
        input_summary: str | None = None,
        output_summary: str | None = None,
        tool_used: str | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        """Write a structured audit row for an agent action or pipeline event."""
        derived_error = error or payload.get("error")
        derived_status = status or payload.get("status") or ("failure" if derived_error else "success")
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_name": agent_name or payload.get("agent_name") or "system",
            "action": action or event,
            "tool_used": tool_used,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "status": derived_status,
            "error": derived_error,
            "event": event,
            "payload": payload,
        }
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

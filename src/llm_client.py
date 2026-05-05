from __future__ import annotations

import json
import urllib.error
import urllib.request


class OllamaClient:
    """Thin, zero-dependency wrapper around the Ollama local REST API.

    Uses only Python stdlib (urllib) so no extra package is required.
    The server must be running: ``ollama serve`` (default port 11434).
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.2",
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_available(self, timeout_seconds: int = 8) -> bool:
        """Return True if the Ollama server responds within timeout_seconds."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=timeout_seconds):
                return True
        except Exception:
            return False

    def generate(self, prompt: str, max_tokens: int = 300) -> str:
        """Send *prompt* to ``/api/generate`` and return the response text.

        Raises ``RuntimeError`` if the server is unreachable or returns an error.
        """
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "").strip()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama unreachable at {self.host}. "
                "Is 'ollama serve' running?"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama generate error: {exc}") from exc

    # ------------------------------------------------------------------
    # Screening-specific prompts
    # ------------------------------------------------------------------

    def screening_rationale(
        self,
        candidate_id: str,
        final_score: float,
        final_label: str,
        tech_rationale: str,
        profile_rationale: str,
    ) -> str:
        """Ask the LLM for a short professional hiring recommendation."""
        prompt = (
            "You are an expert HR analyst. Based on the automated CV screening result "
            "below, write a concise 2-3 sentence professional hiring recommendation.\n\n"
            f"Candidate: {candidate_id}\n"
            f"Overall fit score: {final_score:.3f} — {final_label}\n"
            f"Technical assessment: {tech_rationale}\n"
            f"Profile assessment: {profile_rationale}\n\n"
            "Recommendation:"
        )
        return self.generate(prompt, max_tokens=120)

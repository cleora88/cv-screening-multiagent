from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agents.orchestrator import run_screening
from src.agents.schemas import ScreeningInput
from src.crew_runtime import is_crewai_available, run_with_crewai
from src.config import settings
from src.llm_client import OllamaClient
from src.utils.json_logger import JsonLogger


def _load_sample_data(root: Path) -> tuple[dict, dict]:
    cv = json.loads((root / "data" / "sample_cvs.json").read_text(encoding="utf-8"))[0]
    job = json.loads((root / "data" / "sample_jobs.json").read_text(encoding="utf-8"))[0]
    return cv, job


def _load_json_record(path: Path, index: int = 0) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload[index]
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CV screening multi-agent pipeline.")
    parser.add_argument("--cv-file", type=Path, help="Path to a JSON file containing one CV record or a list of CV records.")
    parser.add_argument("--job-file", type=Path, help="Path to a JSON file containing one job record or a list of job records.")
    parser.add_argument("--cv-index", type=int, default=0, help="Index to use when the CV JSON file contains a list.")
    parser.add_argument("--job-index", type=int, default=0, help="Index to use when the job JSON file contains a list.")
    parser.add_argument(
        "--runtime",
        choices=["auto", "crewai", "deterministic"],
        default="auto",
        help=(
            "Runtime backend: auto (CrewAI primary, deterministic fallback), "
            "crewai (force CrewAI), deterministic (force deterministic orchestrator)."
        ),
    )
    parser.add_argument("--use-crewai", action="store_true", help="Deprecated alias for --runtime crewai.")
    parser.add_argument("--ollama", action="store_true", help="Enable Ollama LLM rationale enrichment (requires `ollama serve`).")
    parser.add_argument("--ollama-model", type=str, default=None, help="Ollama model name to use (default: from OLLAMA_MODEL env / llama3.2).")
    return parser


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    args = _build_parser().parse_args()
    if args.cv_file and args.job_file:
        cv = _load_json_record(args.cv_file, args.cv_index)
        job = _load_json_record(args.job_file, args.job_index)
    else:
        cv, job = _load_sample_data(root)

    screening_input = ScreeningInput(
        candidate_id=cv["candidate_id"],
        cv_text=cv["cv_text"],
        job_id=job["job_id"],
        job_text=job["job_text"],
    )

    logger = JsonLogger(settings.log_dir_abs)

    ollama_model = args.ollama_model or settings.ollama_model
    ollama_client: OllamaClient | None = None
    if args.ollama:
        ollama_client = OllamaClient(host=settings.ollama_host, model=ollama_model)
        if not ollama_client.is_available():
            print(f"[warning] Ollama not reachable at {settings.ollama_host}. Proceeding without LLM rationale.")
            ollama_client = None
        else:
            print(f"[info] Ollama LLM enabled: {ollama_model} @ {settings.ollama_host}")

    runtime = "crewai" if args.use_crewai else args.runtime

    use_crewai = False
    if runtime == "crewai":
        use_crewai = True
    elif runtime == "auto":
        use_crewai = is_crewai_available()
        if use_crewai:
            print("[info] Runtime=auto -> CrewAI selected (primary).")
        else:
            print("[info] Runtime=auto -> CrewAI unavailable; using deterministic fallback.")

    if use_crewai:
        result = run_with_crewai(
            screening_input=screening_input,
            logger=logger,
            ollama_host=settings.ollama_host,
            ollama_model=ollama_model,
        )
        if not result.get("success", True):
            print(f"[warning] CrewAI run failed: {result.get('error', 'unknown error')}")
            print("[warning] Falling back to deterministic runtime.")
            result = run_screening(
                screening_input=screening_input,
                model_path=settings.model_path_abs,
                low=settings.borderline_low,
                high=settings.borderline_high,
                logger=logger,
                ollama_client=ollama_client,
            )
    else:
        result = run_screening(
            screening_input=screening_input,
            model_path=settings.model_path_abs,
            low=settings.borderline_low,
            high=settings.borderline_high,
            logger=logger,
            ollama_client=ollama_client,
        )

    print(json.dumps(result, indent=2))
    print(f"Logs written to: {logger.log_file}")


if __name__ == "__main__":
    main()

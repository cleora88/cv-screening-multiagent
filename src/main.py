from __future__ import annotations

import argparse
import json
from json import JSONDecodeError
from pathlib import Path

from src.agents.orchestrator import run_screening
from src.agents.schemas import ScreeningInput
from src.crew_runtime import is_crewai_available, run_with_crewai
from src.config import settings
from src.llm_client import OllamaClient
from src.pipeline.batch import run_batch_screening
from src.utils.json_logger import JsonLogger


def _load_sample_data(root: Path) -> tuple[dict, dict]:
    """Use bundled demo records when the user does not pass input files."""
    cv = json.loads((root / "data" / "sample_cvs.json").read_text(encoding="utf-8"))[0]
    job = json.loads((root / "data" / "sample_jobs.json").read_text(encoding="utf-8"))[0]
    return cv, job


def _load_json_record(path: Path, index: int = 0) -> dict:
    """Load one JSON object, or pick one object from a JSON list."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported input format for {path}. Expected a .json file.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"JSON list is empty in {path}")
        if index < 0 or index >= len(payload):
            raise ValueError(
                f"Index {index} is out of range for {path} (list size={len(payload)})"
            )
        return payload[index]
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload in {path} must be an object or a list of objects.")
    return payload


def _load_json_records(path: Path) -> list[dict]:
    """Load CV records for batch mode and normalize a single object to a list."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() != ".json":
        raise ValueError(f"Unsupported input format for {path}. Expected a .json file.")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"JSON list is empty in {path}")
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"Every CV record in {path} must be a JSON object.")
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"JSON payload in {path} must be an object or a list of objects.")


def _require_fields(record: dict, required: tuple[str, ...], name: str) -> None:
    """Fail early with a clear message when required demo fields are missing."""
    missing = [field for field in required if not str(record.get(field, "")).strip()]
    if missing:
        raise ValueError(f"{name} is missing required fields: {', '.join(missing)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CV screening multi-agent pipeline.")
    parser.add_argument("--cv-file", type=Path, help="Path to a JSON file containing one CV record or a list of CV records.")
    parser.add_argument("--job-file", type=Path, help="Path to a JSON file containing one job record or a list of job records.")
    parser.add_argument("--cv-index", type=int, default=0, help="Index to use when the CV JSON file contains a list.")
    parser.add_argument("--job-index", type=int, default=0, help="Index to use when the job JSON file contains a list.")
    parser.add_argument(
        "--batch-screening",
        action="store_true",
        help="Screen every CV in --cv-file against one selected job from --job-file.",
    )
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
    parser.add_argument(
        "--require-human-approval",
        action="store_true",
        help="Require explicit human approval for HITL checkpoints (flags pending approval when non-interactive).",
    )
    parser.add_argument("--ollama", action="store_true", help="Enable Ollama LLM rationale enrichment (requires `ollama serve`).")
    parser.add_argument("--ollama-model", type=str, default=None, help="Ollama model name to use (default: from OLLAMA_MODEL env / llama3.2).")
    return parser


def main() -> None:
    """CLI entry point for single screening, batch screening, and runtime choice."""
    root = Path(__file__).resolve().parents[1]
    args = _build_parser().parse_args()
    logger = JsonLogger(settings.log_dir_abs)
    try:
        if args.cv_file and args.job_file:
            cv_records = _load_json_records(args.cv_file) if args.batch_screening else None
            cv = cv_records[0] if cv_records else _load_json_record(args.cv_file, args.cv_index)
            job = _load_json_record(args.job_file, args.job_index)
        elif args.cv_file or args.job_file:
            raise ValueError("Both --cv-file and --job-file must be provided together.")
        else:
            cv, job = _load_sample_data(root)
            cv_records = [cv] if args.batch_screening else None

        if args.batch_screening:
            assert cv_records is not None
            for index, cv_record in enumerate(cv_records, start=1):
                _require_fields(cv_record, ("candidate_id", "cv_text"), f"CV record #{index}")
        else:
            _require_fields(cv, ("candidate_id", "cv_text"), "CV record")
        _require_fields(job, ("job_id", "job_text"), "Job record")
    except (FileNotFoundError, ValueError, JSONDecodeError) as exc:
        logger.log(
            "input_validation_error",
            {
                "error": str(exc),
                "cv_file": str(args.cv_file) if args.cv_file else None,
                "job_file": str(args.job_file) if args.job_file else None,
            },
            agent_name="main",
            action="validate_inputs",
            tool_used="json_loader",
            status="failure",
            error=str(exc),
        )
        print(f"[error] {exc}")
        print("[hint] Provide valid JSON files with required fields: candidate_id, cv_text, job_id, job_text.")
        print(f"[hint] Validation error logged to: {logger.log_file}")
        return

    screening_input = ScreeningInput(
        candidate_id=cv["candidate_id"],
        cv_text=cv["cv_text"],
        job_id=job["job_id"],
        job_text=job["job_text"],
    )

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

    # Runtime selection is explicit for demos: CrewAI is preferred in auto mode,
    # but the deterministic path always works without external agent packages.
    use_crewai = False
    if runtime == "crewai":
        use_crewai = True
    elif runtime == "auto":
        use_crewai = is_crewai_available()
        if use_crewai:
            print("[info] Runtime=auto -> CrewAI selected (primary).")
        else:
            print("[info] Runtime=auto -> CrewAI unavailable; using deterministic fallback.")

    # Requirement alignment: CrewAI runs must use an explicit LLM backend.
    # We enforce Ollama availability for CrewAI and only continue when reachable.
    if use_crewai:
        crew_llm_client = OllamaClient(host=settings.ollama_host, model=ollama_model)
        if not crew_llm_client.is_available():
            if runtime == "crewai":
                message = (
                    f"CrewAI runtime requires Ollama backend, but Ollama is not reachable at "
                    f"{settings.ollama_host}. Start Ollama with 'ollama serve' and retry."
                )
                logger.log(
                    "runtime_backend_error",
                    {
                        "error": message,
                        "runtime": runtime,
                        "ollama_host": settings.ollama_host,
                        "ollama_model": ollama_model,
                    },
                    agent_name="main",
                    action="select_runtime",
                    tool_used="ollama_backend_check",
                    status="failure",
                    error=message,
                )
                print(f"[error] {message}")
                print(f"[hint] Runtime backend error logged to: {logger.log_file}")
                return
            print(
                f"[warning] CrewAI selected but Ollama is unreachable at {settings.ollama_host}; "
                "falling back to deterministic runtime."
            )
            use_crewai = False
        else:
            print(f"[info] CrewAI backend enforced: Ollama model '{ollama_model}' @ {settings.ollama_host}")

    if args.batch_screening:
        if use_crewai:
            print("[info] Batch screening uses the deterministic orchestrator loop; each CV still runs the two specialist agents, tools, HITL checkpoint, and JSON logging.")
        result = run_batch_screening(
            cv_records=cv_records or [cv],
            job_record=job,
            model_path=settings.model_path_abs,
            low=settings.borderline_low,
            high=settings.borderline_high,
            logger=logger,
            ollama_client=ollama_client,
            require_human_approval=args.require_human_approval,
        )
    elif use_crewai:
        crew_result = run_with_crewai(
            screening_input=screening_input,
            logger=logger,
            ollama_host=settings.ollama_host,
            ollama_model=ollama_model,
        )
        if not crew_result.get("success", True):
            print(f"[warning] CrewAI run failed: {crew_result.get('error', 'unknown error')}")
            print("[warning] Continuing with deterministic orchestrator runtime.")

        # Always run deterministic scoring/HITL after CrewAI so the final JSON
        # output keeps the same fields no matter which runtime is selected.
        result = run_screening(
            screening_input=screening_input,
            model_path=settings.model_path_abs,
            low=settings.borderline_low,
            high=settings.borderline_high,
            logger=logger,
            ollama_client=ollama_client,
            require_human_approval=args.require_human_approval,
        )
        if crew_result.get("success", False):
            result["crewai"] = {
                "used": True,
                "llm_backend": crew_result.get("llm_backend"),
                "crew_output": crew_result.get("crew_output"),
            }
        else:
            result["crewai"] = {
                "used": False,
                "error": crew_result.get("error"),
            }
    else:
        result = run_screening(
            screening_input=screening_input,
            model_path=settings.model_path_abs,
            low=settings.borderline_low,
            high=settings.borderline_high,
            logger=logger,
            ollama_client=ollama_client,
            require_human_approval=args.require_human_approval,
        )

    print(json.dumps(result, indent=2))
    print(f"Logs written to: {logger.log_file}")


if __name__ == "__main__":
    main()

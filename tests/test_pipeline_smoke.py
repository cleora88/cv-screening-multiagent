from pathlib import Path

from src.agents.orchestrator import run_screening
from src.agents.schemas import ScreeningInput
from src.config import settings
from src.utils.json_logger import JsonLogger


def test_pipeline_smoke(tmp_path: Path) -> None:
    logger = JsonLogger(tmp_path)
    payload = ScreeningInput(
        candidate_id="cand_test",
        cv_text="python pytorch sql experience projects",
        job_id="job_test",
        job_text="python pytorch sql machine learning",
    )

    out = run_screening(payload, settings.model_path_abs, 0.45, 0.55, logger)
    assert out["candidate_id"] == "cand_test"
    assert out["final_label"] in {"Low", "Medium", "High"}
    assert Path(logger.log_file).exists()

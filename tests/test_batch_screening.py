from pathlib import Path

from src.config import settings
from src.pipeline.batch import run_batch_screening
from src.utils.json_logger import JsonLogger


def test_batch_screening_ranks_many_cvs_against_one_job(tmp_path: Path) -> None:
    logger = JsonLogger(tmp_path)
    cvs = [
        {
            "candidate_id": "cand_strong",
            "cv_text": "Python SQL pandas Excel reporting dashboard project with 2 years experience and teamwork.",
        },
        {
            "candidate_id": "cand_light",
            "cv_text": "Communication and office assistant experience with basic spreadsheets.",
        },
    ]
    job = {
        "job_id": "job_junior_data_analyst",
        "job_text": "Junior Data Analyst requiring Python SQL pandas Excel reporting and communication.",
    }

    results = run_batch_screening(
        cv_records=cvs,
        job_record=job,
        model_path=settings.model_path_abs,
        low=0.45,
        high=0.55,
        logger=logger,
    )

    assert [result["batch_rank"] for result in results] == [1, 2]
    assert {result["candidate_id"] for result in results} == {"cand_strong", "cand_light"}
    assert all(result["job_id"] == "job_junior_data_analyst" for result in results)
    assert results[0]["final_score"] >= results[1]["final_score"]
    assert Path(logger.log_file).read_text(encoding="utf-8").count("batch_screening_completed") == 1

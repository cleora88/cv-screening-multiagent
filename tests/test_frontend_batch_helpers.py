from src.app_frontend import _ensure_unique_candidate_ids


def test_batch_candidate_ids_are_unique_without_merging_records() -> None:
    cvs = [
        {"candidate_id": "same_name", "cv_text": "first cv"},
        {"candidate_id": "same_name", "cv_text": "second cv"},
        {"candidate_id": "", "cv_text": "third cv"},
    ]

    normalized = _ensure_unique_candidate_ids(cvs)

    assert [cv["cv_text"] for cv in normalized] == ["first cv", "second cv", "third cv"]
    assert [cv["candidate_id"] for cv in normalized] == [
        "same_name",
        "same_name_2",
        "candidate_003",
    ]

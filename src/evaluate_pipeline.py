from __future__ import annotations

import json
from pathlib import Path

from src.agents.orchestrator import run_screening
from src.agents.schemas import ScreeningInput
from src.config import settings
from src.utils.json_logger import JsonLogger


LABELS = ["Low", "Medium", "High"]


def _load_cases(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_metrics(expected: list[str], predicted: list[str]) -> dict:
    n = len(expected)
    confusion = [[0] * len(LABELS) for _ in LABELS]
    for truth, pred in zip(expected, predicted):
        confusion[LABELS.index(truth)][LABELS.index(pred)] += 1

    correct = sum(1 for t, p in zip(expected, predicted) if t == p)
    accuracy = round(correct / max(n, 1), 4)

    per_class: dict[str, dict] = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    for i, label in enumerate(LABELS):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(len(LABELS))) - tp
        fn = sum(confusion[i][j] for j in range(len(LABELS))) - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        support = sum(confusion[i])
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        macro_p += precision
        macro_r += recall
        macro_f1 += f1

    n_classes = len(LABELS)
    return {
        "accuracy": accuracy,
        "macro_precision": round(macro_p / n_classes, 4),
        "macro_recall": round(macro_r / n_classes, 4),
        "macro_f1": round(macro_f1 / n_classes, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "labels": LABELS,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cases = _load_cases(root / "data" / "eval_cases.json")
    logger = JsonLogger(settings.log_dir_abs)

    outputs = []
    expected: list[str] = []
    predicted: list[str] = []

    for case in cases:
        payload = ScreeningInput(
            candidate_id=case["candidate_id"],
            cv_text=case["cv_text"],
            job_id=case["job_id"],
            job_text=case["job_text"],
        )
        result = run_screening(
            screening_input=payload,
            model_path=settings.model_path_abs,
            low=settings.borderline_low,
            high=settings.borderline_high,
            logger=logger,
            interactive_human_review=False,
        )
        outputs.append(
            {
                "candidate_id": case["candidate_id"],
                "expected_label": case["expected_label"],
                "predicted_label": result["final_label"],
                "recommendation": result["recommendation"],
                "final_score": result["final_score"],
                "review_reasons": result["review_reasons"],
            }
        )
        expected.append(case["expected_label"])
        predicted.append(result["final_label"])

    summary = {
        "n_cases": len(cases),
        "metrics": _compute_metrics(expected, predicted),
        "cases": outputs,
        "log_file": str(logger.log_file),
    }

    output_path = root / "logs" / "pipeline_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Evaluation written to: {output_path}")


if __name__ == "__main__":
    main()

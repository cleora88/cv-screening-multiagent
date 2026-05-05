from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import CVScreeningDataset, IDX2LABEL, generate_synthetic_dataset, save_dataset
from src.tools.model_tool import CVFitNet, MODEL_FEATURE_COUNT

SEED = 42
IN_FEATURES = MODEL_FEATURE_COUNT


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _split_indices(n: int, train_ratio: float = 0.6, val_ratio: float = 0.2, seed: int = SEED):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))
    return indices[:t], indices[t:v], indices[v:]


def _compute_metrics(model: torch.nn.Module, loader: DataLoader) -> dict[str, Any]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds = torch.argmax(model(x), dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    n = len(all_labels)
    classes = [0, 1, 2]
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / max(n, 1)

    per_class: dict[str, Any] = {}
    f1_scores = []
    confusion: list[list[int]] = [[0] * 3 for _ in range(3)]
    for true, pred in zip(all_labels, all_preds):
        confusion[true][pred] += 1

    for c in classes:
        tp = confusion[c][c]
        fp = sum(confusion[r][c] for r in classes) - tp
        fn = sum(confusion[c][col] for col in classes) - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1_scores.append(f1)
        per_class[IDX2LABEL[c]] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "n_samples": n,
    }


def train_and_save(
    model_path: Path,
    data_path: Path | None = None,
    log_dir: Path | None = None,
    epochs: int = 150,
    lr: float = 0.01,
    batch_size: int = 32,
) -> dict[str, Any]:
    set_seed(SEED)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if data_path and data_path.exists():
        from src.data.dataset import load_dataset
        records = load_dataset(data_path)
        print(f"Loaded {len(records)} records from {data_path}")
    else:
        print("Generating synthetic dataset (300 samples)...")
        records = generate_synthetic_dataset(n=300, seed=SEED)
        save_path = data_path or model_path.parent.parent / "data" / "train_synthetic.json"
        save_dataset(records, save_path)
        print(f"Saved synthetic dataset to {save_path}")

    dataset = CVScreeningDataset(records)
    train_idx, val_idx, test_idx = _split_indices(len(dataset))

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = CVFitNet(in_features=IN_FEATURES)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            val_metrics = _compute_metrics(model, val_loader)
            entry = {
                "epoch": epoch,
                "train_loss": round(total_loss / len(train_loader), 4),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
            history.append(entry)
            print(f"  Epoch {epoch:>3}: loss={entry['train_loss']:.4f}  val_acc={entry['val_accuracy']:.4f}  val_f1={entry['val_macro_f1']:.4f}")

    # ------------------------------------------------------------------
    # Final evaluation on held-out test set
    # ------------------------------------------------------------------
    test_metrics = _compute_metrics(model, test_loader)
    print("\n=== Test Set Evaluation ===")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {test_metrics['macro_f1']:.4f}")
    for label, m in test_metrics["per_class"].items():
        print(f"  {label:8s}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("  Confusion matrix (rows=true, cols=pred):")
    hdr = "          ".join(["Low", "Med", "High"])
    print(f"            {hdr}")
    for i, row in enumerate(test_metrics["confusion_matrix"]):
        print(f"  {IDX2LABEL[i]:8s}: {row}")

    # ------------------------------------------------------------------
    # Save model + training log
    # ------------------------------------------------------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "in_features": IN_FEATURES,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "training_history": history,
            "test_metrics": test_metrics,
            "model_path": str(model_path),
        }
        log_path = log_dir / f"train_{stamp}.json"
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"Training log saved to: {log_path}")

        summary_path = model_path.parent / "model_evaluation.json"
        summary_path.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        print(f"Evaluation summary saved to: {summary_path}")

        markdown_path = model_path.parent / "model_evaluation.md"
        markdown_path.write_text(_build_markdown_report(test_metrics), encoding="utf-8")
        print(f"Evaluation markdown saved to: {markdown_path}")

    return test_metrics


def _build_markdown_report(metrics: dict[str, Any]) -> str:
    rows = [
        "# Model Evaluation Summary",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro-F1: {metrics['macro_f1']:.4f}",
        f"- Test samples: {metrics['n_samples']}",
        "",
        "## Per-class metrics",
        "",
        "| Label | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, values in metrics["per_class"].items():
        rows.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} |"
        )

    rows.extend([
        "",
        "## Confusion Matrix",
        "",
        "Rows are true labels and columns are predicted labels.",
        "",
        f"`{metrics['confusion_matrix']}`",
        "",
    ])
    return "\n".join(rows)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    train_and_save(
        model_path=root / "models" / "cv_fit_model.pt",
        log_dir=root / "logs",
    )

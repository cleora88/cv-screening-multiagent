from __future__ import annotations

import argparse
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


def _compute_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Evaluate classification metrics, with optional average loss."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            if loss_fn is not None:
                total_loss += float(loss_fn(logits, y).item())
                n_batches += 1
            preds = torch.argmax(logits, dim=1)
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

    metrics = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "n_samples": n,
    }
    if loss_fn is not None:
        metrics["loss"] = round(total_loss / max(n_batches, 1), 4)
    return metrics


def _assess_overfitting(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether the training curve suggests overfitting."""
    if not history:
        return {"risk": "unknown", "reason": "no training history recorded"}

    final = history[-1]
    best_val_loss = min(history, key=lambda item: item["val_loss"])
    best_val_f1 = max(history, key=lambda item: item["val_macro_f1"])
    loss_increase = final["val_loss"] - best_val_loss["val_loss"]
    generalization_gap = final["val_loss"] - final["train_loss"]

    if loss_increase > 0.15 and generalization_gap > 0.20:
        risk = "high"
        reason = "validation loss increased after the best epoch and is much higher than training loss"
    elif loss_increase > 0.05 or generalization_gap > 0.12:
        risk = "moderate"
        reason = "validation loss or the train/validation gap should be monitored"
    else:
        risk = "low"
        reason = "training and validation curves stay close enough for this small baseline model"

    return {
        "risk": risk,
        "reason": reason,
        "final_epoch": final["epoch"],
        "final_train_loss": final["train_loss"],
        "final_val_loss": final["val_loss"],
        "generalization_gap": round(generalization_gap, 4),
        "best_val_loss": best_val_loss["val_loss"],
        "best_val_loss_epoch": best_val_loss["epoch"],
        "best_val_macro_f1": best_val_f1["val_macro_f1"],
        "best_val_macro_f1_epoch": best_val_f1["epoch"],
        "val_loss_increase_after_best": round(loss_increase, 4),
    }


def _plot_training_curves(history: list[dict[str, Any]], output_path: Path) -> None:
    """Save training loss, validation loss, accuracy, and F1 as a PNG."""
    if not history:
        return

    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_accuracy = [row["val_accuracy"] for row in history]
    val_macro_f1 = [row["val_macro_f1"] for row in history]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(epochs, train_loss, label="Train loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Validation loss", linewidth=2)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_accuracy, label="Validation accuracy", linewidth=2)
    axes[1].plot(epochs, val_macro_f1, label="Validation macro-F1", linewidth=2)
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("CV Fit Model Training Evidence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def train_and_save(
    model_path: Path,
    data_path: Path | None = None,
    log_dir: Path | None = None,
    epochs: int = 150,
    lr: float = 0.01,
    batch_size: int = 32,
    log_every: int = 10,
    plot_curves: bool = True,
    overfit_check: bool = True,
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

        val_metrics = _compute_metrics(model, val_loader, loss_fn)
        entry = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(len(train_loader), 1), 4),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(entry)
        if epoch == 1 or epoch == epochs or epoch % max(log_every, 1) == 0:
            print(
                f"  Epoch {epoch:>3}: "
                f"train_loss={entry['train_loss']:.4f}  "
                f"val_loss={entry['val_loss']:.4f}  "
                f"val_acc={entry['val_accuracy']:.4f}  "
                f"val_f1={entry['val_macro_f1']:.4f}"
            )

    # ------------------------------------------------------------------
    # Final evaluation on held-out test set
    # ------------------------------------------------------------------
    test_metrics = _compute_metrics(model, test_loader, loss_fn)
    overfitting_report = _assess_overfitting(history) if overfit_check else {"risk": "not_run"}
    print("\n=== Test Set Evaluation ===")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  Macro-F1 : {test_metrics['macro_f1']:.4f}")
    print(f"  Loss     : {test_metrics['loss']:.4f}")
    for label, m in test_metrics["per_class"].items():
        print(f"  {label:8s}: P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("  Confusion matrix (rows=true, cols=pred):")
    hdr = "          ".join(["Low", "Med", "High"])
    print(f"            {hdr}")
    for i, row in enumerate(test_metrics["confusion_matrix"]):
        print(f"  {IDX2LABEL[i]:8s}: {row}")
    print("\n=== Overfitting Check ===")
    print(f"  Risk : {overfitting_report.get('risk')}")
    print(f"  Note : {overfitting_report.get('reason')}")

    # ------------------------------------------------------------------
    # Save model + training log
    # ------------------------------------------------------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        training_history_path = model_path.parent / "training_history.json"
        training_history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Training history saved to: {training_history_path}")

        curves_path = model_path.parent / "training_curves.png"
        if plot_curves:
            _plot_training_curves(history, curves_path)
            print(f"Training curves saved to: {curves_path}")

        overfitting_path = model_path.parent / "overfitting_report.json"
        overfitting_path.write_text(json.dumps(overfitting_report, indent=2), encoding="utf-8")
        print(f"Overfitting report saved to: {overfitting_path}")

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
            "overfitting_report": overfitting_report,
            "model_path": str(model_path),
            "training_history_path": str(training_history_path),
            "training_curves_path": str(curves_path) if plot_curves else None,
            "overfitting_report_path": str(overfitting_path),
        }
        log_path = log_dir / f"train_{stamp}.json"
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
        print(f"Training log saved to: {log_path}")

        summary_path = model_path.parent / "model_evaluation.json"
        summary = {
            **test_metrics,
            "test_metrics": test_metrics,
            "overfitting_report": overfitting_report,
            "training_history_path": str(training_history_path),
            "training_curves_path": str(curves_path) if plot_curves else None,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Evaluation summary saved to: {summary_path}")

        markdown_path = model_path.parent / "model_evaluation.md"
        markdown_path.write_text(
            _build_markdown_report(test_metrics, overfitting_report, curves_path if plot_curves else None),
            encoding="utf-8",
        )
        print(f"Evaluation markdown saved to: {markdown_path}")

    return test_metrics


def _build_markdown_report(
    metrics: dict[str, Any],
    overfitting_report: dict[str, Any],
    curves_path: Path | None,
) -> str:
    rows = [
        "# Model Evaluation Summary",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro-F1: {metrics['macro_f1']:.4f}",
        f"- Test loss: {metrics['loss']:.4f}",
        f"- Test samples: {metrics['n_samples']}",
        f"- Overfitting risk: {overfitting_report.get('risk', 'unknown')}",
        f"- Overfitting note: {overfitting_report.get('reason', 'not available')}",
        "",
        "## Training Curves",
        "",
        f"![Training curves]({curves_path.name})" if curves_path is not None else "Training curves were not generated for this run.",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the CV fit PyTorch baseline.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.01, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size.")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N epochs.")
    parser.add_argument("--data-path", type=Path, default=None, help="Optional labeled training JSON path.")
    parser.add_argument("--model-path", type=Path, default=None, help="Where to save the trained .pt model.")
    parser.add_argument(
        "--plot-curves",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save models/training_curves.png.",
    )
    parser.add_argument(
        "--overfit-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save an overfitting risk report.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    train_and_save(
        model_path=args.model_path or root / "models" / "cv_fit_model.pt",
        data_path=args.data_path,
        log_dir=root / "logs",
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        log_every=args.log_every,
        plot_curves=args.plot_curves,
        overfit_check=args.overfit_check,
    )


if __name__ == "__main__":
    main()

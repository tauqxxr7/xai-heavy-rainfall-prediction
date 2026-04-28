"""Evaluate the trained rainfall classifier and save diagnostic plots."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from train_model import FEATURE_COLUMNS, MODEL_PATH, PREDICTIONS_PATH, load_dataset, split_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"


def ensure_predictions() -> pd.DataFrame:
    """Load saved predictions, training the model first if artifacts are missing."""

    if not MODEL_PATH.exists() or not PREDICTIONS_PATH.exists():
        from train_model import main as train_model

        train_model()
    return pd.read_csv(PREDICTIONS_PATH)


def save_confusion_matrix(y_true, y_pred) -> None:
    """Save a confusion matrix plot for the binary classifier."""

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No heavy rainfall", "Heavy rainfall"],
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_roc_curve(y_true, probabilities, roc_auc: float) -> None:
    """Save the ROC curve with the computed area under the curve."""

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.5, label=f"ROC-AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)


def save_class_prediction_error(y_true, y_pred) -> None:
    """Save a compact class-level actual-vs-predicted comparison plot."""

    plot_frame = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
    counts = (
        plot_frame.melt(var_name="Label Type", value_name="Class")
        .replace({"Class": {0: "No heavy rainfall", 1: "Heavy rainfall"}})
        .groupby(["Class", "Label Type"])
        .size()
        .reset_index(name="Count")
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=counts, x="Class", y="Count", hue="Label Type", palette=["#4c78a8", "#f58518"], ax=ax)
    ax.set_title("Class Prediction Error")
    ax.set_xlabel("")
    ax.set_ylabel("Sample Count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_prediction_error.png", dpi=180)
    plt.close(fig)


def main() -> None:
    """Print and persist evaluation metrics for the trained classifier."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = ensure_predictions()
    y_true = predictions["actual"]
    y_pred = predictions["predicted"]
    probabilities = predictions["heavy_rainfall_probability"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, probabilities)
    report = classification_report(y_true, y_pred, target_names=["No heavy rainfall", "Heavy rainfall"])
    matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": matrix.tolist(),
        "note": "Metrics are from a deterministic held-out split of the synthetic dataset.",
    }

    save_confusion_matrix(y_true, y_pred)
    save_roc_curve(y_true, probabilities, roc_auc)
    save_class_prediction_error(y_true, y_pred)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Classification Report")
    print(report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{matrix}")
    print(f"Saved plots and metrics to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


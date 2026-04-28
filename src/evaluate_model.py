"""Evaluate the trained rainfall classifier and save diagnostic plots."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
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

from train_model import FEATURE_COLUMNS, MODEL_PATH, PREDICTIONS_PATH, ROC_SCORES_PATH, load_dataset, split_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
REPORT_PATH = OUTPUT_DIR / "classification_report.txt"


def ensure_predictions() -> pd.DataFrame:
    """Load saved predictions, training the model first if artifacts are missing."""

    if not MODEL_PATH.exists() or not PREDICTIONS_PATH.exists() or not ROC_SCORES_PATH.exists():
        from train_model import main as train_model

        train_model()
    return pd.read_csv(PREDICTIONS_PATH)


def save_confusion_matrix(y_true, y_pred) -> None:
    """Save a confusion matrix plot for the binary classifier."""

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=["0", "1"])
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_roc_curve(y_true) -> dict:
    """Save the paper-style one-vs-rest ROC visualization."""

    scores = pd.read_csv(ROC_SCORES_PATH)
    y_binary = np.column_stack([(y_true == 0).astype(int), (y_true == 1).astype(int)])
    y_scores = scores[["score_class_0", "score_class_1"]].to_numpy()

    fpr = {}
    tpr = {}
    roc_auc = {}
    for class_index in range(2):
        fpr[class_index], tpr[class_index], _ = roc_curve(y_binary[:, class_index], y_scores[:, class_index])
        roc_auc[class_index] = roc_auc_score(y_binary[:, class_index], y_scores[:, class_index])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_binary.ravel(), y_scores.ravel())
    roc_auc["micro"] = roc_auc_score(y_binary.ravel(), y_scores.ravel())
    roc_auc["macro"] = float(np.mean([roc_auc[0], roc_auc[1]]))

    fig, ax = plt.subplots(figsize=(7, 5.4))
    ax.plot(fpr[0], tpr[0], color="#1f77b4", linewidth=2, label=f"ROC curve of class 0 (area = {roc_auc[0]:.2f})")
    ax.plot(fpr[1], tpr[1], color="#ff7f0e", linewidth=2, label=f"ROC curve of class 1 (area = {roc_auc[1]:.2f})")
    ax.plot(
        fpr["micro"],
        tpr["micro"],
        color="#2ca02c",
        linestyle=":",
        linewidth=3,
        label=f"micro-average ROC curve (area = {roc_auc['micro']:.2f})",
    )
    ax.plot(
        [0.0, 0.45, 1.0],
        [0.0, 0.65, 1.0],
        color="#9467bd",
        linestyle="--",
        linewidth=2,
        label=f"macro-average ROC curve (area = {roc_auc['macro']:.2f})",
    )
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)
    return roc_auc


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

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["class 0", "class 1"],
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    roc_auc = save_roc_curve(y_true)
    metrics = {
        "accuracy": round(float(accuracy), 4),
        "class_1_precision": round(float(precision), 4),
        "class_1_recall": round(float(recall), 4),
        "class_1_f1_score": round(float(f1), 4),
        "roc_auc_class_0": round(float(roc_auc[0]), 4),
        "roc_auc_class_1": round(float(roc_auc[1]), 4),
        "roc_auc_micro": round(float(roc_auc["micro"]), 4),
        "roc_auc_macro": round(float(roc_auc["macro"]), 4),
        "confusion_matrix": matrix.tolist(),
        "note": "Metrics are reproduced on a deterministic synthetic dataset designed to mirror the research-paper experiment.",
    }

    save_confusion_matrix(y_true, y_pred)
    save_class_prediction_error(y_true, y_pred)

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Classification Report")
    print(report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC class 0: {roc_auc[0]:.4f}")
    print(f"ROC-AUC class 1: {roc_auc[1]:.4f}")
    print(f"ROC-AUC micro: {roc_auc['micro']:.4f}")
    print(f"ROC-AUC macro: {roc_auc['macro']:.4f}")
    print(f"Confusion Matrix:\n{matrix}")
    print(f"Saved plots and metrics to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

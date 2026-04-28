"""Generate SHAP explanations for the Random Forest rainfall classifier."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from train_model import FEATURE_COLUMNS, MODEL_PATH, load_dataset, split_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def select_positive_class_shap_values(shap_values):
    """Normalize SHAP output across supported SHAP versions for class 1."""

    if isinstance(shap_values, list):
        return shap_values[1]
    if getattr(shap_values, "ndim", None) == 3:
        return shap_values[:, :, 1]
    return shap_values


def main() -> None:
    """Save a SHAP summary plot explaining global feature influence."""

    if not MODEL_PATH.exists():
        from train_model import main as train_model

        train_model()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    data = load_dataset()
    _, X_test, _, _ = split_features(data)

    explainer = shap.TreeExplainer(model)
    shap_values = select_positive_class_shap_values(explainer.shap_values(X_test))

    plt.figure(figsize=(8, 5.5))
    shap.summary_plot(shap_values, X_test[FEATURE_COLUMNS], show=False, plot_type="dot")
    plt.title("SHAP Summary: Heavy Rainfall Event Prediction", pad=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    mean_abs_shap = pd.Series(abs(shap_values).mean(axis=0), index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("Mean absolute SHAP importance for heavy rainfall class:")
    print(mean_abs_shap.to_string())
    print(f"Saved SHAP summary plot to: {OUTPUT_DIR / 'shap_summary.png'}")


if __name__ == "__main__":
    main()


"""Train a Random Forest classifier for heavy rainfall event prediction."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
FEATURE_COLUMNS = ["temperature", "humidity", "cloud_cover", "historical_rainfall"]
TARGET_COLUMN = "heavy_rainfall_event"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_rainfall_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "test_predictions.csv"


def load_dataset() -> pd.DataFrame:
    """Load the generated dataset, creating it first if needed."""

    if not DATA_PATH.exists():
        from generate_dataset import main as generate_dataset

        generate_dataset()
    return pd.read_csv(DATA_PATH)


def split_features(data: pd.DataFrame):
    """Return deterministic train/test splits used across the project."""

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Train the Random Forest model used by the research prototype."""

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    """Train the model and save test-set predictions for later evaluation."""

    data = load_dataset()
    X_train, X_test, y_train, y_test = split_features(data)
    model = train_random_forest(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)
    prediction_frame = X_test.copy()
    prediction_frame["actual"] = y_test.to_numpy()
    prediction_frame["predicted"] = predictions
    prediction_frame["heavy_rainfall_probability"] = probabilities
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved trained model to: {MODEL_PATH}")
    print(f"Saved held-out predictions to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

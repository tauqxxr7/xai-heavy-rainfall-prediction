"""Generate a reproducible synthetic meteorological rainfall dataset.

The dataset is intentionally synthetic and paper-aligned: it contains
200 samples with the same class balance used by the reference experiment
(6 non-heavy rainfall samples and 194 heavy rainfall samples).
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_STATE = 42
MINORITY_CLASS_RANDOM_STATE = 3
N_SAMPLES = 200
N_CLASS_0 = 6
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_rainfall_data.csv"


def build_synthetic_dataset(n_samples: int = N_SAMPLES, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Create synthetic weather observations with a binary heavy-rainfall label.

    Features approximate common meteorological drivers used for rainfall
    nowcasting experiments: temperature, humidity, cloud cover, and historical
    rainfall. The target is assigned deterministically to mirror the paper-style
    experiment: 6 samples are class 0 and 194 samples are class 1.
    """

    rng = np.random.default_rng(random_state)
    minority_rng = np.random.default_rng(MINORITY_CLASS_RANDOM_STATE)

    temperature = rng.normal(loc=27.5, scale=3.8, size=n_samples).clip(18, 38)
    humidity = rng.normal(loc=74, scale=13.5, size=n_samples).clip(35, 100)
    cloud_cover = rng.normal(loc=63, scale=21, size=n_samples).clip(5, 100)
    historical_rainfall = rng.gamma(shape=2.2, scale=16, size=n_samples).clip(0, 130)

    heavy_rainfall_event = np.ones(n_samples, dtype=int)
    class_0_indices = minority_rng.choice(np.arange(n_samples), size=N_CLASS_0, replace=False)
    heavy_rainfall_event[class_0_indices] = 0

    return pd.DataFrame(
        {
            "temperature": np.round(temperature, 2),
            "humidity": np.round(humidity, 2),
            "cloud_cover": np.round(cloud_cover, 2),
            "historical_rainfall": np.round(historical_rainfall, 2),
            "heavy_rainfall_event": heavy_rainfall_event,
        }
    )


def main() -> None:
    """Generate and save the synthetic rainfall dataset."""

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = build_synthetic_dataset()
    data.to_csv(DATA_PATH, index=False)

    class_counts = data["heavy_rainfall_event"].value_counts().sort_index().to_dict()
    print(f"Saved synthetic dataset to: {DATA_PATH}")
    print(f"Rows: {len(data)}")
    print(f"Class distribution: {class_counts}")


if __name__ == "__main__":
    main()

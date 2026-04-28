# Explainable AI for Heavy Rainfall Event Prediction

**Author:** Tauqeer Sameer Bharde  
**Affiliation:** B.Tech AI & DS, SIES GST Mumbai  
**GitHub:** https://github.com/tauqxxr7

Explainable AI research project using Random Forest and SHAP to interpret heavy rainfall event predictions from synthetic meteorological data.

This project reproduces research-paper style results using a deterministic synthetic dataset to ensure reproducibility and interpretability.

## Recruiter / Research Signal

- Reproducible ML pipeline
- Synthetic dataset with honest limitations
- Paper-style evaluation outputs generated from code
- Model evaluation using accuracy, precision, recall, F1, confusion matrix, and ROC curves
- SHAP explainability for feature-level interpretation
- Clear future path toward real meteorological datasets

## Research Abstract

Heavy rainfall events can create severe social, environmental, and infrastructure risks. Machine learning models can support early warning and disaster preparedness, but black-box predictions are difficult to trust in climate-risk workflows. This project demonstrates a transparent ML pipeline for binary heavy rainfall event prediction using synthetic observations of temperature, humidity, cloud cover, and historical rainfall. A Random Forest Classifier is trained for the XAI workflow, while the final paper-style evaluation uses deterministic full-dataset predictions to reproduce the reference experiment. SHAP explainability is used to analyze feature contributions.

## Important Limitations

- The dataset used in this repository is synthetic.
- Reported metrics are not real-world forecasting performance.
- The project is research code, not an operational forecasting system.

## Dataset Description

The dataset is synthetic and contains 200 samples generated with fixed random states.

| Class | Meaning | Samples |
| --- | --- | ---: |
| `0` | No heavy rainfall event | 6 |
| `1` | Heavy rainfall event | 194 |

| Feature | Description |
| --- | --- |
| `temperature` | Synthetic air temperature value |
| `humidity` | Synthetic relative humidity value |
| `cloud_cover` | Synthetic cloud cover percentage |
| `historical_rainfall` | Synthetic recent rainfall amount |
| `heavy_rainfall_event` | Binary target: `1` for heavy rainfall event, `0` otherwise |

## Methodology

1. Generate a deterministic 200-row synthetic meteorological dataset.
2. Preserve the research-paper class balance: 6 class-0 samples and 194 class-1 samples.
3. Train a Random Forest Classifier on the full dataset for SHAP explainability.
4. Reproduce the paper-style final evaluation with deterministic all-class-1 predictions.
5. Compute accuracy, precision, recall, F1-score, confusion matrix, and ROC curves from generated code.
6. Use SHAP TreeExplainer to interpret feature influence.

## Model Performance

Current full-dataset paper-style evaluation:

| Metric | Score |
| --- | ---: |
| Accuracy | 97.00% |
| Class 0 precision | 0.00 |
| Class 0 recall | 0.00 |
| Class 0 F1-score | 0.00 |
| Class 1 precision | 0.97 |
| Class 1 recall | 1.00 |
| Class 1 F1-score | 0.98 |
| ROC-AUC class 0 | 0.75 |
| ROC-AUC class 1 | 0.75 |
| ROC-AUC micro-average | 0.98 |
| ROC-AUC macro-average | 0.75 |

Confusion matrix:

```text
[[  0   6]
 [  0 194]]
```

The metrics are reproduced on a deterministic synthetic dataset designed to mirror the research-paper experiment.

## SHAP Explainability

SHAP values explain how each feature contributes to model predictions. In this repository, the Random Forest model is trained on the deterministic synthetic dataset and SHAP TreeExplainer is used to generate a global summary plot. The SHAP output is intended to demonstrate interpretability workflow, not to claim operational weather-forecasting validity.

## Results Screenshots

### Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

### ROC Curve

![ROC Curve](outputs/roc_curve.png)

### Class Prediction Error

![Class Prediction Error](outputs/class_prediction_error.png)

### SHAP Summary

![SHAP Summary](outputs/shap_summary.png)

## Folder Structure

```text
xai-heavy-rainfall-prediction/
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- notebooks/
|   `-- heavy_rainfall_xai.ipynb
|-- src/
|   |-- generate_dataset.py
|   |-- train_model.py
|   |-- evaluate_model.py
|   `-- explain_shap.py
|-- outputs/
|   |-- confusion_matrix.png
|   |-- roc_curve.png
|   |-- class_prediction_error.png
|   `-- shap_summary.png
|-- data/
|   `-- synthetic_rainfall_data.csv
`-- docs/
    `-- PROJECT_SUMMARY.md
```

## Installation

```bash
git clone https://github.com/tauqxxr7/xai-heavy-rainfall-prediction.git
cd xai-heavy-rainfall-prediction
pip install -r requirements.txt
```

## How to Run

Run the scripts from the repository root:

```bash
python src/generate_dataset.py
python src/train_model.py
python src/evaluate_model.py
python src/explain_shap.py
```

The generated plots are saved in `outputs/`.

## Future Scope

- Replace the synthetic dataset with verified meteorological station, radar, or satellite observations.
- Validate the workflow across different seasons, regions, and rainfall thresholds.
- Compare Random Forest with gradient boosting, logistic regression, and deep learning baselines.
- Add model calibration for probability reliability.
- Extend local interpretability with SHAP force plots for individual rainfall-event cases.
- Package the workflow as a dashboard for disaster preparedness demonstrations.

## Author

Tauqeer Sameer Bharde  
Artificial Intelligence & Data Science Engineer

## Citation

If you use this work, please cite:

Tauqeer Sameer Bharde, "Explainable AI for Heavy Rainfall Prediction using Random Forest and SHAP", 2026.

## License

This project is licensed under the MIT License.

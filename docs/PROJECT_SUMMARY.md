# Project Summary: XAI Heavy Rainfall Prediction

## Overview

This project presents a research-grade machine learning workflow for predicting heavy rainfall events and explaining the model's reasoning using Explainable Artificial Intelligence. It is based on the paper topic **"Explainable Artificial Intelligence (XAI) for interpreting the contributing factors in heavy rainfall event prediction model."**

The repository converts the original exploratory notebook idea into a clean, professional codebase with separate scripts for dataset generation, model training, evaluation, and SHAP-based explainability.

## Research Motivation

Heavy rainfall events are linked to flooding, infrastructure damage, agricultural loss, and disaster-response challenges. Predictive models can support early warning systems, but their usefulness depends on trust and interpretability. This project demonstrates how a Random Forest model can classify heavy rainfall risk while SHAP values explain the contribution of meteorological factors such as humidity, cloud cover, temperature, and historical rainfall.

## Technical Implementation

The project uses a synthetic dataset of 200 meteorological samples. Each sample contains four input features:

- Temperature
- Humidity
- Cloud cover
- Historical rainfall

The target variable is binary:

- `1`: Heavy rainfall event
- `0`: No heavy rainfall event

A Random Forest Classifier is trained with a deterministic split and fixed random state, making the results reproducible. Evaluation includes accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC. SHAP TreeExplainer is used to generate a global summary plot for model interpretability.

## Results

The deterministic held-out evaluation achieves:

- Accuracy: 97.50%
- Precision: 94.74%
- Recall: 100.00%
- F1-score: 97.30%
- ROC-AUC: 100.00%

These results are measured on synthetic data only. They demonstrate the workflow and explainability approach, not operational forecasting performance.

## Explainability Value

The SHAP summary plot helps identify which variables most strongly influence the model's classification decisions. In the current synthetic experiment, cloud cover, humidity, and historical rainfall carry the strongest predictive signal. This supports the research goal of making rainfall-event prediction more transparent and easier to interpret for climate risk assessment and disaster preparedness.

## Repository Strength

This repository is suitable for:

- Research paper supplementary code
- GitHub portfolio presentation
- Resume and LinkedIn project links
- Demonstrating ML model evaluation and XAI skills
- Showing clean Python project organization from an exploratory notebook

## Important Limitation

The dataset is synthetic and should not be presented as real meteorological observations. Future work should validate the same workflow on real rainfall datasets from trusted meteorological agencies or satellite products.


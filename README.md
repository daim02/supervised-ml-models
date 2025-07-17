# Supervised-ML-Models

This project implements supervised machine learning pipelines for both classification and regression tasks using real-world datasets. The goal is to demonstrate data preprocessing, model training, hyperparameter tuning, and evaluation using Python's `scikit-learn` library.

---

## 🔍 Overview

### 1. Classification - Titanic Dataset
Predicts passenger survival using the Titanic dataset.

- **Dataset:** `Titanic_Dataset_Classification.csv`
- **Models Used:**
  - Naive Bayes (Main Model)
  - Logistic Regression (Baseline)
  - Support Vector Machine (Baseline)
- **Evaluation Metrics:**
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-Score (Weighted Average)
- **Visualizations:**
  - Confusion matrices
  - Bar chart comparison of model metrics

### 2. Regression - California Housing Dataset
Predicts median house values based on various features.

- **Dataset:** `Housing_Dataset_Regression.csv`
- **Models Used:**
  - Random Forest Regressor (Main Model)
  - Linear Regression (Baseline)
  - Ridge Regression (Baseline)
- **Evaluation Metrics:**
  - MAE, MSE, RMSE
  - R² Score
- **Visualizations:**
  - Scatter plot of predicted vs. actual values

---

## 🛠 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

---

## 📌 Notes

- Datasets are included for reproducibility.
- Code is compatible with Google Colab and local Python environments.
- Paths are relative to ensure GitHub compatibility.



## Dataset Description

This project uses the **Heart Disease Prediction Dataset** from the UCI Machine Learning Repository (commonly available on Kaggle).

The dataset contains medical attributes of patients used to predict the presence of heart disease.

The goal of this project is to build machine learning models that classify whether a patient has heart disease based on clinical features.

## Problem Type

- **Binary Classification**

### Output

- **0 → No Heart Disease**
- **1 → Heart Disease Present**

# 📊 Model Performance Results

## ✅ Model Performance Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.795 | 0.879 | 0.756 | 0.874 | 0.811 | 0.597 |
| **Decision Tree** | 0.985 | 0.985 | 1.000 | 0.971 | 0.985 | 0.971 |
| **KNN** | 0.834 | 0.949 | 0.800 | 0.893 | 0.844 | 0.673 |
| **Naive Bayes** | 0.800 | 0.871 | 0.754 | 0.893 | 0.818 | 0.610 |
| **Random Forest (Ensemble)** | 0.985 | 1.000 | 1.000 | 0.971 | 0.985 | 0.971 |
| **XGBoost (Ensemble)** | 0.985 | 0.989 | 1.000 | 0.971 | 0.985 | 0.971 |

---

## 🤖 Model Performance Analysis

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Provided a good baseline performance with moderate accuracy. Works well for simple linear relationships but struggles with complex patterns. |
| **Decision Tree** | Achieved very high accuracy and captured complex feature relationships effectively. However, it may be prone to overfitting. |
| **KNN** | Delivered good performance with strong recall. Performance depends heavily on feature scaling and computational cost increases with data size. |
| **Naive Bayes** | Showed moderate performance. Assumes independence between features, which may not always hold true for medical datasets. |
| **Random Forest (Ensemble)** | One of the best performing models with very high accuracy and precision. Reduces overfitting using multiple decision trees and provides stable predictions. |
| **XGBoost (Ensemble)** | Achieved excellent performance similar to Random Forest. Uses boosting techniques to improve prediction accuracy and handles complex patterns efficiently. |

---

## 📝 Overall Model Comparison Summary

The performance comparison shows that ensemble learning methods such as Random Forest and XGBoost achieved the highest prediction accuracy and reliability. Decision Tree also performed very well but may risk overfitting due to its structure. Logistic Regression, KNN, and Naive Bayes provided moderate performance and served as baseline models for comparison. Overall, ensemble models demonstrated superior ability in capturing complex relationships in heart disease prediction.

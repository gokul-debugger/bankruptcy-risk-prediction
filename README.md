# Bankruptcy Risk Prediction

## Overview

This project predicts corporate bankruptcy risk using financial indicators from Polish companies. The objective is to identify firms that may face financial distress and provide an early-warning risk assessment tool using machine learning.

The project includes:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Engineering
- Model Comparison
- Hyperparameter Tuning
- Business Risk Analysis
- Executive Reporting

---

## Dataset

Source: UCI Machine Learning Repository

Dataset: Polish Companies Bankruptcy Dataset

The dataset contains financial ratios collected from Polish companies between 2000 and 2013.

### Dataset Characteristics

- 43,405 company records
- 64 financial indicators
- Binary target variable
  - 0 = Healthy Company
  - 1 = Bankrupt Company

The original dataset combines multiple forecasting periods ranging from one to five years before bankruptcy.

---

## Project Structure

text bankruptcy-risk-prediction/ │ ├── data/ ├── models/ ├── notebooks/ │   ├── 01_EDA.ipynb │   ├── 02_Preprocessing.ipynb │   ├── 03_Model_Comparison.ipynb │   ├── 04_Hyperparameter_Tuning.ipynb │   ├── 05_Business_Insights.ipynb │   └── 06_Executive_Report.ipynb │ ├── reports/ ├── requirements.txt └── README.md 

---

## Workflow

### 1. Exploratory Data Analysis

- Class distribution analysis
- Correlation analysis
- Outlier detection
- PCA visualization
- Financial ratio exploration

### 2. Data Preprocessing

- Missing value handling
- Outlier treatment
- Feature engineering
- Feature selection
- Dataset preparation

### 3. Model Development

The following algorithms were evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting

### 4. Hyperparameter Tuning

Random Forest was optimized using GridSearchCV.

### 5. Business Insights

- Feature importance analysis
- Bankruptcy risk scoring
- Risk segmentation
- Executive-level insights

---

## Best Model Performance

### Tuned Random Forest

| Metric | Score |
|----------|----------:|
| Accuracy | 0.95 |
| Precision | 0.44 |
| Recall | 0.47 |
| F1 Score | 0.45 |
| ROC-AUC | 0.905 |

### Classification Report

text               precision    recall  f1-score  Healthy          0.97      0.97      0.97 Bankrupt         0.44      0.47      0.45 

The model successfully identifies nearly half of all bankrupt companies while maintaining strong overall predictive performance.

---

## Key Findings

Top bankruptcy predictors identified by the Random Forest model:

| Feature | Importance |
|----------|----------:|
| A27 | 0.1409 |
| A24 | 0.0934 |
| A46 | 0.0610 |
| A26 | 0.0586 |
| A6 | 0.0493 |

These variables showed the strongest influence on bankruptcy prediction.

---

## Risk Segmentation

Companies were grouped according to predicted bankruptcy probability.

| Risk Level | Companies |
|------------|-----------:|
| Low Risk | 36,701 |
| Medium Risk | 3,191 |
| High Risk | 1,913 |

This segmentation can be used as an early-warning financial monitoring framework.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

## Model Files

The repository includes trained model artifacts inside the models/ directory.

These .pkl files are provided for convenience and reproducibility.

They are not required to run the project. Running the notebooks will retrain the models and regenerate the files automatically.

---

## Future Improvements

- XGBoost implementation
- LightGBM implementation
- SHAP explainability
- Time-series bankruptcy forecasting
- Model deployment using Streamlit

---

## Author

Gokul

GitHub: https://github.com/gokul-debugger

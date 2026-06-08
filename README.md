# Bankruptcy Risk Prediction

A machine learning project for predicting corporate bankruptcy risk from financial ratios. The project uses the Polish Companies Bankruptcy dataset to build an early-warning risk model, compare multiple classifiers, and convert model probabilities into practical business risk segments.

## Results Summary

| Best Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Tuned Random Forest | 95.98% | 66.51% | 33.25% | 44.34% | 90.24% |

Because bankruptcy is a rare event, this project does not rely on accuracy alone. The evaluation focuses on precision, recall, F1 score, and ROC-AUC to better understand how well the model identifies financially distressed companies.

## Project Goals

- Predict whether a company is at risk of bankruptcy
- Handle a highly imbalanced financial-risk dataset
- Compare multiple classical machine learning models
- Tune a Random Forest classifier for stronger risk detection
- Estimate bankruptcy probabilities for each company
- Create business-friendly risk segments for monitoring and reporting
- Summarize model findings in an executive-style notebook

## Dataset

Dataset: Polish Companies Bankruptcy Dataset  
Source: UCI Machine Learning Repository

The dataset contains financial ratios from Polish companies collected across multiple bankruptcy forecasting horizons.

| Item | Value |
|---|---:|
| Records | 43,405 |
| Financial indicators | 64 |
| Healthy companies | 41,314 |
| Bankrupt companies | 2,091 |
| Bankruptcy rate | 4.82% |

Target variable:

| Class | Meaning |
|---:|---|
| 0 | Healthy company |
| 1 | Bankrupt company |

## Project Structure

```text
bankruptcy-risk-prediction/
├── data/
│   ├── bankruptcy.csv
│   ├── bankruptcy_processed.csv
│   ├── bankruptcy_risk_scored.csv
│   └── model_results.csv
├── models/
│   ├── random_forest_baseline.pkl
│   └── random_forest_tuned.pkl
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Comparison.ipynb
│   ├── 04_Hyperparameter_Tuning.ipynb
│   ├── 05_Business_Insights.ipynb
│   └── 06_Executive_Report.ipynb
├── requirements.txt
└── README.md
```

## Notebook Workflow

1. `01_EDA.ipynb`  
   Explores data quality, class imbalance, forecasting horizons, feature distributions, outliers, correlations, and PCA structure.

2. `02_Preprocessing.ipynb`  
   Handles missing values, treats outliers, selects useful features, and saves the processed dataset.

3. `03_Model_Comparison.ipynb`  
   Compares Logistic Regression, SVM, Random Forest, and Gradient Boosting.

4. `04_Hyperparameter_Tuning.ipynb`  
   Tunes the Random Forest model and evaluates the final classifier.

5. `05_Business_Insights.ipynb`  
   Analyzes feature importance, scores bankruptcy risk, creates risk segments, and identifies the highest-risk companies.

6. `06_Executive_Report.ipynb`  
   Summarizes the final results for a business audience.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 95.98% | 66.51% | 33.25% | 44.34% | 90.24% |
| Gradient Boosting | 95.84% | 76.15% | 19.86% | 31.50% | 87.89% |
| SVM | 72.02% | 11.55% | 72.25% | 19.91% | 79.64% |
| Logistic Regression | 73.07% | 10.56% | 61.48% | 18.02% | 74.46% |

The tuned Random Forest provided the best overall balance for this project. SVM captured more bankrupt companies, but its low precision would create many false alarms. Gradient Boosting was more precise, but missed more bankrupt companies.

## Final Model

The final model is a balanced Random Forest classifier:

```text
RandomForestClassifier(
    class_weight="balanced",
    max_depth=20,
    min_samples_leaf=2,
    min_samples_split=10,
    n_estimators=500,
    random_state=42
)
```

Why Random Forest:

- Handles nonlinear relationships between financial ratios
- Works well with tabular financial data
- Supports class weighting for imbalance
- Provides feature importance for interpretation
- Produces bankruptcy probabilities for risk segmentation

## Key Predictors

Top bankruptcy predictors identified by the tuned Random Forest model:

| Feature | Importance |
|---|---:|
| A27 | 0.1409 |
| A24 | 0.0934 |
| A46 | 0.0610 |
| A26 | 0.0586 |
| A6 | 0.0493 |

The dataset uses anonymized financial-ratio names, so these features are interpreted as model drivers rather than directly named accounting variables.

## Business Risk Segmentation

The tuned model assigns bankruptcy probability scores and groups companies into practical monitoring bands.

| Risk Level | Companies |
|---|---:|
| Low Risk | 36,701 |
| Medium Risk | 3,191 |
| High Risk | 1,913 |

Example use cases:

- Prioritize companies for financial review
- Flag high-risk accounts for credit monitoring
- Support portfolio risk analysis
- Create executive-level early-warning reports

## Key Learning Points

- Bankruptcy prediction is highly imbalanced, so accuracy can be misleading.
- Recall is important for catching distressed companies, but precision matters to avoid excessive false alarms.
- ROC-AUC helps evaluate how well the model ranks risk across thresholds.
- Probability-based risk segmentation is more useful for business monitoring than a single binary prediction.
- Interpretability matters in financial machine learning because stakeholders need to understand risk drivers.

## How to Run

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Run the notebooks in order from `01_EDA.ipynb` to `06_Executive_Report.ipynb`.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Limitations

- The dataset is historical and may not reflect current financial reporting behavior.
- The model is a decision-support tool, not a replacement for financial due diligence.
- The bankruptcy class is rare, so threshold tuning should depend on the business cost of false positives and false negatives.
- Feature names are anonymized financial indicators, which limits direct financial interpretation.

## Future Improvements

- Add XGBoost or LightGBM
- Add SHAP explanations for model interpretability
- Tune the decision threshold using business cost assumptions
- Build a Streamlit dashboard for interactive risk scoring
- Add a model card documenting intended use and limitations

## Author

Gokul  
GitHub: [gokul-debugger](https://github.com/gokul-debugger)

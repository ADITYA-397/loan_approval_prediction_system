# Loan Approval Prediction System

## 1. Project Overview
This project develops a machine-learning based Loan Approval Prediction System. The system analyzes applicant demographic and financial attributes and estimates whether a loan application is likely to be approved or rejected.

The project follows an end-to-end workflow:
- Data preparation
- Exploratory data analysis
- Feature preprocessing
- Classification model comparison
- Model evaluation
- Feature importance analysis
- Interactive prediction interface
- Business-oriented interpretation of results

## 2. Problem Statement
Financial institutions need consistent ways to assess loan applications. Manual assessment can be time-consuming and may produce inconsistent decisions. This project demonstrates how historical applicant data can be used to build a classification model that estimates loan approval outcomes and highlights the variables that contribute most strongly to the prediction.

## 3. Dataset
For this submission package, a reproducible synthetic dataset with 4,269 applicant records and 13 columns is included as `loan_approval_dataset.csv`. It follows the same structure as the publicly documented Kaggle Loan Approval Prediction dataset, including applicant income, loan amount, CIBIL score, assets and approval status.

Public reference for the dataset structure:
https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

The project submission rules in the Zoom session require a new dataset rather than the exact learning dataset used in the masterclasses. The included dataset is therefore generated independently for this project and is not the masterclass dataset.

## 4. Features
- `no_of_dependents`: Number of dependents
- `education`: Graduate / Not Graduate
- `self_employed`: Yes / No
- `income_annum`: Annual income
- `loan_amount`: Requested loan amount
- `loan_term`: Loan repayment term in years
- `cibil_score`: Credit score
- `residential_assets_value`: Residential asset value
- `commercial_assets_value`: Commercial asset value
- `luxury_assets_value`: Luxury asset value
- `bank_asset_value`: Bank asset value
- `loan_status`: Approved / Rejected

`loan_id` is retained only as an identifier and is excluded from model training.

## 5. Models
Four classification algorithms are compared:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

The final model is selected using ROC-AUC on the held-out test set.

## 6. Model Results
Results generated from the included dataset and fixed random seed:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 75.18% | 76.02% | 78.02% | 77.01% | 0.834 |
| Decision Tree | 72.01% | 78.42% | 65.49% | 71.38% | 0.783 |
| Random Forest | 75.41% | 75.68% | 79.34% | 77.47% | 0.822 |
| Gradient Boosting | 74.82% | 75.75% | 77.58% | 76.66% | 0.823 |

The Logistic Regression model achieved the highest ROC-AUC on the held-out test set in this project.

## 7. Key Insights
- CIBIL score is the strongest predictive feature in the permutation-importance analysis.
- Loan amount and annual income provide additional predictive information.
- Financial asset variables contribute smaller incremental information compared with credit score.
- Model comparison shows that a simple interpretable baseline can perform competitively with more complex tree-based models on this dataset.
- The system is most useful as a decision-support prototype rather than as an autonomous lending system.

## 8. Business Actions
Based on the analysis, a financial institution could:
- Use CIBIL score as one of the major risk-screening variables.
- Review high loan-to-income applications more carefully.
- Combine model probability with existing lending policy and human review.
- Monitor model performance regularly for data drift and changes in applicant behavior.
- Avoid using the prediction as the sole basis for a lending decision.

## 9. Application
The Python file also contains a Streamlit interface.

Run:
```bash
pip install -r requirements.txt
streamlit run loan_approval_prediction.py
```

The application contains:
- Executive Overview
- Model Analysis
- Confusion Matrix
- Feature Importance
- Applicant Prediction Form

## 10. Submission Files
The required four submission files are:
1. `loan_approval_prediction.py`
2. `requirements.txt`
3. `README.md`
4. `Loan_Approval_Prediction_Project_Report.docx`

The dataset is included only for reproducibility and may be placed in the GitHub repository.

## 11. Author
Aditya Kadam

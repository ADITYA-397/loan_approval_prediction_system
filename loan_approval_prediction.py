"""
Loan Approval Prediction System
Author: Aditya Kadam
Purpose: End-to-end loan approval classification and interactive Streamlit app.

Dataset:
    loan_approval_dataset.csv

Run:
    python loan_approval_prediction.py
or:
    streamlit run loan_approval_prediction.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
DATA_FILE = "loan_approval_dataset.csv"


@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(
            f"Dataset not found. Place '{DATA_FILE}' in the same folder as this script."
        )
        st.stop()

    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().lower() for c in df.columns]

    # Clean categorical values
    for col in ["education", "self_employed", "loan_status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


@st.cache_resource
def train_models(df):
    target = "loan_status"

    X = df.drop(columns=[target, "loan_id"], errors="ignore")
    y = df[target].map({"Rejected": 0, "Approved": 1})

    categorical = ["education", "self_employed"]
    numerical = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numerical
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical
            )
        ]
    )

    model_defs = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    trained = {}
    rows = []

    for name, model in model_defs.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, prob)
        })
        trained[name] = pipe

    results = pd.DataFrame(rows)
    best_name = results.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]

    # Permutation importance on original features
    best_model = trained[best_name]
    permutation = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=8, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": permutation.importances_mean
    }).sort_values("Importance", ascending=False)

    return trained, results, importance, X_test, y_test, best_name


def main():
    st.set_page_config(
        page_title="Loan Approval Prediction System",
        page_icon="💳",
        layout="wide"
    )

    st.title("Loan Approval Prediction System")
    st.caption(
        "Machine learning decision-support prototype for estimating loan approval probability."
    )

    df = load_data()
    models, results, importance, X_test, y_test, best_name = train_models(df)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Applications", f"{len(df):,}")
    c2.metric("Approval Rate", f"{(df.loan_status.eq('Approved').mean()*100):.1f}%")
    c3.metric("Best Model", best_name)
    c4.metric("Best ROC-AUC", f"{results['ROC-AUC'].max():.3f}")

    tab1, tab2, tab3 = st.tabs(
        ["Executive Overview", "Model Analysis", "Loan Prediction"]
    )

    with tab1:
        st.subheader("Business Overview")

        col1, col2 = st.columns(2)
        with col1:
            counts = df["loan_status"].value_counts()
            fig, ax = plt.subplots()
            ax.bar(counts.index, counts.values)
            ax.set_ylabel("Applications")
            ax.set_title("Loan Status Distribution")
            st.pyplot(fig, clear_figure=True)

        with col2:
            fig, ax = plt.subplots()
            ax.hist(
                df.loc[df["loan_status"] == "Approved", "cibil_score"],
                bins=25, alpha=0.7, label="Approved"
            )
            ax.hist(
                df.loc[df["loan_status"] == "Rejected", "cibil_score"],
                bins=25, alpha=0.7, label="Rejected"
            )
            ax.set_xlabel("CIBIL Score")
            ax.set_ylabel("Applications")
            ax.set_title("CIBIL Score Distribution")
            ax.legend()
            st.pyplot(fig, clear_figure=True)

        st.subheader("Key Drivers")
        st.dataframe(importance.head(8), use_container_width=True, hide_index=True)

        st.info(
            "Decision-support note: this model is a demonstration and should not "
            "be used as the sole basis for real lending decisions."
        )

    with tab2:
        st.subheader("Model Comparison")
        display_results = results.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
            display_results[col] = display_results[col].round(4)

        st.dataframe(display_results, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots()
        ax.bar(results["Model"], results["ROC-AUC"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("ROC-AUC")
        ax.set_title("ROC-AUC by Model")
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig, clear_figure=True)

        chosen = st.selectbox("Inspect model", list(models.keys()))
        pred = models[chosen].predict(X_test)
        cm = confusion_matrix(y_test, pred)

        st.write("Confusion Matrix")
        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual Rejected", "Actual Approved"],
                columns=["Predicted Rejected", "Predicted Approved"]
            ),
            use_container_width=True
        )

        st.text(classification_report(
            y_test, pred, target_names=["Rejected", "Approved"]
        ))

    with tab3:
        st.subheader("Applicant Prediction")

        left, right = st.columns(2)
        with left:
            dependents = st.number_input("Number of Dependents", 0, 10, 2)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            income = st.number_input("Annual Income", 100000, 50000000, 700000, step=50000)
            loan_amount = st.number_input("Loan Amount", 100000, 50000000, 3000000, step=100000)

        with right:
            loan_term = st.selectbox("Loan Term (years)", [5, 10, 15, 20, 25, 30], index=3)
            cibil = st.slider("CIBIL Score", 300, 900, 700)
            residential = st.number_input("Residential Assets Value", 0, 100000000, 1000000, step=100000)
            commercial = st.number_input("Commercial Assets Value", 0, 100000000, 500000, step=100000)
            luxury = st.number_input("Luxury Assets Value", 0, 100000000, 700000, step=100000)
            bank = st.number_input("Bank Asset Value", 0, 100000000, 500000, step=100000)

        if st.button("Predict Loan Approval", type="primary"):
            applicant = pd.DataFrame([{
                "no_of_dependents": dependents,
                "education": education,
                "self_employed": self_employed,
                "income_annum": income,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil,
                "residential_assets_value": residential,
                "commercial_assets_value": commercial,
                "luxury_assets_value": luxury,
                "bank_asset_value": bank
            }])

            model = models[best_name]
            probability = model.predict_proba(applicant)[0, 1]
            prediction = "Approved" if probability >= 0.50 else "Rejected"

            if prediction == "Approved":
                st.success(f"Prediction: {prediction}")
            else:
                st.warning(f"Prediction: {prediction}")

            st.metric("Estimated Approval Probability", f"{probability*100:.1f}%")
            st.caption(
                "This is a machine-learning estimate, not a financial or lending decision."
            )


if __name__ == "__main__":
    main()

# Databricks Notebook
# Phase: Model Training & Evaluation
# Purpose: Train baseline and improved churn models using sklearn (DB Free compatible)

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# -----------------------------
# Config
# -----------------------------
CATALOG = "end_to_end_churn"
SCHEMA = "churn_analytics"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.telco_gold_features"

# -----------------------------
# Load Gold data
# -----------------------------
spark_df = spark.table(GOLD_TABLE)
pdf = spark_df.toPandas()

print("Total rows:", len(pdf))
print("Columns:", len(pdf.columns))

# -----------------------------
# Train / Test split
# -----------------------------
train_df = pdf[pdf["dataset_split"] == "train"].copy()
test_df = pdf[pdf["dataset_split"] == "test"].copy()

X_train_full = train_df.drop(columns=["customerID", "label_churn", "dataset_split"])
y_train = train_df["label_churn"].astype(int)

X_test_full = test_df.drop(columns=["customerID", "label_churn", "dataset_split"])
y_test = test_df["label_churn"].astype(int)

print("Train size:", X_train_full.shape)
print("Test size:", X_test_full.shape)

# We only impute numeric columns (in this project, all feature cols should be numeric already)
numeric_cols = X_train_full.select_dtypes(include=["float64", "int64"]).columns.tolist()
X_train = X_train_full[numeric_cols]
X_test = X_test_full[numeric_cols]

# Quick NaN check
print("NaNs in X_train:", int(np.isnan(X_train.to_numpy()).sum()))
print("NaNs in X_test:", int(np.isnan(X_test.to_numpy()).sum()))
print("Numeric feature count:", len(numeric_cols))

# -----------------------------
# Helper: evaluation
# -----------------------------
def evaluate_proba(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

def print_metrics(name, metrics):
    print(f"\n--- {name} ---")
    print("accuracy:", metrics["accuracy"])
    print("precision:", metrics["precision"])
    print("recall:", metrics["recall"])
    print("f1:", metrics["f1"])
    print("roc_auc:", metrics["roc_auc"])
    print("confusion_matrix:\n", metrics["confusion_matrix"])

# -----------------------------
# 1) Baseline: Logistic Regression (with imputer + scaler)
# -----------------------------
log_reg_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ))
])

log_reg_pipeline.fit(X_train, y_train)

log_pred = log_reg_pipeline.predict(X_test)
log_prob = log_reg_pipeline.predict_proba(X_test)[:, 1]
log_metrics = evaluate_proba(y_test, log_pred, log_prob)

print_metrics("Logistic Regression", log_metrics)

# -----------------------------
# 2) Improved: Random Forest (with imputer; no scaling needed)
# -----------------------------
rf_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=50,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

rf_pipeline.fit(X_train, y_train)

rf_pred = rf_pipeline.predict(X_test)
rf_prob = rf_pipeline.predict_proba(X_test)[:, 1]
rf_metrics = evaluate_proba(y_test, rf_pred, rf_prob)

print_metrics("Random Forest", rf_metrics)

# -----------------------------
# 3) Model comparison summary
# -----------------------------
summary = pd.DataFrame(
    [
        {"model": "logistic_regression", **{k: v for k, v in log_metrics.items() if k != "confusion_matrix"}},
        {"model": "random_forest", **{k: v for k, v in rf_metrics.items() if k != "confusion_matrix"}},
    ]
)

print("\n=== Model Comparison (Test Set) ===")
display(summary)

# Optional: pick a champion model based on ROC-AUC (you can change criterion)
champion_name = "random_forest" if rf_metrics["roc_auc"] >= log_metrics["roc_auc"] else "logistic_regression"
print("\nChampion model:", champion_name)


# Databricks Notebook
# Phase: Batch Inference
# Purpose: Score all customers using champion churn model

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# -----------------------------
# Config
# -----------------------------
CATALOG = "end_to_end_churn"
SCHEMA = "churn_analytics"

GOLD_FEATURES_TABLE = f"{CATALOG}.{SCHEMA}.telco_gold_features"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.churn_predictions_gold"

MODEL_NAME = "random_forest_churn"
MODEL_VERSION = "v1.0"

# -----------------------------
# Load Gold features
# -----------------------------
spark_df = spark.table(GOLD_FEATURES_TABLE)
pdf = spark_df.toPandas()

print("Scoring rows:", len(pdf))

# -----------------------------
# Prepare features
# -----------------------------
X = pdf.drop(columns=["customerID", "label_churn", "dataset_split"])
y = pdf["label_churn"]  # optional, for offline validation

numeric_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
X = X[numeric_cols]

# -----------------------------
# Re-train champion model
# (In real prod, load from registry)
# -----------------------------
train_mask = pdf["dataset_split"] == "train"
X_train = X[train_mask]
y_train = y[train_mask]

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

# -----------------------------
# Score all customers
# -----------------------------
preds = rf_pipeline.predict(X)
probs = rf_pipeline.predict_proba(X)[:, 1]

results = pd.DataFrame({
    "customerID": pdf["customerID"],
    "prediction": preds.astype(int),
    "churn_probability": probs,
    "model_name": MODEL_NAME,
    "model_version": MODEL_VERSION,
    "scored_ts": datetime.utcnow()
})

print(results.head())

# -----------------------------
# Write predictions to Delta
# -----------------------------
spark_results = spark.createDataFrame(results)

(
    spark_results.write.format("delta")
    .mode("overwrite")
    .saveAsTable(PREDICTIONS_TABLE)
)

print("✅ Predictions written to:", PREDICTIONS_TABLE)

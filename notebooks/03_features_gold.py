# Databricks Notebook
# Phase: Gold Feature Engineering
# Purpose: Create model-ready features from silver table (manual OHE for DB Free)

from pyspark.sql import functions as F

CATALOG = "end_to_end_churn"
SCHEMA = "churn_analytics"

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.telco_silver"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.telco_gold_features"

df = spark.table(SILVER_TABLE)
print("Silver rows:", df.count())

def sanitize(s: str) -> str:
    return (
        s.strip()
         .replace(" ", "_")
         .replace("-", "_")
         .replace("(", "")
         .replace(")", "")
         .replace("/", "_")
         .replace(",", "")
    )

# -----------------------------
# 1) Binary mappings
# -----------------------------
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for c in binary_cols:
    df = df.withColumn(c, F.when(F.lower(F.col(c)) == "yes", 1).otherwise(0))

# -----------------------------
# 2) Multi-category columns (manual one-hot)
# -----------------------------
categorical_map = {
    "gender": ["Male", "Female"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

# Validate categories (real-job check)
print("\n---- Category Validation (Silver vs Expected) ----")
for col, expected in categorical_map.items():
    actual = [r[0] for r in df.select(col).distinct().collect()]
    unexpected = sorted([a for a in actual if a not in expected and a is not None and a != ""])
    missing = sorted([e for e in expected if e not in actual])
    print(f"{col}: distinct={len(actual)} unexpected={unexpected} missing={missing}")

# Create one-hot columns
ohe_cols = []
for col, categories in categorical_map.items():
    for cat in categories:
        new_col = f"{col}_{sanitize(cat)}"
        ohe_cols.append(new_col)
        df = df.withColumn(new_col, F.when(F.col(col) == cat, 1).otherwise(0))

# Optional: add an "unknown" flag per categorical column (helpful in prod)
# If you prefer, uncomment this:
# for col, categories in categorical_map.items():
#     df = df.withColumn(
#         f"{col}_UNKNOWN",
#         F.when(~F.col(col).isin(categories), 1).otherwise(0)
#     )
#     ohe_cols.append(f"{col}_UNKNOWN")

# -----------------------------
# 3) Select final features
# -----------------------------
numeric_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

feature_cols = numeric_features + binary_cols + ohe_cols

gold_df = df.select("customerID", "label_churn", *feature_cols)

# -----------------------------
# 4) Train/Test split flag (reproducible)
# -----------------------------
gold_df = gold_df.withColumn(
    "dataset_split",
    F.when(F.rand(seed=42) < 0.8, "train").otherwise("test")
)

print("Gold rows:", gold_df.count())

# -----------------------------
# 5) Write Gold table
# -----------------------------
(
    gold_df.write.format("delta")
    .mode("overwrite")
    .saveAsTable(GOLD_TABLE)
)

print("✅ Gold feature table written:", GOLD_TABLE)

# -----------------------------
# 6) Split validation
# -----------------------------
gold_df.groupBy("dataset_split", "label_churn").count().orderBy(
    "dataset_split", "label_churn"
).show()


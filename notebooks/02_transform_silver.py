# Databricks Notebook
# Phase: Silver Transformation
# Purpose: Clean and standardize bronze data into analytics-ready silver table

from pyspark.sql import functions as F

CATALOG = "end_to_end_churn"
SCHEMA = "churn_analytics"

BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.telco_bronze"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.telco_silver"

print("BRONZE_TABLE:", BRONZE_TABLE)
print("SILVER_TABLE:", SILVER_TABLE)

df = spark.table(BRONZE_TABLE)

print("Bronze row count:", df.count())

# -----------------------------
# 1) Basic standardization
# -----------------------------

# Trim all string columns to remove hidden spaces
string_cols = [c for c, t in df.dtypes if t == "string"]
for c in string_cols:
    df = df.withColumn(c, F.trim(F.col(c)))

# Clean TotalCharges: dataset has blanks for some customers (often tenure=0)
df = df.withColumn(
    "TotalCharges",
    F.when((F.col("TotalCharges").isNull()) | (F.col("TotalCharges") == ""), None).otherwise(F.col("TotalCharges"))
)

df = df.withColumn("TotalCharges", F.col("TotalCharges").cast("double"))

# Standardize SeniorCitizen to 0/1 int (already int, but keep explicit)
df = df.withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))

# Standardize Churn to label_churn (0/1)
df = df.withColumn(
    "label_churn",
    F.when(F.lower(F.col("Churn")) == "yes", F.lit(1)).otherwise(F.lit(0))
)

# Drop duplicates by customerID (shouldn’t exist, but we enforce)
df = df.dropDuplicates(["customerID"])

# -----------------------------
# 2) Data quality checks (light but real)
# -----------------------------
n_rows = df.count()
n_customers = df.select("customerID").distinct().count()
null_key = df.filter(F.col("customerID").isNull() | (F.col("customerID") == "")).count()
null_label = df.filter(F.col("label_churn").isNull()).count()

print("---- Data Quality ----")
print("Rows:", n_rows)
print("Distinct customerID:", n_customers)
print("Null/blank customerID:", null_key)
print("Null label_churn:", null_label)

# TotalCharges null rate (expected some nulls when tenure=0)
totalcharges_nulls = df.filter(F.col("TotalCharges").isNull()).count()
print("TotalCharges nulls:", totalcharges_nulls, f"({totalcharges_nulls/n_rows:.2%})")

# If you want to enforce “must-have” rules, keep them as asserts (job-style)
assert null_key == 0, "customerID has null/blank values — investigate bronze ingestion/source"
assert null_label == 0, "label_churn has null values — investigate churn parsing"

# -----------------------------
# 3) Write Silver
# -----------------------------
(
    df.write.format("delta")
    .mode("overwrite")
    .saveAsTable(SILVER_TABLE)
)

print("✅ Wrote Silver table:", SILVER_TABLE)

# Quick preview
spark.table(SILVER_TABLE).select(
    "customerID", "tenure", "MonthlyCharges", "TotalCharges", "label_churn"
).show(10, truncate=False)

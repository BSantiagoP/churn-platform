# Databricks Notebook
# Phase: Silver Transformation
# Purpose: Clean and standardize bronze data into analytics-ready silver table

from pyspark.sql import functions as F

# -----------------------------
# Config
# -----------------------------
CATALOG = "end_to_end_churn"
SCHEMA = "churn_analytics"

BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.telco_bronze"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.telco_silver"

df = spark.table(BRONZE_TABLE)

print("Bronze row count:", df.count())

# -----------------------------
# 1) Trim all string columns
# -----------------------------
string_cols = [c for c, t in df.dtypes if t == "string"]

for c in string_cols:
    df = df.withColumn(c, F.trim(F.col(c)))

# -----------------------------
# 2) Fix TotalCharges (string -> double)
# -----------------------------
df = df.withColumn(
    "TotalCharges",
    F.when(
        (F.col("TotalCharges").isNull()) | (F.col("TotalCharges") == ""),
        None
    ).otherwise(F.col("TotalCharges"))
)

df = df.withColumn("TotalCharges", F.col("TotalCharges").cast("double"))

# -----------------------------
# 3) Explicit label column
# -----------------------------
df = df.withColumn(
    "label_churn",
    F.when(F.lower(F.col("Churn")) == "yes", F.lit(1)).otherwise(F.lit(0))
)

# -----------------------------
# 4) Deduplication & key enforcement
# -----------------------------
df = df.dropDuplicates(["customerID"])

# -----------------------------
# 5) Data quality checks
# -----------------------------
n_rows = df.count()
n_customers = df.select("customerID").distinct().count()

null_customer_id = df.filter(
    F.col("customerID").isNull() | (F.col("customerID") == "")
).count()

null_label = df.filter(F.col("label_churn").isNull()).count()

totalcharges_nulls = df.filter(F.col("TotalCharges").isNull()).count()

print("---- Data Quality Report ----")
print(f"Rows: {n_rows}")
print(f"Distinct customerID: {n_customers}")
print(f"Null/blank customerID: {null_customer_id}")
print(f"Null label_churn: {null_label}")
print(f"TotalCharges nulls: {totalcharges_nulls} ({totalcharges_nulls/n_rows:.2%})")

# Enforce critical rules (job-style)
assert null_customer_id == 0, "❌ customerID contains null or blank values"
assert null_label == 0, "❌ label_churn contains null values"

# -----------------------------
# 6) Write Silver Delta table
# -----------------------------
(
    df.write.format("delta")
    .mode("overwrite")
    .saveAsTable(SILVER_TABLE)
)

print("✅ Silver table written:", SILVER_TABLE)

# -----------------------------
# 7) Preview
# -----------------------------
spark.table(SILVER_TABLE).select(
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "InternetService",
    "label_churn"
).show(10, truncate=False)

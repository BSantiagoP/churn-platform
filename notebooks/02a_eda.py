# Databricks Notebook
# Phase: Exploratory Data Analysis (EDA)
# Purpose: Understand raw churn dataset before cleaning & modeling

from pyspark.sql import functions as F

CATALOG = "end_to_end_churn"
SCHEMA = "churn_analytics"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.telco_bronze"

df = spark.table(BRONZE_TABLE)

print("Rows:", df.count())
print("Columns:", len(df.columns))

# -----------------------------
# 1) Schema & basic stats
# -----------------------------
df.printSchema()
df.describe(["tenure", "MonthlyCharges"]).show()

# -----------------------------
# 2) Target variable inspection
# -----------------------------
df.groupBy("Churn").count().show()

# -----------------------------
# 3) Key integrity
# -----------------------------
df.select(
    F.countDistinct("customerID").alias("distinct_customers"),
    F.sum(F.when(F.col("customerID").isNull() | (F.col("customerID") == ""), 1).otherwise(0)).alias("null_customerID")
).show()

# -----------------------------
# 4) Missing & problematic values
# -----------------------------
df.select(
    F.count(
        F.when(
            F.col("TotalCharges").isNull() | (F.trim(F.col("TotalCharges")) == ""),
            1
        )
    ).alias("TotalCharges_nulls"),

    F.count(
        F.when(F.col("MonthlyCharges").isNull(), 1)
    ).alias("MonthlyCharges_nulls"),

    F.count(
        F.when(F.col("tenure").isNull(), 1)
    ).alias("tenure_nulls")
).show()

# -----------------------------
# 5) Categorical value distributions (sample)
# -----------------------------
categorical_cols = [
    "Contract", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport"
]

for c in categorical_cols:
    print(f"\nDistribution for {c}")
    df.groupBy(c).count().orderBy(F.desc("count")).show()

# -----------------------------
# 6) Churn by segment (early insight)
# -----------------------------
df.groupBy("Contract", "Churn").count().orderBy("Contract", "Churn").show()
df.groupBy("InternetService", "Churn").count().orderBy("InternetService", "Churn").show()






# Databricks Notebook
# Phase: Bronze Ingestion
# Purpose: Ingest raw churn data into bronze Delta table

from pyspark.sql import functions as F

# ---- Config (keep simple for Free Edition) ----
CATALOG = "end_to_end_churn"
SCHEMA = "churn_analytics"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.telco_bronze"

RAW_PATH = "dbfs:/Volumes/end_to_end_churn/churn_analytics/telco_raw_data/Telco-Customer-Churn.csv"
SOURCE_FILE = "Telco-Customer-Churn.csv"

print("RAW_PATH:", RAW_PATH)
print("BRONZE_TABLE:", BRONZE_TABLE)

# ---- Read CSV ----
df_raw = (
    spark.read.format("csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load(RAW_PATH)
)

print("Raw row count:", df_raw.count())
print("Raw columns:", len(df_raw.columns))
df_raw.printSchema()

# ---- Add ingestion metadata ----
df_bronze = (
    df_raw
    .withColumn("ingest_ts", F.current_timestamp())
    .withColumn("source_file", F.lit(SOURCE_FILE))
)

# Optional: basic column name hygiene (no spaces)
# (Telco dataset usually doesn't have spaces, but this prevents headaches)
clean_cols = [c.strip().replace(" ", "_") for c in df_bronze.columns]
df_bronze = df_bronze.toDF(*clean_cols)

# ---- Write Bronze Delta ----
(
    df_bronze.write.format("delta")
    .mode("overwrite")  # overwrite is fine for v1; later we can move to append
    .saveAsTable(BRONZE_TABLE)
)

print("✅ Wrote Bronze table:", BRONZE_TABLE)

# ---- Quick validation ----
df_check = spark.table(BRONZE_TABLE)
print("Bronze row count:", df_check.count())
df_check.select("ingest_ts", "source_file").show(5, truncate=False)
df_check.show(5, truncate=False)

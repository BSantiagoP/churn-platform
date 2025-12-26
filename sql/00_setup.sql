-- sql/00_setup.sql

CREATE CATALOG IF NOT EXISTS end_to_end_business_insight_pipeline;

CREATE SCHEMA IF NOT EXISTS end_to_end_business_insight_pipeline.churn_analytics;

CREATE VOLUME IF NOT EXISTS end_to_end_business_insight_pipeline.churn_analytics.telco_raw_data;

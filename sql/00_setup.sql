-- sql/00_setup.sql

CREATE CATALOG IF NOT EXISTS end_to_end_churn;

CREATE SCHEMA IF NOT EXISTS end_to_end_churn.churn_analytics;

CREATE VOLUME IF NOT EXISTS end_to_end_churn.churn_analytics.telco_raw_data;

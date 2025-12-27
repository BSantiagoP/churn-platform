## EDA Findings – Telco Churn Dataset

- Target variable `Churn` is binary (Yes/No) with class imbalance (~27% churn).
- `TotalCharges` is stored as string with blank values for customers with tenure = 0.
- No duplicate `customerID` values observed.
- Several categorical columns contain values such as "No internet service" which need consistent handling.
- Strong churn differences observed by Contract type and InternetService.


## Silver Layer Decisions

- Converted `TotalCharges` from STRING to DOUBLE.
- Empty string values mapped to NULL (observed primarily for tenure = 0).
- Created explicit binary label column `label_churn`.
- Enforced uniqueness and non-null constraint on `customerID`.
- Deferred categorical encoding to Gold layer.
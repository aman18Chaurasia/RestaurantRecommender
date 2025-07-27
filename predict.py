import pandas as pd
import joblib
import os

# Load model
model = joblib.load("model/lgbm_model.pkl")

# Load data
test_customers = pd.read_csv("data/assignment/Test/test_customers.csv")
test_locations = pd.read_csv("data/assignment/Test/test_locations.csv")

# Use one vendor for now (can be modified for looped predictions)
TARGET_VENDOR_ID = 243

# Merge customer and location info
df = test_locations.merge(test_customers, on="customer_id", how="left")
df["vendor_id"] = TARGET_VENDOR_ID

# Feature engineering (customize based on how your model was trained)
df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
df["age"] = 2025 - df["dob"].dt.year.fillna(0)

# Select features (MUST match your model training exactly)
features = ["location_number", "vendor_id", "age"]  # modify if needed
X = df[features].astype(float).fillna(0)

# Predict
preds = model.predict_proba(X)[:, 1]

# Format output
df["CID X LOC_NUM X VENDOR"] = df["customer_id"] + " X " + df["location_number"].astype(str) + " X " + df["vendor_id"].astype(str)
output = df[["CID X LOC_NUM X VENDOR"]].copy()
output["target"] = preds

# Save
output.to_csv("submission.csv", index=False)
print("✅ submission.csv created with", len(output), "rows.")

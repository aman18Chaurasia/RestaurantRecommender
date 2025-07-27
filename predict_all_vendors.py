# predict_all_vendors.py
import pandas as pd
import joblib
from tqdm import tqdm
import os

# === CONFIG ===
MODEL_PATH = "model/lgbm_model.pkl"
OUTPUT_PATH = "submission_all_vendors.csv"

# === LOAD MODEL ===
model = joblib.load(MODEL_PATH)

# === LOAD DATA ===
test_customers = pd.read_csv("data/assignment/Test/test_customers.csv")
test_locations = pd.read_csv("data/assignment/Test/test_locations.csv")
vendors = pd.read_csv("data/assignment/Train/vendors.csv")

# === GENERATE CUSTOMER-LOCATION-VENDOR COMBINATIONS ===
print("Generating customer-location-vendor combinations...")
combo_data = []
for _, row in tqdm(test_locations.iterrows(), total=len(test_locations)):
    for vendor_id in vendors['id'].unique():
        combo_data.append({
            "customer_id": row["customer_id"],
            "location_number": row["location_number"],
            "vendor_id": vendor_id
        })
combo_df = pd.DataFrame(combo_data)

# === BASIC FEATURE ENGINEERING (EDIT THIS TO MATCH TRAINING STAGE) ===
# Placeholder: You must match features used during training
# For example, if model was trained on location_number and vendor_id only:
features = ["location_number", "vendor_id"]  # Adjust based on training
X = combo_df[features]

# === PREDICTION ===
print("Running predictions...")
preds = model.predict_proba(X)[:, 1]  # Predict probabilities for class 1
combo_df["CID X LOC_NUM X VENDOR"] = combo_df["customer_id"] + " X " + combo_df["location_number"].astype(str) + " X " + combo_df["vendor_id"].astype(str)
combo_df["target"] = preds
submission = combo_df[["CID X LOC_NUM X VENDOR", "target"]]

# === SAVE TO CSV ===
submission.to_csv(OUTPUT_PATH, index=False)
print(f"✅ {OUTPUT_PATH} created with {len(submission)} rows.")

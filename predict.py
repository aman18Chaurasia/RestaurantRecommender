import pandas as pd
import joblib

# Load model
model = joblib.load("model/lgbm_model.pkl")

# Sample input
sample = pd.DataFrame([{
    "vendor_rating": 4.5,
    "deliverydistance": 3.2,
    "preparationtime": 15
}])

# Predict
prediction = model.predict_proba(sample)[:, 1]
print(f"Predicted probability of interaction: {prediction[0]:.4f}")

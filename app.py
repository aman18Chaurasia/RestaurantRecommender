# app.py
import streamlit as st
import lightgbm as lgb
import pandas as pd
import joblib

# Load model
model = joblib.load("model/lgbm_model.pkl")  # Ensure this path is correct

# App UI
st.title("Restaurant Recommender System")
st.write("Enter customer and vendor details:")

cid = st.text_input("Customer ID (e.g., 0JP29SK)")
loc_num = st.number_input("Location Number", min_value=0, step=1)
vendor = st.number_input("Vendor ID", min_value=0, step=1)

# Predict
if st.button("Predict"):
    # Prepare features
    df = pd.DataFrame({
        "location_number": [loc_num],
        "vendor_id": [vendor]
        # Add more features here if your model expects them
    })

    # Predict
    prediction = model.predict(df)[0]
    st.success(f"Predicted probability of interaction: {prediction:.4f}")

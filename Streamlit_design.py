import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import math

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("impurity_nn_model.h5", compile=False)
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# Load dataset for threshold lookup
@st.cache_data
def load_threshold_data():
    file_path = r"C:\Users\nalla\OneDrive\Desktop\Food Thresholds Project\cleaned_with_log_applied.xlsx"

    df = pd.read_excel(file_path)
    return df

df_threshold = load_threshold_data()

st.title("Protein Powder Safety Checker")

# Input fields (real values)
labeled_sugars = st.number_input("Labeled Sugars (g)")
total_sugars = st.number_input("Total Sugars (g)")
labeled_protein = st.number_input("Labeled Total Protein Content (g)")
total_protein = st.number_input("Total Protein (g)")
amino_variation = st.number_input("Variation of Amino Acids (µg/g)")
Ar = st.number_input("Arsenic (µg/kg)")
Cd = st.number_input("Cadmium (µg/kg)")
Hg = st.number_input("Mercury (µg/kg)")
Pb = st.number_input("Lead (µg/kg)")
Co = st.number_input("Cobalt (µg/kg)")
Cr = st.number_input("Chromium (µg/kg)")
Mn = st.number_input("Manganese (µg/kg)")
Fe = st.number_input("Iron (µg/kg)")

if st.button("Check Safety"):
    try:
        # Convert real inputs to log
        input_features = [
            math.log(labeled_sugars + 1e-5),
            math.log(total_sugars + 1e-5),
            math.log(labeled_protein + 1e-5),
            math.log(total_protein + 1e-5),
            math.log(amino_variation + 1e-5),
            math.log(Ar + 1e-5),
            math.log(Cd + 1e-5),
            math.log(Hg + 1e-5),
            math.log(Pb + 1e-5),
            math.log(Co + 1e-5),
            math.log(Cr + 1e-5),
            math.log(Mn + 1e-5),
            math.log(Fe + 1e-5)
        ]

        # Scale the input
        input_scaled = scaler.transform([input_features])

        # Predict log(PHA)
        predicted_log_pha = model.predict(input_scaled)[0][0]
        predicted_pha = round(math.exp(predicted_log_pha), 4)

        # Try to find the matching threshold log value
        matched_row = df_threshold[
            (df_threshold['Labeled_total_protein_content'] == labeled_protein) &
            (df_threshold['Total_Protein_g'] == total_protein) &
            (df_threshold['Labeled_sugars_g'] == labeled_sugars) &
            (df_threshold['Total_Sugars_g'] == total_sugars)
        ]

        if not matched_row.empty:
            threshold_log = matched_row['STANDARD_PHA_μgkg_log'].values[0]
            threshold_real = matched_row['STANDARD_PHA_μgkg'].values[0]
            is_safe = predicted_log_pha <= threshold_log

            st.success(f"Predicted PHA: {predicted_pha} µg/kg")
            st.info(f"Threshold for this sample: {threshold_real} µg/kg")
            if is_safe:
                st.markdown("✅ This protein powder is **SAFE** to consume.")
            else:
                st.markdown("⚠️ This protein powder is **RISKY** to consume.")
        else:
            st.warning("Matching threshold not found for given values. Check inputs or update threshold table.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

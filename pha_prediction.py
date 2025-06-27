# ===============================
# 1. TRAIN NEURAL NETWORK + SAVE
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
from tensorflow.keras.utils import plot_model

# Load dataset
file_path = r"C:\Users\nalla\OneDrive\Desktop\Food Thresholds Project\cleaned_with_log_applied.xlsx"
df = pd.read_excel(file_path)

# Define features and corrected target
features = [
    'Labeled_sugars_g', 'Total_Sugars_g', 'Labeled_total_protein_content', 'Total_Protein_g',
    'variation_of_amino_acids_µgg_log', 'Ar_log', 'Cd_log', 'Hg_log', 'Pb_log', 'Co_log',
    'Cr_log', 'Mn_log', 'Fe_log'
]
target = 'STANDARD_PHA_μgkg_log'  

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and compile model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=0)

# Save model and scaler
model.save("nn_model.h5")
joblib.dump(scaler, "scaler.pkl")

import os
import joblib
import pandas as pd
import numpy as np

# ==========================================
# 1. Paths & Configurations
# ==========================================
input_path = "data/mixed_biomass_data"
file_name = "mixed_biomass.csv"
output_name = "mixed_biomass_with_predictions.csv"
output_excel_name = "mixed_biomass_with_predictions_top_last_1000.xlsx"

# Model and preprocessor paths
model_filename = 'model_pkl/trained_gbdt_model.pkl'
scaler_filename = 'model_pkl/scaler.pkl'
encoder_filename = 'model_pkl/encoder.pkl'
tfidf_filename = 'model_pkl/tfidf_vectorizers.pkl'

# ==========================================
# 2. Load Models & Data
# ==========================================
print("Loading models and preprocessors...")
loaded_model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)
encoder = joblib.load(encoder_filename)
tfidf_vectorizers = joblib.load(tfidf_filename)

# Read input data
input_data_path = os.path.join(input_path, file_name)
print(f"Loading input data from {input_data_path}...")
df_input = pd.read_csv(input_data_path)

# Make a copy of the original data for saving the results later
df_original = df_input.copy()

# ==========================================
# 3. Feature Preprocessing
# ==========================================
numerical_features = [
    'cell-con.', 'hemi-con.', 'lig-con.', 'bio-loading', 'tr-time', 'tr-temp',
    'amount-1', 'MW-1', 'Covalent-1', 'XLogP-1', 'Charge-1', 'Complexity-1', 'HB Donor-1', 'HB Acceptor-1', 'Rotatable-1',
    'amount-2', 'MW-2', 'Covalent-2', 'XLogP-2', 'Charge-2', 'Complexity-2', 'HB Donor-2', 'HB Acceptor-2', 'Rotatable-2', 
    'enzy-amount', 'enzy-pH', 'enzy-temp', 'enzy-time', 'enzy-rpm',
]

categorical_features = [
    'biomass-AR', 'biomass-WWC', 'biomass-GV', 'biomass-AMB', 'biomass-FPB', 'biomass-HO', 
    'chemical', 'physical', 
    'phy-microwave', 'phy-extrusion', 'phy-Ultrasonication', 'phy-milling', 'phy-hydrothermal', 'phy-explosion',
    'phy-subW', 'phy-gas', 'phy-radiation', 'phy-UHP', 
    'Ctec', 'HTec', 'Zytex', 'CelluclastTM 1.5L', 'Novo188', 'other'
]

textual_features = ['biomass name', 'chemical-1', 'Canonical Smiles-1', 'chemical-2', 'Canonical Smiles-2']

print("Preprocessing features...")

# 3.1 Numerical features: Standardization (using transform)
X_num = scaler.transform(df_input[numerical_features])

# 3.2 Categorical features: One-Hot Encoding (using transform)
X_cat = encoder.transform(df_input[categorical_features])

# 3.3 Textual features: TF-IDF Encoding (using transform)
X_text_list = []
for feature in textual_features:
    tfidf_vectorizer = tfidf_vectorizers[feature]
    # Convert to dense array for subsequent merging
    X_text_list.append(tfidf_vectorizer.transform(df_input[feature]).toarray())

# ==========================================
# 4. Merge Features & Predict
# ==========================================
# Merge using np.hstack to ensure the data type and order match the training phase exactly
X_new = np.hstack([X_num, X_cat] + X_text_list)

print(f"Feature shape ready for prediction: {X_new.shape}")
print("Predicting yields...")

# Make predictions
y_new_pred = loaded_model.predict(X_new)

# Append the prediction results to the original DataFrame
df_original['Prediction'] = y_new_pred

# ==========================================
# 5. Save Results
# ==========================================
output_path = os.path.join(input_path, output_name)
if os.path.exists(output_path):
    print(f"File {output_path} already exists. Overwriting...")
    os.remove(output_path)
df_original.to_csv(output_path, index=False)

# Sort descending by prediction, extract Top 1000 and Last 1000
df_sorted = df_original.sort_values(by='Prediction', ascending=False)
top_1000 = df_sorted.head(1000)
last_1000 = df_sorted.tail(1000)

output_excel_path = os.path.join(input_path, output_excel_name)
print(f"Saving top and last 1000 results to {output_excel_path}...")
with pd.ExcelWriter(output_excel_path) as writer:
    top_1000.to_excel(writer, sheet_name='Top 1000', index=False)
    last_1000.to_excel(writer, sheet_name='Last 1000', index=False)

print("✅ Prediction completed successfully!")
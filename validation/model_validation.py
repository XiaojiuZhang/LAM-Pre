import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# ==========================================
# 1. Paths & Configurations
# ==========================================
MODEL_DIR = 'model_pkl'
INPUT_DATA_PATH = 'data/validation/validation_input.xlsx'
OUTPUT_DIR = 'results/validation_output'

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

model_filename = os.path.join(MODEL_DIR, 'trained_gbdt_model.pkl')
scaler_filename = os.path.join(MODEL_DIR, 'scaler.pkl')
encoder_filename = os.path.join(MODEL_DIR, 'encoder.pkl')
tfidf_filename = os.path.join(MODEL_DIR, 'tfidf_vectorizers.pkl')

# ==========================================
# 2. Load Models & Data
# ==========================================
print("Loading models and preprocessors...")
loaded_model = joblib.load(model_filename)
scaler = joblib.load(scaler_filename)
encoder = joblib.load(encoder_filename)
tfidf_vectorizers = joblib.load(tfidf_filename)

print(f"Loading validation data from {INPUT_DATA_PATH}...")
df_input = pd.read_excel(INPUT_DATA_PATH)

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

print("Preprocessing validation features...")

# Standardize numerical features
X_num = scaler.transform(df_input[numerical_features])

# One-Hot Encode categorical features
X_cat = encoder.transform(df_input[categorical_features])

# TF-IDF Encode text features
X_text_list = []
for feature in textual_features:
    tfidf_vectorizer = tfidf_vectorizers[feature]
    X_text_list.append(tfidf_vectorizer.transform(df_input[feature]).toarray())

# Merge features using np.hstack
X_new = np.hstack([X_num, X_cat] + X_text_list)

# ==========================================
# 4. Prediction & Evaluation
# ==========================================
print("Predicting yields...")
y_new_pred = loaded_model.predict(X_new)

# Check if ground truth column 'glu-yield' exists and has valid values
has_ground_truth = 'glu-yield' in df_input.columns and not df_input['glu-yield'].isna().all()

if not has_ground_truth:
    print("No ground truth values found. Saving predictions only.")
    output_df = pd.DataFrame({'Prediction': y_new_pred})
    output_path = os.path.join(OUTPUT_DIR, 'validation_prediction_results_GBDT.xlsx')
    output_df.to_excel(output_path, index=False)
    
else:
    print("Ground truth values found. Calculating evaluation metrics...")
    y_true = df_input['glu-yield'].values
    
    # Save predictions alongside true values and difference
    output_df = pd.DataFrame({
        'True Values': y_true,
        'Prediction': y_new_pred,
        'Difference': y_true - y_new_pred
    })
    
    output_path = os.path.join(OUTPUT_DIR, 'validation_prediction_results_GBDT.xlsx')
    output_df.to_excel(output_path, index=False)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_new_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_new_pred)
    r2 = r2_score(y_true, y_new_pred)
    pcc, _ = pearsonr(y_true, y_new_pred)
    
    print("-" * 30)
    print("Validation Metrics:")
    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")
    print(f"PCC:  {pcc:.3f}")
    print("-" * 30)
    
    # Save metrics to a separate file (moved inside the conditional block!)
    output_metrics = pd.DataFrame({
        'RMSE': [f'{rmse:.3f}'],
        'MAE': [f'{mae:.3f}'],
        'R2': [f'{r2:.3f}'],
        'PCC': [f'{pcc:.3f}'],
    })
    metrics_path = os.path.join(OUTPUT_DIR, 'validation_metrics_GBDT.xlsx')
    output_metrics.to_excel(metrics_path, index=False)

print(f"✅ Process completed. Results saved in '{OUTPUT_DIR}' directory.")